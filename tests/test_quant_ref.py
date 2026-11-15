"""Unit tests for quant_ref.py (CPU). Run: pytest -q test_quant_ref.py"""
import math, torch, pytest
from kvpick.ref.quant_ref import QuantSpec, VARIANTS, quantize, dequantize, roundtrip, bytes_per_token, _to_groups

torch.manual_seed(0)
SPECS = [QuantSpec(bits=b, symmetric=s, **kw) for kw in VARIANTS.values() for b in (8, 4) for s in (False, True)]


def group_bound(x, spec):
    """Per-element allowed error: scale/2 (+ fp16 rounding slack on the zero point) computed from the same groups."""
    xg, pad = _to_groups(x.float(), spec)
    if spec.symmetric:
        scale = (xg.abs().amax(-1, keepdim=True) * (1.0 / spec.qmax)).clamp_min(1e-8).half().float()
        slack = 0
    else:
        mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        scale = ((mx - mn.half().float()) * (1.0 / spec.qmax)).clamp_min(1e-8).half().float()
        slack = (mn.abs() + mx.abs()) * 2 ** -10          # fp16 rounding of zero point / scale
    bound = (0.5 * scale + slack + 1e-6).expand_as(xg)
    from kvpick.ref.quant_ref import _from_groups
    return _from_groups(bound, spec, tuple(x.shape), pad)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_roundtrip_bounded_random(spec):
    x = torch.randn(97, 8, 128) * 3        # T=97 is not a multiple of 32 -> exercises chan padding
    err = (roundtrip(x, spec) - x).abs()
    assert (err <= group_bound(x, spec) * (1 + 2 ** -10)).all(), float((err - group_bound(x, spec)).max())


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_edge_rows(spec):
    """Edge cases laid out as 32-token blocks so that they are whole groups for BOTH row and chan variants."""
    T, H, D = 160, 8, 128
    x = torch.randn(T, H, D)
    x[0:32] = 0.0                                   # zero block
    x[32:64] = 0.75                                 # constant block
    x[64:96] = -torch.rand(32, H, D) - 0.1          # negative-only block
    x[96, 0, 5] = 1000.0                            # one huge outlier
    y = roundtrip(x, spec)
    assert torch.isfinite(y).all()
    whole_group = spec.axis == "row" or spec.group == 32     # chan_full groups span all T, so blocks are not whole groups
    if whole_group:
        assert torch.equal(y[0:32], torch.zeros(32, H, D))                        # zeros round-trip exactly
        assert torch.allclose(y[32:64], torch.full((32, H, D), 0.75), atol=1e-3)   # constants round-trip (fp16 zero point)
        assert (y[64:96] <= 0).all()                                              # sign preserved
    assert abs(float(y[96, 0, 5]) - 1000.0) <= 1000.0 * 2 ** -9              # outlier itself is representable
    err = (y - x).abs()
    assert (err <= group_bound(x, spec) * (1 + 2 ** -10)).all()


def test_nan_rejected():
    x = torch.randn(8, 8, 128); x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        quantize(x, QuantSpec(8, "row", None))


def test_bf16_input_and_T1():
    x = torch.randn(1, 8, 128).to(torch.bfloat16)
    for spec in SPECS:
        y = roundtrip(x, spec)
        assert y.shape == x.shape and torch.isfinite(y).all()


def test_outlier_channel_hurts_row_but_not_chan():
    """The KIVI argument: a channel that is 50x larger ruins per-row 4-bit codes but not per-channel codes."""
    x = torch.randn(256, 8, 128); x[:, :, 7] *= 50
    row = QuantSpec(4, "row", None); chan = QuantSpec(4, "chan", 32)
    mask = torch.ones(128, dtype=torch.bool); mask[7] = False
    e_row = (roundtrip(x, row) - x)[..., mask].norm() / x[..., mask].norm()
    e_chan = (roundtrip(x, chan) - x)[..., mask].norm() / x[..., mask].norm()
    assert e_chan < 0.5 * e_row, (float(e_row), float(e_chan))


def test_bytes_table():
    # spec section 5: INT8 per-row symmetric with fp32 scale = 132 B/row -> here fp16 scale -> 130 B/row -> 1040 per K per token (8 heads)
    b = bytes_per_token(QuantSpec(8, "row", None, symmetric=True))
    assert b["payload"] == 1024 and b["meta"] == 16
    b4 = bytes_per_token(QuantSpec(4, "row", 32, symmetric=True))
    assert b4["payload"] == 512 and b4["meta"] == 8 * 4 * 2
    bc = bytes_per_token(QuantSpec(4, "chan", 32), T=2048)
    assert bc["payload"] == 512 and math.isclose(bc["meta"], 8 * 128 * 4 / 32)


def test_hand_computed_int4_group():
    """One 32-value group, asymmetric 4-bit, worked by hand: values 0..31 -> min 0, max 31, scale 31/15."""
    x = torch.zeros(1, 1, 128); x[0, 0, :32] = torch.arange(32.0)
    q = quantize(x, QuantSpec(4, "row", 32))
    scale = (torch.tensor(31.0) * (1.0 / 15)).half().float()
    expected = torch.round(torch.arange(32.0) * (1.0 / scale)).clamp(0, 15).to(torch.uint8)
    assert torch.equal(q.codes[0, 0, :32], expected)
    assert float(q.scale[0, 0, 0]) == pytest.approx(float(scale))
    assert float(q.zero[0, 0, 0]) == 0.0


def test_constant_group_codes_finite_after_eps_fix():
    """D21: EPS must survive fp16 rounding so constant groups give code 0, not NaN-derived garbage."""
    from kvpick.ref.quant_ref import EPS
    assert float(torch.tensor(EPS).half()) > 0
    x = torch.full((64, 8, 128), 0.5)
    for spec in SPECS:
        q = quantize(x, spec)
        assert torch.isfinite(dequantize(q)).all()
        if not spec.symmetric:                      # asymmetric: zero point absorbs the constant -> code 0, exact
            assert int(q.codes.max()) == 0 and torch.equal(roundtrip(x, spec), x)
