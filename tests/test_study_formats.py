import torch, pytest
from kvpick.study import formats as F
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import roundtrip_wire


def rnd(T=70, H=8, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, H, T, D, generator=g) * 2).to(torch.bfloat16), (torch.randn(1, H, T, D, generator=g)).to(torch.bfloat16)


def rel(a, b): return float((a.float() - b.float()).norm() / a.float().norm())


def test_raw_is_identity():
    k, v = rnd(); k2, v2 = F.roundtrip_layer(k, v, "raw"); assert torch.equal(k, k2) and torch.equal(v, v2)


@pytest.mark.parametrize("fmt,bound", [("int8_row", 0.01), ("fp8", 0.06), ("kv4", 0.15), ("int4_g32", 0.15)])
def test_roundtrip_error_bounds_and_shape(fmt, bound):
    k, v = rnd(); k2, v2 = F.roundtrip_layer(k, v, fmt, use_kernel=False)
    assert k2.shape == k.shape and v2.dtype == torch.bfloat16
    assert rel(k, k2) < bound and rel(v, v2) < bound and rel(k, k2) > 0


def test_fp8_scale_is_per_tensor_and_uses_range():
    x = torch.randn(1, 8, 64, 128).to(torch.bfloat16) * 3
    q, s = F.fp8_compress(x)
    assert q.dtype == torch.float8_e4m3fn and s.numel() == 1
    assert float(s) == pytest.approx(float(x.float().abs().max()) / 448, rel=1e-6)
    assert float(F.fp8_decompress(q, s, torch.bfloat16).float().abs().max()) <= float(x.float().abs().max()) * 1.07


def test_tail_group_not_multiple_of_32_roundtrips_without_touching_other_tokens():
    k, v = rnd(T=37)                                           # one full group + 5 tail tokens
    k2, _ = F.roundtrip_layer(k, v, "kv4", use_kernel=False)
    assert rel(k[:, :, 32:], k2[:, :, 32:]) < 0.2             # tail group is quantized with its own stats, not dropped


def test_layers_with_left_padding_leave_pad_region_untouched():
    k, v = rnd(T=40); k = torch.cat([k, k]); v = torch.cat([v, v])
    out = F.roundtrip_cache_layers([(k, v)], "int8_row", "cpu", starts=[0, 8])
    k2, _ = out[0]
    assert torch.equal(k2[1, :, :8], k[1, :, :8]) and not torch.equal(k2[1, :, 8:], k[1, :, 8:])
    assert torch.equal(F.roundtrip_cache_layers([(k, v)], "int8_row", "cpu")[0][0][0], F.roundtrip_cache_layers([(k, v)], "int8_row", "cpu", starts=[0, 8])[0][0][0])


def test_int8_matches_reference_wire():
    k, v = rnd(); k2, _ = F.roundtrip_layer(k, v, "int8_row")
    ref = roundtrip_wire(k.permute(0, 2, 1, 3).contiguous(), QuantSpec(8, "row", None)).permute(0, 2, 1, 3)
    assert torch.equal(k2, ref)
