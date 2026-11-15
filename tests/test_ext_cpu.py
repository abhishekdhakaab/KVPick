"""C++ CPU reference must produce the same bytes as the Python reference (quant_wire) and invert exactly."""
import torch, pytest
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire, dequantize_wire
from kvpick import ext

K_SPEC = QuantSpec(4, "chan", 32); V_SPEC = QuantSpec(4, "row", 32)


@pytest.fixture(scope="module")
def mod():
    return ext.kvq(force_cpu=True)


def cases():
    g = torch.Generator().manual_seed(0)
    yield "normal_T97", (torch.randn(2, 97, 8, 128, generator=g) * 2).to(torch.bfloat16)
    yield "T1", torch.randn(1, 1, 8, 128, generator=g).to(torch.bfloat16)
    yield "T64", torch.randn(1, 64, 8, 128, generator=g).to(torch.bfloat16)
    x = torch.randn(1, 160, 8, 128, generator=g); x[:, 0:32] = 0; x[:, 32:64] = 0.75; x[:, 64:96] = -x[:, 64:96].abs() - 0.1; x[0, 96, 0, 5] = 1000.0
    yield "edges", x.to(torch.bfloat16)
    x = torch.randn(1, 256, 8, 128, generator=g); m = torch.rand(x.shape, generator=g) < 0.001; x[m] *= 50
    yield "outliers", x.to(torch.bfloat16)
    x = torch.randn(1, 64, 8, 128, generator=g); x[..., 7] = 20 + 2 * x[..., 7]
    yield "offset_channel", x.to(torch.bfloat16)


@pytest.mark.parametrize("name,x", list(cases()), ids=[c[0] for c in cases()])
def test_k4_matches_python(mod, name, x):
    p, s, z = ext.quantize_k4(x)
    rp, rs, rz = quantize_wire(x, K_SPEC)
    assert torch.equal(p, rp), f"K payload mismatch: {(p != rp).float().mean():.4f}"
    assert torch.equal(s, rs) and torch.equal(z, rz)
    k = ext.dequantize_k4(p, s, z, x.shape[1])
    assert torch.equal(k, dequantize_wire(rp, rs, rz, K_SPEC, x.shape[1]))


@pytest.mark.parametrize("name,x", list(cases()), ids=[c[0] for c in cases()])
def test_v4_matches_python(mod, name, x):
    p, s, z = ext.quantize_v4(x)
    rp, rs, rz = quantize_wire(x, V_SPEC)
    assert torch.equal(p, rp), f"V payload mismatch: {(p != rp).float().mean():.4f}"
    assert torch.equal(s, rs) and torch.equal(z, rz)
    v = ext.dequantize_v4(p, s, z)
    assert torch.equal(v, dequantize_wire(rp, rs, rz, V_SPEC, x.shape[1]))


def test_fused_equals_separate(mod):
    g = torch.Generator().manual_seed(1)
    k = torch.randn(3, 100, 8, 128, generator=g).to(torch.bfloat16); v = torch.randn(3, 100, 8, 128, generator=g).to(torch.bfloat16)
    (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(k, v)
    assert all(torch.equal(a, b) for a, b in zip((kp, ks, kz), ext.quantize_k4(k)))
    assert all(torch.equal(a, b) for a, b in zip((vp, vs, vz), ext.quantize_v4(v)))


def test_constant_group_well_defined(mod):
    """D21: all-constant groups must give finite codes (0) and exact reconstruction."""
    x = torch.full((1, 32, 8, 128), 0.5, dtype=torch.bfloat16)
    p, s, z = ext.quantize_k4(x)
    assert int(p.max()) == 0 and torch.equal(ext.dequantize_k4(p, s, z, 32), x)
    p, s, z = ext.quantize_v4(x)
    assert int(p.max()) == 0 and torch.equal(ext.dequantize_v4(p, s, z), x)


def test_shape_checks(mod):
    x = torch.randn(1, 8, 8, 128).to(torch.bfloat16)
    p, s, z = ext.alloc_k4(x)
    with pytest.raises(RuntimeError):
        mod.quantize_k4(x.float(), p, s, z, 2)             # wrong dtype
    with pytest.raises(RuntimeError):
        mod.quantize_k4(x.transpose(1, 2), p, s, z, 2)      # non-contiguous
