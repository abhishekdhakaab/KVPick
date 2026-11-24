"""Part A4: the existing kernel tests on the new layer counts (32 = Llama-3.1-8B, 80 = Llama-3.1-70B) and the per-rank shape of
8-way tensor parallel 70B (1 KV head), synthetic data. CPU: C++ reference vs Python reference. GPU: CUDA v0/v1/v2 vs Python reference."""
import pytest, torch
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire, dequantize_wire
from kvpick import ext

K_SPEC, V_SPEC = QuantSpec(4, "chan", 32), QuantSpec(4, "row", 32)
SHAPES = [(32, 40, 8), (80, 33, 8), (80, 64, 1), (36, 97, 8), (32, 1, 8)]   # (L, T, H); T not a multiple of 32 in most cases


def data(L, T, H, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randn(L, T, H, 128, generator=g) * 2
    x[..., 3] *= 12                                                           # an outlier channel, like real keys
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("L,T,H", SHAPES)
def test_cpu_reference_matches_python_on_layer_counts(L, T, H):
    mod = ext.kvq(force_cpu=True); x = data(L, T, H)
    p, s, z = ext.quantize_k4(x); rp, rs, rz = quantize_wire(x, K_SPEC)
    assert torch.equal(p, rp) and torch.equal(s, rs) and torch.equal(z, rz)
    p, s, z = ext.quantize_v4(x); rp, rs, rz = quantize_wire(x, V_SPEC)
    assert torch.equal(p, rp) and torch.equal(s, rs) and torch.equal(z, rz)
    assert torch.equal(ext.dequantize_k4(*ext.quantize_k4(x), T), dequantize_wire(*quantize_wire(x, K_SPEC), K_SPEC, T))


def test_buffer_layout_sizes_for_new_models():
    for L, raw, kv4 in ((32, 131072, 40960), (80, 327680, 102400)):
        offs, total = ext.kv4_buffer_layout(L, 1024, 8)
        assert total >= kv4 * 1024 and total <= kv4 * 1024 + 6 * 16                # exact up to 16-byte section alignment
        assert L * 2 * 8 * 128 * 2 == raw


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("L,T,H", SHAPES)
def test_cuda_matches_python_on_layer_counts(L, T, H, version):
    mod = ext.kvq(); x = data(L, T, H).cuda(); y = data(L, T, H, seed=1).cuda()
    (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(x, y, version=version)
    rk, rv = quantize_wire(x, K_SPEC), quantize_wire(y, V_SPEC)
    assert torch.equal(kp, rk[0]) and torch.equal(ks, rk[1]) and torch.equal(kz, rk[2])
    assert torch.equal(vp, rv[0]) and torch.equal(vs, rv[1]) and torch.equal(vz, rv[2])
    assert torch.equal(ext.dequantize_k4(kp, ks, kz, T, version=version), dequantize_wire(*rk, K_SPEC, T))
    assert torch.equal(ext.dequantize_v4(vp, vs, vz, version=version), dequantize_wire(*rv, V_SPEC, T))
