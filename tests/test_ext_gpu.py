"""CUDA kernels (v0, v1, v2) must be bit-identical to the Python reference and to the C++ CPU reference."""
import torch, pytest
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire, dequantize_wire
from kvpick import ext

pytestmark = pytest.mark.gpu
K_SPEC = QuantSpec(4, "chan", 32); V_SPEC = QuantSpec(4, "row", 32)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@pytest.fixture(scope="module")
def mod():
    m = ext.kvq(); assert m.with_cuda; return m


def cases():
    g = torch.Generator().manual_seed(0)
    for name, shape in [("T1", (1, 1, 8, 128)), ("T7", (1, 7, 8, 128)), ("T97", (2, 97, 8, 128)), ("T2048", (2, 2048, 8, 128)), ("H6", (1, 40, 6, 128)), ("H1", (1, 33, 1, 128))]:
        yield name, (torch.randn(*shape, generator=g) * 2).to(torch.bfloat16)
    x = torch.randn(1, 160, 8, 128, generator=g); x[:, 0:32] = 0; x[:, 32:64] = 0.75; x[:, 64:96] = -x[:, 64:96].abs() - 0.1; x[0, 96, 0, 5] = 1000.0
    yield "edges", x.to(torch.bfloat16)
    x = torch.randn(1, 256, 8, 128, generator=g); m = torch.rand(x.shape, generator=g) < 0.001; x[m] *= 50
    yield "outliers", x.to(torch.bfloat16)
    x = torch.randn(1, 64, 8, 128, generator=g); x[..., 7] = 20 + 2 * x[..., 7]
    yield "offset_channel", x.to(torch.bfloat16)


CASES = list(cases())


@needs_cuda
@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("name,x", CASES, ids=[c[0] for c in CASES])
def test_quant_kv4_bitexact(mod, version, name, x):
    xc = x.cuda()
    (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(xc, xc, version=version)
    rp, rs, rz = quantize_wire(xc, K_SPEC)
    assert torch.equal(kp, rp), f"K payload mismatch frac {(kp != rp).float().mean():.5f}"
    assert torch.equal(ks, rs) and torch.equal(kz, rz)
    rp, rs, rz = quantize_wire(xc, V_SPEC)
    assert torch.equal(vp, rp), f"V payload mismatch frac {(vp != rp).float().mean():.5f}"
    assert torch.equal(vs, rs) and torch.equal(vz, rz)
    # also equal to the C++ CPU reference bytes
    cp, cs, cz = ext.quantize_k4(x)
    assert torch.equal(kp.cpu(), cp) and torch.equal(ks.cpu(), cs) and torch.equal(kz.cpu(), cz)


@needs_cuda
@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("name,x", CASES, ids=[c[0] for c in CASES])
def test_dequant_kv4_bitexact(mod, version, name, x):
    xc = x.cuda(); T = x.shape[1]
    p, s, z = quantize_wire(xc, K_SPEC)
    assert torch.equal(ext.dequantize_k4(p, s, z, T, version=version), dequantize_wire(p, s, z, K_SPEC, T))
    p, s, z = quantize_wire(xc, V_SPEC)
    assert torch.equal(ext.dequantize_v4(p, s, z, version=version), dequantize_wire(p, s, z, V_SPEC, T))


@needs_cuda
def test_deterministic_and_input_unchanged(mod):
    x = torch.randn(4, 300, 8, 128, device="cuda").to(torch.bfloat16); x0 = x.clone()
    a = ext.quantize_kv4(x, x, version=2); b = ext.quantize_kv4(x, x, version=2)
    assert all(torch.equal(p, q) for grp_a, grp_b in zip(a, b) for p, q in zip(grp_a, grp_b))
    assert torch.equal(x, x0)


@needs_cuda
def test_sentinel_fully_overwritten(mod):
    x = torch.randn(2, 100, 8, 128, device="cuda").to(torch.bfloat16)
    outs = []
    for sent in (0xA5, 0x5A):
        kp, ks, kz = ext.alloc_k4(x); vp, vs, vz = ext.alloc_v4(x)
        for t in (kp, vp): t.fill_(sent)
        for t in (ks, kz, vs, vz): t.view(torch.uint8).fill_(sent)
        mod.quantize_kv4(x, x, kp, ks, kz, vp, vs, vz, 2); outs.append([kp.clone(), ks.clone(), kz.clone(), vp.clone(), vs.clone(), vz.clone()])
    assert all(torch.equal(a, b) for a, b in zip(*outs))


@needs_cuda
def test_non_default_stream(mod):
    x = torch.randn(2, 500, 8, 128, device="cuda").to(torch.bfloat16)
    ref = ext.quantize_kv4(x, x, version=2)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        junk = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")   # work in flight
        out = ext.quantize_kv4(x, x, version=2)
    torch.cuda.synchronize()
    assert all(torch.equal(p, q) for ga, gb in zip(ref, out) for p, q in zip(ga, gb))


@needs_cuda
def test_rejects_bad_inputs(mod):
    x = torch.randn(1, 8, 8, 128, device="cuda").to(torch.bfloat16); p, s, z = ext.alloc_k4(x)
    with pytest.raises(RuntimeError):
        mod.quantize_k4(x.float(), p, s, z, 2)
    with pytest.raises(RuntimeError):
        mod.quantize_k4(x.transpose(1, 2), p, s, z, 2)
    with pytest.raises(RuntimeError):
        mod.quantize_k4(x, p[:, :4].contiguous(), s, z, 2)
