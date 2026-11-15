import torch, pytest
from quant_ref import QuantSpec, quantize, dequantize, VARIANTS
from quant_wire import quantize_wire, dequantize_wire

SPECS = [QuantSpec(b, **kw) for kw in VARIANTS.values() for b in (8, 4)]

@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
@pytest.mark.parametrize("T", [1, 97, 256])
def test_wire_matches_ref(spec, T):
    x = torch.randn(3, T, 8, 128).to(torch.bfloat16)
    p, s, z = quantize_wire(x, spec)
    for l in range(3):
        q = quantize(x[l], spec)
        codes = torch.stack([p[l] & 0xF, p[l] >> 4], -1).reshape(T, 8, 128) if spec.bits == 4 else p[l]
        assert torch.equal(codes, q.codes), spec.name
        assert torch.equal(s[l], q.scale) and torch.equal(z[l], q.zero)
        dq_ref = dequantize(q)
        dq_wire = dequantize_wire(p, s, z, spec, T, out_dtype=torch.float32)[l]
        assert torch.allclose(dq_ref, dq_wire, atol=1e-6, rtol=1e-6)

def test_bytes_4bit_row_g32_full_cache():
    x = torch.randn(36, 2048, 8, 128).to(torch.bfloat16)
    p, s, z = quantize_wire(x, QuantSpec(4, "row", 32))
    per_token_per_layer = (p.numel() + 2 * s.numel() + 2 * z.numel()) / (36 * 2048)
    assert per_token_per_layer == 512 + 32 * 4          # payload + 32 groups x (scale+zero)
