import torch, pytest
from kvpick import ext
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire


def test_layout_sizes_match_config():
    offs, total = ext.kv4_buffer_layout(36, 2048)
    assert total / (36 * 2048) == pytest.approx(1280, rel=0.01)          # bytes per token per layer incl. alignment padding (<1%)
    for name, (o, sz) in offs.items():
        assert o % 16 == 0


def test_buffer_views_receive_cpu_reference_output():
    x = torch.randn(2, 70, 8, 128).to(torch.bfloat16)
    flat, v = ext.alloc_kv4_buffer(2, 70, device="cpu")
    ext.quantize_kv4_into_buffer(x, x, flat, v)
    rp, rs, rz = quantize_wire(x, QuantSpec(4, "chan", 32)); assert torch.equal(v["kp"], rp) and torch.equal(v["ks"], rs) and torch.equal(v["kz"], rz)
    rp, rs, rz = quantize_wire(x, QuantSpec(4, "row", 32)); assert torch.equal(v["vp"], rp) and torch.equal(v["vs"], rs) and torch.equal(v["vz"], rz)
    # the flat blob is exactly what a receiver would reinterpret: re-view it and dequantize
    flat2 = flat.clone(); _, v2 = ext.alloc_kv4_buffer(2, 70, device="cpu"); offs, _ = ext.kv4_buffer_layout(2, 70)
    k = ext.dequantize_k4(flat2[offs["kp"][0]:offs["kp"][0] + offs["kp"][1]].view(2, 70, 8, 64), v["ks"], v["kz"], 70)
    assert k.shape == x.shape
