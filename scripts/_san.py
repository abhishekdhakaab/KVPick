import torch
from kvpick import ext
m = ext.kvq()
for shape in [(1, 97, 6, 128), (1, 1, 8, 128), (2, 40, 8, 128)]:
    x = torch.randn(*shape, device="cuda").to(torch.bfloat16)
    for ver in (0, 1, 2):
        (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(x, x, version=ver)
        ext.dequantize_k4(kp, ks, kz, shape[1], version=ver); ext.dequantize_v4(vp, vs, vz, version=ver)
torch.cuda.synchronize(); print("sanitizer workload done")
