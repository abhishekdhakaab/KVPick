"""Round-trip (compress -> restore) of ONE layer's cache for every format under test, plus the FP8 / INT8 cast kernels
used by the B1 benchmark.  Layer cache layout (HF): k, v = [1, H, T, D] bf16.

  raw        no change (same code path, paired baseline)
  int8_row   one (scale, zero) per (token, head) row, 8-bit            (torch reference path, 1.94x)
  fp8        E4M3 with ONE scale per tensor per layer = amax/448, computed from the prompt cache itself (2x)  [D-ext]
  kv4        keys per channel / 32 tokens, values per token / 32 dims, 4-bit (shipped kernel on CUDA; reference on CPU) (3.2x)
  int4_g32   keys and values both per row, groups of 32, 4-bit         (3.2x)
"""
from __future__ import annotations
import torch
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import roundtrip_wire

FP8_MAX = 448.0
SPECS = {
    "int8_row": (QuantSpec(8, "row", None), QuantSpec(8, "row", None)),
    "int4_g32": (QuantSpec(4, "row", 32), QuantSpec(4, "row", 32)),
    "kv4": (QuantSpec(4, "chan", 32), QuantSpec(4, "row", 32)),
}
FORMATS = ("raw", "int8_row", "fp8", "kv4", "int4_g32")
_kernel_checked = {"done": False}


# ---- FP8 per-tensor (the "just use FP8" baseline). compress = amax reduction + scale + cast; decompress = cast * scale
def fp8_compress(x: torch.Tensor):
    amax = x.float().abs().amax().clamp_min(1e-12)
    scale = amax / FP8_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_decompress(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16):
    return (q.to(torch.float32) * scale).to(dtype)


def fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    q, s = fp8_compress(x); return fp8_decompress(q, s, x.dtype)


def _to_tld(x):                                    # [1,H,T,D] -> [1,T,H,D] contiguous
    return x.permute(0, 2, 1, 3).contiguous()


def _from_tld(x):
    return x.permute(0, 2, 1, 3).contiguous()


def kv4_roundtrip_kernel(xk: torch.Tensor, xv: torch.Tensor, check: bool = True):
    from kvpick import ext
    (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(xk, xv, version=2)
    yk = ext.dequantize_k4(kp, ks, kz, xk.shape[1], version=2); yv = ext.dequantize_v4(vp, vs, vz, version=2)
    if check and not _kernel_checked["done"]:                      # once per process: kernel path == reference path, bit for bit
        rk, rv = SPECS["kv4"]
        assert torch.equal(yk, roundtrip_wire(xk, rk)) and torch.equal(yv, roundtrip_wire(xv, rv)), "kernel != reference"
        _kernel_checked["done"] = True
    return yk, yv


def roundtrip_layer(k: torch.Tensor, v: torch.Tensor, fmt: str, use_kernel: bool = True):
    """Compress and restore one layer's K and V. Returns new tensors (same shape/dtype) or the inputs for raw."""
    if fmt == "raw":
        return k, v
    if fmt == "fp8":
        return fp8_roundtrip(k), fp8_roundtrip(v)
    xk, xv = _to_tld(k), _to_tld(v)                               # [1, T, H, D]
    if fmt == "kv4" and use_kernel and k.is_cuda:
        yk, yv = kv4_roundtrip_kernel(xk, xv)
    else:
        sk, sv = SPECS[fmt]
        yk, yv = roundtrip_wire(xk, sk, out_dtype=k.dtype), roundtrip_wire(xv, sv, out_dtype=v.dtype)
    return _from_tld(yk), _from_tld(yv)


def iter_roundtrip_layers(layers, fmt: str, device, use_kernel: bool = True, starts=None):
    """Generator: yields restored (k, v) ON `device`, one layer at a time (so a caller that moves each layer into a cache and
    drops it keeps only ONE extra layer in memory; HF's cache copies on update, so a full list would double the cache)."""
    for k, v in layers:
        k, v = k.to(device, non_blocking=True), v.to(device, non_blocking=True)
        if fmt == "raw":
            yield k, v; continue                                # DynamicCache.update concatenates (new tensors); the source is never mutated
        if starts is None or all(int(s) == 0 for s in starts):
            ks, vs = [], []
            for b in range(k.shape[0]):
                a, c = roundtrip_layer(k[b:b + 1], v[b:b + 1], fmt, use_kernel); ks.append(a); vs.append(c)
            yield torch.cat(ks), torch.cat(vs)
        else:
            k2, v2 = k.clone(), v.clone()
            for b in range(k.shape[0]):
                s = int(starts[b]); a, c = roundtrip_layer(k[b:b + 1, :, s:], v[b:b + 1, :, s:], fmt, use_kernel)
                k2[b:b + 1, :, s:], v2[b:b + 1, :, s:] = a, c
            yield k2, v2


def roundtrip_cache_layers(layers, fmt: str, device, use_kernel: bool = True, starts=None):
    """List form of iter_roundtrip_layers (fine for short contexts and tests)."""
    return list(iter_roundtrip_layers(layers, fmt, device, use_kernel, starts))
