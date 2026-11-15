"""Loads the kvq PyTorch extension (JIT via torch.utils.cpp_extension) and offers allocating wrappers.

CPU builds use the C++ reference only; CUDA builds add the kernels (KVQ_WITH_CUDA). FMA contraction is disabled
on both sides so the arithmetic matches the Python reference bit for bit.
"""
from __future__ import annotations
import os, pathlib, torch
from torch.utils.cpp_extension import load

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
_mod = None


def kvq(force_cpu: bool = False, verbose: bool = False, arch: str | None = None):
    global _mod
    if _mod is not None:
        return _mod
    cuda = torch.cuda.is_available() and not force_cpu
    sources = [str(CSRC / "kvq_ref.cpp"), str(CSRC / "kvq_bind.cpp")] + ([str(CSRC / "kvq_kernels.cu")] if cuda else [])
    # torch >= 2.13 headers use C++20 concepts; older torch (the pod image's 2.8) builds with C++17
    std = "-std=c++20" if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 13) else "-std=c++17"
    cflags = ["-O3", std, "-ffp-contract=off"] + (["-DKVQ_WITH_CUDA"] if cuda else [])
    cuda_flags = ["-O3", "--fmad=false", "-lineinfo", std, "-DKVQ_WITH_CUDA"]
    if arch:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    _mod = load(name="kvq_cuda" if cuda else "kvq_cpu", sources=sources, extra_cflags=cflags,
                extra_cuda_cflags=cuda_flags, extra_include_paths=[str(CSRC)], verbose=verbose)
    return _mod


def k_groups(T: int) -> int:
    return (T + 31) // 32


def alloc_k4(k: torch.Tensor):
    L, T, H, D = k.shape
    dev = k.device
    return (torch.empty(L, T, H, D // 2, dtype=torch.uint8, device=dev),
            torch.empty(L, k_groups(T), H, D, dtype=torch.float16, device=dev),
            torch.empty(L, k_groups(T), H, D, dtype=torch.float16, device=dev))


def alloc_v4(v: torch.Tensor):
    L, T, H, D = v.shape
    dev = v.device
    return (torch.empty(L, T, H, D // 2, dtype=torch.uint8, device=dev),
            torch.empty(L, T, H, D // 32, dtype=torch.float16, device=dev),
            torch.empty(L, T, H, D // 32, dtype=torch.float16, device=dev))


def quantize_k4(k: torch.Tensor, version: int = 2):
    p, s, z = alloc_k4(k); kvq().quantize_k4(k, p, s, z, version); return p, s, z


def quantize_v4(v: torch.Tensor, version: int = 2):
    p, s, z = alloc_v4(v); kvq().quantize_v4(v, p, s, z, version); return p, s, z


def quantize_kv4(k: torch.Tensor, v: torch.Tensor, version: int = 2):
    kp, ks, kz = alloc_k4(k); vp, vs, vz = alloc_v4(v)
    kvq().quantize_kv4(k, v, kp, ks, kz, vp, vs, vz, version); return (kp, ks, kz), (vp, vs, vz)


def dequantize_k4(p, s, z, T: int, version: int = 2):
    L, _, H, D = s.shape
    k = torch.empty(L, T, H, D, dtype=torch.bfloat16, device=p.device); kvq().dequantize_k4(p, s, z, k, version); return k


def dequantize_v4(p, s, z, version: int = 2):
    L, T, H, _ = s.shape
    v = torch.empty(L, T, H, 128, dtype=torch.bfloat16, device=p.device); kvq().dequantize_v4(p, s, z, v, version); return v


# ---------------------------------------------------------------------------- one contiguous transfer buffer
def kv4_buffer_layout(L: int, T: int, H: int = 8, D: int = 128, align: int = 16):
    """Byte offsets of [kp | ks | kz | vp | vs | vz] inside one flat uint8 buffer (each section 16-byte aligned)."""
    Tg = k_groups(T)
    sizes = {"kp": L * T * H * D // 2, "ks": L * Tg * H * D * 2, "kz": L * Tg * H * D * 2,
             "vp": L * T * H * D // 2, "vs": L * T * H * (D // 32) * 2, "vz": L * T * H * (D // 32) * 2}
    offs, cur = {}, 0
    for name, sz in sizes.items():
        offs[name] = (cur, sz); cur += (sz + align - 1) // align * align
    return offs, cur


def alloc_kv4_buffer(L: int, T: int, H: int = 8, D: int = 128, device="cuda"):
    """Allocate ONE flat buffer and return (flat, views) where views are the six kernel outputs viewing into it.
    The fused kernel writes straight into this buffer; the transfer sends `flat` as a single contiguous blob."""
    offs, total = kv4_buffer_layout(L, T, H, D)
    flat = torch.empty(total, dtype=torch.uint8, device=device); Tg = k_groups(T)
    def view(name, dtype, shape):
        o, sz = offs[name]; return flat[o:o + sz].view(dtype).view(*shape)
    views = {"kp": view("kp", torch.uint8, (L, T, H, D // 2)), "ks": view("ks", torch.float16, (L, Tg, H, D)), "kz": view("kz", torch.float16, (L, Tg, H, D)),
             "vp": view("vp", torch.uint8, (L, T, H, D // 2)), "vs": view("vs", torch.float16, (L, T, H, D // 32)), "vz": view("vz", torch.float16, (L, T, H, D // 32))}
    return flat, views


def quantize_kv4_into_buffer(k: torch.Tensor, v: torch.Tensor, flat: torch.Tensor, views: dict, version: int = 2):
    kvq().quantize_kv4(k, v, views["kp"], views["ks"], views["kz"], views["vp"], views["vs"], views["vz"], version)
    return flat
