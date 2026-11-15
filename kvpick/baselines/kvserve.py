"""JIT loader for the KVServe-kernel benchmark harness (csrc/baselines/kvserve_kernels.cu).
CUDA-only (KVServe ships no CPU reference for these kernels). See that file's header and
DECISIONS.md D36 for what is and is not a fair comparison here."""
from __future__ import annotations
import pathlib
import torch
from torch.utils.cpp_extension import load

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "baselines"
_mod = None


def kvserve_bench(verbose: bool = False):
    global _mod
    if _mod is not None:
        return _mod
    assert torch.cuda.is_available(), "KVServe's kernels are CUDA-only"
    _mod = load(name="kvserve_bench", sources=[str(CSRC / "kvserve_kernels.cu")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3", "-std=c++17"],
                verbose=verbose)
    return _mod


def quantize_channel(x: torch.Tensor, levels: int = 16):
    """x: bf16 CUDA [T, H, D]. One (min, scale) per (head, dim) over the WHOLE T axis
    (no 32-token grouping -- see csrc/baselines/kvserve_kernels.cu header)."""
    return kvserve_bench().quantize_channel(x, levels)


def quantize_token(x: torch.Tensor, levels: int = 16):
    """x: bf16 CUDA [T, H, D]. One (min, scale) per token over ALL heads/dims
    (no 32-dim grouping -- see csrc/baselines/kvserve_kernels.cu header)."""
    return kvserve_bench().quantize_token(x, levels)


def dequantize_channel(payload, aux_min, aux_scale):
    return kvserve_bench().dequantize_channel(payload, aux_min, aux_scale)


def dequantize_token(payload, aux_min, aux_scale):
    return kvserve_bench().dequantize_token(payload, aux_min, aux_scale)
