"""Thin wrapper around KIVI's own pack kernels (github.com/jy-yuan/KIVI, MIT), cloned
read-only at feasibility/external/kivi (see feasibility/results/C2_kivi.md). Not vendored:
imported directly from the clone, unmodified, for benchmarking only (DECISIONS.md D36).

KIVI ships no standalone quantize *kernel* for the pack step (see C2_kivi.md: its CUDA
code is a fused dequant+GEMV for decode attention). `quant_and_pack_kcache` is plain
PyTorch; `quant_and_pack_vcache`'s packing has a pure-PyTorch and a Triton path -- this
wraps both so the comparison can show either.

KIVI's own tensor layout is [B, nh, T, D]; KVPick/KVServe benchmarks use [T, H, D].
These wrappers do the permute so callers never see KIVI's layout.
"""
from __future__ import annotations
import pathlib
import sys
import torch

KIVI_ROOT = pathlib.Path(__file__).resolve().parents[2] / "feasibility" / "external" / "kivi"
if str(KIVI_ROOT) not in sys.path:
    sys.path.insert(0, str(KIVI_ROOT))

from quant.new_pack import (  # noqa: E402
    quant_and_pack_kcache,
    quant_and_pack_vcache,
    unpack_and_dequant_kcache,
    unpack_and_dequant_vcache,
)


def _to_bnhd(x: torch.Tensor) -> torch.Tensor:
    """[T, H, D] -> [1, H, T, D] (KIVI's layout, batch size 1)."""
    return x.permute(1, 0, 2).unsqueeze(0).contiguous()


def quant_and_pack_k(k_thd: torch.Tensor, group_size: int = 32, bits: int = 4):
    """k_thd: [T, H, D]. Per-channel over token groups, KIVI's native scheme (matches
    KVPick's K axis choice; different group packing: 32/bits int32 words vs 2 codes/byte)."""
    return quant_and_pack_kcache(_to_bnhd(k_thd), group_size, bits)


def quant_and_pack_v(v_thd: torch.Tensor, group_size: int = 32, bits: int = 4):
    """v_thd: [T, H, D]. Per-token over dim groups (matches KVPick's V axis choice).
    KIVI also ships a Triton-accelerated packer (`triton_quantize_and_pack_along_last_dim`),
    but it groups over the TOKEN axis of a [B, nh, D, T] tensor -- i.e. it is their
    Triton path for the K-style (per-channel) packing, not a V-axis kernel -- so it is
    not wired in here; `quant_and_pack_vcache` (plain PyTorch) is KIVI's actual V packer."""
    return quant_and_pack_vcache(_to_bnhd(v_thd), group_size, bits)


def unpack_and_dequant_k(code, scale, mn, group_size: int = 32, bits: int = 4):
    return unpack_and_dequant_kcache(code, scale, mn, group_size, bits)


def unpack_and_dequant_v(code, scale, mn, group_size: int = 32, bits: int = 4):
    return unpack_and_dequant_vcache(code, scale, mn, group_size, bits)
