"""Slow, obvious PyTorch reference quantize/dequantize for KV tensors.

Tensor layout everywhere: x[T, H, D]  (tokens, kv heads, head_dim).  A "row" is one (t, h) vector of D values.

Variants (G2 of the plan):
  axis="row",  group=None : one (scale, zero) per row                       -> spec's per-row design
  axis="row",  group=32   : groups of 32 along D within a row               -> per-row g32
  axis="chan", group=32   : groups of 32 consecutive tokens for fixed (h, d) -> KIVI-style per-channel keys
  axis="chan", group=None : whole sequence per (h, d)                        -> KVServe "channel" axis

Scheme: asymmetric min/max with zero point (default) or symmetric amax (--symmetric).
Code = round((x - zero) * (1/scale)): the reciprocal is computed once per group so the kernel does one multiply per element.
Scales and zero points are rounded to fp16 (what goes on the wire); codes are kept unpacked as uint8/int8
for clarity; byte counts are computed analytically by `bytes_per_token`.
All arithmetic in float32. Readability over speed. No in-place tricks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

EPS = 2.0 ** -24   # smallest fp16 subnormal: survives the fp16 rounding of the scale (D21)
WIRE_DTYPE = torch.float16   # dtype of scale / zero on the wire

VARIANTS = {
    "row":       dict(axis="row", group=None),
    "row_g32":   dict(axis="row", group=32),
    "chan_g32":  dict(axis="chan", group=32),
    "chan_full": dict(axis="chan", group=None),
}


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    axis: str            # "row" | "chan"
    group: int | None    # group size along that axis; None = whole axis
    symmetric: bool = False

    @property
    def qmax(self) -> int:
        return (2 ** (self.bits - 1) - 1) if self.symmetric else (2 ** self.bits - 1)

    @property
    def name(self) -> str:
        g = "full" if self.group is None else f"g{self.group}"
        return f"{self.axis}_{g}_{'sym' if self.symmetric else 'asym'}_{self.bits}b"


@dataclass
class Quantized:
    spec: QuantSpec
    codes: torch.Tensor      # [T, H, D] integer codes (uint8 for asym, int8 for sym)
    scale: torch.Tensor      # grouped shape, fp16
    zero: torch.Tensor | None  # grouped shape, fp16 (asym only; stores the group minimum)
    shape: tuple           # original (T, H, D)


def _to_groups(x: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, int]:
    """Return x reshaped to [..., G, g] where the last dim is the reduction group, plus padding added."""
    T, H, D = x.shape
    if spec.axis == "row":
        g = D if spec.group is None else spec.group
        assert D % g == 0, "head_dim must be a multiple of the group size"
        return x.reshape(T, H, D // g, g), 0                       # groups along D
    elif spec.axis == "chan":
        g = T if spec.group is None else spec.group
        pad = (-T) % g
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, H, D)], dim=0)   # pad by repeating the last token (does not widen the range)
        Tp = T + pad
        # [Tp, H, D] -> [Tp/g, g, H, D] -> [Tp/g, H, D, g]
        return x.reshape(Tp // g, g, H, D).permute(0, 2, 3, 1), pad
    raise ValueError(spec.axis)


def _from_groups(xg: torch.Tensor, spec: QuantSpec, shape: tuple, pad: int) -> torch.Tensor:
    T, H, D = shape
    if spec.axis == "row":
        return xg.reshape(T, H, D)
    x = xg.permute(0, 3, 1, 2).reshape(T + pad, H, D)
    return x[:T]


def quantize(x: torch.Tensor, spec: QuantSpec) -> Quantized:
    assert x.dim() == 3, "expected [T, H, D]"
    if not torch.isfinite(x).all():
        raise ValueError("input contains NaN or Inf; refusing to quantize silently")
    shape = tuple(x.shape)
    xf = x.detach().to(torch.float32)
    xg, pad = _to_groups(xf, spec)
    if spec.symmetric:
        amax = xg.abs().amax(dim=-1, keepdim=True)
        scale = (amax * (1.0 / spec.qmax)).clamp_min(EPS).to(WIRE_DTYPE).to(torch.float32)      # explicit multiply by the fp32 constant 1/qmax (D29)
        codes = torch.round(xg / scale).clamp(-spec.qmax, spec.qmax)
        zero = None
        codes = codes.to(torch.int8)
    else:
        mn = xg.amin(dim=-1, keepdim=True)
        mx = xg.amax(dim=-1, keepdim=True)
        zero = mn.to(WIRE_DTYPE).to(torch.float32)
        scale = ((mx - zero) * (1.0 / spec.qmax)).clamp_min(EPS).to(WIRE_DTYPE).to(torch.float32)  # explicit multiply by the fp32 constant 1/qmax (D29)
        inv = 1.0 / scale                                   # one IEEE reciprocal per group (D24: kernels use __frcp_rn)
        codes = torch.round((xg - zero) * inv).clamp(0, spec.qmax).to(torch.uint8)
    codes = _from_groups(codes, spec, shape, pad)
    return Quantized(spec, codes, scale.squeeze(-1).to(WIRE_DTYPE),
                     None if zero is None else zero.squeeze(-1).to(WIRE_DTYPE), shape)


def dequantize(q: Quantized) -> torch.Tensor:
    spec = q.spec
    cg, pad = _to_groups(q.codes.to(torch.float32), spec)
    scale = q.scale.to(torch.float32).unsqueeze(-1)
    if spec.symmetric:
        xg = cg * scale
    else:
        xg = cg * scale + q.zero.to(torch.float32).unsqueeze(-1)
    return _from_groups(xg, spec, q.shape, pad)


def roundtrip(x: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    return dequantize(quantize(x, spec))


def bytes_per_token(spec: QuantSpec, H: int = 8, D: int = 128, T: int = 2048) -> dict:
    """Bytes for ONE of K or V, per token, for one layer. Scale/zero are fp16 (2 B each)."""
    payload = spec.bits * H * D / 8
    meta_per_group = 2 if spec.symmetric else 4
    if spec.axis == "row":
        groups_per_token = H * (1 if spec.group is None else D // spec.group)
        meta = groups_per_token * meta_per_group
    else:
        g = T if spec.group is None else spec.group
        n_groups_total = math.ceil(T / g) * H * D
        meta = n_groups_total * meta_per_group / T          # amortised over tokens
    return {"payload": payload, "meta": meta, "total": payload + meta}


def kv_bytes_per_token_per_layer(k_spec: QuantSpec, v_spec: QuantSpec, H=8, D=128, T=2048) -> float:
    return bytes_per_token(k_spec, H, D, T)["total"] + bytes_per_token(v_spec, H, D, T)["total"]


RAW_BYTES_PER_TOKEN_PER_LAYER = 2 * 8 * 128 * 2  # 4096


def compression_ratio(k_spec: QuantSpec, v_spec: QuantSpec, H=8, D=128, T=2048) -> float:
    return RAW_BYTES_PER_TOKEN_PER_LAYER / kv_bytes_per_token_per_layer(k_spec, v_spec, H, D, T)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    x = torch.randn(100, 8, 128)
    out = {}
    for name, kw in VARIANTS.items():
        for bits in (8, 4):
            for sym in (False, True):
                spec = QuantSpec(bits=bits, symmetric=sym, **kw)
                err = (roundtrip(x, spec) - x).norm() / x.norm()
                out[spec.name] = {"rel_l2": float(err), **bytes_per_token(spec)}
    print(json.dumps(out, indent=1))
