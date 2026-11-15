"""Wire-format quantize/dequantize on full caches, written with plain PyTorch ops (no permutes), used as the
B0 "PyTorch reference" baseline for kernel timing (G4) and as the thing torch.compile compiles (B1).

Input  x: bf16 [L, T, H, D] (whole cache, one of K or V) or [T, H, D].
Output payload: uint8 packed (4-bit: two codes per byte, low nibble first, adjacent d; 8-bit: one code per byte),
        scale, zero: fp16 with the group layout of the spec.
Equivalent to quant_ref (checked by test_quant_wire.py) but vectorised over layers.
"""
from __future__ import annotations
import torch
from kvpick.ref.quant_ref import QuantSpec, EPS, WIRE_DTYPE


def _group_dims(x4: torch.Tensor, spec: QuantSpec):
    """Return (xg, reduce_dim, pad) where xg has an explicit group axis to reduce over."""
    L, T, H, D = x4.shape
    if spec.axis == "row":
        g = D if spec.group is None else spec.group
        return x4.reshape(L, T, H, D // g, g), -1, 0
    g = T if spec.group is None else spec.group
    pad = (-T) % g
    if pad:
        x4 = torch.cat([x4, x4[:, -1:].expand(L, pad, H, D)], dim=1)
    return x4.reshape(L, (T + pad) // g, g, H, D), 2, pad


def quantize_wire(x: torch.Tensor, spec: QuantSpec):
    squeeze = x.dim() == 3
    x4 = x.unsqueeze(0) if squeeze else x
    L, T, H, D = x4.shape
    xg, rd, pad = _group_dims(x4.float(), spec)
    mn = xg.amin(dim=rd, keepdim=True); mx = xg.amax(dim=rd, keepdim=True)
    zero = mn.to(WIRE_DTYPE).float()
    scale = ((mx - zero) * (1.0 / spec.qmax)).clamp_min(EPS).to(WIRE_DTYPE).float()   # D29
    inv = 1.0 / scale
    codes = ((xg - zero) * inv).round_().clamp_(0, spec.qmax).to(torch.uint8)
    if spec.axis == "row":
        codes = codes.reshape(L, T, H, D)
    else:
        codes = codes.reshape(L, T + pad, H, D)[:, :T]
    if spec.bits == 4:
        payload = codes[..., 0::2] | (codes[..., 1::2] << 4)            # [L, T, H, D/2]
    else:
        payload = codes
    scale = scale.squeeze(rd).to(WIRE_DTYPE); zero = zero.squeeze(rd).to(WIRE_DTYPE)
    if squeeze:
        payload, scale, zero = payload[0], scale[0], zero[0]
    return payload, scale, zero


def dequantize_wire(payload: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, spec: QuantSpec, T: int, out_dtype=torch.bfloat16):
    squeeze = payload.dim() == 3
    if squeeze:
        payload, scale, zero = payload.unsqueeze(0), scale.unsqueeze(0), zero.unsqueeze(0)
    L, Tp, H, Dp = payload.shape
    if spec.bits == 4:
        codes = torch.stack([payload & 0xF, payload >> 4], dim=-1).reshape(L, T, H, Dp * 2)
    else:
        codes = payload
    D = codes.shape[-1]
    if spec.axis == "row":
        g = D if spec.group is None else spec.group
        cg = codes.reshape(L, T, H, D // g, g).float()
        x = cg * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)
        x = x.reshape(L, T, H, D)
    else:
        g = T if spec.group is None else spec.group
        pad = (-T) % g
        if pad:
            codes = torch.cat([codes, codes[:, -1:].expand(L, pad, H, D)], dim=1)
        cg = codes.reshape(L, (T + pad) // g, g, H, D).float()
        x = cg * scale.float().unsqueeze(2) + zero.float().unsqueeze(2)
        x = x.reshape(L, T + pad, H, D)[:, :T]
    x = x.to(out_dtype)
    return x[0] if squeeze else x


def roundtrip_wire(x: torch.Tensor, spec: QuantSpec, out_dtype=torch.bfloat16):
    p, s, z = quantize_wire(x, spec)
    return dequantize_wire(p, s, z, spec, x.shape[-3], out_dtype)


def wire_bytes(payload, scale, zero) -> int:
    return payload.numel() * payload.element_size() + scale.numel() * 2 + zero.numel() * 2
