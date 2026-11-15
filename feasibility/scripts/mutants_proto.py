"""Candidate quantizers for the checker prototype: 3 good ones and 6 deliberately broken ones (G8).

Candidate interface (caller allocates outputs so memory can be measured):
    cand(x: Tensor[T,H,D], out_codes: Tensor, out_scale: Tensor fp16, out_zero: Tensor fp16) -> None
Output shapes follow quant_ref.quantize(x, spec) for the candidate's spec.
"""
from __future__ import annotations
import torch
from quant_ref import QuantSpec, quantize, _to_groups, _from_groups, WIRE_DTYPE, EPS

SPECS = {
    "row_8b": QuantSpec(8, "row", None),
    "row_g32_4b": QuantSpec(4, "row", 32),
    "chan_g32_4b": QuantSpec(4, "chan", 32),
}


def _write(q, out_codes, out_scale, out_zero):
    out_codes.copy_(q.codes); out_scale.copy_(q.scale); out_zero.copy_(q.zero)


# ---------------- good candidates: an independent vectorised implementation (not the reference code path)
def make_good(spec: QuantSpec):
    def good(x, out_codes, out_scale, out_zero):
        xg, pad = _to_groups(x.float(), spec)
        mn = xg.amin(-1, keepdim=True); mx = xg.amax(-1, keepdim=True)
        zero = mn.to(WIRE_DTYPE).float()
        scale = ((mx - zero) / spec.qmax).clamp_min(EPS).to(WIRE_DTYPE).float()
        codes = ((xg - zero) / scale).round_().clamp_(0, spec.qmax).to(torch.uint8)
        out_codes.copy_(_from_groups(codes, spec, tuple(x.shape), pad))
        out_scale.copy_(scale.squeeze(-1).to(WIRE_DTYPE)); out_zero.copy_(zero.squeeze(-1).to(WIRE_DTYPE))
    good.__name__ = f"good_{spec.name}"
    return good


# ---------------- mutants (all built on the row_g32_4b spec unless noted)
def make_mutants(spec: QuantSpec):
    M = {}

    def m1_skip_groups(x, oc, os_, oz):          # 1. skips every other group (writes zeros there)
        q = quantize(x, spec)
        cg, pad = _to_groups(q.codes.float(), spec)
        cg = cg.clone(); cg[..., 1::2, :] = 0 if spec.axis == "row" else cg[..., 1::2, :]
        if spec.axis == "chan":
            cg[1::2] = 0
        oc.copy_(_from_groups(cg, spec, tuple(x.shape), pad).to(torch.uint8)); os_.copy_(q.scale); oz.copy_(q.zero)
    M["m1_skip_groups"] = m1_skip_groups

    state = {}
    def m2_cached(x, oc, os_, oz):               # 2. returns the previous call's output
        if "prev" in state:
            pc, ps, pz = state["prev"]
            if pc.shape == oc.shape:
                oc.copy_(pc); os_.copy_(ps); oz.copy_(pz); return
        q = quantize(x, spec); _write(q, oc, os_, oz)
        state["prev"] = (oc.clone(), os_.clone(), oz.clone())
    M["m2_cached_output"] = m2_cached

    def m3_first_row_scale(x, oc, os_, oz):      # 3. computes scales from the first row only
        xg, pad = _to_groups(x.float(), spec)
        first = xg[:1] if spec.axis == "chan" else xg[:1, :1]          # one row's groups
        mn = first.amin(-1, keepdim=True); mx = first.amax(-1, keepdim=True)
        zero = mn.to(WIRE_DTYPE).float(); scale = ((mx - zero) / spec.qmax).clamp_min(EPS).to(WIRE_DTYPE).float()
        zero = zero.expand_as(xg[..., :1]); scale = scale.expand_as(xg[..., :1])
        codes = ((xg - zero) / scale).round_().clamp_(0, spec.qmax).to(torch.uint8)
        oc.copy_(_from_groups(codes, spec, tuple(x.shape), pad)); os_.copy_(scale.squeeze(-1).to(WIRE_DTYPE)); oz.copy_(zero.squeeze(-1).to(WIRE_DTYPE))
    M["m3_first_row_scale"] = m3_first_row_scale

    def m4_clipped_amax(x, oc, os_, oz):         # 4. scale from a clipped range (99.9th percentile): outliers saturate
        xg, pad = _to_groups(x.float(), spec)
        lo = torch.quantile(xg.flatten(), 0.001); hi = torch.quantile(xg.flatten(), 0.999)
        xc = xg.clamp(lo, hi)
        mn = xc.amin(-1, keepdim=True); mx = xc.amax(-1, keepdim=True)
        zero = mn.to(WIRE_DTYPE).float(); scale = ((mx - zero) / spec.qmax).clamp_min(EPS).to(WIRE_DTYPE).float()
        codes = ((xg - zero) / scale).round_().clamp_(0, spec.qmax).to(torch.uint8)
        oc.copy_(_from_groups(codes, spec, tuple(x.shape), pad)); os_.copy_(scale.squeeze(-1).to(WIRE_DTYPE)); oz.copy_(zero.squeeze(-1).to(WIRE_DTYPE))
    M["m4_clipped_amax"] = m4_clipped_amax

    def m5_hidden_buffer(x, oc, os_, oz):        # 5. allocates a large hidden temporary buffer
        junk = torch.empty(256 << 20, dtype=torch.uint8, device=x.device); junk.fill_(1)   # 256 MB, touched
        q = quantize(x, spec); _write(q, oc, os_, oz)
        oc[0, 0, 0] = oc[0, 0, 0] + int(junk[0]) - 1     # keep junk alive until here
    M["m5_hidden_buffer"] = m5_hidden_buffer

    def m6_truncate(x, oc, os_, oz):             # 6. truncates instead of rounds
        xg, pad = _to_groups(x.float(), spec)
        mn = xg.amin(-1, keepdim=True); mx = xg.amax(-1, keepdim=True)
        zero = mn.to(WIRE_DTYPE).float(); scale = ((mx - zero) / spec.qmax).clamp_min(EPS).to(WIRE_DTYPE).float()
        codes = ((xg - zero) / scale).trunc_().clamp_(0, spec.qmax).to(torch.uint8)
        oc.copy_(_from_groups(codes, spec, tuple(x.shape), pad)); os_.copy_(scale.squeeze(-1).to(WIRE_DTYPE)); oz.copy_(zero.squeeze(-1).to(WIRE_DTYPE))
    M["m6_truncate"] = m6_truncate
    return M


def all_candidates():
    """{name: (spec, fn, expected_verdict)}"""
    C = {}
    for n, s in SPECS.items():
        C[f"good_{n}"] = (s, make_good(s), "PASS")
    for n, fn in make_mutants(SPECS["row_g32_4b"]).items():
        C[n] = (SPECS["row_g32_4b"], fn, "REJECT")
    return C
