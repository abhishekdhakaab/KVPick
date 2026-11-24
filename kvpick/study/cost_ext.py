"""Extended cost model (B4). Times in ms, sizes in bytes, links in Gbps. Pure Python, no GPU.

A GPU *profile* (results/ext/profile_<gpu>.json, built by scripts/ext_build_profile.py from measured JSONs) holds
  copy_ceiling_GBps, peak_tflops, kernel points per format and direction [(bytes_in, ms)], and per-model prefill fits.
Kernel time:  t = a + bytes_in / bw   fitted per (format, direction) by least squares over measured points; the
intercept a is the per-launch fixed cost (kept, per the spec).  Beyond the largest measured size the same line is used
and the value is labelled "extrapolated".
Prefill time: t = c0 + c1 n + c2 n^2 + c3 n m (fitted to measured points).  Models/GPUs without a fit use the FLOPs model
t = flops / (tp * peak * eff) with eff measured on the reference model, labelled "modeled".
"""
from __future__ import annotations
import json, math, pathlib
from dataclasses import dataclass, field
import numpy as np
from kvpick.study.models import MODELS, FMT_BYTES_PER_TOKEN_PER_LAYER, RAW_BYTES_PER_TOKEN_PER_LAYER, flops_prefill
from kvpick.study.overlap import exposed_after_prefill

RTT_MS = 0.5                    # t_fixed of a transfer (assumption inherited from the original study)
TP_EFFICIENCY = 0.8             # tensor-parallel efficiency for the modeled 70B prefill (assumption, prereg addendum 1)
MODELED = {"lossless90": (0.9, "lossless"), "lossless50": (0.5, "lossless"), "fp8_ideal": (0.9, "fp8")}   # strategy -> (fraction of copy ceiling, wire format)
LOSSLESS_FRACS = {k: v[0] for k, v in MODELED.items()}
KERNEL_FORMATS = ("int8_row", "fp8", "kv4")
SEND_FORMATS = ("raw", "int8_row", "fp8", "kv4")
LABEL_RANK = {"measured": 0, "fitted": 1, "extrapolated": 2, "modeled": 3}
THRESH_FRAC, THRESH_MS = 0.10, 10.0
NVLINK_GBPS = 900 * 8           # 900 GB/s per direction


def _worse(*labels: str) -> str:
    return max(labels, key=lambda l: LABEL_RANK[l])


@dataclass
class LineFit:
    a_ms: float                 # intercept (fixed per launch)
    bw_Bps: float               # bytes_in per second
    max_bytes: float
    n: int
    max_rel_err: float

    @staticmethod
    def fit(points):            # points: [(bytes_in, ms)]
        x = np.array([p[0] for p in points], float); y = np.array([p[1] for p in points], float)
        A = np.stack([np.ones_like(x), x], 1); w = 1.0 / y
        (a, b), *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        a = max(float(a), 0.0)
        if b <= 0:
            raise ValueError("kernel fit has non-positive slope")
        pred = a + b * x
        return LineFit(a, 1e3 / float(b), float(x.max()), len(x), float(np.max(np.abs(pred - y) / y)))

    def predict(self, nbytes: float):
        return self.a_ms + nbytes / self.bw_Bps * 1e3, ("fitted" if nbytes <= self.max_bytes else "extrapolated")


@dataclass
class Profile:
    gpu: str
    copy_ceiling_GBps: float
    peak_tflops: float
    kernels: dict = field(default_factory=dict)       # (fmt, dir) -> LineFit
    prefill: dict = field(default_factory=dict)       # model -> {"coef": [...], "max_total": int, "ref_eff": float?}
    eff_ref: float | None = None                      # achieved fraction of peak on the measured reference model (FLOPs model)

    @staticmethod
    def from_json(path_or_dict):
        d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
        p = Profile(d["gpu"], d["copy_ceiling_GBps"], d["peak_tflops"], eff_ref=d.get("eff_ref"))
        for fmt, dirs in d["kernel_points"].items():
            for direction, pts in dirs.items():
                p.kernels[(fmt, direction)] = LineFit.fit(pts)
        p.prefill = d.get("prefill", {})
        return p


class CostEngine:
    def __init__(self, model: str, profile: Profile, tp: int = 1, rtt_ms: float = RTT_MS):
        self.model, self.cfg, self.p, self.tp, self.rtt = model, MODELS[model], profile, tp, rtt_ms
        self.L = self.cfg["layers"]

    # ---------------------------------------------------------------- bytes
    def raw_bytes_shard(self, n: int) -> float:
        return RAW_BYTES_PER_TOKEN_PER_LAYER * self.L * n / self.tp

    def wire_bytes_shard(self, fmt: str, n: int) -> float:
        return FMT_BYTES_PER_TOKEN_PER_LAYER[fmt] * self.L * n / self.tp

    @staticmethod
    def wire_ms(nbytes: float, gbps: float) -> float:
        return nbytes * 8 / (gbps * 1e9) * 1e3

    # ---------------------------------------------------------------- kernels (per shard; `layers` = how many layers one launch covers)
    def kernel_ms(self, fmt: str, direction: str, n: int, layers: int | None = None):
        layers = self.L if layers is None else layers
        if fmt == "raw":
            return 0.0, "measured"
        raw = RAW_BYTES_PER_TOKEN_PER_LAYER * layers * n / self.tp
        if fmt in LOSSLESS_FRACS:
            return raw / (LOSSLESS_FRACS[fmt] * self.p.copy_ceiling_GBps * 1e9) * 1e3, "modeled"
        return self.p.kernels[(fmt, direction)].predict(raw)

    # ---------------------------------------------------------------- prefill (recompute side)
    def prefill_ms(self, n: int, m: int = 0):
        fit = self.p.prefill.get(self.model)
        if fit is not None:
            c = fit["coef"]
            t = max(0.0, c[0] + c[1] * n + c[2] * n * n + c[3] * n * m)
            lab = fit.get("label", "fitted")
            if n + m > fit["max_total"] or n + m > self.cfg["native_ctx"]:
                lab = "extrapolated"
            return t, lab
        if self.p.eff_ref is None:
            raise KeyError(f"no prefill fit for {self.model} on {self.p.gpu} and no reference efficiency")
        t = flops_prefill(self.model, n, m) / (self.tp * (TP_EFFICIENCY if self.tp > 1 else 1.0) * self.p.peak_tflops * 1e12 * self.p.eff_ref) * 1e3
        return t, "modeled"

    # ---------------------------------------------------------------- strategies
    def send_ms(self, fmt: str, n: int, gbps: float, overlapped: bool, prefill_ms: float | None = None, prefill_slowdown: float = 0.0):
        """Time until the KV is ready on the decode GPU, measured from the moment the prefill GPU finishes (see prereg addendum 1)."""
        wire_fmt = MODELED[fmt][1] if fmt in MODELED else fmt
        wire = self.wire_ms(self.wire_bytes_shard(wire_fmt, n), gbps)
        if not overlapped:
            tc, lc = self.kernel_ms(fmt, "compress", n); td, ld = self.kernel_ms(fmt, "decompress", n)
            return tc + wire + td + self.rtt, _worse(lc, ld)
        assert prefill_ms is not None
        N = self.L
        tc, lc = self.kernel_ms(fmt, "compress", n, layers=1); td, ld = self.kernel_ms(fmt, "decompress", n, layers=1)
        a = prefill_ms * (1 + prefill_slowdown) / N
        b = tc + wire / N; d = td
        return exposed_after_prefill(a, b, d, N) + self.rtt, _worse(lc, ld)

    def all_times(self, n: int, m: int, gbps: float, overlapped: bool, prefill_slowdown: float = 0.0, decode_batch: int = 0, itl_slo_ms: float = 60.0):
        """-> {strategy: (ms, label)}. Strategies: raw, int8_row, fp8, kv4, lossless90, lossless50, recompute."""
        P, plab = self.prefill_ms(n, m)
        out = {}
        for f in SEND_FORMATS + tuple(MODELED):
            out[f] = self.send_ms(f, n, gbps, overlapped, prefill_ms=P, prefill_slowdown=prefill_slowdown)
        out["recompute"] = (P, plab)
        out["_recompute_blocked"] = decode_batch > 0 and P > itl_slo_ms          # ITL-SLO rule retained from the original planner
        out["_prefill_ms"] = P
        return out


def pays_off(times: dict) -> dict:
    """Prereg rule. times from CostEngine.all_times. Returns verdict and savings."""
    t = {k: v[0] for k, v in times.items() if not k.startswith("_")}
    alt_names = [k for k in ("raw", "int8_row", "fp8", "recompute") if not (k == "recompute" and times["_recompute_blocked"])]
    best_alt = min(alt_names, key=lambda k: t[k]); tk = t["kv4"]; ta = t[best_alt]
    save_ms = ta - tk; save_frac = save_ms / ta if ta > 0 else 0.0
    s = min(save_frac / THRESH_FRAC, save_ms / THRESH_MS) - 1.0
    if tk < t["raw"] and (times["_recompute_blocked"] or tk < t["recompute"]) and s >= 0:
        verdict = "pays_off"
    elif save_ms > 0:
        verdict = "marginal"
    else:
        verdict = "does_not_pay_off"
    ta_l = min(ta, t["lossless90"]); ta_f = min(ta, t["fp8_ideal"]); tk_ = tk
    return {"pays_off_vs_fp8_ideal": bool(tk_ < t["raw"] and (ta_f - tk_) >= THRESH_MS and (ta_f - tk_) / ta_f >= THRESH_FRAC), "verdict": verdict, "best_alt": best_alt, "save_ms": save_ms, "save_frac": save_frac, "score": s,
            "save_vs_raw_ms": t["raw"] - tk, "pays_off_vs_incl_lossless90": (ta_l - tk) >= THRESH_MS and (ta_l - tk) / ta_l >= THRESH_FRAC and tk < t["raw"]}


def winner(times: dict, include=("raw", "int8_row", "fp8", "kv4", "recompute")) -> str:
    cand = [k for k in include if not (k == "recompute" and times["_recompute_blocked"])]
    return min(cand, key=lambda k: times[k][0])


def crossings(links: list[float], scores: list[float]):
    """Interpolate (log10 link) where the score s changes sign between ADJACENT grid points (no extrapolation).
    Returns dict(lower, upper, status). upper = first link above which KV4 stops paying off after having paid off;
    lower = first link at which it starts to pay off. None when no crossing inside the grid."""
    lo = up = None
    for i in range(len(links) - 1):
        s0, s1 = scores[i], scores[i + 1]
        if s0 == s1 or (s0 >= 0) == (s1 >= 0):
            continue
        x0, x1 = math.log10(links[i]), math.log10(links[i + 1]); x = x0 + (0 - s0) * (x1 - x0) / (s1 - s0)
        if s0 < 0 <= s1 and lo is None:
            lo = 10 ** x
        if s0 >= 0 > s1:
            up = 10 ** x
    pays = [s >= 0 for s in scores]
    status = "never" if not any(pays) else ("all_grid" if all(pays) else "crossing")
    return {"lower": lo, "upper": up, "status": status, "pays_at_min_link": pays[0], "pays_at_max_link": pays[-1]}
