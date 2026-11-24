"""Build results/ext/profile_<gpu>.json (the cost-model inputs for one GPU type) from the measured B1 and B2 JSONs.
  kernel points: (raw K+V bytes of the measured chunk, median ms) per format and direction   [measured]
  prefill fit:   ms = c0 + c1 n + c2 n^2 + c3 n m  by relative least squares on the B2 points; error per point reported  [fitted]
  eff_ref:       achieved fraction of peak bf16 FLOPs on the measured Llama-3.1-8B points with n >= 4096 (for the FLOPs model of other models)
Usage: python scripts/ext_build_profile.py --kernels results/ext/kernels_X.json --prefill results/ext/prefill_llama-3.1-8b_X.json [--prefill ...]
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study.models import flops_prefill

ROOT = pathlib.Path(__file__).resolve().parents[1]
PEAK_TFLOPS = {"a40": 149.7, "l40s": 362.0, "rtx 6000 ada": 364.0, "a6000": 154.8, "h100 sxm": 989.0, "h100 80gb hbm3": 989.0, "h100 pcie": 756.0, "h100 nvl": 835.0,
               "h100": 756.0, "a100": 312.0}


def peak_for(gpu: str) -> float:
    g = gpu.lower()
    for k in sorted(PEAK_TFLOPS, key=len, reverse=True):
        if k in g:
            return PEAK_TFLOPS[k]
    raise KeyError(f"peak TFLOPs unknown for {gpu}; add it to PEAK_TFLOPS")


def fit_prefill(points):
    X = np.array([[1.0, p["n"], p["n"] ** 2, p["n"] * p["m"]] for p in points]); y = np.array([p["ms"] for p in points]); w = 1 / y
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    pred = X @ coef; rel = np.abs(pred - y) / y
    per = [{"n": p["n"], "m": p["m"], "measured_ms": p["ms"], "fitted_ms": float(q), "rel_err": float(r)} for p, q, r in zip(points, pred, rel)]
    return [float(c) for c in coef], {"max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean()), "n_points": len(points), "per_point": per,
                                      "max_rel_err_n_ge_32k": float(max([r for p, r in zip(points, rel) if p["n"] + p["m"] >= 32768], default=0.0))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kernels", required=True); ap.add_argument("--prefill", action="append", default=[]); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kj = json.load(open(a.kernels)); gpu = kj["gpu"]; tag = gpu.lower().replace("nvidia ", "").replace(" ", "_")
    kp = {}; ceil = []
    for key, c in kj["cases"].items():
        if "copy_ceiling" not in c: continue
        ceil.append((c["raw_bytes_KV_measured"], c["copy_ceiling"]["GBps_moved"]))
        for d, dn in (("quant", "compress"), ("dequant", "decompress")):
            for fmt, r in c[d].items():
                kp.setdefault(fmt, {}).setdefault(dn, []).append([c["raw_bytes_KV_measured"], r["median_us"] / 1e3])
    big = [g for b, g in ceil if b >= 1e9]
    out = {"gpu": gpu, "label": "measured/fitted", "copy_ceiling_GBps": statistics.median(big or [g for _, g in ceil]), "peak_tflops": peak_for(gpu),
           "kernel_points": kp, "prefill": {}, "sources": {"kernels": a.kernels, "prefill": a.prefill}}
    effs = []
    for pf in a.prefill:
        pj = json.load(open(pf)); assert pj["gpu"] == gpu, f"GPU mismatch {pj['gpu']} vs {gpu}: never mix GPU types in one profile"
        coef, fit = fit_prefill(pj["points"])
        out["prefill"][pj["model"]] = {"coef": coef, "fit": fit, "max_total": max(p["n"] + p["m"] for p in pj["points"]), "points": pj["points"], "label": "fitted"}
        if pj["model"] == "llama-3.1-8b":
            effs = [flops_prefill(pj["model"], p["n"], p["m"]) / (p["ms"] / 1e3) / (out["peak_tflops"] * 1e12) for p in pj["points"] if p["n"] >= 4096]
    out["eff_ref"] = statistics.median(effs) if effs else None
    p = pathlib.Path(a.out) if a.out else ROOT / "results" / "ext" / f"profile_{tag}.json"
    p.write_text(json.dumps(out, indent=1)); print("saved", p, "eff_ref", out["eff_ref"], {m: (v["fit"]["max_rel_err"], v["fit"]["max_rel_err_n_ge_32k"]) for m, v in out["prefill"].items()})


if __name__ == "__main__":
    main()
