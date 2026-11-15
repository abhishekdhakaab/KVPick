"""Build results/cost_tables_<gpu>.json from the measured feasibility JSONs, with provenance per entry, and fit
  t_prefill(n_new, m_cached) = c0 + c1*n + c2*n*(n/2 + m)        (ms; 14 measured points: G5 m=0 sweep + E5 grid)
  t_decode_step(batch)       = d0 + d1*batch                      (ms; E5, ctx 2048)
Kernel times: taken from results/kernels_<gpu>.json if present (our v2 kernel), else from feasibility G4 torch.compile
numbers, flagged as placeholder.  Quality classes: from feasibility G3 and configs/quality_budgets.yaml.
"""
from __future__ import annotations
import json, pathlib, sys, argparse
import numpy as np, yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]; FEAS = ROOT / "results" / "feasibility"


def load(name):
    return json.load(open(FEAS / name))["measurements"]


def fit_prefill(points):
    """points: list of (n, m, ms). Model ms = c0 + c1*n + c2*n^2 + c3*n*m, least squares on RELATIVE error
    (rows scaled by 1/ms) so the small-n points that matter for multi-round chat are fitted as well as the big ones."""
    X = np.array([[1.0, n, n * n, n * m] for n, m, _ in points]); y = np.array([ms for *_, ms in points])
    w = 1.0 / y
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None); pred = X @ coef
    rel = np.abs(pred - y) / y
    return [float(c) for c in coef], {"max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean()), "n_points": len(points),
                                       "per_point_rel_err": [round(float(r), 3) for r in rel]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu-tag", default="a40"); ap.add_argument("--kernels", default=None, help="results/kernels_<gpu>.json from our kernel benchmark")
    ap.add_argument("--quality", default=None, help="quality JSON to classify formats from (default: results/quality_a40.json if present, else feasibility G3)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/planner.yaml")); budgets = yaml.safe_load(open(ROOT / "configs/quality_budgets.yaml"))
    g5 = load("G5_crossover.json"); e5 = load("E5_prefill_cached.json"); g4 = load("G4_headroom.json")
    qpath = pathlib.Path(a.quality) if a.quality else (ROOT / "results" / "quality_a40.json" if (ROOT / "results" / "quality_a40.json").exists() else FEAS / "G3_quality.json")
    g3 = json.load(open(qpath))["measurements"]
    out = {"gpu": g5["device"] + " / " + e5.get("gpu", "?"), "model": g5["model"], "sources": {}, "prefill": {}, "decode": {}, "kernel_ms": {}, "bytes_per_token": {}, "quality": {}}
    # ---- prefill
    pts = [(r["n"], 0, r["median_s"] * 1e3) for r in g5["prefill"]]
    for k, ms in e5["prefill_ms"].items():
        n, m = (int(s[1:]) for s in k.split("_")); pts.append((n, m, ms))
    coef, fitq = fit_prefill(pts)
    out["prefill"] = {"model": "ms = c0 + c1*n + c2*n^2 + c3*n*m  (relative least squares)", "coef": coef, "fit": fitq, "points": [{"n": n, "m": m, "ms": ms} for n, m, ms in pts]}
    out["sources"]["prefill"] = "feasibility G5 (n sweep, m=0, 5 reps) + E5 (n x m grid, 3 reps), Qwen3-4B bf16 HF eager on A40"
    # ---- decode step
    b = np.array([int(k[5:]) for k in e5["decode_step_ms"]]); d = np.array(list(e5["decode_step_ms"].values()))
    A = np.stack([np.ones_like(b, dtype=float), b.astype(float)], 1); dc, *_ = np.linalg.lstsq(A, d, rcond=None)
    out["decode"] = {"model": "ms_per_step = d0 + d1*batch (ctx 2048)", "coef": [float(x) for x in dc], "points": {int(k[5:]): v for k, v in e5["decode_step_ms"].items()}}
    out["sources"]["decode"] = "feasibility E5 decode step at batch 1/8/16, ctx 2048, HF eager on A40 (no CUDA graphs; conservative)"
    # ---- bytes
    for f, spec in cfg["formats"].items():
        out["bytes_per_token"][f] = spec["bytes_per_token"]
    # ---- kernel times (ms) per format as a function of T: table at T in {512, 2048, 8192}; linear interpolation in between
    kern = {}
    if a.kernels and pathlib.Path(a.kernels).exists():
        kj = json.load(open(a.kernels))
        for Tk, row in kj["by_T"].items():
            T = int(Tk[1:]); kern.setdefault("int4_kivi", {})[T] = {"compress": row["quant"]["v2"]["median_us"] / 1e3, "decompress": row["dequant"]["v2"]["median_us"] / 1e3}
        out["sources"]["kernel_int4_kivi"] = f"our fused v2 kernel, {kj['gpu']} ({a.kernels})"
    g4b = load("G4b_compile_modes.json") if (FEAS / "G4b_compile_modes.json").exists() else {}
    for Tk, row in g4["by_T"].items():
        T = int(Tk[1:])
        for fname, key in (("int8_row", "int8_row"), ("int4_kivi", "int4_kivi")):
            if fname in kern and T in kern[fname]: continue
            r = row["formats"][key]
            comp = r["best_compress_us_K_plus_V"] / 1e3
            if fname == "int4_kivi" and Tk in g4b:          # G4 chan_g32 T=8192 compile fell back to eager (dynamo cache); use G4b max-autotune K + row_g32 V
                comp = (g4b[Tk]["int4_chan_g32"]["max_autotune_nocg"]["median_us"] + g4b[Tk]["int4_row_g32"]["max_autotune_nocg"]["median_us"]) / 1e3
            kern.setdefault(fname, {})[T] = {"compress": comp, "decompress": r["best_decompress_us_K_plus_V"] / 1e3, "placeholder": "torch.compile (feasibility G4/G4b); replaced by our kernel after the A40 results session"}
    kern["raw"] = {T: {"compress": 0.0, "decompress": 0.0} for T in (512, 2048, 8192)}
    out["kernel_ms"] = {f: {str(T): v for T, v in sorted(d_.items())} for f, d_ in kern.items()}
    out["sources"].setdefault("kernel_int8_row", "torch.compile best-of (feasibility G4, A40); no custom kernel by design")
    # ---- quality classes from G3
    pf = g3["tasks"]; order = budgets["order"]
    for f in ["raw"] + [x for x in g3["formats"] if x != "raw"]:
        if f == "raw":
            out["quality"][f] = {"class": "strict", "losses": {}}; continue
        ppl = pf["ppl"]["per_format"][f]["rel_change"] * 100
        gsm = -pf["gsm8k"]["per_format"][f]["delta_ci95"]["point"] * 100 if "gsm8k" in pf else 0
        qa = -pf["qasper"]["per_format"][f]["delta_ci95"]["point"] * 100 if "qasper" in pf else 0
        cls = None
        for bname in order:
            lim = budgets["budgets"][bname]
            if ppl <= lim["max_ppl_rel_pct"] and max(gsm, qa, 0) <= lim["max_task_points"]:
                cls = bname; break
        out["quality"][f] = {"class": cls or "none", "losses": {"ppl_rel_pct": ppl, "gsm8k_points": gsm, "qasper_points": qa}}
    out["sources"]["quality"] = f"{qpath.name} on Qwen3-4B (n=50 ppl windows, 100 GSM8K, 50 Qasper; kernel in loop: {g3.get('kernel_in_loop')}), classes per configs/quality_budgets.yaml (point estimates)"
    p = ROOT / "results" / f"cost_tables_{a.gpu_tag}.json"; p.write_text(json.dumps(out, indent=1))
    print("prefill fit:", coef, fitq); print("decode fit:", out["decode"]["coef"])
    print("quality classes:", {f: v["class"] for f, v in out["quality"].items()}); print("kernel ms:", json.dumps(out["kernel_ms"])); print("saved", p)


if __name__ == "__main__":
    main()
