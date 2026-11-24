"""B5: the breakeven sweep, computed from the cost model (CPU only). Every row carries a label.
Grid: model x ctx x cached fraction x link x GPU profile x mode(serial / overlapped / overlapped with 10% prefill slowdown).
Strategies: raw, int8_row, fp8, kv4, lossless90/50 (modeled), recompute.  Output results/ext/sweep[_provisional].json
Usage: python scripts/ext_sweep.py --profile results/ext/profile_a40.json --profile results/ext/profile_h100.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import cost_ext as C
from kvpick.study.models import MODELS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CTX = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
LINKS = [1, 3, 10, 25, 50, 100, 200, 400, 800, C.NVLINK_GBPS]
MODES = {"serial": (False, 0.0), "overlapped": (True, 0.0), "overlapped_slow10": (True, 0.10)}
TP = {"qwen3-4b": 1, "llama-3.1-8b": 1, "llama-3.1-70b": 8}


def row_label(times):
    return max((v[1] for k, v in times.items() if k in ("raw", "int8_row", "fp8", "kv4", "recompute")), key=lambda l: C.LABEL_RANK[l])   # modeled-by-design strategies (lossless, fp8_ideal) are labeled separately


def build(profiles, qclasses=None):
    rows, be = [], []
    for prof in profiles:
        for model in MODELS:
            eng = C.CostEngine(model, prof, tp=TP[model])
            try:
                eng.prefill_ms(1024)
            except KeyError:
                continue                                                           # no reference efficiency on this profile
            for mode, (ov, slow) in MODES.items():
                for ctx in CTX:
                    for mf in (0.0, 0.5):
                        n, m = ctx, int(ctx * mf); scores = []
                        for g in LINKS:
                            t = eng.all_times(n, m, g, ov, prefill_slowdown=slow)
                            po = C.pays_off(t); scores.append(po["score"])
                            P_src = eng.prefill_ms(n, m)[0]
                            win = C.winner(t); fp8_better = (t["fp8"][0] - t["kv4"][0] >= C.THRESH_MS and (t["fp8"][0] - t["kv4"][0]) / t["fp8"][0] >= C.THRESH_FRAC)
                            rows.append({"model": model, "gpu": prof.gpu, "mode": mode, "ctx": ctx, "cached_frac": mf, "link_gbps": g,
                                         "ms": {k: v[0] for k, v in t.items() if not k.startswith("_")}, "labels": {k: v[1] for k, v in t.items() if not k.startswith("_")},
                                         "label": row_label(t), "link_label": "modeled", "winner": win, "winner_incl_lossless90": C.winner(t, ("raw", "int8_row", "fp8", "kv4", "recompute", "lossless90")),
                                         "kv4_verdict": po["verdict"], "kv4_score": po["score"], "best_alt": po["best_alt"], "save_vs_best_alt_ms": po["save_ms"], "save_vs_best_alt_frac": po["save_frac"],
                                         "save_vs_raw_ms": po["save_vs_raw_ms"], "pays_off_incl_lossless90": po["pays_off_vs_incl_lossless90"], "pays_off_vs_fp8_ideal": po["pays_off_vs_fp8_ideal"], "kv4_beats_fp8_time_rule": bool(fp8_better),
                                         "ttft_incl_source_prefill_ms": {k: (v[0] + (P_src if k != "recompute" else 0.0)) for k, v in t.items() if not k.startswith("_")},
                                         "prefill_ms": t["_prefill_ms"]})
                        be.append({"model": model, "gpu": prof.gpu, "mode": mode, "ctx": ctx, "cached_frac": mf, **C.crossings(LINKS, scores)})
    return rows, be


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", action="append", required=True); ap.add_argument("--out", default=None); ap.add_argument("--provisional", action="store_true")
    a = ap.parse_args()
    profiles = [C.Profile.from_json(p) for p in a.profile]
    rows, be = build(profiles)
    out = {"label": "modeled (cost model on measured/fitted kernel and prefill inputs; links always modeled)", "provisional": a.provisional, "profiles": a.profile,
           "ctx": CTX, "links_gbps": LINKS, "modes": MODES, "tp": TP, "assumptions": {"rtt_ms": C.RTT_MS, "tp_efficiency": C.TP_EFFICIENCY, "lossless_ratio": 1.33, "decode_batch": 0}, "rows": rows, "breakeven": be}
    p = pathlib.Path(a.out) if a.out else ROOT / "results" / "ext" / ("sweep_provisional.json" if a.provisional else "sweep.json")
    p.write_text(json.dumps(out)); print("saved", p, len(rows), "rows,", len(be), "breakeven series")


if __name__ == "__main__":
    main()
