"""B2: real prefill time as a function of new tokens and cached tokens, via ONE chunked-prefill trace per model.
Random token ids, chunk 4096, logits for the last token only. T(n new | m cached) = sum of measured chunk times whose start
lies in [m, m+n)   (n, m multiples of the chunk). Small n (< chunk) is timed directly with m = 0.
Several repetitions; the per-chunk median is used.  Output: results/ext/prefill_<model>_<gpu>.json
--cpu-smoke: tiny random model, tiny sizes (plumbing only).
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import runlog
from kvpick.study.longctx import load_model, chunked_prefill

ROOT = pathlib.Path(__file__).resolve().parents[1]


def window_sum(medians, chunk, m, n):
    idx = range(m // chunk, (m + n) // chunk)
    return sum(medians[i] for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=4096); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--small-n", default="256,512,1024,2048"); ap.add_argument("--cpu-smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cpu" if a.cpu_smoke else "cuda"
    if a.cpu_smoke: a.max_ctx, a.chunk, a.reps, a.small_n = 512, 128, 1, "32,64"
    info = runlog.start(extra={"model": a.model, "chunk": a.chunk}); tag = (info["gpu"] or "cpu").lower().replace("nvidia ", "").replace(" ", "_")
    model, tok, cfginfo = load_model(a.model, dev, tiny=a.cpu_smoke); info["config_check"] = cfginfo
    g = torch.Generator().manual_seed(0); N = a.max_ctx
    ids = torch.randint(1000, 30000, (1, N), generator=g).to(dev)
    chunk_times = []                                                            # reps x n_chunks
    for r in range(a.reps + 1):                                                  # first rep is warm-up
        tr = []; t0 = time.time(); chunked_prefill(model, ids, a.chunk, tr)
        if r: chunk_times.append([t for _, _, t in tr])
        print(f"rep {r}: {time.time()-t0:.1f}s for {N} tokens", flush=True)
        if dev == "cuda": torch.cuda.empty_cache()
    med = [statistics.median(c[i] for c in chunk_times) for i in range(len(chunk_times[0]))]
    points = []
    n = a.chunk
    while n <= N:
        for m in sorted({0, n // 2, n}):
            if m % a.chunk == 0 and m + n <= N:
                points.append({"n": n, "m": m, "ms": window_sum(med, a.chunk, m, n) * 1e3, "source": "trace"})
        n *= 2
    for sn in [int(x) for x in a.small_n.split(",") if x]:                       # direct small-n, m = 0, own forward pass
        ts = []
        for r in range(a.reps + 1):
            tr = []; chunked_prefill(model, ids[:, :sn], sn, tr)
            if r: ts.append(tr[0][2])
        points.append({"n": sn, "m": 0, "ms": statistics.median(ts) * 1e3, "source": "direct"})
    out = {"info": runlog.finish(info), "label": "measured", "model": a.model, "gpu": info["gpu"], "chunk": a.chunk, "reps": a.reps,
           "max_ctx": N, "chunk_ms_median": [t * 1e3 for t in med], "chunk_ms_all": [[t * 1e3 for t in c] for c in chunk_times], "points": points}
    p = pathlib.Path(a.out) if a.out else ROOT / "results" / "ext" / f"prefill_{a.model}_{tag}{'_cpusmoke' if a.cpu_smoke else ''}.json"
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))
    for pt in points: print(pt)
    print("saved", p, "throttled:", out["info"]["throttled"])


if __name__ == "__main__":
    main()
