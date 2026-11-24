"""Dump EVERY recorded number of the extension study into docs/experiment_log.md (script-generated from results/ext/*.json and the
pod logs; nothing hand-typed). Re-run any time; it reflects whatever is in results/ext/ at that moment."""
from __future__ import annotations
import collections, glob, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results/ext"
out = []; P = out.append
J = lambda p: json.load(open(p))
def tbl(head, rows):
    P("| " + " | ".join(head) + " |"); P("|" + "---|" * len(head))
    for r in rows: P("| " + " | ".join(str(x) for x in r) + " |")
    P("")
f1 = lambda x, n=1: "" if x is None else f"{x:.{n}f}"

P(f"# Extension experiment log (auto-generated {time.strftime('%Y-%m-%d %H:%M:%S')} UTC-local; source: results/ext/*.json, logs/pod_ext/*)\n")
P("Hand-written chronology, bugs and fixes: docs/experiment_timeline.md. Every number below is read from a result file.\n")

# ---------------------------------------------------------------- environment
k = J(R / "kernels_a40.json"); info = k["info"]
P("## 1. Environment (from the kernel run's runlog)\n")
tbl(["field", "value"], [[a, info.get(a)] for a in ("gpu", "python", "platform", "torch", "cuda", "transformers", "datasets", "numpy", "accelerate", "commit", "dirty", "seed", "command", "seconds", "throttled")])
for tag in ("gpu_start", "gpu_end"):
    for g in info[tag]["gpus"]:
        P(f"- {tag} @ {info[tag]['t']}: " + ", ".join(f"{a}={v}" for a, v in g.items()))
P("\nPod env file (logs/pod_ext/ext_env_gpu.txt):\n\n```\n" + open(ROOT / "logs/pod_ext/ext_env_gpu.txt").read() + "```\n")

# ---------------------------------------------------------------- B1
P("## 2. B1 kernel benchmark (measured, NVIDIA A40; 3 warm-up + 20 timed runs, L2 flushed; us = microseconds)\n")
P(f"kv4 bit-exact vs Python reference on this GPU (small shape): {info.get('kv4_bitexact_K_small')}. FP8: A40 has no native FP8; torch.compile fails (Triton: 'type fp8e4nv not supported in this architecture'); eager software cast is the only FP8 measurement on this GPU.\n")
rows = []
for key, c in k["cases"].items():
    if "copy_ceiling" not in c: rows.append([key, json.dumps(c), "", "", "", "", ""]); continue
    cc = c["copy_ceiling"]
    rows.append([key, c["layers_measured"], c["chunked"], c["raw_bytes_KV_measured"], f1(cc["median_us"]), f1(cc["p5_us"]), f1(cc["p95_us"]) + f" | {cc['GBps_moved']:.1f} GB/s"])
P("### 2a copy ceiling (D2D copy of K; GB/s = bytes read + written / time)\n")
tbl(["case", "layers measured", "chunked", "raw K+V bytes measured", "median us", "p5 us", "p95 us | GB/s moved"], rows)
for d, name in (("quant", "compress"), ("dequant", "decompress")):
    for fmt in ("kv4", "int8_row", "fp8"):
        P(f"### 2b {fmt} {name}\n"); rows = []
        for key, c in k["cases"].items():
            if "copy_ceiling" not in c or fmt not in c[d]: continue
            r = c[d][fmt]
            rows.append([key, r.get("chosen", "kernel"), f1(r["median_us"]), f1(r["p5_us"]), f1(r["p95_us"]), f1(r["GBps_moved"]), f1(r["GBps_input"]), f1(r["pct_of_copy_ceiling"]),
                         r["meets_90pct_target"], f1(r.get("eager_median_us")), f1(r.get("compile_median_us")), (r.get("compile_error", "")[:60].replace("\n", " "))])
        tbl(["case", "chosen", "median us", "p5 us", "p95 us", "GB/s moved", "GB/s input", "% of copy ceiling", ">=90%", "eager median us", "compile median us", "compile error (first 60 chars)"], rows)
P("### 2c wire sizes\n"); rows = []
for key, c in k["cases"].items():
    if "copy_ceiling" in c: rows.append([key, c["raw_bytes_KV_measured"], c.get("kv4_wire_bytes"), f1(c["raw_bytes_KV_measured"] / c["kv4_wire_bytes"], 3), c.get("int8_wire_bytes"), f1(c["raw_bytes_KV_measured"] / c["int8_wire_bytes"], 3), c.get("fp8_wire_bytes"), f1(c["raw_bytes_KV_measured"] / c["fp8_wire_bytes"], 3)])
tbl(["case", "raw bytes", "kv4 bytes", "kv4 ratio", "int8 bytes", "int8 ratio", "fp8 bytes", "fp8 ratio"], rows)

# ---------------------------------------------------------------- B2
P("## 3. B2 prefill timing (measured; chunk 4096, random tokens, last-token logits)\n")
for f in sorted(glob.glob(str(R / "prefill_*_a40.json"))):
    j = J(f); P(f"### {j['model']} on {j['gpu']} (max ctx {j['max_ctx']}, reps {j['reps']}, throttled {j['info']['throttled']}, wall {j['info']['seconds']:.0f} s)\n")
    P("Repetition totals (s): " + ", ".join(f"{sum(c)/1e3:.2f}" for c in j["chunk_ms_all"]) + "\n")
    P("Per-chunk median ms (chunk index i covers tokens 4096*i .. 4096*(i+1)): " + ", ".join(f"{x:.1f}" for x in j["chunk_ms_median"]) + "\n")
    tbl(["n new", "m cached", "ms", "source"], [[p["n"], p["m"], f1(p["ms"], 2), p["source"]] for p in j["points"]])
if (R / "profile_a40.json").exists():
    pr = J(R / "profile_a40.json")
    P(f"### Cost-model profile (A40): copy ceiling {pr['copy_ceiling_GBps']:.2f} GB/s, peak {pr['peak_tflops']} TFLOPs, eff_ref {pr['eff_ref']:.4f}\n")
    for m, v in pr["prefill"].items():
        P(f"**{m}** t = c0 + c1 n + c2 n^2 + c3 n m; coef = {v['coef']}; max rel err {v['fit']['max_rel_err']:.4f}, mean {v['fit']['mean_rel_err']:.4f}, max for n+m >= 32k {v['fit']['max_rel_err_n_ge_32k']:.4f}; points {v['fit']['n_points']}\n")
        tbl(["n", "m", "measured ms", "fitted ms", "rel err"], [[p["n"], p["m"], f1(p["measured_ms"], 2), f1(p["fitted_ms"], 2), f"{p['rel_err']:.4f}"] for p in v["fit"]["per_point"]])
    from kvpick.study.cost_ext import LineFit
    P("### Kernel line fits t = a + bytes/bw (fitted from the B1 points)\n"); rows = []
    for fmt, dd in pr["kernel_points"].items():
        for dn, pts in dd.items():
            lf = LineFit.fit(pts); rows.append([fmt, dn, len(pts), f"{lf.a_ms:.4f}", f"{lf.bw_Bps/1e9:.1f}", f"{lf.max_rel_err:.3f}", f"{lf.max_bytes:.3g}"])
    tbl(["format", "direction", "points", "a (ms, fixed)", "bw (GB/s of raw input)", "max rel err of fit", "max bytes measured"], rows)

# ---------------------------------------------------------------- outliers, determinism
P("## 4. Key-channel outliers, Llama-3.1-8B (post-RoPE keys; 5 WikiText-2 windows of 2048 + 2 PG-19 prefixes of 8192)\n")
o = J(R / "outliers_llama-3.1-8b.json")
P(f"layers {o['layers']}, prompts {o['n_prompts']}, layers with top-1%/median >= 5: {o['layers_with_ratio_ge_5']}, median ratio {o['median_ratio']:.3f}, max max/median {o['max_ratio']:.3f}\n")
tbl(["layer", "top-1% amax / median", "max amax / median"], [[i, f"{a:.3f}", f"{b:.3f}"] for i, (a, b) in enumerate(zip(o["top1pct_over_median_per_layer"], o["max_over_median_per_layer"]))])
P("## 5. Determinism gate\n\n```\n" + json.dumps(J(R / "determinism_llama-3.1-8b.json")["result"], indent=1) + "\n```\n")

# ---------------------------------------------------------------- runs, pilots, accuracy
P("## 6. Every task run (pilot and full): wall time, samples, GPU state\n"); rows = []
for f in sorted(glob.glob(str(R / "acc/*/*/run_*.json"))):
    j = J(f); p = j["pilot"]; i = j["info"]
    rows.append([pathlib.Path(f).parent.parent.name, p["task"], "+".join(p["formats"]), p["limit"], p["samples_total_in_sink"], f1(p["seconds"], 1), f1(p["seconds_per_sample"], 2), i.get("lengths", ""), i["throttled"], i["gpu_start"]["gpus"][0]["clocks.sm"] if i["gpu_start"].get("gpus") else "", i["gpu_end"]["gpus"][0]["clocks.sm"] if i["gpu_end"].get("gpus") else "", i["gpu_end"]["gpus"][0]["clocks_throttle_reasons.active"] if i["gpu_end"].get("gpus") else ""])
tbl(["model", "task", "formats", "limit", "samples in sink", "seconds", "s/sample", "lengths", "throttled", "sm clock start MHz", "sm clock end MHz", "throttle reasons end"], rows)
P("## 7. Accuracy samples recorded so far (partial results are NOT final results; n shown everywhere)\n")
for model in sorted(p.name for p in (R / "acc").glob("*") if p.is_dir() and not p.name.endswith("_tiny")):
    for tdir in sorted((R / "acc" / model).glob("*")):
        data = {}
        for f in tdir.glob("*.jsonl"): data[f.stem] = [json.loads(l) for l in open(f) if l.strip()]
        if not data: continue
        P(f"### {model} / {tdir.name}\n"); rows = []
        for fmt, recs in sorted(data.items()):
            by = collections.defaultdict(list)
            for r in recs:
                if "skipped" in r: continue
                by[r["ctx"]].append(r)
            for ctx, rs in sorted(by.items()):
                if "score" in rs[0]: rows.append([fmt, ctx, len(rs), f"{sum(r['score'] for r in rs)/len(rs):.4f}", sum(r["score"] for r in rs), ""])
                else:
                    import math; m = sum(r["nll"] for r in rs) / len(rs); rows.append([fmt, ctx, len(rs), f"mean nll {m:.5f}", f"ppl {math.exp(m):.4f}", ""])
        tbl(["format", "ctx", "n", "mean score / nll", "correct count / ppl", ""], rows)
        if tdir.name == "niah":
            P("Per-(ctx, depth) recall by format (n per cell in parentheses):\n"); cells = collections.defaultdict(lambda: collections.defaultdict(list))
            for fmt, recs in data.items():
                for r in recs: cells[(r["ctx"], r["depth"])][fmt].append(r["score"])
            fm = sorted(data); rows = [[c, d] + [f"{sum(v[f])/len(v[f]):.2f} ({len(v[f])})" if f in v else "" for f in fm] for (c, d), v in sorted(cells.items())]
            tbl(["ctx", "depth %"] + fm, rows)
            P("Every sample where any format disagrees with the baseline:\n"); rows = []
            base = {r["id"]: r for r in data.get("raw", [])}
            for fmt, recs in data.items():
                for r in recs:
                    b = base.get(r["id"])
                    if fmt != "raw" and b and (b["score"] != r["score"] or b["output"] != r["output"]): rows.append([fmt, r["id"], b["score"], r["score"], repr(b["output"][:40]), repr(r["output"][:40]), r["gold"]])
            tbl(["format", "sample id", "baseline score", "format score", "baseline output", "format output", "gold"], rows or [["(none)", "", "", "", "", "", ""]])

# ---------------------------------------------------------------- sweep
for sf in ("sweep.json", "sweep_a40_partial.json"):
    if (R / sf).exists():
        s = J(R / sf); P(f"## 8. Sweep summary ({sf}; modeled; profiles {s['profiles']})\n"); cnt = collections.Counter((r['model'], r['gpu'], r['mode'], r['kv4_verdict']) for r in s["rows"])
        tbl(["model", "gpu", "mode", "verdict", "grid points"], [[*k_, v] for k_, v in sorted(cnt.items())]); break
P("## 9. Pod logs (verbatim tails)\n")
for f in ("ext_tests_cpu.log", "ext_tests_gpu.log", "ext_session_a40.out"):
    p = ROOT / "logs/pod_ext" / f
    if p.exists(): P(f"### {f}\n\n```\n" + "".join(open(p).readlines()[-40:]).replace("```", "'''") + "```\n")
(ROOT / "docs/experiment_log.md").write_text("\n".join(out)); print("wrote docs/experiment_log.md", len(out), "lines")
