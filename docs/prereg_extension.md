# Pre-registration: KVPick extension (long-context breakeven + full accuracy study)

Written 2026-11-23, before any GPU run of the extension. Nothing below is changed after seeing data; a missed target is reported as missed.

## Budget
BUDGET_USD = 10 (target 8-9). Interim report at $6, stop at $9. Every pod start/stop/price goes in `docs/budget_log.md`.

## Labels
Every number is one of: **measured** (real hardware, this run), **fitted** (cost model fitted to measured points), **modeled** (extrapolated: 70B prefill, links faster than hardware, lossless codec, overlapped mode). Different GPU types are never mixed in one curve.

## Quality budget classes (existing repo classes, by point estimate)
- strict: <= 0.5 points loss on every task; normal: <= 2.5; lenient: <= 5.
- Perplexity uses relative change: strict 0.5%, normal 2%, lenient 5%.
- NIAH and RULER: loss = drop in average score at that context length.
- A difference counts only if the paired-bootstrap 95% interval excludes zero; otherwise "no clear difference".

## Definitions
- **KV4 pays off** at a grid point: time-to-KV-ready lower than raw AND lower than recompute, and saving vs the best alternative >= 10% AND >= 10 ms. Otherwise "marginal" (positive but under threshold) or "does not pay off".
- **Breakeven link speed**: interpolated link speed at which the best alternative first beats KV4 as winner by the rule above. Interpolation only, never extrapolation.
- **KV4 better than FP8** at a point only if time is lower by >= 10% and >= 10 ms and KV4's quality class is no worse than FP8's, or FP8 misses the quality budget while KV4 meets it. Otherwise "no clear advantage over FP8".

## Targets
- Kernel: compress and decompress >= 90% of the device copy ceiling on both GPUs at T >= 8k.
- Cost model: error <= 10% against measured points, including long context.
- Quality claim: KV4 is "usable" at a context length only if its class at that length (long-context tasks) is <= lenient. If worse at long than short context, say so in the first paragraph of the report.
- Baseline determinism: two baseline runs on a small slice must match before any format comparison.

## Protocol fixed in advance
Greedy decoding; only the prompt cache is compressed (generated tokens keep a 16-bit cache); same prompts/prefill/samples for every format; MC and perplexity tasks compress everything except the final scoring tokens. Benchmarks run at full size or are dropped whole, in this priority order (drop from the bottom): (1) Part A, B1, B2 on 48GB card; (2) NIAH 8k/32k/128k + PG-19 8k/32k, baseline/INT8/KV4; (3) full GSM8K + WikiText-2, Llama-3.1-8B and Qwen3-4B, baseline/INT8/KV4/INT4-g32, FP8 on Llama only (first to drop); (4) <=45 min fast-GPU session; (5) cost model and sweep (CPU); (6) PG-19 128k if pilot fits; (7) RULER. Cut: MMLU, LongBench, Qwen3-8B, two-GPU validation.
Pilot rule: 20-sample pilot per task, extrapolate cost, write the plan before the full run; if total estimate > $9 drop whole benchmarks from the bottom.

## Addendum 1 (2026-11-23, written before any extension data exists; clarifies definitions, changes no threshold)
- **Time convention.** "Time until KV ready" for the send strategies is the exposed time after the prefill GPU finishes the prompt (compress + wire + decompress + fixed overhead; in overlapped mode, what is left after the prefill ends). For recompute it is the prefill time of (new tokens | cached tokens) on the decode GPU. This is the convention of the original study. A TTFT-inclusive variant (send adds the source-GPU prefill) is stored beside it in the sweep JSON and reported as a sensitivity, not as the pre-registered result.
- **Best alternative** in the "pays off" rule = the fastest of raw, INT8, FP8 and recompute. The modeled lossless format is excluded from that set (it is a model of a published idea); a second flag reports whether KV4 still pays off when lossless-at-90%-of-ceiling is included.
- **Winner** in heatmaps = fastest of raw, INT8, FP8, KV4, recompute (time only, ignoring quality). Winners under quality budgets (strict/normal/lenient) use the classes measured in Part C; before those exist the old Qwen3-4B classes are used and marked provisional.
- **Breakeven link speed** is reported as two numbers: the upper one (above it, a faster link makes raw/INT8/FP8 better than KV4 by the rule) which is the pre-registered one, and the lower one (below it recompute beats KV4). Both are interpolated in log10(link) on s = min(saving_fraction/0.10, saving_ms/10) - 1 (pays off iff s >= 0), between adjacent grid points only.
- **Decode GPU** is idle in the sweep (decode batch 0), so the ITL-SLO rule excludes nothing; the rule stays implemented and tested.
- **70B (modeled)**: tensor parallel 8 ranks. Each rank compresses and sends its 1/8 shard of the KV over its own link of the stated speed; prefill is FLOPs-based with the achieved-efficiency factor from the measured 8B points and a fixed tensor-parallel efficiency of 0.8 (assumption, stated in the report).
- **Link fixed overhead** t_fixed = 0.5 ms (the existing rtt assumption). Kernel launch overhead is fitted per format from the measured kernel points (intercept of time vs bytes).
- **Prefill measurement** (B2) comes from chunked-prefill traces (chunk 4096, last-token logits): T(n new | m cached) is the sum of the measured chunk times between positions m and m+n. Small n (<= 2048) is timed directly.

## Addendum 2 (2026-11-24; written after the A40 kernel benchmark showed FP8 cannot be compiled on Ampere; changes no threshold, adds a sensitivity)
- The A40 (sm_86) has no native FP8: Triton refuses to compile FP8 casts for it, so FP8 on the A40 is the PyTorch eager software cast (measured 8-17% of the copy ceiling). That measured number stays the pre-registered FP8 time for the A40 profile.
- Because that would flatter KV4 against FP8, every sweep row also carries a modeled `fp8_ideal` strategy (FP8 cast at 90% of the copy ceiling of that GPU, ratio 2.0, labeled modeled, as hardware FP8 casts behave) and the flag `pays_off_vs_fp8_ideal`. The report states both the pre-registered verdict and this sensitivity, and says where they differ. The H100/A100 profile measures FP8 with torch.compile (native FP8 hardware) and needs no such substitute.
