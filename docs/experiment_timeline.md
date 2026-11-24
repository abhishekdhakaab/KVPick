# Extension study: chronological log (hand-written events; all numbers also appear in docs/experiment_log.md, which is script-generated)

Times: Mac local clock for development, pod UTC for pod steps. $/h of the A40 pod: NOT YET PROVIDED by the user at the time of writing (budget log therefore carries an assumed $0.44/h, marked as assumption).

## Development (no GPU cost)
1. Pre-registration committed : thresholds, definitions, budget rules. Addendum 1 (time convention, best-alternative set, breakeven definition, 70B TP-8 assumption with TP efficiency 0.8, rtt 0.5 ms, prefill measured from chunk traces) committed before any extension data.
2. Code written and committed : `kvpick/study/` (models, overlap, cost_ext, formats, longctx, tasks, stats, runlog), scripts `ext_*`, tests. CPU tests: 129 passed (73 GPU tests deselected locally).
3. Bug found locally: `LineFit` bandwidth unit (ms/byte -> bytes/s) was off by 1e6; caught by `test_study_cost.py`, fixed before use.
4. Bug found locally: `datasets.load_dataset("emozilla/pg19")` downloads the 11 GB train split too and hung; replaced by reading only the needed parquet (D42).
5. Hugging Face check from the Mac: token valid, Llama-3.1-8B-Instruct access approved (config.json downloaded); Llama config verified: 32 layers, 8 KV heads, hidden 4096, 32 attention heads, max_position_embeddings 131072, rope_scaling llama3 factor 8.

## Pod session 1 (NVIDIA A40, 46068 MiB, driver 595.112.07, torch 2.8.0+cu128, CUDA 12.8, 503 GB host RAM shown, 50 GB container disk, 96 cores)
- Setup 1 failed at the Llama download (401 gated): setup script sets HF_HOME=/workspace/hf, token file was in ~/.cache/huggingface. Fixed by copying the token to /workspace/hf/token. Setup 2: Llama (12 files) fetched in 38 s, Qwen (10 files) in 13 s; total /workspace/hf 23 GB; extension built (`kvq built, cuda = True`).
- GPU tests alone: 73 passed in 52.25 s (new shapes L=32/80, H=1, T not multiple of 32 included; versions v0/v1/v2).
- Session script `ext_session_a40.sh` started 00:49:01 UTC: tests_cpu, tests_gpu (+17 s), bench_kernels 00:49:24 -> 00:51:57 (153 s), prefill_llama 00:51:57 -> 01:01:29 (572 s), prefill_qwen -> outliers 01:02:31 -> determinism 01:02:54 -> pilots; session finished after 1697 s (about 28 min).
- First bench_kernels run (overwritten later): FP8 8-9% / 16-17% of ceiling (eager), OOM at L80_T4096, L80_T16384, L80_T131072, skipped L32_T524288 and above ("does not fit even for one layer"). Causes: (a) FP8 torch.compile raised InductorError (Triton: `type fp8e4nv not supported in this architecture`; A40 is sm_86 Ampere, no native FP8) so FP8 falls back to eager software cast; (b) memory not freed between cases (closure/GC) and the 12x sizing factor too small. Fix: independent compile attempts, record compile errors in the JSON, gc.collect + empty_cache before measuring free memory, sizing factor 20, ring of 3 inputs only for tiny cases. Rerun: 184 s, only the three 1M-token cases do not fit.
- Outliers (Llama-3.1-8B): 0 of 32 layers with top-1%/median >= 5; median ratio 2.934; max over layers of max/median 6.4. (Qwen3-4B earlier: >= 5x in 27 of 28 layers of the 0.6B preview; 4B figure in results/feasibility/G1_kv_stats.json.)
- Determinism gate: PASS (GSM8K 32 samples identical, NIAH 3 identical, raw-injection first token equal).
- Pilot NIAH 128k failed: `torch.OutOfMemoryError` at `DynamicCache.update` (256 MiB request, 38.97 GiB allocated). Cause: HF cache copies on update, and my restore built a full list of restored layers first (about 2 x 16.7 GB). Fix: generator that restores and inserts one layer at a time (commit "restore cache layer by layer"). Rerun: 2 samples in 299.3 s total = 149.6 s/sample for 3 formats (prefill 139.9 s).
- Pilot WikiText failed: OOM (43.9 GiB allocated): the task forgot `torch.no_grad()`. Fixed for all tasks; rerun 8 windows, 4 formats, 9.0 s.
- Pilot timings (s/sample, 3 formats unless noted): NIAH 8k 2.66; NIAH 32k 16.03; NIAH 128k 149.6; PG-19 8k 3.19; PG-19 32k 17.34; GSM8K (4 formats) 3.01; WikiText (4 formats) 1.13.
- Code on the pod = the commit recorded in each JSON as `commit` (an intermediate commit, squashed before publishing) plus later copied patches (tasks.py no_grad, formats.py/longctx.py layer-by-layer, ext_bench_kernels.py). The `commit` field therefore understates the patches; the final repo commit is the source of truth.

## Full runs on the A40 (started ~01:28 UTC; `ext_run_full.sh`, FP8=1)
- niah_8k finished 01:28:06 -> niah_32k finished 01:45:47: 55 trials per format at 8k and 32k; all four formats (raw, int8_row, fp8, kv4) 55/55 at both lengths.
- niah_128k started 01:45:47. Interim 02:41:02 UTC: raw 21/22, int8_row 20/21, fp8 20/21, kv4 20/21 (interim). At 03:55:57: 50/51 for every format. Finished 04:04:13 UTC (about 2 h 19 min): 54/55 for raw, int8_row, fp8 and kv4; all compressed outputs identical to baseline outputs; the single miss (id 131072_0_3, depth 0%, output '1', gold 3246204) is also a baseline miss.
- Step times (UTC): pg19_8k 04:04:13 -> 04:08:19 (4 min); pg19_32k 04:08:19 -> 04:27:25 (19 min); gsm8k_llama 04:27:25 -> 05:44:40 (77 min, 5 formats, 1319 questions); wiki_llama 05:44:40 -> 05:47:15 (2.6 min); gsm8k_qwen 05:47:15 -> 06:42:31 (55 min); wiki_qwen 06:42:31 -> ALLDONE about 06:45.
- 06:12 UTC: one ssh call returned 'Connection reset by peer' (client side); the pod and run were unaffected.
- 07:26:47 UTC: the next status check found the run had finished at about 06:45; the pod idled about 40 minutes (about $0.30 at the assumed $0.44/h). All results were copied off and committed before the pod was terminated.

## Final A40 accuracy results (script: scripts/ext_acc_report.py; interval = paired bootstrap 95%, 2000 resamples; class by point estimate)
Llama-3.1-8B baseline: GSM8K 81.9% (n=1319); WikiText-2 ppl 6.526 (141 windows); PG-19 ppl 10.102 (95 books, 8k) and 9.723 (77 books, 32k); NIAH 100/100/98.2% at 8k/32k/128k (55 trials each).
| task | format | n | change [interval] | R>W / W>R | outputs differ | class |
|---|---|---|---|---|---|---|
| GSM8K | int8_row | 1319 | +0.45 pts [-0.76, +1.67] | 31/37 | 552 | strict |
| GSM8K | fp8 | 1319 | -1.59 pts [-2.96, -0.15] | 58/37 | 775 | normal |
| GSM8K | kv4 | 1319 | -0.99 pts [-2.50, +0.45] | 55/42 | 870 | normal |
| GSM8K | int4_g32 | 1319 | -1.74 pts [-3.34, -0.15] | 73/50 | 994 | normal |
| WikiText-2 | int8_row | 141 | -0.01% [-0.04, +0.01] | | | strict |
| WikiText-2 | fp8 | 141 | +0.08% [+0.04, +0.11] | | | strict |
| WikiText-2 | kv4 | 141 | +0.19% [+0.14, +0.25] | | | strict |
| WikiText-2 | int4_g32 | 141 | +0.22% [+0.17, +0.28] | | | strict |
| PG-19 8k | int8_row | 95 | +0.02% [-0.02, +0.05] | | | strict |
| PG-19 8k | kv4 | 95 | +0.15% [+0.07, +0.22] | | | strict |
| PG-19 32k | int8_row | 77 | +0.04% [0.00, +0.08] | | | strict |
| PG-19 32k | kv4 | 77 | +0.25% [+0.14, +0.36] | | | strict |
| NIAH 8k/32k/128k | int8_row, fp8, kv4 | 55 each | 0.00 pts [0, 0] | 0/0 | 0, 0 (kv4 32k: 1 text differs, same score), 0 | strict |
Qwen3-4B baseline: GSM8K 90.5% (n=1319), WikiText-2 145 windows.
| task | format | n | change [interval] | R>W / W>R | outputs differ | class |
|---|---|---|---|---|---|---|
| GSM8K | int8_row | 1319 | +0.15 pts [-0.38, +0.68] | 6/8 | 351 | strict |
| GSM8K | kv4 | 1319 | -0.61 pts [-1.52, +0.30] | 23/15 | 710 | normal |
| GSM8K | int4_g32 | 1319 | -0.45 pts [-1.44, +0.53] | 26/20 | 1007 | strict |
| WikiText-2 | int8_row | 145 | +0.03% [-0.00, +0.07] | | | strict |
| WikiText-2 | kv4 | 145 | +0.07% [-0.00, +0.14] | | | strict |
| WikiText-2 | int4_g32 | 145 | +1.11% [+0.94, +1.29] | | | normal |
Note: Qwen GSM8K KV4 (-0.61 pts) is class normal (loss above 0.5 pts), not strict.
PG-19 books skipped because shorter than prefix+512 tokens: 5 (8k), 23 (32k).


## Estimates used for planning (from pilots)
- Remaining A40 time at 02:00 UTC: about 4.8 h (NIAH 128k 2.0, PG-19 0.6, Llama GSM8K+Wiki with FP8 1.5, Qwen 0.7).
- Cost estimate at the assumed $0.44/h: about $2.1 for that remainder.
