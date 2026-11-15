### Kernel benchmark (NVIDIA A40, torch 2.8.0+cu128, CUDA 12.8; 36-layer cache, K and V; median of 100 runs, L2 flushed)

| T | copy ceiling | metric | eager PyTorch | torch.compile (max-autotune) | v0 naive | v1 tiled | v2 fused |
|---|---|---|---|---|---|---|---|
| 512 | 545 GB/s, 139 us | compress | 4086 us (4%) | 335 us (54%) | 376 us (48%) | 177 us (103%) | 173 us (105%) |
| 512 | 545 GB/s, 139 us | decompress | 2404 us (8%) | 182 us (100%) | 198 us (92%) | 190 us (96%) | 190 us (96%) |
| 2048 | 558 GB/s, 541 us | compress | 16292 us (4%) | 1297 us (55%) | 1439 us (49%) | 665 us (107%) | 675 us (105%) |
| 2048 | 558 GB/s, 541 us | decompress | 9437 us (8%) | 711 us (100%) | 767 us (92%) | 740 us (96%) | 740 us (96%) |
| 8192 | 559 GB/s, 2161 us | compress | 64821 us (4%) | 5104 us (56%) | 5753 us (49%) | 2630 us (108%) | 2725 us (104%) |
| 8192 | 559 GB/s, 2161 us | decompress | 37431 us (8%) | 2824 us (100%) | 3094 us (92%) | 2939 us (97%) | 2939 us (96%) |

At T=8192: compress v2 is 1.9x faster than torch.compile and 24x faster than eager; compression ratio 3.20x; v2 bit-exact vs reference: K=True, V=True.

### Quality vs RAW (Qwen/Qwen3-4B-Instruct-2507, bf16, cuda; kernel in the loop for KV4: True; 95% paired bootstrap CIs)

| format | ratio | WikiText-2 ppl (n=50) | ppl change | GSM8K acc (n=100) | change (pts) | Qasper F1 (n=50) | change (pts) | quality class |
|---|---|---|---|---|---|---|---|---|
| raw | 1.0x | 8.943 |  | 0.89 |  | 0.395 |  | strict |
| int8_row | 1.94x | 8.942 | -0.01 [-0.08, +0.06]% | 0.90 | +1.00 [+0.00, +3.00] | 0.392 | -0.29 [-0.85, +0.00] | strict |
| int4_row | 3.76x | 14.176 | +58.52 [+53.25, +64.45]% | 0.64 | -25.00 [-36.00, -15.00] | 0.245 | -15.05 [-24.49, -5.82] | none |
| int4_row_g32 | 3.2x | 9.045 | +1.14 [+0.81, +1.53]% | 0.88 | -1.00 [-6.00, +4.00] | 0.375 | -2.04 [-6.03, +1.72] | normal |
| int4_kivi | 3.2x | 8.949 | +0.08 [-0.04, +0.20]% | 0.86 | -3.00 [-7.00, +1.00] | 0.372 | -2.31 [-5.21, -0.49] | lenient |

### Transfer model vs two-GPU measurement (2x NVIDIA A40, 36 layers, rtt 0.5 ms, pinned-staged copies; 24 points, max |err| 3.8%, mean 0.36%)

| n new tokens | format | link Gbps | measured serial ms | predicted ms | error | pipelined (36 chunks) ms |
|---|---|---|---|---|---|---|
| 512 | raw | 1 | 604.5 | 604.5 | +0.0% | 604.6 |
| 512 | raw | 3 | 201.9 | 201.8 | +0.0% | 201.9 |
| 512 | raw | 10 | 60.9 | 60.9 | +0.0% | 61.0 |
| 512 | raw | 25 | 24.7 | 24.7 | +0.1% | 24.7 |
| 512 | int4_kivi | 1 | 190.0 | 189.6 | +0.2% | 189.8 |
| 512 | int4_kivi | 3 | 63.9 | 63.8 | +0.3% | 63.6 |
| 512 | int4_kivi | 10 | 20.0 | 19.7 | +1.1% | 19.6 |
| 512 | int4_kivi | 25 | 8.7 | 8.4 | +3.8% | 14.8 |
| 2048 | raw | 1 | 2416.4 | 2416.4 | +0.0% | 2416.5 |
| 2048 | raw | 3 | 805.8 | 805.8 | +0.0% | 806.0 |
| 2048 | raw | 10 | 242.1 | 242.1 | +0.0% | 242.3 |
| 2048 | raw | 25 | 97.2 | 97.1 | +0.0% | 97.3 |
| 2048 | int4_kivi | 1 | 757.4 | 756.9 | +0.1% | 756.1 |
| 2048 | int4_kivi | 3 | 254.0 | 253.6 | +0.2% | 252.8 |
| 2048 | int4_kivi | 10 | 77.8 | 77.4 | +0.6% | 76.6 |
| 2048 | int4_kivi | 25 | 32.7 | 32.1 | +1.8% | 31.1 |
| 8192 | raw | 1 | 9664.2 | 9664.2 | +0.0% | 9664.4 |
| 8192 | raw | 3 | 3221.8 | 3221.7 | +0.0% | 3221.9 |
| 8192 | raw | 10 | 966.9 | 966.9 | +0.0% | 967.0 |
| 8192 | raw | 25 | 387.1 | 387.0 | +0.0% | 387.2 |
| 8192 | int4_kivi | 1 | 3026.6 | 3026.1 | +0.0% | 3020.8 |
| 8192 | int4_kivi | 3 | 1013.2 | 1012.8 | +0.0% | 1007.8 |
| 8192 | int4_kivi | 10 | 308.5 | 308.2 | +0.1% | 303.1 |
| 8192 | int4_kivi | 25 | 127.2 | 127.0 | +0.2% | 121.8 |

### Recompute vs KV4 send, measured end to end on two GPUs

| n new tokens | real recompute on GPU 1 (ms) | model (ms) | measured crossover (Gbps) | predicted (Gbps) |
|---|---|---|---|---|
| 512 | 61 | 68 | 3.5 | 2.9 |
| 2048 | 246 | 219 | 3.5 | 3.5 |

### Planner crossovers (cost model; decode GPU idle)

| budget | cached tokens | new tokens | fastest action by link speed |
|---|---|---|---|
| normal | 0 | 128 | >= 1.0 Gbps: recompute; >= 2.37 Gbps: int8_row |
| normal | 0 | 512 | >= 1.0 Gbps: recompute; >= 4.73 Gbps: int8_row |
| normal | 0 | 2048 | >= 1.0 Gbps: recompute; >= 5.96 Gbps: int8_row |
| normal | 0 | 8192 | >= 1.0 Gbps: recompute; >= 4.47 Gbps: int8_row |
| normal | 2048 | 128 | >= 1.0 Gbps: recompute; >= 2.0 Gbps: int8_row |
| normal | 2048 | 512 | >= 1.0 Gbps: recompute; >= 3.55 Gbps: int8_row |
| normal | 2048 | 2048 | >= 1.0 Gbps: recompute; >= 4.22 Gbps: int8_row |
| normal | 2048 | 8192 | >= 1.0 Gbps: recompute; >= 3.35 Gbps: int8_row |
| normal | 8192 | 128 | >= 1.0 Gbps: recompute; >= 1.41 Gbps: int8_row |
| normal | 8192 | 512 | >= 1.0 Gbps: recompute; >= 2.0 Gbps: int8_row |
| normal | 8192 | 2048 | >= 1.0 Gbps: recompute; >= 2.11 Gbps: int8_row |
| normal | 8192 | 8192 | >= 1.0 Gbps: recompute; >= 1.88 Gbps: int8_row |
| lenient | 0 | 128 | >= 1.0 Gbps: recompute; >= 1.41 Gbps: int4_kivi |
| lenient | 0 | 512 | >= 1.0 Gbps: recompute; >= 2.99 Gbps: int4_kivi |
| lenient | 0 | 2048 | >= 1.0 Gbps: recompute; >= 3.55 Gbps: int4_kivi |
| lenient | 0 | 8192 | >= 1.0 Gbps: recompute; >= 2.66 Gbps: int4_kivi |
| lenient | 2048 | 128 | >= 1.0 Gbps: recompute; >= 1.19 Gbps: int4_kivi |
| lenient | 2048 | 512 | >= 1.0 Gbps: recompute; >= 2.11 Gbps: int4_kivi |
| lenient | 2048 | 2048 | >= 1.0 Gbps: recompute; >= 2.51 Gbps: int4_kivi |
| lenient | 2048 | 8192 | >= 1.0 Gbps: recompute; >= 2.0 Gbps: int4_kivi |
| lenient | 8192 | 128 | >= 1.0 Gbps: int4_kivi |
| lenient | 8192 | 512 | >= 1.0 Gbps: recompute; >= 1.19 Gbps: int4_kivi |
| lenient | 8192 | 2048 | >= 1.0 Gbps: recompute; >= 1.33 Gbps: int4_kivi |
| lenient | 8192 | 8192 | >= 1.0 Gbps: recompute; >= 1.19 Gbps: int4_kivi |

### Planner vs fixed policies (synthetic multi-round workload, 1798 requests, cost model)

| link Gbps | planner mean ms | best fixed policy | its mean ms | planner gain (mean) | planner action mix |
|---|---|---|---|---|---|
| 1 | 314.6 | always_kv4 | 337.3 | +6.7% | int4_kivi 18%, int8_row 45%, recompute 36% |
| 2 | 164.4 | always_kv4 | 169.2 | +2.8% | int4_kivi 26%, int8_row 52%, recompute 22% |
| 3 | 112.3 | always_kv4 | 113.1 | +0.7% | int4_kivi 28%, int8_row 63%, recompute 9% |
| 5 | 68.3 | always_kv4 | 68.3 | +0.0% | int4_kivi 28%, int8_row 72% |
| 10 | 34.6 | always_kv4 | 34.6 | +0.0% | int4_kivi 28%, int8_row 72% |
| 25 | 14.4 | always_kv4 | 14.4 | +0.0% | int4_kivi 28%, int8_row 72% |
| 50 | 7.7 | always_kv4 | 7.7 | +0.0% | int4_kivi 28%, int8_row 72% |
| 100 | 4.3 | always_kv4 | 4.3 | +0.0% | int4_kivi 28%, int8_row 72% |

### Facts from the feasibility study (Qwen3-4B, A40)

- Key outlier channels: top-1% channel amax / median = 38.2x at layer 0; >= 5x in 13/36 layers (25 prompts).
- 4-bit key attention-output error: per-row 0.141, per-row g32 0.079, per-channel g32 0.056 (60% lower than per-row).
- Total paid GPU time: feasibility ~50 min 1x A40; kernel debugging ~55 min 1x RTX A5000; results ~45 min 2x A40.