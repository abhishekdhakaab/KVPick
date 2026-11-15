# C2.1 KVServe (github.com/hpdps-group/KVServe) — read-only review, 2026-10-16

Clone: `feasibility/external/kvserve` (shallow, HEAD of main on 2026-10-16). Nothing executed.

| Question | Finding |
|---|---|
| License | Apache-2.0 (`LICENSE`, `pyproject.toml`). Permits use and comparison. |
| What it is | A vLLM V1 **KV connector** (`kvserve_v1.connector.compressed_kv_connector`) plus a transport-independent "compression runtime". Pipeline: `Raw KV -> Transform (Hadamard) -> Quantizer -> Codec (nvCOMP ANS or LC)`. |
| vLLM version | `requirements.txt`: `vllm>=0.18.0`, `torch>=2.10.0`, `transformers>=4.57.6`, `triton>=3.6.0`, `flashinfer-python>=0.6.6`, `ray`, `cupy-cuda12x`, `nvidia-nvcomp-cu12`. A `kvsfuse-vllm0.30.0/` tree targets vLLM 0.30.0 (Mooncake connector). The plan's "vLLM 0.10.1" is stale. |
| Install | `pip install -e .` inside an env that already has vLLM. Runtime extra: `kvserve_v1[runtime,runtime-cu12]` pulls `humming-kernels` and nvCOMP 5.3. Optional `[fused]` pulls TileLang 0.1.9 (JIT, CUDA Linux only). |
| Quantizers | `quantizer_func.py`: min/max asymmetric, axis `channel` (reduce over blocks+tokens, i.e. **whole-sequence per channel**), `token` (per token over heads+dims), or `tensor`. Output is **uint8 per value** with `max_value` 12 or 16 (defaults: high=16, low=12). Hybrid precision: 30% of heads (by precomputed score CSV per model) get the lower level count. Scale/min stored in bf16/fp16. |
| Transforms | Random-sign Hadamard (`hadamard_func.py`, 7 butterfly stages for head_dim 128). |
| Codecs | nvCOMP ANS (`codec_type=nvcomp`), LC framework (`codec_type=lc`), or none. **The byte reduction comes from the codec**, not from the quantizer (which stays 1 B/value). |
| Kernel type | Three implementations: (1) plain PyTorch (`quantizer_func.py`); (2) CUDA runtime `compression/runtime/csrc/kvs_compress.cu` (736 lines): `quantize_channel_kernel` (8 channels per block, 32 token lanes, warp shuffles), `quantize_token_kernel` (one warp per token), scalar `dequantize_*` kernels (loop over tokens per thread, no vector loads), gather/scatter/FWHT stage kernels; (3) TileLang fused Hadamard+quant (`tilelang_ops/kv_hadamard_quant.py`, 1936 lines), restricted to `axis_key=channel`, `axis_value=token`. |
| Speed benchmark | `scripts/bench_codec_throughput.py` times compress/decompress of synthetic bf16 KV through `KVCompressionAdapter` for 4 profiles and prints "theoretical" transfer times at 20/50/100/200 Gbps. `offline_search/evaluation/component_speed/{quantizer,kivi,hadamard,ans}_speed.py` time individual stages but hard-code `/root/workspace/models`. Reusable in principle; all require the vLLM-side environment (`kvserve_v1.compression.manager` imports vLLM-adjacent utilities) and nvCOMP. |
| GPUs needed | Two engines (one producer, one consumer), each one GPU. TP=1 is fine. Tested configs in README use 2 GPUs. No requirement beyond 2. |
| Reported numbers | README badges: up to 9-10x KV compression, 8-9x lower PD communication time, accuracy "preserved". The plan's "8.28x at ~100% accuracy on Qwen2.5-7B" matches the paper (arXiv 2605.13734). |
| Relevance to R6 | Usable as a baseline **only for a quantize+codec throughput number**, and only if nvCOMP installs on the pod. Its quantizer alone is not comparable to a bit-packed INT4 kernel (different output format). Decision: in G4, time KVServe's `original_nvcomp` profile via `bench_codec_throughput.py` if the env builds in <= 3 attempts; otherwise mark R6 `BLOCKED` and limit claims to our own reference baselines. |

Observations relevant to the project's positioning:
1. KVServe already ships fused CUDA/TileLang kernels for exactly the KIVI-style axes (per-channel K, per-token V). "Our own fused kernels" is therefore an engineering exercise with an existing open-source analogue, not a novel artifact.
2. KVServe's CUDA dequantize kernels are scalar and token-strided; its quantize kernels do not use vectorized loads. There is plausibly headroom over KVServe's kernels specifically, but R5 is defined against PyTorch/torch.compile, not against KVServe.
3. KVServe's `controller/` implements a bandwidth/SLO/accuracy-aware profile selector (bandit + analytical model). "Choose compression by link speed and quality budget" is fully covered. RECOMPUTE is **not** an action in KVServe's controller (checked: `controller/online_controller.py`, `analytical_model.py` names only compression profiles).
