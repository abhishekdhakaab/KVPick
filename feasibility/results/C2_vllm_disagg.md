# C2.3 vLLM disaggregated prefill — docs review, 2026-10-16

Sources (fetched from `vllm-project/vllm` main on 2026-10-16, saved in `feasibility/external/vllm_docs/`):
`docs/features/disagg_prefill.md`, `docs/features/nixl_connector_usage.md`. Latest PyPI release: **vLLM 0.31.0 (2026-10-05)**. NIXL on PyPI: **1.5.0** (`uv pip install nixl`).

## Status
Still marked **experimental** ("subject to change"). Nine connector types listed: ExampleConnector, LMCacheConnectorV1, **NixlConnector**, MooncakeConnector, MoRIIO (ROCm), MultiConnector, OffloadingConnector, FlexKVConnectorV1. No built-in KV compression option in the docs; KVServe is the out-of-tree one.

## Exact two-instance commands (NIXL connector, same host, two GPUs)
```bash
uv pip install vllm nixl          # pin exact versions in logs/env_*.txt at session time
export UCX_TLS=all UCX_NET_DEVICES=all

# prefiller on GPU 0
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# decoder on GPU 1
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# proxy (ships in the vLLM source tree, not the wheel; fetch the file from GitHub at the installed tag)
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200
# then POST /v1/completions to localhost:8192
```
Requirements: NIXL wheel (bundles UCX for CUDA on x86 Linux), unique `VLLM_NIXL_SIDE_CHANNEL_PORT` per instance, both instances on one node with two GPUs is the documented basic case ("Basic Usage (on the same host)"). `--enforce-eager` is in the doc example; drop it for realistic TTFT once it works.

## Alternative path if NIXL fails
LMCacheConnectorV1 example: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh` (also uses NIXL underneath, so it is not independent of NIXL). ExampleConnector (`shared_storage_path`) uses the filesystem and is the most robust fallback for a "does the split pair return a correct completion" smoke test, at the cost of unrealistic transfer times.

## Multi-turn note
NIXL connector now supports **bidirectional** transfer: on turn 2+, the prefiller pulls the decoder's cached blocks and prefills only new tokens. This is exactly the "m_cached" prefix-cache situation the KVPick planner models. The docs confirm the real system has the per-round state we assumed; it does not implement any send-vs-recompute decision.

## Risk for G7
Installing vLLM 0.31 + NIXL on a RunPod pod is a 10-20 minute install plus an 8 GB model download. Within the 1-hour two-GPU budget this leaves about 30 minutes for the run. G7 remains "only if 2a finished early", as the plan says.
