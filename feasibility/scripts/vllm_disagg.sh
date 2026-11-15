#!/usr/bin/env bash
# G7.2: vLLM disaggregated prefill with the NIXL connector on two GPUs (one node). Best effort, bounded attempts.
# Usage: bash scripts/vllm_disagg.sh   (from feasibility/). Writes results/G7_vllm_disagg.json and logs/vllm_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUT=results/G7_vllm_disagg.json; START=$(date -u +%FT%TZ)
status=BLOCKED; notes=()
mkdir -p logs
log(){ echo "[$(date -u +%T)] $*" | tee -a logs/vllm_disagg.log; }
json_out(){ python - "$status" "$START" "${notes[*]}" <<'PY'
import json,sys,datetime
st,start,notes=sys.argv[1],sys.argv[2],sys.argv[3]
d={"check_id":"G7.2","status":st,"started_at":start,"ended_at":datetime.datetime.utcnow().isoformat()+"Z","commands":["bash scripts/vllm_disagg.sh"],
   "versions":{},"measurements":{},"notes":notes.split(" | ")}
try:
    import subprocess; d["versions"]["pip"]=subprocess.run(["bash","-lc",".vllm/bin/pip list 2>/dev/null | grep -Ei 'vllm|nixl|torch|ucx'"],capture_output=True,text=True).stdout
except Exception as e: d["versions"]["error"]=repr(e)
try: d["measurements"]=json.load(open("logs/vllm_disagg_measurements.json"))
except Exception: pass
json.dump(d,open("results/G7_vllm_disagg.json","w"),indent=1)
PY
}
trap 'pkill -f "vllm serve" 2>/dev/null; pkill -f toy_proxy_server 2>/dev/null; json_out' EXIT

# ---- install (attempt 1: wheels)
if [ ! -x .vllm/bin/python ]; then
  log "creating venv .vllm and installing vllm + nixl (attempt 1)"
  (uv venv --python 3.12 .vllm && . .vllm/bin/activate && uv pip install vllm nixl) >> logs/vllm_install.log 2>&1 || { notes+=("install attempt 1 failed, see logs/vllm_install.log"); log "install failed"; exit 0; }
fi
. .vllm/bin/activate
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo unknown); log "vllm $VLLM_VER"
# proxy script lives in the source tree; fetch it for the installed version (fallback: main)
if [ ! -f scripts/toy_proxy_server.py ]; then
  curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VER}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" -o scripts/toy_proxy_server.py \
  || curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/main/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" -o scripts/toy_proxy_server.py \
  || { notes+=("could not fetch toy_proxy_server.py"); exit 0; }
fi
export UCX_TLS=all UCX_NET_DEVICES=all
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 nohup vllm serve "$MODEL" --port 8100 --max-model-len 10240 --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' > logs/vllm_prefill.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 nohup vllm serve "$MODEL" --port 8200 --max-model-len 10240 --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' > logs/vllm_decode.log 2>&1 &
log "waiting for both instances (up to 15 min)"
for i in $(seq 1 90); do
  ok=0; curl -sf localhost:8100/health >/dev/null && curl -sf localhost:8200/health >/dev/null && ok=1
  [ $ok = 1 ] && break; sleep 10
done
[ "${ok:-0}" = 1 ] || { notes+=("instances did not become healthy in 15 min; see logs/vllm_prefill.log logs/vllm_decode.log"); exit 0; }
python scripts/toy_proxy_server.py --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200 > logs/vllm_proxy.log 2>&1 &
sleep 5
# ---- correctness: one request through the split pair
python - "$MODEL" <<'PY' && status=PASS || notes+=("request through proxy failed or wrong")
import json, sys, time, urllib.request
model=sys.argv[1]
def post(port, prompt, n=16):
    req=urllib.request.Request(f"http://localhost:{port}/v1/completions", data=json.dumps({"model":model,"prompt":prompt,"max_tokens":n,"temperature":0}).encode(), headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); r=json.load(urllib.request.urlopen(req, timeout=300)); return r["choices"][0]["text"], time.perf_counter()-t0
txt,_=post(8192, "The capital of France is")
print("split pair says:", repr(txt)); assert "Paris" in txt
# ---- TTFT, 4096-token prompts, 10 requests, concurrency 1: split vs decode-only instance (single)
long=("The quick brown fox jumps over the lazy dog. "*600)
m={}
for name,port in (("split",8192),("single",8200)):
    ts=[]
    for i in range(10):
        _,dt=post(port, long+f" Request {i}. Summary:", n=1); ts.append(dt)
    ts.sort(); m[name]={"ttft_ms_median":1e3*ts[5],"ttft_ms_p90":1e3*ts[8],"all_ms":[1e3*t for t in ts]}
    print(name, m[name]["ttft_ms_median"])
m["note"]="max_tokens=1 end-to-end latency used as TTFT proxy; prompt ~4-5k tokens of repeated text"
json.dump(m, open("logs/vllm_disagg_measurements.json","w"), indent=1)
PY
log "done: $status"
