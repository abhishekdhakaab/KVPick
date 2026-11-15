#!/usr/bin/env bash
# KVPick feasibility GPU session.  Run from feasibility/:   bash run_gpu_session.sh part2a   (one A40, <= 2 h)
#                                                           bash run_gpu_session.sh part2b   (two A40s, <= 1 h)
# Every step is time-boxed; logs in logs/, results in results/.  Set HF_HOME to a persistent volume first.
set -uo pipefail
cd "$(dirname "$0")"
PART=${1:-part2a}
PY=${PYTHON:-python}
KVDIR=${KVDIR:-${HF_HOME:-/workspace}/kvpick_kv}
T0=$(date +%s); START=$(date -u +%FT%TZ)
mkdir -p results logs
LOGS=$PWD/logs
SESSION=$PWD/results/gpu_session_${PART}.json
echo "{\"part\":\"$PART\",\"started_at\":\"$START\",\"steps\":[]}" > $SESSION
step() {   # step NAME TIMEOUT_S cmd...
  local name=$1 tmo=$2; shift 2
  echo "== [$(date -u +%T)] $name (timeout ${tmo}s): $*"
  local s=$(date +%s)
  timeout --signal=INT --kill-after=30 "$tmo" "$@" > "$LOGS/${name}.log" 2>&1; local rc=$?
  local d=$(( $(date +%s) - s ))
  echo "   -> exit $rc after ${d}s  (total $(( $(date +%s) - T0 ))s)"
  $PY - "$SESSION" "$name" "$rc" "$d" <<'PY'
import json,sys; p,n,rc,d=sys.argv[1:]; j=json.load(open(p)); j["steps"].append({"name":n,"exit":int(rc),"seconds":int(d)}); json.dump(j,open(p,"w"),indent=1)
PY
  tail -3 "$LOGS/${name}.log"
}
finish() { $PY - "$SESSION" "$T0" <<'PY'
import json,sys,datetime,subprocess; p,t0=sys.argv[1],int(sys.argv[2]); j=json.load(open(p))
j["ended_at"]=datetime.datetime.utcnow().isoformat()+"Z"; j["wall_s"]=int(__import__("time").time())-t0
try: j["gpu"]=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
except Exception as e: j["gpu"]=repr(e)
json.dump(j,open(p,"w"),indent=1); print(json.dumps(j,indent=1))
PY
}
trap finish EXIT

cd scripts
$PY env_log.py "$PART"
if [ "$PART" = part2a ]; then
  # order: capture -> axes -> checker -> headroom -> crossover -> quality -> link
  step G1_capture      1500 $PY kv_capture.py --save-kv-dir "$KVDIR" --n-wikitext 20 --n-long 5
  step G2_axes         900  $PY axes_eval.py --kv-dir "$KVDIR" --max-files 25
  step G8_checker      600  $PY checker_proto.py --device cuda
  step G4_headroom     1500 $PY bench_quant.py --compile
  step G5_crossover    900  $PY crossover.py
  step G3_quality      2700 $PY quality_roundtrip.py
  step G6_link         600  $PY link_gpu_test.py
else
  step G7_p2p_copy     300  $PY p2p_copy.py
  step G6_link_2gpu    600  $PY link_gpu_test.py
  step G7_vllm_disagg  2700 bash vllm_disagg.sh
fi
echo "ALL DONE in $(( $(date +%s) - T0 ))s"
