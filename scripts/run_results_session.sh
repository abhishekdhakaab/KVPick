#!/usr/bin/env bash
# Final results session on a 2x A40 pod. Run from the repo root:  bash scripts/run_results_session.sh
# Part A uses GPU 0 only; part B uses both. Each step is time-boxed and logged. Reportable numbers come from here.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6} PYTHONPATH=$PWD PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs results figures
T0=$(date +%s); S=results/results_session.json; echo "{\"started_at\":\"$(date -u +%FT%TZ)\",\"steps\":[]}" > $S
step() { local name=$1 tmo=$2; shift 2; echo "== [$(date -u +%T) +$(( $(date +%s) - T0 ))s] $name"; local s=$(date +%s)
  timeout --signal=INT --kill-after=30 "$tmo" "$@" > "logs/${name}.log" 2>&1; local rc=$?; local d=$(( $(date +%s) - s ))
  echo "   -> exit $rc after ${d}s"; tail -2 "logs/${name}.log"
  python - "$S" "$name" "$rc" "$d" <<'PY'
import json,sys; p,n,rc,d=sys.argv[1:]; j=json.load(open(p)); j["steps"].append({"name":n,"exit":int(rc),"seconds":int(d)}); json.dump(j,open(p,"w"),indent=1)
PY
}
pip install -q --break-system-packages ninja pytest numpy scipy matplotlib pyyaml huggingface_hub transformers datasets safetensors accelerate 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee logs/env_gpus.txt; python -m pip list 2>/dev/null | grep -iE "^(torch|transformers|ninja) " >> logs/env_gpus.txt
step build        900  python -c "from kvpick import ext; m=ext.kvq(verbose=True); print('cuda', m.with_cuda)"
step gpu_tests    600  python -m pytest -q tests/test_ext_gpu.py
step sanitizer    600  bash -c 'for t in memcheck racecheck synccheck initcheck; do compute-sanitizer --tool $t --error-exitcode 1 python scripts/_san.py || echo "$t FAILED"; done'
step manifests    600  bash -c 'cd feasibility/scripts && python c1_manifests.py'
# ---- Part A (GPU 0): reportable kernel numbers + quality with the kernel in the loop
step kernels_a40  1200 env CUDA_VISIBLE_DEVICES=0 python scripts/bench_kernels.py
step quality_a40  2400 env CUDA_VISIBLE_DEVICES=0 python kvpick/quality/roundtrip.py
# ---- Part B (both GPUs): transfer validation incl. real crossover; peer copy bandwidth
step p2p_copy     300  bash -c 'cd feasibility/scripts && python p2p_copy.py && cp ../results/G7_p2p_copy.json ../../results/'
step transfer     1800 python scripts/transfer_validate.py --with-model
python - "$S" "$T0" <<'PY'
import json,sys,datetime,time; p,t0=sys.argv[1],int(sys.argv[2]); j=json.load(open(p)); j["ended_at"]=datetime.datetime.utcnow().isoformat()+"Z"; j["wall_s"]=int(time.time())-t0; json.dump(j,open(p,"w"),indent=1); print(json.dumps(j["steps"]))
PY
echo "ALL DONE in $(( $(date +%s) - T0 ))s"
