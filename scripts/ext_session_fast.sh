#!/usr/bin/env bash
# FAST-GPU SESSION (H100 / A100 80 GB), hard cap 45 minutes of rental. Only Llama-3.1-8B is downloaded (scripts/ext_pod_setup.sh meta-llama/Llama-3.1-8B-Instruct).
# Order: B1 kernels + copy ceiling (time-boxed), B2 prefill trace to 128k, B3 contention only if time is left. Stops starting steps after ~38 min.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONPATH=$PWD PATH=/usr/local/cuda/bin:$PATH
T0=$(date +%s); left() { echo $(( 38*60 - ($(date +%s) - T0) )); }
step() { local name=$1 tmo=$2; shift 2; echo "== [$(date -u +%T) +$(( $(date +%s) - T0 ))s] $name"; timeout --signal=INT --kill-after=20 "$tmo" "$@" > "logs/ext_fast_${name}.log" 2>&1; echo "   -> exit $?"; tail -2 "logs/ext_fast_${name}.log"; }
step tests_gpu 300 python -m pytest -q tests/test_ext_gpu.py tests/test_shapes_ext.py -m gpu
step bench_kernels 1200 python scripts/ext_bench_kernels.py --max-seconds 1000
[ "$(left)" -gt 300 ] && step prefill_llama 600 python scripts/ext_bench_prefill.py --model llama-3.1-8b --max-ctx 131072 --reps 2
[ "$(left)" -gt 300 ] && step contention 420 python scripts/ext_contention.py --model llama-3.1-8b --ctx 32768 || echo "B3 contention skipped: not enough time (report must say overlap contention was not measured)"
echo "FAST SESSION DONE after $(( $(date +%s) - T0 ))s. Copy results/ext and logs, then TERMINATE the pod."
