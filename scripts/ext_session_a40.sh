#!/usr/bin/env bash
# SESSION 1 (48 GB card, A40): Part A checks, B1, B2, outliers, determinism, 20-sample pilots. Each step is time-boxed; results in results/ext/.
# Usage: bash scripts/ext_session_a40.sh   (after ext_pod_setup.sh). Stops at the first hard failure of the determinism gate.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONPATH=$PWD PATH=/usr/local/cuda/bin:$PATH
T0=$(date +%s)
step() { local name=$1 tmo=$2; shift 2; echo "== [$(date -u +%T) +$(( $(date +%s) - T0 ))s] $name"; timeout --signal=INT --kill-after=30 "$tmo" "$@" > "logs/ext_${name}.log" 2>&1; local rc=$?; echo "   -> exit $rc"; tail -3 "logs/ext_${name}.log"; return $rc; }
step tests_cpu     600 python -m pytest -q tests -m "not gpu"
step tests_gpu     900 python -m pytest -q tests -m gpu
step bench_kernels 1500 python scripts/ext_bench_kernels.py
step prefill_llama 1500 python scripts/ext_bench_prefill.py --model llama-3.1-8b --max-ctx 131072
step prefill_qwen  900  python scripts/ext_bench_prefill.py --model qwen3-4b --max-ctx 32768
step outliers      600  python scripts/ext_outliers.py --model llama-3.1-8b
step determinism   900  python scripts/ext_determinism.py --model llama-3.1-8b || { echo "DETERMINISM GATE FAILED: stop and fix before comparing formats"; exit 3; }
F=raw,int8_row,kv4
step pilot_niah8k   600 python scripts/ext_accuracy.py --task niah --lengths 8192 --formats $F --limit 20
step pilot_niah32k  900 python scripts/ext_accuracy.py --task niah --lengths 32768 --formats $F --limit 10
step pilot_niah128k 900 python scripts/ext_accuracy.py --task niah --lengths 131072 --formats $F --limit 2
step pilot_pg19_8k  600 python scripts/ext_accuracy.py --task pg19 --prefix 8192 --formats $F --limit 20
step pilot_pg19_32k 900 python scripts/ext_accuracy.py --task pg19 --prefix 32768 --formats $F --limit 10
step pilot_gsm8k    600 python scripts/ext_accuracy.py --task gsm8k --formats raw,int8_row,kv4,int4_g32 --limit 32
step pilot_wiki     600 python scripts/ext_accuracy.py --task wikitext --formats raw,int8_row,kv4,int4_g32 --limit 8
echo "SESSION A40 PILOTS DONE in $(( $(date +%s) - T0 ))s: now run scripts/ext_plan.py --price <live $/h> and decide the full-run list"
