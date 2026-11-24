#!/usr/bin/env bash
# Full runs in the pre-registered priority order. Pass the list of steps decided from the pilot plan, e.g.
#   bash scripts/ext_run_full.sh niah_8k niah_32k niah_128k pg19_8k pg19_32k gsm8k_llama wiki_llama gsm8k_qwen wiki_qwen
# Resumable: finished samples are skipped, so an interrupted pod can be restarted and the same command re-issued.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONPATH=$PWD PATH=/usr/local/cuda/bin:$PATH
FP8=${FP8:-1}                       # set FP8=0 to drop FP8 (first thing to drop when the budget is tight; say so in the report)
L=raw,int8_row,kv4; LF=$L; [ "$FP8" = 1 ] && LF=raw,int8_row,fp8,kv4
S=raw,int8_row,kv4,int4_g32; SF=$S; [ "$FP8" = 1 ] && SF=raw,int8_row,fp8,kv4,int4_g32
run() { local name=$1; shift; echo "== [$(date -u +%T)] $name"; python scripts/ext_accuracy.py "$@" > "logs/ext_full_${name}.log" 2>&1; echo "   exit $?"; tail -1 "logs/ext_full_${name}.log"; }
for s in "$@"; do case $s in
  niah_8k)       run $s --task niah --lengths 8192 --formats $LF ;;
  niah_32k)      run $s --task niah --lengths 32768 --formats $LF ;;
  niah_128k)     run $s --task niah --lengths 131072 --formats $LF ;;
  pg19_8k)       run $s --task pg19 --prefix 8192 --formats $L ;;
  pg19_32k)      run $s --task pg19 --prefix 32768 --formats $L ;;
  gsm8k_llama)   run $s --task gsm8k --model llama-3.1-8b --formats $SF ;;
  wiki_llama)    run $s --task wikitext --model llama-3.1-8b --formats $SF ;;
  gsm8k_qwen)    run $s --task gsm8k --model qwen3-4b --formats $S ;;
  wiki_qwen)     run $s --task wikitext --model qwen3-4b --formats $S ;;
  pg19_128k)     run $s --task pg19 --prefix 131072 --formats $L ;;
  *) echo "unknown step $s" ;;
esac; done
