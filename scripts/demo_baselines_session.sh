#!/usr/bin/env bash
# Scripted terminal demo (recorded with asciinema): KVPick's v2 kernel vs KVServe's actual CUDA
# quantize kernel and KIVI's pack step, same GPU, same tensors, same run. ~1.5 minutes.
# Run from /workspace/kvpick on the rented pod (needs a GPU; see DECISIONS.md D36).
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export PYTHONWARNINGS=ignore::UserWarning   # silences KIVI's benign advanced-indexing deprecation notice
cd /workspace/kvpick
say_run() { printf '\n\033[1;34m$ %s\033[0m\n' "$1"; sleep 1.2; eval "$1"; sleep 2; }
printf '\033[1mKVPick vs KVServe vs KIVI: a live kernel-speed comparison\033[0m\n'; sleep 2
say_run "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
printf '\n\033[2mKVServe'"'"'s kernel is vendored verbatim (not reimplemented) -- see the attribution header:\033[0m\n'; sleep 1
say_run "sed -n '1,9p' csrc/baselines/kvserve_kernels.cu"
say_run "python3 scripts/bench_baselines.py --T 2048,8192"
printf '\n\033[1mFull numbers: results/baselines_a40.json. Methodology and granularity caveats: DECISIONS.md D36.\033[0m\n'; sleep 3
