#!/usr/bin/env bash
# Run once on a fresh pod (container disk >= 60 GB, no volume). Needs HF_TOKEN in the environment (never written to disk by this script
# except through huggingface_hub's login cache on the pod, which dies with the pod).  Usage: HF_TOKEN=... bash scripts/ext_pod_setup.sh [models...]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0} PYTHONPATH=$PWD PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs results/ext data
pip install -q --break-system-packages -r env/requirements.txt 2>&1 | tail -1 || pip install -q -r env/requirements.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee logs/ext_env_gpu.txt
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)" | tee -a logs/ext_env_gpu.txt
free -g | head -2 | tee -a logs/ext_env_gpu.txt; df -h /workspace . | tee -a logs/ext_env_gpu.txt
MODELS=${@:-"meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-4B-Instruct-2507"}
for m in $MODELS; do
  echo "downloading $m"; python - "$m" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"], token=os.environ.get("HF_TOKEN"))
PY
done
python -c "from kvpick import ext; m=ext.kvq(verbose=False); print('kvq built, cuda =', m.with_cuda)"
echo SETUP_OK
