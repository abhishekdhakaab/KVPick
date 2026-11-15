#!/usr/bin/env bash
# One-shot pod setup (container disk only; /workspace is a plain directory, no volume needed).
# Run on the pod as:  bash /workspace/kvpick/feasibility/pod_setup.sh
# Installs pinned deps into a venv, logs the environment, downloads the model, runs a 1-minute GPU smoke.
set -euo pipefail
mkdir -p /workspace && cd /workspace/kvpick
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
T0=$(date +%s)
echo "== deps (reusing the image's torch $(python3 -c "import torch;print(torch.__version__)"))"
pip install -q numpy scipy matplotlib pytest huggingface_hub hf_transfer transformers datasets safetensors accelerate pyyaml 2>&1 | grep -v "already satisfied" | tail -3 || true
cd feasibility && mkdir -p results logs
python scripts/env_log.py setup
echo "== manifests (datasets download)"; python scripts/c1_manifests.py > logs/c1_pod.log 2>&1 && tail -2 logs/c1_pod.log
echo "== model download (8 GB)"; python - <<'PY'
from huggingface_hub import snapshot_download; import time; t=time.time()
p=snapshot_download("Qwen/Qwen3-4B-Instruct-2507", allow_patterns=["*.json","*.safetensors","*.txt"]); print(p, f"{time.time()-t:.0f}s")
PY
echo "== unit tests"; (cd scripts && python -m pytest -q test_quant_ref.py test_quant_wire.py link_emulator.py 2>&1 | tail -1)
echo "== GPU smoke"; (cd scripts && python kv_capture.py --smoke --tiny-random --device cuda 2>&1 | tail -3 && python bench_quant.py --smoke --device cuda 2>&1 | tail -2)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "SETUP DONE in $(( $(date +%s) - T0 ))s. Next:  nohup bash run_gpu_session.sh part2a > logs/part2a.out 2>&1 &"
