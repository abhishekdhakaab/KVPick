#!/usr/bin/env bash
# Kernel debug/benchmark session on a cheap pod. Run from the repo root on the pod:  bash scripts/pod_kernel_session.sh
# 1) deps  2) build extension  3) GPU tests  4) compute-sanitizer on a small case  5) benchmark  (each step logged)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
T0=$(date +%s); log(){ echo "[$(date -u +%T) +$(( $(date +%s) - T0 ))s] $*"; }
log "deps"; pip install -q --break-system-packages ninja pytest numpy pyyaml 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda, 'sm', torch.cuda.get_device_capability())"
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch;print("%d.%d"%torch.cuda.get_device_capability())')"
log "build (arch $TORCH_CUDA_ARCH_LIST)"; python -c "from kvpick import ext; m=ext.kvq(verbose=True); print('built, cuda =', m.with_cuda)" > logs/build.log 2>&1; rc=$?; tail -3 logs/build.log
if [ $rc -ne 0 ]; then grep -E "error" logs/build.log | head -30; log "BUILD FAILED"; exit 1; fi
log "gpu tests"; python -m pytest -q tests/test_ext_gpu.py -x 2>&1 | tail -15
log "sanitizer (memcheck + racecheck on T=97, H=6)"
cat > /tmp/san.py <<'PY'
import torch
from kvpick import ext
m = ext.kvq()
for shape in [(1, 97, 6, 128), (1, 1, 8, 128)]:
    x = torch.randn(*shape, device="cuda").to(torch.bfloat16)
    for ver in (0, 1, 2):
        (kp, ks, kz), (vp, vs, vz) = ext.quantize_kv4(x, x, version=ver)
        ext.dequantize_k4(kp, ks, kz, shape[1], version=ver); ext.dequantize_v4(vp, vs, vz, version=ver)
torch.cuda.synchronize(); print("sanitizer workload done")
PY
if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode 1 python /tmp/san.py > logs/sanitizer_memcheck.log 2>&1; echo "memcheck exit $?"; tail -2 logs/sanitizer_memcheck.log
  compute-sanitizer --tool racecheck --error-exitcode 1 python /tmp/san.py > logs/sanitizer_racecheck.log 2>&1; echo "racecheck exit $?"; tail -2 logs/sanitizer_racecheck.log
else
  SAN=$(ls /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | head -1)
  if [ -n "$SAN" ]; then $SAN --tool memcheck --error-exitcode 1 python /tmp/san.py > logs/sanitizer_memcheck.log 2>&1; echo "memcheck exit $?"; tail -2 logs/sanitizer_memcheck.log
       $SAN --tool racecheck --error-exitcode 1 python /tmp/san.py > logs/sanitizer_racecheck.log 2>&1; echo "racecheck exit $?"; tail -2 logs/sanitizer_racecheck.log
  else echo "compute-sanitizer not found on this image"; fi
fi
log "benchmark smoke"; python scripts/bench_kernels.py --smoke 2>&1 | tail -8
log "benchmark full"; python scripts/bench_kernels.py 2>&1 | tail -14
log "DONE"
