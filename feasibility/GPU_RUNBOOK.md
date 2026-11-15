# GPU session runbook (Stage 2)

Budget: part 2a = one A40 (48 GB), up to 2 h.  part 2b = two A40s on one machine, up to 1 h.  Expected part 2a wall time ~60 min.

## Before renting
- Everything in Stage 1 passes locally: `cd feasibility/scripts && python -m pytest -q test_quant_ref.py test_quant_wire.py link_emulator.py`.
- Have an HF token (read) ready as an environment variable; it is never printed or saved.

## Pod spec (RunPod or similar)
- Template: a PyTorch CUDA 12.x image (e.g. `runpod/pytorch:2.x-py3.11-cuda12.x`), Ubuntu, with `nvidia-smi` and `python3`.
- Disk: >= 40 GB persistent volume mounted at `/workspace` (model 8 GB + captured KV 12 GB + pip caches).
- Expose nothing; no ports needed.

## Steps on the pod (copy-paste)
```bash
export HF_HOME=/workspace/hf HF_TOKEN=<your token>   # not echoed anywhere
cd /workspace && git clone <this repo> kvpick || true      # or scp the feasibility/ folder
cd kvpick && python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install torch numpy scipy matplotlib pytest huggingface_hub transformers datasets safetensors accelerate pyyaml
cd feasibility && python scripts/c1_manifests.py          # rebuilds manifests on the pod (or copy manifests/ over)
python scripts/kv_capture.py --smoke --tiny-random        # 1-min GPU smoke
nohup bash run_gpu_session.sh part2a > logs/part2a.out 2>&1 &
tail -f logs/part2a.out
```
When it prints `ALL DONE`, copy `feasibility/results/` and `feasibility/logs/` off the pod, then stop the pod.

Part 2b (only if 2a finished with time to spare): rent 2x A40 on one node, repeat the setup, run `bash run_gpu_session.sh part2b`.

## What each step produces
| step | time box | output |
|---|---|---|
| G1_capture | 25 min | results/G1_kv_stats.json, G1_key_channel_amax.png, KV files in $KVDIR |
| G2_axes | 15 min | results/G2_axes.json |
| G8_checker | 10 min | results/G8_checker.json |
| G4_headroom | 25 min | results/G4_headroom.json |
| G5_crossover | 15 min | results/G5_crossover.json, G5_crossover_n2048.png |
| G3_quality | 45 min | results/G3_quality.json |
| G6_link | 10 min | results/G6_link.json |

If a step exits non-zero, the session continues with the next one; the log is in `logs/<step>.log`.
