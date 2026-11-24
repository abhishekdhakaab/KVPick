"""Environment/run logging required by the spec: GPU, driver, CUDA, library versions, commit, command, seed, clocks and
throttle reasons at the start and end of a run. A run is flagged `throttled` if any throttle reason other than idle appeared."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time

_Q = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used,memory.total"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def gpu_state() -> dict:
    out = _run(["nvidia-smi", f"--query-gpu={_Q}", "--format=csv,noheader,nounits"])
    if not out:
        return {}
    rows = []
    for line in out.splitlines():
        v = [x.strip() for x in line.split(",")]
        rows.append(dict(zip(_Q.split(","), v)))
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpus": rows}


def throttled(state: dict) -> bool:
    for g in state.get("gpus", []):
        r = g.get("clocks_throttle_reasons.active", "0x0000000000000000")
        try:
            val = int(r, 16)
        except ValueError:
            continue
        if val & ~0x1 & ~0x4:            # ignore "GPU idle" (0x1) and "applications clocks setting" (0x4); anything else counts
            return True
    return False


def start(seed: int = 0, extra: dict | None = None) -> dict:
    import torch
    info = {"command": " ".join(sys.argv), "seed": seed, "commit": _run(["git", "rev-parse", "HEAD"]) or (open(".commit").read().strip() if os.path.exists(".commit") else ""), "dirty": bool(_run(["git", "status", "--porcelain"])),
            "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gpu_start": gpu_state(), "t_start": time.time()}
    for m in ("transformers", "datasets", "numpy", "accelerate"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            pass
    if extra:
        info.update(extra)
    return info


def finish(info: dict) -> dict:
    info["gpu_end"] = gpu_state(); info["seconds"] = time.time() - info["t_start"]
    info["throttled"] = throttled(info.get("gpu_start", {})) or throttled(info["gpu_end"])
    return info
