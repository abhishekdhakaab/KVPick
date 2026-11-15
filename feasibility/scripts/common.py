"""Shared helpers for feasibility scripts: paths, result-file convention, env capture."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

FEAS = Path(__file__).resolve().parents[1]
RESULTS = FEAS / "results"
LOGS = FEAS / "logs"
MANIFESTS = FEAS / "manifests"
for _d in (RESULTS, LOGS, MANIFESTS):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "datasets", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


class Result:
    """Builds a result JSON following section 6 of the plan."""

    def __init__(self, check_id: str):
        self.d = {"check_id": check_id, "status": "INFO", "started_at": now(), "ended_at": None,
                  "commands": [" ".join(sys.argv)], "versions": versions(), "measurements": {},
                  "notes": [], "workarounds": []}

    def note(self, s: str):
        self.d["notes"].append(s)

    def workaround(self, s: str):
        self.d["workarounds"].append(s)

    def save(self, status: str | None = None, name: str | None = None) -> Path:
        if status:
            self.d["status"] = status
        self.d["ended_at"] = now()
        p = RESULTS / f"{name or self.d['check_id']}.json"
        p.write_text(json.dumps(self.d, indent=1, default=str))
        return p


def save_env_logs(tag: str = ""):
    tag = tag or time.strftime("%Y%m%d_%H%M%S")
    for name, cmd in {"nvidia_smi": ["nvidia-smi"], "python": [sys.executable, "-V"],
                      "pip_freeze": [sys.executable, "-m", "pip", "freeze"], "uname": ["uname", "-a"]}.items():
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except Exception as e:
            out = f"unavailable: {e!r}"
        (LOGS / f"env_{name}_{tag}.txt").write_text(out)
