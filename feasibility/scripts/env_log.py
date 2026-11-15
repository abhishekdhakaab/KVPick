"""Save nvidia-smi, python -V, pip freeze, uname to logs/env_*.txt (rule 7)."""
import sys
from common import save_env_logs
save_env_logs(sys.argv[1] if len(sys.argv) > 1 else "")
print("env logs saved")
