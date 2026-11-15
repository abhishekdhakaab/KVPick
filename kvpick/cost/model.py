"""CostModel: every cost the planner uses, read from results/cost_tables_<gpu>.json (built by scripts/build_cost_tables.py).
Units: milliseconds, bytes, Gbps. Nothing here is invented; the JSON carries provenance per entry."""
from __future__ import annotations
import json, pathlib, bisect

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUALITY_ORDER = ["strict", "normal", "lenient", "none"]


class CostModel:
    def __init__(self, path: str | pathlib.Path = ROOT / "results" / "cost_tables_a40.json"):
        self.path = pathlib.Path(path); self.d = json.load(open(self.path))
        self.gpu = self.d["gpu"]; self.formats = list(self.d["bytes_per_token"].keys())

    # ---- compute
    def t_prefill_ms(self, n_new: int, m_cached: int = 0) -> float:
        c = self.d["prefill"]["coef"]
        return max(0.0, c[0] + c[1] * n_new + c[2] * n_new * n_new + c[3] * n_new * m_cached)

    def t_decode_step_ms(self, batch: int) -> float:
        d0, d1 = self.d["decode"]["coef"]; return d0 + d1 * batch

    # ---- kernels: linear interpolation in T over the measured table (extrapolate with the nearest rate)
    def _kernel_ms(self, fmt: str, n: int, key: str) -> float:
        tab = self.d["kernel_ms"][fmt]; Ts = sorted(int(t) for t in tab)
        vals = [tab[str(t)][key] for t in Ts]
        if n <= Ts[0]: return vals[0] * n / Ts[0]
        if n >= Ts[-1]: return vals[-1] * n / Ts[-1]
        i = bisect.bisect_right(Ts, n); t0, t1 = Ts[i - 1], Ts[i]; v0, v1 = vals[i - 1], vals[i]
        return v0 + (v1 - v0) * (n - t0) / (t1 - t0)

    def t_compress_ms(self, fmt: str, n: int) -> float: return self._kernel_ms(fmt, n, "compress")
    def t_decompress_ms(self, fmt: str, n: int) -> float: return self._kernel_ms(fmt, n, "decompress")
    def kernel_is_placeholder(self, fmt: str) -> bool:
        return any("placeholder" in v for v in self.d["kernel_ms"][fmt].values())

    # ---- bytes and wire
    def bytes(self, fmt: str, n: int) -> int: return self.d["bytes_per_token"][fmt] * n
    @staticmethod
    def t_wire_ms(nbytes: int, gbps: float, rtt_ms: float) -> float: return nbytes * 8 / (gbps * 1e9) * 1e3 + rtt_ms
    def t_send_ms(self, fmt: str, n: int, gbps: float, rtt_ms: float) -> float:
        return self.t_compress_ms(fmt, n) + self.t_wire_ms(self.bytes(fmt, n), gbps, rtt_ms) + self.t_decompress_ms(fmt, n)

    # ---- quality
    def quality_class(self, fmt: str) -> str: return self.d["quality"].get(fmt, {}).get("class", "none")
    def meets_budget(self, fmt: str, budget: str) -> bool:
        return QUALITY_ORDER.index(self.quality_class(fmt)) <= QUALITY_ORDER.index(budget)
