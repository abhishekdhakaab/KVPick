import math, pytest
from kvpick.cost.model import CostModel
from kvpick.planner.planner import plan, Request, Link, DecodeLoad, allowed_formats

cost = CostModel()


def test_prefill_fit_quality():
    assert cost.d["prefill"]["fit"]["max_rel_err"] < 0.20, cost.d["prefill"]["fit"]


def test_never_violates_budget():
    for budget in ("strict", "normal", "lenient"):
        for gbps in (0.5, 1, 3, 10, 100, 1000):
            for n in (128, 512, 2048, 8192):
                for m in (0, 2048, 8192):
                    for batch in (0, 8):
                        d = plan(Request(n, m, budget), Link(gbps), DecodeLoad(batch), cost)
                        if d.action != "recompute":
                            assert cost.meets_budget(d.action, budget), (budget, gbps, n, d.action)


def test_fast_link_prefers_highest_quality_send():
    d = plan(Request(2048, 0, "normal"), Link(100_000.0), DecodeLoad(0), cost)
    assert d.action == "raw"            # wire time ~0 -> all sends tie -> highest quality wins


def test_slow_link_recomputes_when_decode_idle():
    d = plan(Request(2048, 0, "normal"), Link(0.5), DecodeLoad(0), cost)
    assert d.action == "recompute"


def test_busy_decode_with_tight_itl_blocks_recompute():
    d = plan(Request(2048, 0, "normal", itl_slo_ms=60), Link(0.5), DecodeLoad(8), cost)
    assert d.action != "recompute"
    assert any(c.action == "recompute" and not c.feasible for c in d.candidates)


def test_strict_budget_excludes_kv4():
    assert "int4_kivi" not in allowed_formats(cost, "strict") and "int8_row" in allowed_formats(cost, "strict")


def test_cached_prefix_moves_crossover_down():
    """Recomputing gets dearer with a long cached history, so recompute should stop winning at a lower link speed."""
    def crossover(m):
        for gbps in [x / 10 for x in range(5, 300)]:
            if plan(Request(512, m, "normal"), Link(gbps), DecodeLoad(0), cost).action != "recompute":
                return gbps
        return float("inf")
    assert crossover(8192) < crossover(0)


def test_stale_link_fallback():
    d = plan(Request(512, 0, "normal"), Link(10, stale=True), DecodeLoad(0), cost)
    assert d.action == "recompute" and "stale" in d.note
    d = plan(Request(8192, 0, "normal", itl_slo_ms=10), Link(10, stale=True), DecodeLoad(8), cost)
    assert d.action == "raw"
