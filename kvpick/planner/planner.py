"""Per-request planner: choose RAW / INT8 / KV4 send, or RECOMPUTE on the decode GPU, to minimise the KV-ready
contribution to TTFT, subject to (1) the request's quality budget and (2) not stalling running decodes past the ITL SLO.

Pure function of (request, link, decode load, cost model). No GPU, no hidden state; the link estimate is an input.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from kvpick.cost.model import CostModel

ACTIONS_SEND = ("raw", "int8_row", "int4_kivi")
QUALITY_RANK = {"raw": 0, "recompute": 0, "int8_row": 1, "int4_kivi": 2}     # lower = higher quality; raw before recompute on ties
TIE_TOL = 0.01                                                              # actions within 1% of the best time tie


@dataclass(frozen=True)
class Request:
    n_new: int                 # tokens whose KV the decode side does not have
    m_cached: int = 0          # tokens already cached on the decode side (multi-round prefix)
    budget: str = "normal"     # strict | normal | lenient
    itl_slo_ms: float = 60.0   # max stall tolerated by decodes already running on the decode GPU


@dataclass(frozen=True)
class Link:
    gbps: float
    rtt_ms: float = 0.5
    stale: bool = False        # estimator has no recent sample -> conservative fallback


@dataclass(frozen=True)
class DecodeLoad:
    batch: int = 0             # sequences currently decoding on the target decode GPU


@dataclass
class Candidate:
    action: str
    est_ms: float
    feasible: bool
    reason: str = ""


@dataclass
class Decision:
    action: str
    est_ms: float
    candidates: list[Candidate] = field(default_factory=list)
    note: str = ""


def allowed_formats(cost: CostModel, budget: str) -> list[str]:
    return [f for f in ACTIONS_SEND if f in cost.formats and cost.meets_budget(f, budget)]


def plan(req: Request, link: Link, load: DecodeLoad, cost: CostModel) -> Decision:
    cands: list[Candidate] = []
    if link.stale:
        # no trustworthy link estimate: do not gamble on the wire; recompute if it will not hurt running decodes, else RAW-safe
        stall = cost.t_prefill_ms(req.n_new, req.m_cached)
        if load.batch == 0 or stall <= req.itl_slo_ms:
            return Decision("recompute", stall, [Candidate("recompute", stall, True, "stale link estimate")], note="stale link: recompute")
        best = allowed_formats(cost, req.budget)[0]
        return Decision(best, float("nan"), [Candidate(best, float("nan"), True, "stale link estimate, decode busy")], note="stale link: highest-quality send")
    for fmt in allowed_formats(cost, req.budget):
        cands.append(Candidate(fmt, cost.t_send_ms(fmt, req.n_new, link.gbps, link.rtt_ms), True))
    for fmt in ACTIONS_SEND:
        if fmt in cost.formats and not cost.meets_budget(fmt, req.budget):
            cands.append(Candidate(fmt, cost.t_send_ms(fmt, req.n_new, link.gbps, link.rtt_ms), False, f"quality class {cost.quality_class(fmt)} violates budget {req.budget}"))
    stall = cost.t_prefill_ms(req.n_new, req.m_cached)
    if load.batch > 0 and stall > req.itl_slo_ms:
        cands.append(Candidate("recompute", stall, False, f"would stall {load.batch} running decodes by {stall:.0f} ms > ITL SLO {req.itl_slo_ms:.0f} ms"))
    else:
        cands.append(Candidate("recompute", stall, True))
    feas = [c for c in cands if c.feasible]
    best_t = min(c.est_ms for c in feas)
    tied = [c for c in feas if c.est_ms <= best_t * (1 + TIE_TOL)]
    choice = min(tied, key=lambda c: (QUALITY_RANK[c.action], c.est_ms))
    return Decision(choice.action, choice.est_ms, cands)
