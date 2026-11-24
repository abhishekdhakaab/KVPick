"""Layer-by-layer pipelined transfer (what real disaggregation systems do), as a 3-stage tandem line.

Per layer: stage 1 compute (a), stage 2 compress + send (b), stage 3 decompress (d).  Stages are single servers with
unbounded buffers, layers arrive in order, a layer enters stage j+1 when it leaves stage j and stage j+1 is free:
    F[i][j] = max(F[i-1][j], F[i][j-1]) + t_j          (finish time of layer i at stage j)
With identical layers this telescopes to
    makespan = (a + b + d) + (N - 1) * max(a, b, d)
Stage 1 never waits (F[i][1] = i*a, the prefill itself, P = N*a), so the time exposed AFTER prefill ends is
    exposed = makespan - N*a = b + d + (N - 1) * (max(a, b, d) - a)  ... (equivalent form: a+b+d+(N-1)max - N a)
It is at least b + d (the last layer must still be compressed, sent and decompressed) and equals b + d when compute is
the slowest stage. Derivation and test: tests/test_study_cost.py::test_overlap_closed_form_matches_simulation.
"""
from __future__ import annotations


def pipeline_makespan(a: float, b: float, d: float, n_layers: int) -> float:
    return a + b + d + (n_layers - 1) * max(a, b, d)


def exposed_after_prefill(a: float, b: float, d: float, n_layers: int) -> float:
    return pipeline_makespan(a, b, d, n_layers) - n_layers * a


def simulate_pipeline(stage_times: list[float], n_layers: int) -> float:
    """Brute-force tandem-line simulation (the test oracle)."""
    S = len(stage_times); F = [[0.0] * S for _ in range(n_layers)]
    for i in range(n_layers):
        for j in range(S):
            prev_item = F[i - 1][j] if i else 0.0
            prev_stage = F[i][j - 1] if j else 0.0
            F[i][j] = max(prev_item, prev_stage) + stage_times[j]
    return F[-1][-1]
