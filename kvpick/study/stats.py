"""Paired bootstrap statistics for the accuracy study (95% intervals, 2000 resamples, fixed seed)."""
from __future__ import annotations
import math
import numpy as np


def paired_bootstrap(base, other, stat="mean_diff", n_boot=2000, seed=0):
    """stat: 'mean_diff' = mean(other) - mean(base) (accuracy, F1, recall);
             'ppl_rel'   = exp(mean(other) - mean(base)) - 1 on per-sample mean NLL (relative perplexity change)."""
    a, b = np.asarray(base, float), np.asarray(other, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    f = (lambda x, y: float(np.mean(y) - np.mean(x))) if stat == "mean_diff" else (lambda x, y: float(math.exp(np.mean(y) - np.mean(x)) - 1))
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([f(a[i], b[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": f(a, b), "lo": float(lo), "hi": float(hi), "n": int(len(a)), "excludes_zero": bool(lo > 0 or hi < 0)}


def flips(base, other, thresh=0.5):
    """Counts for 0/1 scores: right->wrong, wrong->right, and the number of samples whose score changed at all."""
    a, b = np.asarray(base, float), np.asarray(other, float)
    return {"right_to_wrong": int(((a >= thresh) & (b < thresh)).sum()), "wrong_to_right": int(((a < thresh) & (b >= thresh)).sum()),
            "score_changed": int((a != b).sum())}


def verdict(ci: dict, higher_is_better=True) -> str:
    if not ci["excludes_zero"]:
        return "no clear difference"
    better = (ci["point"] > 0) == higher_is_better
    return "better" if better else "worse"


def budget_class(loss_points: float | None = None, ppl_rel_pct: float | None = None) -> str:
    """Pre-registered classes by POINT ESTIMATE. loss_points >= 0 means a loss (accuracy points); ppl_rel_pct is a relative increase in %."""
    lim = {"strict": (0.5, 0.5), "normal": (2.0, 2.5), "lenient": (5.0, 5.0)}      # (ppl %, task points)
    for name, (lp, lt) in lim.items():
        ok_p = ppl_rel_pct is None or ppl_rel_pct <= lp
        ok_t = loss_points is None or loss_points <= lt
        if ok_p and ok_t:
            return name
    return "none"
