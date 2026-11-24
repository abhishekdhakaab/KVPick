import numpy as np, pytest
from kvpick.study import stats as S, runlog


def test_bootstrap_detects_real_difference_and_null():
    rng = np.random.default_rng(1); a = rng.integers(0, 2, 400).astype(float)
    b = a.copy(); b[:60] = 0                                         # lose up to 60 correct answers
    r = S.paired_bootstrap(a, b); assert r["point"] < 0 and r["excludes_zero"] and r["n"] == 400
    assert not S.paired_bootstrap(a, a.copy())["excludes_zero"] or S.paired_bootstrap(a, a.copy())["point"] == 0.0
    assert S.verdict(r) == "worse"


def test_flips_counts():
    f = S.flips([1, 1, 0, 0, 1], [0, 1, 1, 0, 1]); assert f == {"right_to_wrong": 1, "wrong_to_right": 1, "score_changed": 2}


def test_ppl_rel_matches_definition():
    a = np.full(50, 2.0); b = a + 0.01
    assert S.paired_bootstrap(a, b, "ppl_rel")["point"] == pytest.approx(np.exp(0.01) - 1)


def test_budget_class_boundaries():
    assert S.budget_class(0.4, 0.3) == "strict" and S.budget_class(2.0, 1.0) == "normal" and S.budget_class(3.0, 4.9) == "lenient" and S.budget_class(5.1, 1) == "none"
    assert S.budget_class(None, 0.08) == "strict" and S.budget_class(0.0, 58.5) == "none"


def test_throttle_logic():
    mk = lambda r: {"gpus": [{"clocks_throttle_reasons.active": r}]}
    assert not runlog.throttled(mk("0x0000000000000001")) and not runlog.throttled(mk("0x0000000000000000"))
    assert runlog.throttled(mk("0x0000000000000008")) and runlog.throttled(mk("0x0000000000000080"))
