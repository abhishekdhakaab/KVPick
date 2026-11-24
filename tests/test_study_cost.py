import math, random, pytest
from kvpick.study.overlap import pipeline_makespan, exposed_after_prefill, simulate_pipeline
from kvpick.study.models import MODELS, FMT_BYTES_PER_TOKEN_PER_LAYER, kv_bytes_per_token, verify_config, flops_prefill
from kvpick.study import cost_ext as C


def synthetic_profile(ceiling=600.0, peak=150.0, with_prefill=True):
    pts = lambda bw, a: [[b, a + b / (bw * 1e9) * 1e3] for b in (1e8, 1e9, 4e9, 1.6e10)]
    d = {"gpu": "synthetic", "copy_ceiling_GBps": ceiling, "peak_tflops": peak, "eff_ref": 0.5,
         "kernel_points": {"int8_row": {"compress": pts(ceiling * 0.9, 0.02), "decompress": pts(ceiling * 0.9, 0.02)},
                           "fp8": {"compress": pts(ceiling * 0.9, 0.02), "decompress": pts(ceiling * 0.9, 0.02)},
                           "kv4": {"compress": pts(ceiling * 1.05, 0.02), "decompress": pts(ceiling * 0.95, 0.02)}}}
    if with_prefill:
        d["prefill"] = {"llama-3.1-8b": {"coef": [5.0, 0.08, 1e-6, 1e-6], "max_total": 131072}}
    return C.Profile.from_json(d)


def test_overlap_closed_form_matches_simulation():
    rnd = random.Random(0)
    for _ in range(200):
        a, b, d = (rnd.uniform(0.01, 5) for _ in range(3)); N = rnd.randint(1, 90)
        assert pipeline_makespan(a, b, d, N) == pytest.approx(simulate_pipeline([a, b, d], N), rel=1e-12)
        assert exposed_after_prefill(a, b, d, N) == pytest.approx(simulate_pipeline([a, b, d], N) - N * a, rel=1e-9, abs=1e-9)
        assert exposed_after_prefill(a, b, d, N) >= b + d - 1e-9


def test_overlap_limits():
    assert exposed_after_prefill(10, 1, 1, 32) == pytest.approx(2.0)                    # compute-bound: only last layer's b + d is exposed
    assert exposed_after_prefill(1, 10, 1, 4) == pytest.approx(1 + 10 + 1 + 3 * 10 - 4)   # send-bound
    assert exposed_after_prefill(0.0, 5, 0.0, 10) == pytest.approx(50)                  # no prefill to hide behind: whole serial transfer


def test_byte_table_matches_spec():
    assert kv_bytes_per_token("qwen3-4b", "raw") == 147456 and kv_bytes_per_token("qwen3-4b", "kv4") == 46080
    assert kv_bytes_per_token("llama-3.1-8b", "raw") == 131072 and kv_bytes_per_token("llama-3.1-8b", "kv4") == 40960
    assert kv_bytes_per_token("llama-3.1-70b", "raw") == 327680 and kv_bytes_per_token("llama-3.1-70b", "kv4") == 102400
    assert 4096 / FMT_BYTES_PER_TOKEN_PER_LAYER["kv4"] == 3.2 and 4096 / FMT_BYTES_PER_TOKEN_PER_LAYER["int8_row"] == pytest.approx(1.94, abs=0.01)


def test_verify_config_catches_mismatch():
    ok = dict(num_hidden_layers=32, num_key_value_heads=8, hidden_size=4096, num_attention_heads=32, max_position_embeddings=131072)
    assert verify_config("llama-3.1-8b", ok)["raw_bytes_per_token"] == 131072
    with pytest.raises(ValueError):
        verify_config("llama-3.1-8b", {**ok, "num_hidden_layers": 36})


def test_line_fit_recovers_parameters():
    f = C.LineFit.fit([[b, 0.05 + b / 500e9 * 1e3] for b in (1e8, 1e9, 1e10)])
    assert f.a_ms == pytest.approx(0.05, rel=1e-6) and f.bw_Bps == pytest.approx(500e9, rel=1e-6)
    assert f.predict(1e9)[1] == "fitted" and f.predict(1e12)[1] == "extrapolated"


def test_kv4_wins_at_mid_link_loses_at_extremes():
    eng = C.CostEngine("llama-3.1-8b", synthetic_profile())
    t = lambda g, ov: eng.all_times(32768, 0, g, ov)
    assert C.pays_off(t(25, False))["verdict"] == "pays_off"
    assert C.winner(t(25, False)) == "kv4"
    assert C.winner(t(1, False)) == "recompute"                      # slow link: do not send
    assert C.pays_off(t(7200, False))["verdict"] != "pays_off"        # NVLink: wire time ~0, kernels only cost
    assert C.winner(t(7200, False)) == "raw"


def test_overlap_never_slower_than_serial():
    eng = C.CostEngine("llama-3.1-8b", synthetic_profile())
    for g in (3, 25, 100, 800):
        for f in C.SEND_FORMATS:
            ts = eng.all_times(32768, 0, g, False)[f][0]; to = eng.all_times(32768, 0, g, True)[f][0]
            assert to <= ts + 1.0                                       # layer launches add a little fixed cost, never more than ~1 ms here


def test_prefill_slowdown_sensitivity_increases_overlapped_time():
    eng = C.CostEngine("llama-3.1-8b", synthetic_profile())
    a = eng.all_times(8192, 0, 400, True)["kv4"][0]; b = eng.all_times(8192, 0, 400, True, prefill_slowdown=0.10)["kv4"][0]
    assert b >= a


def test_flops_model_used_when_no_fit_and_labelled_modeled():
    prof = synthetic_profile(with_prefill=False)
    e8 = C.CostEngine("llama-3.1-8b", prof); e70 = C.CostEngine("llama-3.1-70b", prof, tp=8)
    t8, l8 = e8.prefill_ms(32768); t70, l70 = e70.prefill_ms(32768)
    assert l8 == l70 == "modeled" and t70 > 0 and t8 > 0
    assert flops_prefill("llama-3.1-70b", 32768) > flops_prefill("llama-3.1-8b", 32768)


def test_decode_slo_blocks_recompute():
    eng = C.CostEngine("llama-3.1-8b", synthetic_profile())
    t = eng.all_times(32768, 0, 1, False, decode_batch=8)
    assert t["_recompute_blocked"] and C.winner(t) != "recompute"


def test_crossings_interpolates_inside_grid_only():
    links = [1, 3, 10, 25, 100]; s = [-1.0, 0.5, 1.0, 0.2, -0.4]
    r = C.crossings(links, s)
    assert 1 < r["lower"] < 3 and 25 < r["upper"] < 100 and r["status"] == "crossing"
    assert C.crossings(links, [1, 1, 1, 1, 1])["upper"] is None and C.crossings(links, [1] * 5)["status"] == "all_grid"
    assert C.crossings(links, [-1] * 5)["status"] == "never"


def test_lossless_kernel_time_scales_with_ceiling_fraction():
    eng = C.CostEngine("llama-3.1-8b", synthetic_profile())
    t90, _ = eng.kernel_ms("lossless90", "compress", 8192); t50, _ = eng.kernel_ms("lossless50", "compress", 8192)
    assert t50 == pytest.approx(t90 * 0.9 / 0.5)
