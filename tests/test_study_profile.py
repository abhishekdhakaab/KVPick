import importlib.util, pathlib, numpy as np, pytest
spec = importlib.util.spec_from_file_location("bp", pathlib.Path(__file__).resolve().parents[1] / "scripts/ext_build_profile.py"); bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)


def test_peak_lookup():
    assert bp.peak_for("NVIDIA A40") == 149.7 and bp.peak_for("NVIDIA H100 80GB HBM3") == 989.0 and bp.peak_for("NVIDIA H100 PCIe") == 756.0
    with pytest.raises(KeyError): bp.peak_for("Mystery GPU")


def test_prefill_fit_recovers_known_quadratic_and_reports_long_context_error():
    c = [20.0, 0.05, 2e-6, 1e-6]; pts = []
    for n in (256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072):
        for m in (0, n // 2):
            if m + n <= 131072: pts.append({"n": n, "m": m, "ms": c[0] + c[1] * n + c[2] * n * n + c[3] * n * m})
    coef, fit = bp.fit_prefill(pts)
    assert np.allclose(coef, c, rtol=1e-4) and fit["max_rel_err"] < 1e-6 and fit["max_rel_err_n_ge_32k"] < 1e-6 and fit["n_points"] == len(pts)
