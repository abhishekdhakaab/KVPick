"""Checker prototype (G8): decide PASS / REJECT for a candidate quantizer, with reasons.

Checks (all recorded; verdict = REJECT if any hard check fails):
  C2 match_normal      codes/scale/zero identical to the reference on random normal inputs (2 shapes incl. T=97)
  C3 match_outlier     same, with 0.1% of entries amplified 50x
  C3 match_edge        constant rows, zero rows, one huge value, negative-only
  C5 memory            peak allocation delta <= input + outputs + slack   (GPU: torch.cuda.max_memory_allocated; CPU: RSS)
  C6 speed             median time vs reference (informational on CPU; threshold applied on GPU)
  C7a sentinel         outputs pre-filled with 0xA5 then 0x5A: results must be identical (every byte written) and deterministic
  C7c input_unchanged  input bytes identical after the call
  C7d perturbation     changing one input element changes only that element's group (codes + that group's scale/zero)
Thresholds are fixed here before running (memory slack 32 MB on CPU / 1 MB on GPU; speed factor only on GPU).
"""
from __future__ import annotations
import argparse, json, time, os, resource, sys
import torch
from common import Result, RESULTS
from quant_ref import QuantSpec, quantize, _to_groups, _from_groups

THRESH = {"mem_slack_bytes_cpu": 32 << 20, "mem_slack_bytes_gpu": 1 << 20, "min_speedup_gpu": 1.0, "n_timing": 20}


def alloc_outputs(x, spec):
    q = quantize(x, spec)
    return (torch.empty_like(q.codes), torch.empty_like(q.scale), torch.empty_like(q.zero)), q


def peak_mem_delta(fn, device):
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated(); fn(); torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base
    import gc; gc.collect()
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fn()
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024      # ru_maxrss is bytes on macOS, KB on Linux
    return max(0, (r1 - r0) * scale)


def run_checks(name, spec, cand, device="cpu", seed=0) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    res = {"candidate": name, "spec": spec.name, "checks": {}, "verdict": "PASS", "reasons": []}

    def match(x, tag):
        q = quantize(x, spec)
        (oc, os_, oz), _ = alloc_outputs(x, spec)
        cand(x, oc, os_, oz)
        ok_c = torch.equal(oc, q.codes); ok_s = torch.equal(os_, q.scale); ok_z = torch.equal(oz, q.zero)
        frac = float((oc != q.codes).float().mean())
        res["checks"][tag] = {"pass": ok_c and ok_s and ok_z, "codes_mismatch_frac": frac,
                              "scale_match": ok_s, "zero_match": ok_z}
        if not (ok_c and ok_s and ok_z):
            res["reasons"].append(f"{tag}: codes mismatch {frac:.4f}, scale {ok_s}, zero {ok_z}")
        return q, oc, os_, oz

    # memory check (FIRST call of the candidate: on CPU, freed pages are reused and RSS would not grow on later calls)
    x = torch.randn(2048, 8, 128, generator=gen, device=device)
    (oc, os_, oz), _ = alloc_outputs(x, spec)
    delta = peak_mem_delta(lambda: cand(x, oc, os_, oz), device)
    budget = x.numel() * 4 + oc.numel() + os_.numel() * 2 + oz.numel() * 2
    slack = THRESH["mem_slack_bytes_gpu" if device == "cuda" else "mem_slack_bytes_cpu"]
    # the reference itself needs fp32 temporaries of the input size; allow 6x input (fp32 copy, groups, codes float) + slack
    allowed = 6 * x.numel() * 4 + budget + slack
    ok = delta <= allowed
    res["checks"]["memory"] = {"pass": ok, "peak_delta_MB": delta / 2**20, "allowed_MB": allowed / 2**20}
    if not ok:
        res["reasons"].append(f"memory: peak delta {delta/2**20:.0f} MB > allowed {allowed/2**20:.0f} MB")


    for T in (97, 512):
        x = torch.randn(T, 8, 128, generator=gen, device=device)
        match(x, f"match_normal_T{T}")
    # outlier inputs: 0.1% entries x50
    x = torch.randn(512, 8, 128, generator=gen, device=device)
    m = torch.rand(x.shape, generator=gen, device=device) < 0.001; x[m] *= 50
    match(x, "match_outlier_0.1pct_x50")
    # edge rows
    x = torch.randn(160, 8, 128, generator=gen, device=device)
    x[0:32] = 0; x[32:64] = 0.75; x[64:96] = -x[64:96].abs() - 0.1; x[96, 0, 5] = 1000.0
    match(x, "match_edge_rows")

    # speed vs reference
    x = torch.randn(2048, 8, 128, generator=gen, device=device)
    (oc, os_, oz), _ = alloc_outputs(x, spec)
    def timeit(fn):
        ts = []
        for _ in range(THRESH["n_timing"]):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); fn()
            if device == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2]
    t_ref = timeit(lambda: quantize(x, spec)); t_c = timeit(lambda: cand(x, oc, os_, oz))
    speedup = t_ref / max(t_c, 1e-9)
    ok = True if device != "cuda" else speedup >= THRESH["min_speedup_gpu"]
    res["checks"]["speed"] = {"pass": ok, "median_ref_ms": t_ref * 1e3, "median_cand_ms": t_c * 1e3, "speedup": speedup,
                              "note": "informational on CPU"}
    if not ok:
        res["reasons"].append(f"speed: {speedup:.2f}x < {THRESH['min_speedup_gpu']}x")

    # sentinel + determinism: two runs with different pre-fills must agree byte for byte
    x = torch.randn(256, 8, 128, generator=gen, device=device)
    outs = []
    for sent in (0xA5, 0x5A):
        (oc, os_, oz), _ = alloc_outputs(x, spec)
        oc.fill_(sent); os_.view(torch.uint8).fill_(sent); oz.view(torch.uint8).fill_(sent)
        cand(x, oc, os_, oz); outs.append((oc.clone(), os_.clone(), oz.clone()))
    ok = all(torch.equal(a, b) for a, b in zip(outs[0], outs[1]))
    unwritten = float((outs[0][0] != outs[1][0]).float().mean())
    res["checks"]["sentinel_determinism"] = {"pass": ok, "codes_differ_frac": unwritten}
    if not ok:
        res["reasons"].append(f"sentinel: {unwritten:.4f} of code bytes depend on pre-fill (not written or nondeterministic)")

    # input unchanged
    x0 = x.clone(); (oc, os_, oz), _ = alloc_outputs(x, spec); cand(x, oc, os_, oz)
    ok = torch.equal(x, x0); res["checks"]["input_unchanged"] = {"pass": ok}
    if not ok:
        res["reasons"].append("input modified")

    # perturbation: change one element -> only its group's outputs may change
    x = torch.randn(256, 8, 128, generator=gen, device=device)
    (oc1, os1, oz1), _ = alloc_outputs(x, spec); cand(x, oc1, os1, oz1)
    x2 = x.clone(); t, h, d = 100, 3, 40; x2[t, h, d] += 7.0
    (oc2, os2, oz2), _ = alloc_outputs(x2, spec); cand(x2, oc2, os2, oz2)
    # group mask in grouped coordinates
    cg1, pad = _to_groups(oc1.float(), spec); cg2, _ = _to_groups(oc2.float(), spec)
    changed_groups = (cg1 != cg2).any(-1)                         # [... groups]
    tgt = torch.zeros_like(changed_groups)
    if spec.axis == "row":
        tgt[t, h, d // (spec.group or 128)] = True
    else:
        g = spec.group or 256
        tgt[t // g, h, d] = True
    outside = bool((changed_groups & ~tgt).any()); inside = bool(changed_groups[tgt].all())
    scale_outside = bool(((os1 != os2) & ~tgt).any() or ((oz1 != oz2) & ~tgt).any())
    ok = (not outside) and inside and (not scale_outside)
    res["checks"]["perturbation"] = {"pass": ok, "changed_outside_group": outside, "changed_inside_group": inside, "scale_changed_outside": scale_outside}
    if not ok:
        res["reasons"].append(f"perturbation: outside={outside} inside={inside} scale_outside={scale_outside}")

    hard = [k for k, v in res["checks"].items() if not v["pass"] and k != "speed"] + (["speed"] if device == "cuda" and not res["checks"]["speed"]["pass"] else [])
    res["verdict"] = "REJECT" if hard else "PASS"
    res["failed_checks"] = hard
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    from mutants_proto import all_candidates
    r = Result("G8")
    rows = []
    for name, (spec, fn, expected) in all_candidates().items():
        t0 = time.time(); res = run_checks(name, spec, fn, a.device); res["runtime_s"] = round(time.time() - t0, 2)
        res["expected"] = expected; res["as_expected"] = res["verdict"] == expected
        rows.append(res)
        print(f"{name:28s} expected {expected:6s} got {res['verdict']:6s} {'OK ' if res['as_expected'] else 'XX '} failed={res['failed_checks']} ({res['runtime_s']}s)")
    good = [x for x in rows if x["expected"] == "PASS"]; bad = [x for x in rows if x["expected"] == "REJECT"]
    summary = {"mutants_rejected": sum(x["verdict"] == "REJECT" for x in bad), "mutants_total": len(bad),
               "false_rejects": sum(x["verdict"] == "REJECT" for x in good), "good_total": len(good),
               "catch_table": {x["candidate"]: x["failed_checks"] for x in bad},
               "R10_pass": sum(x["verdict"] == "REJECT" for x in bad) >= 5 and sum(x["verdict"] == "REJECT" for x in good) == 0,
               "thresholds": THRESH, "device": a.device}
    r.d["measurements"] = {"summary": summary, "candidates": rows}
    p = r.save("PASS" if summary["R10_pass"] else "FAIL", name="G8_checker" + ("_cpu" if a.device == "cpu" else ""))
    print(json.dumps(summary, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
