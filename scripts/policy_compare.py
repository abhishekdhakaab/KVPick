"""Compare the planner with fixed policies on a synthetic multi-round workload, under the measured cost model.
This is a cost-model evaluation (per-request KV-ready time), NOT a queueing simulation: no arrival process, no queues.
Workload (documented, synthetic): sessions of R rounds; round r adds n_new ~ lognormal(median 400, sigma 1.0) clipped
[32, 8192] tokens; the decode side already caches the previous rounds (m_cached grows by n_new + 200 generated tokens);
budget strict / normal / lenient with probs 0.2 / 0.5 / 0.3 (D34); decode GPU batch ~ Poisson(6) at the time of the request; ITL SLO 60 ms.
Policies: always RAW / INT8 / KV4 (fall back to INT8 when KV4 violates the budget) / RECOMPUTE (fall back to INT8 when
infeasible) / link-threshold (KV4 if link < 10 Gbps else RAW) / planner.  Metrics per link speed: mean and p95 KV-ready ms,
and the planner's improvement over the best fixed policy.  Output: results/policy_compare.json, figures/policy_compare.png
"""
from __future__ import annotations
import json, pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.cost.model import CostModel
from kvpick.planner.planner import plan, Request, Link, DecodeLoad

ROOT = pathlib.Path(__file__).resolve().parents[1]


def workload(seed=0, sessions=400, max_rounds=8):
    rng = np.random.default_rng(seed); reqs = []
    for _ in range(sessions):
        R = int(rng.integers(1, max_rounds + 1)); m = 0; u = rng.random(); budget = "strict" if u < 0.2 else ("normal" if u < 0.7 else "lenient")
        for r in range(R):
            n = int(np.clip(rng.lognormal(np.log(400), 1.0), 32, 8192)); batch = int(rng.poisson(6))
            reqs.append((n, m, budget, batch)); m += n + 200
    return reqs


def cost_of(action, n, m, batch, budget, link, cost):
    if action == "recompute":
        t = cost.t_prefill_ms(n, m); feasible = batch == 0 or t <= 60.0
        return t, feasible
    return cost.t_send_ms(action, n, link.gbps, link.rtt_ms), cost.meets_budget(action, budget)


def main():
    cost = CostModel(); reqs = workload(); links = [1, 2, 3, 5, 10, 25, 50, 100]
    pol_names = ["always_raw", "always_int8", "always_kv4", "always_recompute", "threshold_10gbps", "planner"]
    out = {"n_requests": len(reqs), "links_gbps": links, "policies": {p: {} for p in pol_names}, "improvement_vs_best_fixed_pct": {}}
    for g in links:
        link = Link(float(g)); times = {p: [] for p in pol_names}; mix = {}
        for n, m, budget, batch in reqs:
            for p in pol_names:
                if p == "planner":
                    d = plan(Request(n, m, budget), link, DecodeLoad(batch), cost); t = d.est_ms; mix[d.action] = mix.get(d.action, 0) + 1
                else:
                    a = {"always_raw": "raw", "always_int8": "int8_row", "always_kv4": "int4_kivi", "always_recompute": "recompute",
                         "threshold_10gbps": ("int4_kivi" if g < 10 else "raw")}[p]
                    t, ok = cost_of(a, n, m, batch, budget, link, cost)
                    if not ok: t, _ = cost_of("int8_row", n, m, batch, budget, link, cost)     # documented fallback
                times[p].append(t)
        for p in pol_names:
            arr = np.array(times[p]); out["policies"][p][str(g)] = {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95))}
        best_fixed = min((p for p in pol_names if p != "planner"), key=lambda p: out["policies"][p][str(g)]["mean_ms"])
        imp = 100 * (1 - out["policies"]["planner"][str(g)]["mean_ms"] / out["policies"][best_fixed][str(g)]["mean_ms"])
        imp95 = 100 * (1 - out["policies"]["planner"][str(g)]["p95_ms"] / out["policies"][best_fixed][str(g)]["p95_ms"])
        out["improvement_vs_best_fixed_pct"][str(g)] = {"best_fixed": best_fixed, "mean": imp, "p95": imp95, "planner_action_mix": {k: v / len(reqs) for k, v in mix.items()}}
        print(f"{g:4d} Gbps: planner mean {out['policies']['planner'][str(g)]['mean_ms']:7.1f} ms vs best fixed {best_fixed:16s} {out['policies'][best_fixed][str(g)]['mean_ms']:7.1f} ms -> {imp:+5.1f}% mean, {imp95:+5.1f}% p95; mix " + ", ".join(f"{k} {v/len(reqs):.0%}" for k, v in sorted(mix.items())))
    (ROOT / "results/policy_compare.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    col = {"always_raw": "#2a78d6", "always_int8": "#eb6834", "always_kv4": "#1baf7a", "always_recompute": "#8c8b86", "threshold_10gbps": "#c3c2b7", "planner": "#0b0b0b"}
    for p in pol_names:
        ys = [out["policies"][p][str(g)]["mean_ms"] for g in links]
        ax.plot(links, ys, color=col[p], lw=2 if p == "planner" else 1.4, ls="--" if p == "threshold_10gbps" else "-", marker="o", ms=3, label=p)
        ax.annotate(p, (links[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color="#52514e", va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("emulated link speed (Gbps)"); ax.set_ylabel("mean KV-ready time per request (ms)")
    ax.set_title("Synthetic multi-round workload (%d requests), measured cost model" % len(reqs), fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(ROOT / "figures/policy_compare.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print("saved results/policy_compare.json, figures/policy_compare.png")


if __name__ == "__main__":
    main()
