"""Decision map: which action the planner picks over (link speed) x (new tokens) for each (cached prefix, budget, decode load).
Cost-model evaluation (not a queueing simulation). Outputs figures/decision_map.png and results/decision_map.json with the
crossover link speeds (where the chosen action changes) per cell row.
"""
from __future__ import annotations
import json, pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.cost.model import CostModel
from kvpick.planner.planner import plan, Request, Link, DecodeLoad

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTIONS = ["raw", "int8_row", "int4_kivi", "recompute"]
LABEL = {"raw": "RAW", "int8_row": "INT8", "int4_kivi": "KV4", "recompute": "RECOMP"}
COLOR = {"raw": "#2a78d6", "int8_row": "#eb6834", "int4_kivi": "#1baf7a", "recompute": "#fcfcfb"}   # 3 validated hues (all-pairs PASS); recompute = surface + hatch + label
HATCH = {"raw": "", "int8_row": "", "int4_kivi": "", "recompute": "////"}
EDGE = {"raw": "#fcfcfb", "int8_row": "#fcfcfb", "int4_kivi": "#fcfcfb", "recompute": "#8c8b86"}


def main():
    cost = CostModel()
    gbps = np.logspace(0, 2, 81)                     # 1 .. 100 Gbps
    n_news = [128, 512, 2048, 8192]; m_cacheds = [0, 2048, 8192]
    panels = [("strict", 0), ("normal", 0), ("lenient", 0), ("lenient", 8)]      # budgets with decode idle, plus the busy case
    grid = {}; crossovers = []
    for budget, batch in panels:
        for m in m_cacheds:
            for n in n_news:
                acts = [plan(Request(n, m, budget), Link(g), DecodeLoad(batch), cost).action for g in gbps]
                grid[(budget, batch, m, n)] = acts
                ch = [(round(float(gbps[i]), 2), acts[i]) for i in range(len(acts)) if i == 0 or acts[i] != acts[i - 1]]
                crossovers.append({"budget": budget, "decode_batch": batch, "m_cached": m, "n_new": n, "action_from_gbps": ch})
    # ---- figure: rows = panels, cols = m_cached; cells coloured by action, direct text labels on runs
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(len(panels), len(m_cacheds), figsize=(11, 7.2), sharex=True, sharey=True)
    edges = np.sqrt(gbps[:-1] * gbps[1:]); edges = np.concatenate([[gbps[0] ** 2 / edges[0]], edges, [gbps[-1] ** 2 / edges[-1]]])
    for r, (budget, batch) in enumerate(panels):
        for c, m in enumerate(m_cacheds):
            ax = axes[r, c]
            for yi, n in enumerate(n_news):
                acts = grid[(budget, batch, m, n)]
                i = 0
                while i < len(acts):
                    j = i
                    while j + 1 < len(acts) and acts[j + 1] == acts[i]: j += 1
                    ax.fill_betweenx([yi - 0.45, yi + 0.45], edges[i], edges[j + 1], facecolor=COLOR[acts[i]], hatch=HATCH[acts[i]], edgecolor=EDGE[acts[i]], linewidth=0.8)
                    if np.log10(edges[j + 1] / edges[i]) > 0.22:
                        ax.text(np.sqrt(edges[i] * edges[j + 1]), yi, LABEL[acts[i]], ha="center", va="center", fontsize=7, color="#0b0b0b" if acts[i] != "raw" else "#fcfcfb",
                                bbox=dict(boxstyle="round,pad=0.15", facecolor="#fcfcfb", edgecolor="none", alpha=0.85) if acts[i] == "recompute" else None)
                    i = j + 1
            ax.set_xscale("log"); ax.set_xlim(gbps[0], gbps[-1]); ax.set_ylim(-0.6, len(n_news) - 0.4)
            ax.set_yticks(range(len(n_news))); ax.set_yticklabels([f"{n} new" for n in n_news]); ax.grid(False)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            if r == 0: ax.set_title(f"{m} tokens already cached on decode GPU", fontsize=8.5, color="#0b0b0b")
            if c == 0: ax.set_ylabel(f"budget {budget}\ndecode GPU {'idle' if batch == 0 else f'busy (batch {batch})'}", fontsize=8, color="#0b0b0b")
            if r == len(panels) - 1: ax.set_xlabel("emulated link speed (Gbps)")
    fig.legend(handles=[Patch(facecolor=COLOR[a], hatch=HATCH[a], edgecolor=EDGE[a], label=LABEL[a]) for a in ACTIONS], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"KVPick planner decision map: fastest KV-ready action per request ({cost.gpu.split('/')[-1].strip()}, costs measured; transfer modelled as bytes/link + {cost.d.get('rtt_ms', 0.5)} ms rtt)", y=1.035, fontsize=9, color="#0b0b0b")
    cls = cost.d["quality"]
    fig.text(0.5, -0.01, f"Cost-model evaluation, not a queueing simulation. ITL SLO 60 ms. Quality classes from the measured eval (pre-set budgets): INT8 {cls['int8_row']['class']}, KV4 {cls['int4_kivi']['class']}.", ha="center", fontsize=7, color="#52514e")
    (ROOT / "figures").mkdir(exist_ok=True); fig.savefig(ROOT / "figures/decision_map.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    out = {"gpu": cost.gpu, "gbps_grid": [float(g) for g in gbps], "crossovers": crossovers,
           "note": "cost-model evaluation; prefill fit max rel err %.1f%%" % (100 * cost.d["prefill"]["fit"]["max_rel_err"])}
    (ROOT / "results/decision_map.json").write_text(json.dumps(out, indent=1))
    for row in crossovers:
        if row["decode_batch"] == 0 and row["budget"] in ("normal", "lenient"):
            print(f"m={row['m_cached']:5d} n={row['n_new']:5d}: " + "  ".join(f">={g}Gbps {LABEL[a]}" for g, a in row["action_from_gbps"]))
    print("saved figures/decision_map.png, results/decision_map.json")


if __name__ == "__main__":
    main()
