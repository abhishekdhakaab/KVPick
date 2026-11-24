"""B6 figures from results/ext/sweep.json: winner heatmaps (ctx x link, breakeven contour) and breakeven-vs-context lines."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT = pathlib.Path(__file__).resolve().parents[1]
WIN = ["raw", "int8_row", "fp8", "kv4", "recompute"]; COL = ["#9e9e9e", "#4c78a8", "#72b7b2", "#e45756", "#f2cf5b"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", default=str(ROOT / "results/ext/sweep.json")); ap.add_argument("--outdir", default=str(ROOT / "figures/ext"))
    a = ap.parse_args(); d = json.load(open(a.sweep)); out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = d["rows"]; ctx = d["ctx"]; links = d["links_gbps"]; gpus = sorted({r["gpu"] for r in rows}); tag = "PROVISIONAL " if d.get("provisional") else ""
    for model in ("qwen3-4b", "llama-3.1-8b", "llama-3.1-70b"):
        fig, axs = plt.subplots(len(gpus), 2, figsize=(13, 4.2 * len(gpus)), squeeze=False)
        for gi, gpu in enumerate(gpus):
            for mi, mode in enumerate(("serial", "overlapped")):
                ax = axs[gi][mi]; W = np.full((len(links), len(ctx)), np.nan); S = np.full_like(W, np.nan)
                for r in rows:
                    if r["model"] == model and r["gpu"] == gpu and r["mode"] == mode and r["cached_frac"] == 0.0:
                        i, j = links.index(r["link_gbps"]), ctx.index(r["ctx"]); W[i, j] = WIN.index(r["winner"]); S[i, j] = r["kv4_score"]
                if np.isnan(W).all(): ax.axis("off"); continue
                ax.imshow(W, origin="lower", aspect="auto", cmap=ListedColormap(COL), vmin=-0.5, vmax=4.5)
                if (S >= 0).any() and (S < 0).any(): ax.contour(S, levels=[0], colors="k", linewidths=1.6)
                ax.set_xticks(range(len(ctx))); ax.set_xticklabels([f"{c//1024}k" if c < 1048576 else "1M" for c in ctx], rotation=45)
                ax.set_yticks(range(len(links))); ax.set_yticklabels([("NVLink" if l == 7200 else f"{l:g}") for l in links])
                ax.set_xlabel("new tokens (context)"); ax.set_ylabel("link, Gbps"); ax.set_title(f"{model} | {gpu} | {mode}", fontsize=9)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL]; fig.legend(h, WIN, loc="lower center", ncol=5)
        fig.suptitle(f"{tag}Fastest strategy (time only). Black line = KV4 'pays off' boundary. cached = 0%. MODELED links.", fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(out / f"heatmap_{model}.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, len(gpus), figsize=(6.5 * len(gpus), 4.5), squeeze=False)
    for gi, gpu in enumerate(gpus):
        ax = axs[0][gi]
        for model, ls in (("qwen3-4b", ":"), ("llama-3.1-8b", "-"), ("llama-3.1-70b", "--")):
            for mode, c in (("serial", "tab:red"), ("overlapped", "tab:blue")):
                xs, ys = [], []
                for b in d["breakeven"]:
                    if b["gpu"] == gpu and b["model"] == model and b["mode"] == mode and b["cached_frac"] == 0.0 and b["upper"]:
                        xs.append(b["ctx"]); ys.append(b["upper"])
                if xs: ax.plot(xs, ys, ls, color=c, marker="o", ms=3, label=f"{model} {mode}")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("new tokens"); ax.set_ylabel("upper breakeven link, Gbps (above: KV4 stops paying off)")
        ax.set_title(f"{tag}{gpu} (modeled)", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "breakeven_vs_context.png", dpi=130); plt.close(fig); print("saved figures to", out)


if __name__ == "__main__":
    main()
