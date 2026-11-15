"""Generate the README figures from results/*.json (palette: validated slots blue/orange/aqua + neutral text tokens).
  figures/kernel_roofline.png     kernels_a40.json: % of copy ceiling per implementation per T (quantize and dequantize)
  figures/transfer_validation.png transfer_validate_a40.json: measured vs predicted KV-ready time
  figures/format_quality.png      quality_a40.json: per-format deltas with 95% CIs on three tasks (skipped if absent)
"""
from __future__ import annotations
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b"})
BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8c8b86", "#fcfcfb"


def kernel_roofline():
    k = json.load(open(R / "kernels_a40.json")); Ts = sorted(int(t[1:]) for t in k["by_T"])
    impls = [("B0_eager", "eager PyTorch", MUTED), ("B1_compile", "torch.compile max-autotune", ORANGE), ("v0", "v0 naive", "#9ec5f4"), ("v1", "v1 tiled", "#5598e7"), ("v2", "v2 fused (shipped)", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, which, title in zip(axes, ("quant", "dequant"), ("compress (bf16 -> KV4)", "decompress (KV4 -> bf16)")):
        x = np.arange(len(Ts)); w = 0.16
        for i, (key, label, col) in enumerate(impls):
            vals = [k["by_T"][f"T{T}"][which][key].get("pct_of_copy_ceiling", 0) for T in Ts]
            bars = ax.bar(x + (i - 2) * w, vals, w * 0.92, color=col, label=label if which == "quant" else None, linewidth=0)
            for b, v in zip(bars, vals):
                if v >= 20: ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=6, color=INK)
        ax.axhline(100, color=INK, lw=1, ls="--"); ax.text(-0.45, 110, "device copy ceiling = 100", ha="left", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f"T={T}" for T in Ts]); ax.set_title(title, fontsize=9); ax.set_ylim(0, 118)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("% of device-copy bandwidth (36-layer cache)")
    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.suptitle(f"KV4 kernels on {k['gpu']}: achieved bandwidth relative to a plain device-to-device copy", y=1.08, fontsize=9, color=INK)
    fig.savefig(F / "kernel_roofline.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def transfer_validation():
    tv = json.load(open(R / "transfer_validate_a40.json")); pts = tv["points"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for fmt, col, lab in (("raw", BLUE, "RAW (bf16)"), ("int4_kivi", AQUA, "KV4 (our kernel)")):
        xs = [p["predicted_serial_ms_final_model"] for p in pts if p["format"] == fmt]; ys = [p["serial_ms_median"] for p in pts if p["format"] == fmt]
        ax.scatter(xs, ys, s=22, color=col, label=lab, zorder=3, edgecolor=SURF, linewidth=0.6)
    lim = [5, 20000]; ax.plot(lim, lim, color=INK, lw=1, ls="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    errs = [abs(p["err_pct_serial_final_model"]) for p in pts]
    ax.set_xlabel("predicted KV-ready time, cost model (ms)"); ax.set_ylabel("measured on 2x A40, pinned-staged copy (ms)")
    ax.set_title(f"Transfer model vs hardware: {len(pts)} points, max |error| {max(errs):.1f}%, mean {np.mean(errs):.2f}%", fontsize=8.5)
    ax.text(0.03, 0.97, "n = 512 / 2048 / 8192 tokens\nemulated link 1 / 3 / 10 / 25 Gbps, rtt 0.5 ms\n36 layers, Qwen3-4B cache sizes", transform=ax.transAxes, va="top", fontsize=7, color="#52514e")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(F / "transfer_validation.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def format_quality():
    p = R / "quality_a40.json"
    if not p.exists(): print("quality_a40.json not present yet; skipping format_quality"); return
    q = json.load(open(p))["measurements"]; T = q["tasks"]
    fmts = [f for f in q["formats"] if f != "raw"]; names = {"int8_row": "INT8 per-row (1.94x)", "int4_row": "INT4 per-row (3.76x)", "int4_row_g32": "INT4 per-row g32 (3.2x)", "int4_kivi": "KV4 per-channel (3.2x)"}
    panels = [("ppl", "WikiText-2 perplexity change (%)", lambda e: (e["rel_change"] * 100, [c * 100 for c in e["rel_change_ci95"]["ci95"]])),
              ("gsm8k", "GSM8K accuracy change (points)", lambda e: (e["delta_ci95"]["point"] * 100, [c * 100 for c in e["delta_ci95"]["ci95"]])),
              ("qasper", "Qasper F1 change (points)", lambda e: (e["delta_ci95"]["point"] * 100, [c * 100 for c in e["delta_ci95"]["ci95"]]))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6)); fig.subplots_adjust(top=0.72, wspace=0.12)
    for ax, (task, title, get) in zip(axes, panels):
        if task not in T: ax.set_visible(False); continue
        for i, f in enumerate(fmts):
            pt, (lo, hi) = get(T[task]["per_format"][f]); col = AQUA if f == "int4_kivi" else (ORANGE if f == "int4_row" else MUTED)
            ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, ms=5, capsize=3, lw=1.2)
            ax.text(pt, i + 0.28, f"{pt:+.1f}", ha="center", fontsize=6.5, color=INK)
        ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(len(fmts))); ax.set_yticklabels([names[f] for f in fmts] if ax is axes[0] else [""] * len(fmts))
        ax.set_title(f"{title}\nn={T[task]['n']}", fontsize=8, pad=6); ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if task == "ppl":
            ax.set_xscale("symlog", linthresh=2); ax.set_xticks([-1, 0, 1, 2, 10, 50]); ax.set_xticklabels(["-1", "0", "1", "2", "10", "50"])
    fig.suptitle(f"Quality cost of each wire format vs RAW, {q['model'].split('/')[-1]} on A40 (95% bootstrap CIs; shipped kernel in the loop for KV4)", fontsize=9, color=INK, y=0.95)
    fig.savefig(F / "format_quality.png", dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    kernel_roofline(); transfer_validation(); format_quality(); print("figures written to", F)
