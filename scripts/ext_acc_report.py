"""C4/C5: statistics, tables and figures from the per-sample JSONLs under results/ext/acc/.
Paired bootstrap 95% CIs over samples, flips, number of samples whose output differs from baseline, sample counts everywhere,
quality class per format and context length (worst class over the long-context tasks at that length, point estimates).
Outputs: results/ext/acc_summary.json, docs/tables_accuracy.md, figures/ext/accuracy_vs_context.png, figures/ext/niah_grid_<model>.png, figures/ext/quality_vs_speed.png"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import stats as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORDER = ["strict", "normal", "lenient", "none"]; RATIO = {"raw": 1.0, "int8_row": 1.94, "fp8": 2.0, "kv4": 3.2, "int4_g32": 3.2}


def load(root, model):
    data = {}
    for tdir in sorted((root / model).glob("*")):
        for f in tdir.glob("*.jsonl"):
            recs = [json.loads(l) for l in open(f) if l.strip()]
            data.setdefault(tdir.name, {})[f.stem] = {r["id"]: r for r in recs if "skipped" not in r}
    return data


def compare(task_data, key, stat):
    """-> {fmt: {...}} paired against raw over the ids present in both."""
    base = task_data["raw"]; out = {}
    for f, recs in task_data.items():
        ids = sorted(set(base) & set(recs)); a = [base[i][key] for i in ids]; b = [recs[i][key] for i in ids]
        e = {"n": len(ids), "mean": float(np.mean(b)) if ids else None}
        if f != "raw" and ids:
            ci = S.paired_bootstrap(a, b, stat); e["delta"] = ci; e["verdict"] = S.verdict(ci, higher_is_better=(stat == "mean_diff"))
            if stat == "mean_diff":
                e["flips"] = S.flips(a, b)
            if "output" in base[ids[0]]:
                e["outputs_differ_from_baseline"] = int(sum(base[i].get("output") != recs[i].get("output") for i in ids))
        out[f] = e
    return out


def split_by_ctx(task_data):
    ctxs = sorted({r["ctx"] for recs in task_data.values() for r in recs.values()}); res = {}
    for c in ctxs:
        res[c] = {f: {i: r for i, r in recs.items() if r["ctx"] == c} for f, recs in task_data.items()}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(ROOT / "results/ext/acc")); ap.add_argument("--sweep", default=str(ROOT / "results/ext/sweep.json")); a = ap.parse_args()
    root = pathlib.Path(a.root); summary = {}; md = []
    for model in sorted(p.name for p in root.glob("*") if p.is_dir()):
        data = load(root, model); summary[model] = {}
        for task, td in data.items():
            if "raw" not in td: continue
            is_ppl = "nll" in next(iter(next(iter(td.values())).values()), {"score": 0})
            key, stat = ("nll", "ppl_rel") if is_ppl else ("score", "mean_diff")
            if task.startswith("niah"):
                bycx = split_by_ctx(td); summary[model][task] = {}
                for c, tdc in bycx.items():
                    cmp_ = compare(tdc, key, stat); base_mean = cmp_["raw"]["mean"]; cmp_["_baseline_uninformative"] = bool(base_mean is not None and base_mean < 0.90)
                    summary[model][task][str(c)] = cmp_
            else:
                cmp_ = compare(td, key, stat); summary[model][task] = {"all": cmp_}
                if task.startswith("pg19") or task.startswith("ruler"): pass
        # ---- markdown
        md.append(f"## {model}\n"); md.append("| task | ctx | format | ratio | n | score / ppl | change vs baseline [95% CI] | verdict | flips R>W / W>R | outputs differ | class |"); md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for task, per in summary[model].items():
            for c, cmp_ in per.items():
                unin = cmp_.get("_baseline_uninformative")
                for f, e in cmp_.items():
                    if f.startswith("_"): continue
                    is_ppl = task.startswith(("pg19", "wikitext")); mean = e["mean"]
                    sc = (f"{np.exp(mean):.3f}" if is_ppl else f"{100*mean:.1f}") if mean is not None else ""
                    if "delta" in e:
                        d = e["delta"]; sc_d = (100 if not is_ppl else 100); ch = f"{sc_d*d['point']:+.2f}{'%' if is_ppl else ' pts'} [{sc_d*d['lo']:+.2f}, {sc_d*d['hi']:+.2f}]"
                        loss = (d["point"] * 100) if is_ppl else (-d["point"] * 100)
                        cls = S.budget_class(None, loss) if is_ppl else S.budget_class(loss, None)
                        if unin: cls = "baseline fails: uninformative"
                        e["quality_class"] = cls
                        fl = e.get("flips"); flt = f"{fl['right_to_wrong']} / {fl['wrong_to_right']}" if fl else ""
                        md.append(f"| {task} | {c} | {f} | {RATIO.get(f,'')}x | {e['n']} | {sc} | {ch} | {e['verdict']}{' (n<200: wide interval)' if e['n'] < 200 else ''} | {flt} | {e.get('outputs_differ_from_baseline','')} | {cls} |")
                    else:
                        md.append(f"| {task} | {c} | {f} (baseline) | 1.0x | {e['n']} | {sc} | | | | | |")
        md.append("")
    # class per format and context length from the long-context tasks (NIAH, PG-19, RULER)
    classes = {}
    for model, tasks in summary.items():
        for task, per in tasks.items():
            if not task.startswith(("niah", "pg19", "ruler")): continue
            for c, cmp_ in per.items():
                ctx = c if task.startswith("niah") else (task.split("_")[1] if task.startswith("pg19") else "ruler")
                for f, e in cmp_.items():
                    if f.startswith("_") or "quality_class" not in e: continue
                    cur = classes.setdefault(model, {}).setdefault(f, {}).get(str(ctx))
                    new = e["quality_class"]
                    if new.startswith("baseline"): continue
                    if cur is None or ORDER.index(new) > ORDER.index(cur): classes[model][f][str(ctx)] = new
    summary["_long_context_class"] = classes
    (ROOT / "results/ext/acc_summary.json").write_text(json.dumps(summary, indent=1)); (ROOT / "docs/tables_accuracy.md").write_text("\n".join(md)); print("tables written")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figdir = ROOT / "figures/ext"; figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); have = False
    for model, tasks in summary.items():
        if model.startswith("_"): continue
        for task, per in tasks.items():
            if task == "niah":
                for f in {f for cm in per.values() for f in cm if not f.startswith("_") and f != "raw"}:
                    xs = sorted(int(c) for c in per if f in per[c]); ys = [100 * per[str(c)][f]["delta"]["point"] for c in xs]; lo = [100 * per[str(c)][f]["delta"]["lo"] for c in xs]; hi = [100 * per[str(c)][f]["delta"]["hi"] for c in xs]
                    ax[0].errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], marker="o", capsize=3, label=f); have = True
        pg = sorted((int(t.split("_")[1]), t) for t in tasks if t.startswith("pg19_"))
        for f in {f for _, t in pg for f in tasks[t]["all"] if f != "raw"}:
            xs = [c for c, t in pg if f in tasks[t]["all"]]; ys = [100 * tasks[t]["all"][f]["delta"]["point"] for c, t in pg if f in tasks[t]["all"]]
            ax[1].plot(xs, ys, marker="o", label=f); have = True
    if have:
        ax[0].set_xscale("log", base=2); ax[0].set_xlabel("context length"); ax[0].set_ylabel("NIAH recall change (points)"); ax[0].axhline(0, color="k", lw=.5); ax[0].legend()
        ax[1].set_xscale("log", base=2); ax[1].set_xlabel("prefix length"); ax[1].set_ylabel("PG-19 perplexity change (%)"); ax[1].axhline(0, color="k", lw=.5); ax[1].legend()
        fig.tight_layout(); fig.savefig(figdir / "accuracy_vs_context.png", dpi=130)
    plt.close(fig)
    for model, tasks in summary.items():
        if model.startswith("_") or "niah" not in tasks: continue
        d = load(root, model)["niah"]; ctxs = sorted(int(c) for c in tasks["niah"]); fmts = [f for f in ("raw", "kv4") if f in d]
        fig, axs = plt.subplots(len(fmts), len(ctxs), figsize=(4 * len(ctxs), 3 * len(fmts)), squeeze=False)
        for i, f in enumerate(fmts):
            for j, c in enumerate(ctxs):
                depths = sorted({r["depth"] for r in d[f].values() if r["ctx"] == c}); g = np.full((1, len(depths)), np.nan)
                for k, dp in enumerate(depths):
                    v = [r["score"] for r in d[f].values() if r["ctx"] == c and r["depth"] == dp]; g[0, k] = np.mean(v) if v else np.nan
                axs[i][j].imshow(g, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"); axs[i][j].set_yticks([]); axs[i][j].set_xticks(range(len(depths))); axs[i][j].set_xticklabels(depths, fontsize=6)
                axs[i][j].set_title(f"{f}, {c//1024}k", fontsize=9)
        fig.suptitle(f"{model} NIAH recall by depth (%) per length"); fig.tight_layout(); fig.savefig(figdir / f"niah_grid_{model}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
