"""B6 tables from results/ext/sweep.json -> docs/tables_extension.md  (every number script-produced).
Sections: realistic cases, breakeven link speeds, 'Where KV4 does not help' (every region where KV4 loses or is marginal, with reason)."""
from __future__ import annotations
import argparse, json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
NV = 7200


def fmtlink(g): return "NVLink" if g == NV else f"{g:g} Gbps"
def fmtctx(c): return f"{c // 1024}k" if c < 1048576 else "1M"


def reason(r):
    ms = r["ms"]; alt = r["best_alt"]
    if alt == "recompute":
        return "link too slow: recomputing is cheaper than sending"
    why = []
    if r["mode"] != "serial" and ms["raw"] < 0.05 * r["prefill_ms"]:
        why.append("overlap hides the transfer behind prefill")
    if alt == "raw":
        why.append("link too fast: kernel time exceeds the wire time saved")
    else:
        why.append(f"{alt} already shrinks the wire enough (KV4 gain under 10% / 10 ms)")
    return "; ".join(why)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", default=str(ROOT / "results/ext/sweep.json")); ap.add_argument("--out", default=str(ROOT / "docs/tables_extension.md"))
    a = ap.parse_args(); d = json.load(open(a.sweep)); rows = d["rows"]; out = []
    if d.get("provisional"): out.append("> **PROVISIONAL: built from pre-extension numbers to test the code. Not a result.**\n")
    gpus = sorted({r["gpu"] for r in rows})
    out.append("All times are ms until the KV is ready on the decode GPU (see prereg addendum 1). Link speeds are modeled; kernel/prefill inputs carry the label in the last column.\n")
    out.append("### Realistic cases\n")
    cases = [("llama-3.1-8b", [32768, 131072], [25, 100, 400]), ("llama-3.1-70b", [32768, 131072, 1048576], [100, 400, NV])]
    for gpu in gpus:
        for mode in ("serial", "overlapped"):
            out.append(f"**{gpu}, mode = {mode}, cached = 0%**\n"); out.append("| model | ctx | link | raw | INT8 | FP8 | KV4 | recompute | KV4 saving vs raw | KV4 saving vs best alt (alt) | verdict | label |"); out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for model, ctxs, links in cases:
                for c in ctxs:
                    for g in links:
                        r = next((x for x in rows if x["gpu"] == gpu and x["model"] == model and x["mode"] == mode and x["ctx"] == c and x["link_gbps"] == g and x["cached_frac"] == 0.0), None)
                        if r is None: continue
                        m = r["ms"]; sv = m["raw"] - m["kv4"]
                        out.append(f"| {model} | {fmtctx(c)} | {fmtlink(g)} | {m['raw']:.0f} | {m['int8_row']:.0f} | {m['fp8']:.0f} | {m['kv4']:.0f} | {m['recompute']:.0f} | {sv:.0f} ms ({100*sv/m['raw']:.0f}%) | {r['save_vs_best_alt_ms']:.0f} ms ({100*r['save_vs_best_alt_frac']:.0f}%) ({r['best_alt']}) | {r['kv4_verdict']} | {r['label']} |")
            out.append("")
    out.append("### Breakeven link speed (interpolated between grid points, no extrapolation)\n")
    out.append("`upper` = link speed above which KV4 stops being the winner by the pre-registered rule; `lower` = below it recompute (or another alternative) wins. `>max` = still pays off at the fastest link in the grid, `never` = never pays off.\n")
    for gpu in gpus:
        out.append(f"**{gpu}**\n"); out.append("| model | mode | cached | " + " | ".join(fmtctx(c) for c in d["ctx"]) + " |"); out.append("|---|---|---|" + "---|" * len(d["ctx"]))
        for model in ("qwen3-4b", "llama-3.1-8b", "llama-3.1-70b"):
            for mode in ("serial", "overlapped"):
                for mf in (0.0, 0.5):
                    cells = []
                    for c in d["ctx"]:
                        b = next((x for x in d["breakeven"] if x["gpu"] == gpu and x["model"] == model and x["mode"] == mode and x["ctx"] == c and x["cached_frac"] == mf), None)
                        if b is None: cells.append(""); continue
                        if b["status"] == "never": cells.append("never")
                        elif b["upper"]: cells.append(f"{b['upper']:.3g}" + (f" (from {b['lower']:.2g})" if b["lower"] else ""))
                        else: cells.append(">max" if b["pays_at_max_link"] else "-")
                    if any(cells): out.append(f"| {model} | {mode} | {int(mf*100)}% | " + " | ".join(cells) + " |")
        out.append("")
    out.append("### Where KV4 does not help\n")
    out.append("Every grid point where KV4 is `marginal` or `does_not_pay_off`, counted by reason (serial and overlapped modes, both cached fractions, all links).\n")
    out.append("| gpu | model | mode | verdict | reason | grid points | link range (Gbps) | ctx range |"); out.append("|---|---|---|---|---|---|---|---|")
    grp = collections.defaultdict(list)
    for r in rows:
        if r["mode"] == "overlapped_slow10" or r["kv4_verdict"] == "pays_off": continue
        grp[(r["gpu"], r["model"], r["mode"], r["kv4_verdict"], reason(r))].append(r)
    for (gpu, model, mode, v, why), rs in sorted(grp.items()):
        out.append(f"| {gpu} | {model} | {mode} | {v} | {why} | {len(rs)} | {min(r['link_gbps'] for r in rs):g} to {max(r['link_gbps'] for r in rs):g} | {fmtctx(min(r['ctx'] for r in rs))} to {fmtctx(max(r['ctx'] for r in rs))} |")
    tot = len(rows); pays = sum(r["kv4_verdict"] == "pays_off" for r in rows)
    out.append(f"\nKV4 pays off at {pays} of {tot} grid points ({100*pays/tot:.0f}%).\n")
    fp8 = collections.Counter(("kv4 faster than FP8 by the rule" if r["kv4_beats_fp8_time_rule"] else "no clear time advantage over FP8") for r in rows if r["mode"] == "serial")
    fi = collections.Counter(("pays off" if r["pays_off_vs_fp8_ideal"] else "does not pay off") for r in rows if r["mode"] == "serial" and r["kv4_verdict"] == "pays_off")
    out.append("Sensitivity (prereg addendum 2): among serial-mode points where KV4 pays off against the measured FP8, KV4 still " + ", ".join(f"{k}: {v}" for k, v in fi.items()) + " against a modeled FP8 cast running at 90% of the copy ceiling.\n")
    out.append("KV4 vs FP8 on time alone (serial mode, all points): " + ", ".join(f"{k}: {v}" for k, v in fp8.items()) + ". The quality condition of the pre-registered rule is applied in the report from Part C.\n")
    pathlib.Path(a.out).write_text("\n".join(out)); print("saved", a.out)


if __name__ == "__main__":
    main()
