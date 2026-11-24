"""Pilot -> plan: reads pilot run JSONs (results/ext/acc/**/run_*.json with a `limit`), extrapolates full-run cost at a given $/h and
writes docs/pilot_plan.md. Full sizes: GSM8K 1319; WikiText-2 windows = measured count; NIAH 55 per length; PG-19 100 books.
Usage: python scripts/ext_plan.py --price 0.44 --formats 3 [--budget-left 9]"""
import argparse, json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
FULL = {"gsm8k": 1319, "wikitext": 165, "pg19_8192": 100, "pg19_32768": 100, "pg19_131072": 100}
PRIORITY = ["niah_8192", "niah_32768", "niah_131072", "pg19_8192", "pg19_32768", "gsm8k", "wikitext", "pg19_131072", "ruler"]
ap = argparse.ArgumentParser(); ap.add_argument("--price", type=float, required=True); ap.add_argument("--budget-left", type=float, default=9.0); a = ap.parse_args()
est = {}
for f in (ROOT / "results/ext/acc").glob("*/*/run_*.json"):
    j = json.load(open(f)); p = j["pilot"]; key = p["task"] if not p["task"].startswith("niah") else p["task"] + "_" + str(j["info"].get("lengths", ""))
    est[(p["model"], p["task"], tuple(p["formats"]))] = p
rows = []
for (model, task, fm), p in sorted(est.items()):
    n_full = 55 if task.startswith("niah") else FULL.get(task, 100)
    hrs = p["seconds_per_sample"] * n_full / 3600; rows.append((model, task, "+".join(fm), p["seconds_per_sample"], n_full, hrs, hrs * a.price))
tot = sum(r[6] for r in rows)
md = [f"# Pilot plan at ${a.price}/h (budget left ${a.budget_left})", "", "| model | task | formats | s/sample (pilot) | full n | hours | cost $ |", "|---|---|---|---|---|---|---|"]
md += [f"| {m} | {t} | {f} | {s:.1f} | {n} | {h:.2f} | {c:.2f} |" for m, t, f, s, n, h, c in rows]
md += ["", f"Total if everything runs: ${tot:.2f} ({'fits' if tot <= a.budget_left else 'DOES NOT FIT: drop whole benchmarks from the bottom of the priority list: ' + ' < '.join(PRIORITY[::-1])})"]
(ROOT / "docs/pilot_plan.md").write_text("\n".join(md)); print("\n".join(md))
