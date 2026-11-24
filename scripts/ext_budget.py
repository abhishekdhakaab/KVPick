"""Budget log helper: python scripts/ext_budget.py add "event" --gpu A40 --rate 0.44 --hours 2 [--spent X]  /  show
Appends to docs/budget_log.md and prints remaining against the $10 cap (interim report at $6, stop at $9)."""
import argparse, pathlib, re
P = pathlib.Path(__file__).resolve().parents[1] / "docs/budget_log.md"; CAP, INTERIM, STOP = 10.0, 6.0, 9.0


def spent():
    return sum(float(m) for m in re.findall(r"\| spent \$([\d.]+) \|", P.read_text()))


ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["add", "show"]); ap.add_argument("event", nargs="?", default="")
ap.add_argument("--gpu", default="-"); ap.add_argument("--rate", type=float, default=0.0); ap.add_argument("--hours", type=float, default=0.0); ap.add_argument("--spent", type=float, default=None)
a = ap.parse_args()
if a.cmd == "add":
    planned = a.rate * a.hours; s = a.spent if a.spent is not None else 0.0; rem = CAP - spent() - s - (planned if a.spent is None else 0)
    if a.spent is None and rem < 0: raise SystemExit(f"REFUSED: planned ${planned:.2f} exceeds remaining ${CAP - spent():.2f}")
    import time
    P.write_text(P.read_text() + f"| {time.strftime('%Y-%m-%d %H:%M')} | {a.event} | {a.gpu} | {a.rate:.3f} | {a.hours:.2f} | {planned:.2f} | spent ${s:.2f} | remaining after ${rem:.2f} |\n")
t = spent(); print(f"spent ${t:.2f} of ${CAP:.0f}; remaining ${CAP - t:.2f}; " + ("STOP (>= $9)" if t >= STOP else "INTERIM REPORT DUE (>= $6)" if t >= INTERIM else "ok"))
