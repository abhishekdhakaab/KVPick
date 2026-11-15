"""G5: recompute versus send.

1. Measure prefill time t(n) for n in {128, 512, 2048, 4096, 8192} new tokens (0 cached), 5 reps each; fit
   t = c0 + c1*n (+ c2*n^2 if it improves adjusted R^2).  Saves coefficients and tokens/s.
2. With KV bytes/token (C1) and compression ratios + compress/decompress times (G4 json if present, else analytic
   ratios and zero kernel time, flagged), compute for each action the link speed at which sending == recomputing,
   for n in {512, 2048, 8192}, and the best action over 1..100 Gbps.
3. Plot time vs link speed at n=2048.  R8: PASS if the best action changes across 1..100 Gbps for at least one n.
--smoke: tiny random model, n in {64,128,256}.
"""
from __future__ import annotations
import argparse, json, time, math
import torch
from common import Result, RESULTS, MODEL_ID, SEED

KV_BYTES_PER_TOKEN = 147456          # from C1 arithmetic (36 layers x 4096 B)
ACTIONS_DEFAULT = {                   # analytic ratios (fp16 scale+zero), used when no G4 json is available
    "raw": 1.0,
    "int8_row": 4096 / (2 * 1056),
    "int4_row_g32": 4096 / (2 * 640),
    "int4_kivi": 4096 / (640 + 544 + 0),   # K chan_g32 (512 + 32 B amortised meta per token), V row_g32 (512+128)
}


def fit(ns, ts, quadratic=False):
    import numpy as np
    X = np.stack([np.ones(len(ns)), np.array(ns, float)] + ([np.array(ns, float) ** 2] if quadratic else []), 1)
    y = np.array(ts)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef; ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    n, k = len(y), X.shape[1]
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
    return [float(c) for c in coef], r2, adj


def measure_prefill(model, ns, reps, device):
    rows = []
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for n in ns:
            ts = []
            for rep in range(reps + 1):
                ids = torch.randint(0, 1000, (1, n), generator=gen).to(device)
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); model(input_ids=ids, use_cache=True)
                if device == "cuda": torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if rep > 0: ts.append(dt)          # first is warmup
            rows.append({"n": n, "times_s": ts, "median_s": sorted(ts)[len(ts) // 2]})
    return rows


def crossover_table(prefill_coef, actions, kernel_us, ns, rates=range(1, 101)):
    """actions: {name: ratio}; kernel_us: {name: {n: (compress_us, decompress_us)}} (0 if unknown)."""
    def t_prefill(n):
        c = prefill_coef; return c[0] + c[1] * n + (c[2] * n * n if len(c) > 2 else 0)
    out = {}
    for n in ns:
        V = KV_BYTES_PER_TOKEN * n
        tp = t_prefill(n)
        per_action = {}
        for name, ratio in actions.items():
            kq, kdq = kernel_us.get(name, {}).get(n, (0.0, 0.0))
            k_s = (kq + kdq) / 1e6
            # send time at rate B Gbps: k_s + V/ratio*8/(B*1e9)
            # equality with recompute: B* = V/ratio*8 / ((tp - k_s)*1e9)
            B_eq = (V / ratio * 8) / ((tp - k_s) * 1e9) if tp > k_s else float("inf")
            per_action[name] = {"ratio": ratio, "kernel_ms": k_s * 1e3, "bytes": V / ratio,
                                "link_gbps_where_send_equals_recompute": B_eq,
                                "send_ms_at": {B: 1e3 * (k_s + V / ratio * 8 / (B * 1e9)) for B in (1, 2, 5, 10, 25, 50, 100)}}
        best = {}
        for B in rates:
            cands = {"recompute": tp}
            for name, ratio in actions.items():
                kq, kdq = kernel_us.get(name, {}).get(n, (0.0, 0.0))
                cands[name] = (kq + kdq) / 1e6 + V / ratio * 8 / (B * 1e9)
            best[B] = min(cands, key=cands.get)
        changes = [(B, best[B]) for i, B in enumerate(rates) if i == 0 or best[B] != best[rates[i - 1]]]
        out[n] = {"recompute_ms": tp * 1e3, "raw_MB": V / 1e6, "actions": per_action, "best_action_by_gbps": changes,
                  "best_changes_in_1_100": len(set(best.values())) > 1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ns", default="128,512,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--g4-json", default=str(RESULTS / "G4_headroom.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-random", action="store_true")
    a = ap.parse_args()
    r = Result("G5")
    from kv_capture import load_model
    tiny = a.tiny_random or (a.smoke and a.model == MODEL_ID)
    if a.smoke:
        a.ns, a.reps = "64,128,256", 2
    model, tok, cfg = load_model(a.model, a.device, tiny)
    ns = [int(x) for x in a.ns.split(",")]
    rows = measure_prefill(model, ns, a.reps, a.device)
    med = [x["median_s"] for x in rows]
    lin, r2l, adjl = fit(ns, med, False); quad, r2q, adjq = fit(ns, med, True)
    coef = quad if (adjq > adjl + 0.002 and len(ns) >= 4) else lin
    fitres = {"linear": {"coef": lin, "r2": r2l, "adj_r2": adjl}, "quadratic": {"coef": quad, "r2": r2q, "adj_r2": adjq},
              "chosen": "quadratic" if coef is quad else "linear", "coef": coef,
              "tokens_per_s_at_2048": 2048 / (coef[0] + coef[1] * 2048 + (coef[2] * 2048 ** 2 if len(coef) > 2 else 0))}
    # kernel times + ratios from G4 if available
    actions = dict(ACTIONS_DEFAULT); kernel_us = {}
    try:
        g4 = json.load(open(a.g4_json))["measurements"]["by_T"]
        for Tk, ent in g4.items():
            T = int(Tk[1:])
            for f, row in ent["formats"].items():
                ratio = (2 * ent["raw_bytes_one_of_KV"]) / (row["k"]["wire_bytes"] + row["v"]["wire_bytes"])
                actions[f] = ratio
                kernel_us.setdefault(f, {})[T] = (row["best_compress_us_K_plus_V"], row["best_decompress_us_K_plus_V"])
        r.note(f"kernel times and ratios taken from {a.g4_json}")
    except Exception as e:
        r.note(f"G4 json not usable ({e!r}); using analytic ratios and ZERO kernel time (optimistic for compression)")
    cross_ns = [512, 2048, 8192] if not a.smoke else ns
    table = crossover_table(coef, actions, kernel_us, cross_ns)
    r.d["measurements"] = {"model": "tiny-random" if tiny else a.model, "device": a.device, "prefill": rows, "fit": fitres,
                           "actions_ratio": actions, "crossover": {str(k): v for k, v in table.items()},
                           "R8": {"best_action_changes_for_some_n": any(v["best_changes_in_1_100"] for v in table.values()),
                                  "per_n": {str(n): v["best_action_by_gbps"] for n, v in table.items()}}}
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n0 = 2048 if 2048 in table else cross_ns[-1]
        Bs = [b / 2 for b in range(2, 201)]
        plt.figure(figsize=(6, 4))
        for name, ratio in actions.items():
            kq, kdq = kernel_us.get(name, {}).get(n0, (0, 0))
            plt.plot(Bs, [1e3 * ((kq + kdq) / 1e6 + KV_BYTES_PER_TOKEN * n0 / ratio * 8 / (B * 1e9)) for B in Bs], label=f"send {name} ({ratio:.2f}x)")
        plt.axhline(table[n0]["recompute_ms"], color="k", ls="--", label="recompute")
        plt.xscale("log"); plt.yscale("log"); plt.xlabel("emulated link (Gbps)"); plt.ylabel("time to KV ready (ms)")
        plt.title(f"n={n0} new tokens, {r.d['measurements']['model']} on {a.device}"); plt.legend(fontsize=7); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(RESULTS / "G5_crossover_n2048.png", dpi=120)
    except Exception as e:
        r.note(f"plot skipped: {e!r}")
    status = "INFO" if (tiny or a.device != "cuda") else ("PASS" if r.d["measurements"]["R8"]["best_action_changes_for_some_n"] else "FAIL")
    p = r.save(status, name="G5_crossover")
    print(json.dumps({"fit": fitres, "R8": r.d["measurements"]["R8"]}, indent=1))
    for n, v in table.items():
        print(f"n={n}: recompute {v['recompute_ms']:.1f} ms, raw {v['raw_MB']:.0f} MB; send==recompute at Gbps: " +
              ", ".join(f"{k} {x['link_gbps_where_send_equals_recompute']:.1f}" for k, x in v["actions"].items()))
    print("saved", p)


if __name__ == "__main__":
    main()
