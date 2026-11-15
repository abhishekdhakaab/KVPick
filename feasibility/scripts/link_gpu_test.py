"""G6: link emulator on a real GPU. Buffers 10 MB / 100 MB / 1 GB at 1, 5, 10, 25, 100 Gbps.
Real copy = device -> pinned host -> device (one GPU) or device0 -> device1 if two GPUs are visible.
Reports achieved goodput (excluding the one-time rtt) vs set rate, single-shot and pipelined in 36 chunks.
R7: PASS if every point is within 5%.  --smoke: CPU, pure emulation, small buffers.
"""
from __future__ import annotations
import argparse, json, torch
from common import Result, RESULTS
from link_emulator import LinkEmulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mb", default="10,100,1000"); ap.add_argument("--rates", default="1,5,10,25,100")
    ap.add_argument("--rtt-ms", type=float, default=0.5); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = Result("G6")
    cuda = torch.cuda.is_available() and not a.smoke
    if a.smoke:
        a.sizes_mb, a.rates, a.reps = "1,4", "10,100", 2
    two = cuda and torch.cuda.device_count() >= 2
    rows = []
    for mb in [int(x) for x in a.sizes_mb.split(",")]:
        n = mb << 20
        if cuda:
            src = torch.empty(n, dtype=torch.uint8, device="cuda:0").fill_(1)
            if two:
                dst = torch.empty(n, dtype=torch.uint8, device="cuda:1")
                def copy(lo=0, hi=n): dst[lo:hi].copy_(src[lo:hi]); torch.cuda.synchronize()
            else:
                host = torch.empty(n, dtype=torch.uint8, pin_memory=True); dst = torch.empty_like(src)
                def copy(lo=0, hi=n): host[lo:hi].copy_(src[lo:hi]); dst[lo:hi].copy_(host[lo:hi]); torch.cuda.synchronize()
            # raw copy time without emulation
            import time
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter(); copy(); ts.append(time.perf_counter() - t0)
            raw_copy_s = sorted(ts)[len(ts) // 2]
        else:
            copy = None; raw_copy_s = 0.0
        for rate in [float(x) for x in a.rates.split(",")]:
            link = LinkEmulator(rate, a.rtt_ms / 1e3)
            single, piped = [], []
            for _ in range(a.reps):
                link.reset(); d = link.send(n, copy); single.append(n * 8 / (d.wall - link.rtt_s) / 1e9)
                link.reset(); chunk = n // 36
                sizes = [chunk] * 35 + [n - 35 * chunk]; offs = [sum(sizes[:i]) for i in range(36)]
                fns = [(lambda lo=o, hi=o + sz: copy(lo, hi)) for o, sz in zip(offs, sizes)] if copy else None   # each chunk copies ITS slice
                ds = link.send_chunks(sizes, fns)
                piped.append(n * 8 / (ds[-1].t_delivered - ds[0].t_start - link.rtt_s) / 1e9)
            s_med = sorted(single)[len(single) // 2]; p_med = sorted(piped)[len(piped) // 2]
            rows.append({"MB": mb, "rate_gbps": rate, "raw_copy_ms": raw_copy_s * 1e3, "raw_copy_GBps": (n / raw_copy_s / 1e9) if raw_copy_s else None,
                         "achieved_gbps_single": s_med, "achieved_gbps_pipelined36": p_med,
                         "err_pct_single": 100 * (s_med - rate) / rate, "err_pct_pipelined": 100 * (p_med - rate) / rate,
                         "link_dominated": (raw_copy_s < n * 8 / (rate * 1e9)),
                         "spin_window_ms": link.spin_window_s * 1e3, "sleep_overshoot_ms": link.sleep_overshoot_s * 1e3})
            print(f"{mb:5d} MB @ {rate:5.0f} Gbps: single {s_med:7.2f} ({rows[-1]['err_pct_single']:+.2f}%)  pipelined {p_med:7.2f} ({rows[-1]['err_pct_pipelined']:+.2f}%)  raw copy {raw_copy_s*1e3:.1f} ms")
    worst = max(max(abs(x["err_pct_single"]), abs(x["err_pct_pipelined"])) for x in rows)
    r.d["measurements"] = {"rows": rows, "two_gpu_copy": two, "copy_path": "d2d peer" if two else ("d2h2d pinned" if cuda else "none (cpu emulation)"),
                           "worst_abs_err_pct": worst, "R7_pass": worst <= 5.0, "rtt_ms": a.rtt_ms}
    p = r.save("INFO" if a.smoke else ("PASS" if worst <= 5.0 else "FAIL"), name="G6_link")
    print(json.dumps({k: v for k, v in r.d["measurements"].items() if k != "rows"}, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
