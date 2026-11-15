"""G4: kernel headroom. Time quantize / dequantize of a full 36-layer cache at T in {512, 2048, 8192} for the chosen
formats, with (a) eager PyTorch (quant_wire), (b) torch.compile of the same, (c) a device-to-device copy of the
input as the bandwidth ceiling. CUDA events, 20 warmup, 100 timed runs, ring of 3 fresh inputs, L2 flush between runs.

Reports median/p95 us, achieved GB/s = (bytes read + bytes written)/time, and % of the copy ceiling.
R5: PASS (headroom exists) if the better of (a),(b) is below 50% of the ceiling at T=8192 for at least one format.
--smoke: CPU, tiny sizes, perf_counter timing, no compile unless --compile.
"""
from __future__ import annotations
import argparse, json, time, statistics
import torch
from common import Result, RESULTS
from quant_ref import QuantSpec
from quant_wire import quantize_wire, dequantize_wire, wire_bytes

FORMATS = {
    "int8_row":        {"k": QuantSpec(8, "row", None), "v": QuantSpec(8, "row", None)},
    "int4_row_g32":    {"k": QuantSpec(4, "row", 32),   "v": QuantSpec(4, "row", 32)},
    "int4_kivi":       {"k": QuantSpec(4, "chan", 32),  "v": QuantSpec(4, "row", 32)},
}


class Timer:
    def __init__(self, device, warmup=20, runs=100, flush_mb=256):
        self.device, self.warmup, self.runs = device, warmup, runs
        self.flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device=device) if device == "cuda" else None

    def time(self, fn, inputs: list):
        """fn(input) ; inputs is a ring of fresh tensors. Returns list of seconds."""
        ts = []
        n = len(inputs)
        for i in range(self.warmup + self.runs):
            x = inputs[i % n]
            if self.device == "cuda":
                self.flush.fill_(i & 0xFF)                       # evict L2
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(x); e.record(); torch.cuda.synchronize()
                dt = s.elapsed_time(e) / 1e3
            else:
                t0 = time.perf_counter(); fn(x); dt = time.perf_counter() - t0
            if i >= self.warmup:
                ts.append(dt)
        return ts


def stats(ts, bytes_moved):
    ts = sorted(ts); med = statistics.median(ts); p95 = ts[int(0.95 * (len(ts) - 1))]
    return {"median_us": med * 1e6, "p95_us": p95 * 1e6, "GBps": bytes_moved / med / 1e9, "n": len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--T", default="512,2048,8192")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--compile", action="store_true", help="also time torch.compile versions")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.layers, a.T, a.runs = 2, "64,256", 5
    r = Result("G4")
    dev = a.device
    timer = Timer(dev, warmup=3 if a.smoke else 20, runs=a.runs)
    Ts = [int(t) for t in a.T.split(",")]
    H, D, L = 8, 128, a.layers
    out = {}
    gen = torch.Generator(device=dev).manual_seed(0)
    for T in Ts:
        ring_n = 3 if not a.smoke else 2
        xs = [torch.randn(L, T, H, D, generator=gen, device=dev).to(torch.bfloat16) for _ in range(ring_n)]
        in_bytes = xs[0].numel() * 2
        # (c) copy ceiling: a copy reads N and writes N bytes
        dst = torch.empty_like(xs[0])
        ts = timer.time(lambda x: dst.copy_(x), xs)
        ceil = stats(ts, 2 * in_bytes)
        out[f"T{T}"] = {"raw_bytes_one_of_KV": in_bytes, "raw_MB_full_cache_K_and_V": 2 * in_bytes / 1e6, "copy_ceiling": ceil, "formats": {}}
        for fname in a.formats.split(","):
            fmt = FORMATS[fname]; row = {}
            for which in ("k", "v"):
                spec = fmt[which]
                p, s, z = quantize_wire(xs[0], spec); out_bytes = wire_bytes(p, s, z)
                q_fn = lambda x, spec=spec: quantize_wire(x, spec)
                dq_in = [quantize_wire(x, spec) for x in xs]
                dq_fn = lambda psz, spec=spec, T=T: dequantize_wire(*psz, spec, T)
                ent = {"spec": spec.name, "wire_bytes": out_bytes, "compression_ratio": in_bytes / out_bytes}
                ent["eager_quant"] = stats(timer.time(q_fn, xs), in_bytes + out_bytes)
                ent["eager_dequant"] = stats(timer.time(dq_fn, dq_in), in_bytes + out_bytes)
                if a.compile:
                    try:
                        t0 = time.time()
                        cq = torch.compile(q_fn, dynamic=False); cdq = torch.compile(dq_fn, dynamic=False)
                        cq(xs[0]); cdq(dq_in[0])
                        if dev == "cuda": torch.cuda.synchronize()
                        ent["compile_s"] = time.time() - t0
                        ent["compiled_quant"] = stats(timer.time(cq, xs), in_bytes + out_bytes)
                        ent["compiled_dequant"] = stats(timer.time(cdq, dq_in), in_bytes + out_bytes)
                    except Exception as e:
                        ent["compile_error"] = repr(e)[:500]
                for k in ("eager_quant", "eager_dequant", "compiled_quant", "compiled_dequant"):
                    if k in ent:
                        ent[k]["pct_of_copy_ceiling"] = 100 * ent[k]["GBps"] / ceil["GBps"]
                row[which] = ent
            # combined K+V compress / decompress time (what G5 needs)
            best_q = sum(min(row[w].get("eager_quant", {}).get("median_us", 1e18), row[w].get("compiled_quant", {}).get("median_us", 1e18)) for w in "kv")
            best_dq = sum(min(row[w].get("eager_dequant", {}).get("median_us", 1e18), row[w].get("compiled_dequant", {}).get("median_us", 1e18)) for w in "kv")
            row["best_compress_us_K_plus_V"] = best_q; row["best_decompress_us_K_plus_V"] = best_dq
            row["best_pct_of_ceiling"] = max(max(row[w].get(k, {}).get("pct_of_copy_ceiling", 0) for k in ("eager_quant", "compiled_quant", "eager_dequant", "compiled_dequant")) for w in "kv")
            out[f"T{T}"]["formats"][fname] = row
        del xs, dst
        if dev == "cuda": torch.cuda.empty_cache()
    # R5 verdict at the largest T
    Tmax = f"T{max(Ts)}"
    verdict = {f: out[Tmax]["formats"][f]["best_pct_of_ceiling"] for f in out[Tmax]["formats"]}
    r.d["measurements"] = {"by_T": out, "R5": {"best_pct_of_ceiling_at_Tmax": verdict, "threshold_pct": 50,
                           "R5_pass_headroom_exists": any(v < 50 for v in verdict.values()), "note": "pass means PyTorch/compile stay below 50% of the copy ceiling, i.e. a custom kernel has room"}}
    status = "INFO" if (a.smoke or dev != "cuda") else ("PASS" if r.d["measurements"]["R5"]["R5_pass_headroom_exists"] else "FAIL")
    p = r.save(status, name="G4_headroom")
    for T in Ts:
        c = out[f"T{T}"]["copy_ceiling"]
        print(f"T={T}: copy ceiling {c['GBps']:.1f} GB/s ({c['median_us']:.0f} us)")
        for f, row in out[f"T{T}"]["formats"].items():
            for w in "kv":
                e = row[w]
                line = f"  {f:14s} {w} {e['spec']:22s} ratio {e['compression_ratio']:.2f} | eager q {e['eager_quant']['median_us']:.0f}us {e['eager_quant']['pct_of_copy_ceiling']:.0f}% dq {e['eager_dequant']['median_us']:.0f}us {e['eager_dequant']['pct_of_copy_ceiling']:.0f}%"
                if "compiled_quant" in e:
                    line += f" | compiled q {e['compiled_quant']['median_us']:.0f}us {e['compiled_quant']['pct_of_copy_ceiling']:.0f}% dq {e['compiled_dequant']['median_us']:.0f}us {e['compiled_dequant']['pct_of_copy_ceiling']:.0f}%"
                if "compile_error" in e:
                    line += f" | compile FAILED: {e['compile_error'][:80]}"
                print(line)
    print(json.dumps(r.d["measurements"]["R5"], indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
