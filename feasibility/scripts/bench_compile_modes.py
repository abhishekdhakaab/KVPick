"""G4b: can torch.compile close the remaining gap on 4-bit QUANTIZE? Tries compile modes on int4 row_g32 and
chan_g32 quantize at T in {2048, 8192}: default, dynamic=True, max-autotune-no-cudagraphs. Also retries the
chan_g32 T=8192 case that fell back to eager in G4. Output: results/G4b_compile_modes.json"""
from __future__ import annotations
import argparse, json, time, torch
from common import Result, RESULTS
from quant_ref import QuantSpec
from quant_wire import quantize_wire, wire_bytes
from bench_quant import Timer, stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--T", default="2048,8192"); ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--runs", type=int, default=50); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke: a.T, a.layers, a.runs = "64", 2, 3
    r = Result("G4b"); timer = Timer(dev, warmup=5 if a.smoke else 10, runs=a.runs); out = {}
    specs = {"int4_row_g32": QuantSpec(4, "row", 32), "int4_chan_g32": QuantSpec(4, "chan", 32)}
    modes = {"default": dict(), "dynamic": dict(dynamic=True), "max_autotune_nocg": dict(mode="max-autotune-no-cudagraphs")}
    for T in [int(t) for t in a.T.split(",")]:
        xs = [torch.randn(a.layers, T, 8, 128, device=dev).to(torch.bfloat16) for _ in range(3)]
        in_bytes = xs[0].numel() * 2; dst = torch.empty_like(xs[0])
        ceil = stats(timer.time(lambda x: dst.copy_(x), xs), 2 * in_bytes); out[f"T{T}"] = {"copy_GBps": ceil["GBps"]}
        for sname, spec in specs.items():
            fn = lambda x, spec=spec: quantize_wire(x, spec); ob = wire_bytes(*quantize_wire(xs[0], spec)); ent = {}
            ent["eager"] = stats(timer.time(fn, xs), in_bytes + ob)
            for mname, kw in modes.items():
                try:
                    torch._dynamo.reset(); t0 = time.time(); c = torch.compile(fn, **kw); c(xs[0]); c(xs[1])
                    if dev == "cuda": torch.cuda.synchronize()
                    ent[mname] = stats(timer.time(c, xs), in_bytes + ob); ent[mname]["compile_s"] = time.time() - t0
                except Exception as e:
                    ent[mname] = {"error": repr(e)[:300]}
            for k, v in ent.items():
                if "GBps" in v: v["pct_of_copy"] = 100 * v["GBps"] / ceil["GBps"]
            out[f"T{T}"][sname] = ent
            print(f"T={T} {sname}: " + " | ".join(f"{k} {v.get('median_us', 0):.0f}us {v.get('pct_of_copy', 0):.0f}%" if "GBps" in v else f"{k} ERR" for k, v in ent.items()))
        del xs, dst
        if dev == "cuda": torch.cuda.empty_cache()
    r.d["measurements"] = out; p = r.save("INFO", name="G4b_compile_modes"); print("saved", p)

if __name__ == "__main__":
    main()
