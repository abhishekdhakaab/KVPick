"""B1: kernel benchmark on synthetic tensors with the real per-layer shape and real layer counts (36, 32, 80).
Formats: kv4 (our fused v2 compress, v1 decompress), int8_row and fp8 (better of eager PyTorch and torch.compile
max-autotune, dynamic shapes; never a hand-written kernel), plus the device-to-device copy ceiling on the same GPU.
Large cases are chunked over layers (the measured chunk is reported with its byte count; fits use measured chunks only).
3 warm-up runs, >= 20 measured runs, median / p5 / p95, L2 flushed between runs.   Output: results/ext/kernels_<gpu>.json
--cpu-smoke: tiny sizes on CPU, perf_counter timing, plumbing check only (never reported).
"""
from __future__ import annotations
import argparse, gc, json, pathlib, statistics, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire, dequantize_wire
from kvpick.study import runlog
from kvpick.study.formats import FP8_MAX

ROOT = pathlib.Path(__file__).resolve().parents[1]
H, D = 8, 128
INT8 = QuantSpec(8, "row", None)
K_SPEC, V_SPEC = QuantSpec(4, "chan", 32), QuantSpec(4, "row", 32)


# ---- FP8 per tensor per layer (batched over the layer axis): amax pass + scale + cast / cast + scale
def fp8_c(x):
    amax = x.float().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12); scale = amax / FP8_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_d(q, scale):
    return (q.to(torch.float32) * scale).to(torch.bfloat16)


class Timer:
    def __init__(self, cuda, warmup=3, runs=20, flush_mb=256):
        self.cuda, self.warmup, self.runs = cuda, warmup, runs
        self.flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device="cuda") if cuda else None

    def time(self, fn, inputs):
        ts = []
        for i in range(self.warmup + self.runs):
            x = inputs[i % len(inputs)]
            if self.cuda:
                self.flush.fill_(i & 0xFF)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(x); e.record(); torch.cuda.synchronize(); dt = s.elapsed_time(e) / 1e3
            else:
                t0 = time.perf_counter(); fn(x); dt = time.perf_counter() - t0
            if i >= self.warmup:
                ts.append(dt)
        return ts


def stats(ts, bytes_moved, bytes_in, ceiling_GBps=None):
    s = sorted(ts); med = statistics.median(s); q = lambda p: s[int(round(p * (len(s) - 1)))]
    r = {"median_us": med * 1e6, "p5_us": q(0.05) * 1e6, "p95_us": q(0.95) * 1e6, "GBps_moved": bytes_moved / med / 1e9,
         "GBps_input": bytes_in / med / 1e9, "n": len(s)}
    if ceiling_GBps:
        r["pct_of_copy_ceiling"] = 100 * r["GBps_moved"] / ceiling_GBps
    return r


def safe(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": (type(e).__name__ + ": " + str(e))[:600]}


def best_of(eager, comp):
    c = {"eager": eager}
    if comp is not None and "median_us" in comp:
        c["compile"] = comp
    name = min(c, key=lambda k: c[k]["median_us"]); out = dict(c[name]); out["chosen"] = name; out["eager_median_us"] = eager["median_us"]
    out["compile_median_us"] = comp.get("median_us") if comp else None
    if comp is not None and "error" in comp:
        out["compile_error"] = comp["error"]
    return out


def layers_that_fit(L, T, free_bytes, factor=20):
    per_layer = T * H * D * 2
    return max(1, min(L, int(free_bytes * 0.8 // (factor * per_layer))))


def bench_case(mod, L, T, Lc, timer, compiled, cuda, ceiling_cache):
    dev = "cuda" if cuda else "cpu"; ring = 3 if (Lc * T * H * D * 2 < 1e8) else 1
    gen = torch.Generator(device=dev).manual_seed(0)
    mk = lambda: (torch.randn(Lc, T, H, D, generator=gen, device=dev) * 2).to(torch.bfloat16)
    ks = [mk() for _ in range(ring)]; vs = [mk() for _ in range(ring)]
    one = ks[0].numel() * 2; kv_in = 2 * one                                  # bytes of K+V raw
    row = {"L_requested": L, "layers_measured": Lc, "chunked": Lc < L, "T": T, "raw_bytes_KV_measured": kv_in,
           "raw_bytes_KV_full": kv_in * L // Lc if Lc else None}
    # copy ceiling: device-to-device copy of K (read + write = 2 x bytes)
    dst = torch.empty_like(ks[0]); key = (T, Lc)
    ceil = stats(timer.time(lambda x: dst.copy_(x), ks), 2 * one, one)
    row["copy_ceiling"] = ceil; cg = ceil["GBps_moved"]
    pairs = list(zip(ks, vs)); row["quant"] = {}; row["dequant"] = {}
    if mod is not None:                                                       # ---- KV4 (CUDA kernel)
        from kvpick import ext
        flat, views = ext.alloc_kv4_buffer(Lc, T, device=dev); out_b = flat.numel()
        fn = lambda kv: mod.quantize_kv4(kv[0], kv[1], views["kp"], views["ks"], views["kz"], views["vp"], views["vs"], views["vz"], 2)
        row["quant"]["kv4"] = stats(timer.time(fn, pairs), kv_in + out_b, kv_in, cg)
        kd = torch.empty_like(ks[0]); vd = torch.empty_like(vs[0]); fn(pairs[0])
        def dq(_):
            mod.dequantize_k4(views["kp"], views["ks"], views["kz"], kd, 2); mod.dequantize_v4(views["vp"], views["vs"], views["vz"], vd, 2)
        row["dequant"]["kv4"] = stats(timer.time(dq, [0]), kv_in + out_b, kv_in, cg)
        row["kv4_wire_bytes"] = out_b; del flat, views, kd, vd
    # ---- INT8 per row
    def i8c(kv): return quantize_wire(kv[0], INT8), quantize_wire(kv[1], INT8)
    i8 = i8c(pairs[0]); i8_out = sum(a.numel() * a.element_size() for t in i8 for a in t)
    def i8d(_): return dequantize_wire(*i8[0], INT8, T), dequantize_wire(*i8[1], INT8, T)
    e_c = stats(timer.time(i8c, pairs), kv_in + i8_out, kv_in, cg); e_d = stats(timer.time(i8d, [0]), kv_in + i8_out, kv_in, cg)
    c_c = c_d = None
    if compiled:
        c_c = safe(lambda: stats(timer.time(lambda kv: (compiled["i8c"](kv[0]), compiled["i8c"](kv[1])), pairs), kv_in + i8_out, kv_in, cg))
        c_d = safe(lambda: stats(timer.time(lambda _: (compiled["i8d"](*i8[0], T), compiled["i8d"](*i8[1], T)), [0]), kv_in + i8_out, kv_in, cg))
    row["quant"]["int8_row"] = best_of(e_c, c_c); row["dequant"]["int8_row"] = best_of(e_d, c_d); row["int8_wire_bytes"] = i8_out; del i8
    # ---- FP8 per tensor per layer
    def f8c(kv): return fp8_c(kv[0]), fp8_c(kv[1])
    f8 = f8c(pairs[0]); f8_out = sum(q.numel() + s.numel() * 4 for q, s in f8)
    def f8d(_): return fp8_d(*f8[0]), fp8_d(*f8[1])
    e_c = stats(timer.time(f8c, pairs), kv_in + f8_out, kv_in, cg); e_d = stats(timer.time(f8d, [0]), kv_in + f8_out, kv_in, cg)
    c_c = c_d = None
    if compiled:
        c_c = safe(lambda: stats(timer.time(lambda kv: (compiled["f8c"](kv[0]), compiled["f8c"](kv[1])), pairs), kv_in + f8_out, kv_in, cg))
        c_d = safe(lambda: stats(timer.time(lambda _: (compiled["f8d"](*f8[0]), compiled["f8d"](*f8[1])), [0]), kv_in + f8_out, kv_in, cg))
    row["quant"]["fp8"] = best_of(e_c, c_c); row["dequant"]["fp8"] = best_of(e_d, c_d); row["fp8_wire_bytes"] = f8_out
    for d in ("quant", "dequant"):
        for f, r in row[d].items():
            r["meets_90pct_target"] = bool(r.get("pct_of_copy_ceiling", 0) >= 90)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="36,32,80"); ap.add_argument("--T", default="1024,4096,16384,65536,131072,262144,524288,1048576")
    ap.add_argument("--runs", type=int, default=20); ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true"); ap.add_argument("--cpu-smoke", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=1e9, help="stop starting new cases after this wall time (fast-GPU time box)")
    a = ap.parse_args()
    cuda = torch.cuda.is_available() and not a.cpu_smoke
    if a.cpu_smoke: a.layers, a.T, a.runs, a.warmup, a.no_compile = "2,3", "64,100", 2, 1, True
    mod = None
    if cuda:
        from kvpick import ext
        mod = ext.kvq(); assert mod.with_cuda
    info = runlog.start(extra={"layers": a.layers, "T": a.T}); gpu = info["gpu"] or "cpu"; tag = gpu.lower().replace("nvidia ", "").replace(" ", "_")
    compiled = None
    if cuda and not a.no_compile:
        torch._dynamo.reset(); mode = "max-autotune-no-cudagraphs"
        compiled = {"i8c": torch.compile(lambda x: quantize_wire(x, INT8), mode=mode, dynamic=True),
                    "i8d": torch.compile(lambda p, s, z, T: dequantize_wire(p, s, z, INT8, T), mode=mode, dynamic=True),
                    "f8c": torch.compile(fp8_c, mode=mode, dynamic=True), "f8d": torch.compile(fp8_d, mode=mode, dynamic=True)}
    timer = Timer(cuda, a.warmup, a.runs); t0 = time.time(); cases = {}
    if mod is not None:                                                       # bit-exactness of the shipped kernel on this GPU, small shape
        from kvpick import ext
        x = (torch.randn(2, 97, H, D, device="cuda") * 2).to(torch.bfloat16)
        kp, ks_, kz = ext.quantize_k4(x); rp, rs, rz = quantize_wire(x, K_SPEC)
        info["kv4_bitexact_K_small"] = bool(torch.equal(kp, rp) and torch.equal(ks_, rs) and torch.equal(kz, rz))
    for L in [int(x) for x in a.layers.split(",")]:
        for T in [int(x) for x in a.T.split(",")]:
            if time.time() - t0 > a.max_seconds:
                cases[f"L{L}_T{T}"] = {"skipped": "time box"}; continue
            gc.collect()
            if cuda: torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0] if cuda else 2e9
            Lc = layers_that_fit(L, T, free)
            if cuda and Lc == 1 and T * H * D * 2 * 20 > free * 0.8:
                cases[f"L{L}_T{T}"] = {"skipped": "does not fit even for one layer"}; continue
            try:
                cases[f"L{L}_T{T}"] = bench_case(mod, L, T, Lc, timer, compiled, cuda, {})
                r = cases[f"L{L}_T{T}"]; print(f"[{time.time()-t0:6.0f}s] L{L} T{T} Lc={Lc}: copy {r['copy_ceiling']['GBps_moved']:.0f} GB/s; " +
                      ", ".join(f"{f} c{r['quant'][f]['pct_of_copy_ceiling']:.0f}%/d{r['dequant'][f]['pct_of_copy_ceiling']:.0f}%" for f in r["quant"]), flush=True)
            except torch.OutOfMemoryError as e:
                cases[f"L{L}_T{T}"] = {"error": "OOM"}; torch.cuda.empty_cache()
            gc.collect()
            if cuda: torch.cuda.empty_cache()
    out = {"info": runlog.finish(info), "label": "measured", "gpu": gpu, "cases": cases}
    p = pathlib.Path(a.out) if a.out else ROOT / "results" / "ext" / (f"kernels_{tag}" + ("_cpusmoke" if a.cpu_smoke else "") + ".json")
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1)); print("saved", p, "throttled:", out["info"]["throttled"])


if __name__ == "__main__":
    main()
