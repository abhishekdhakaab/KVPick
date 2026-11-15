"""E1: kernel benchmark. For KV4 at T in {512, 2048, 8192} on the full 36-layer cache (K and V):
  B0 eager PyTorch (quant_wire), B1 torch.compile (max-autotune-no-cudagraphs), v0, v1, v2 (fused single launch),
  B3 device-to-device copy of the input as the bandwidth ceiling.  Quantize and dequantize separately.
CUDA events, 20 warmup, 100 timed runs, ring of 3 fresh inputs, L2 flush between runs.
Output: results/kernels_<gpu>.json  (never mix GPU types).  --smoke: tiny sizes, few runs.
"""
from __future__ import annotations
import argparse, json, statistics, time, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import quantize_wire, dequantize_wire, wire_bytes
from kvpick import ext

K_SPEC, V_SPEC = QuantSpec(4, "chan", 32), QuantSpec(4, "row", 32)


class Timer:
    def __init__(self, warmup=20, runs=100, flush_mb=256):
        self.warmup, self.runs = warmup, runs
        self.flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device="cuda")

    def time(self, fn, inputs):
        ts = []
        for i in range(self.warmup + self.runs):
            x = inputs[i % len(inputs)]
            self.flush.fill_(i & 0xFF)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            if i >= self.warmup: ts.append(s.elapsed_time(e) / 1e3)
        return ts


def stats(ts, bytes_moved):
    ts = sorted(ts); med = statistics.median(ts)
    return {"median_us": med * 1e6, "p95_us": ts[int(0.95 * (len(ts) - 1))] * 1e6, "min_us": ts[0] * 1e6, "GBps": bytes_moved / med / 1e9, "n": len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="512,2048,8192"); ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--runs", type=int, default=100); ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke: a.T, a.layers, a.runs = "64,256", 2, 5
    mod = ext.kvq(); assert mod.with_cuda
    gpu = torch.cuda.get_device_name(0); tag = gpu.lower().replace("nvidia ", "").replace(" ", "_")
    timer = Timer(warmup=3 if a.smoke else 20, runs=a.runs)
    out = {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "layers": a.layers, "by_T": {}}
    for T in [int(t) for t in a.T.split(",")]:
        gen = torch.Generator(device="cuda").manual_seed(0)
        ks = [torch.randn(a.layers, T, 8, 128, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)]
        vs = [torch.randn(a.layers, T, 8, 128, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)]
        in_bytes = ks[0].numel() * 2                       # one of K/V
        kp, ks_, kz = ext.alloc_k4(ks[0]); vp, vs_, vz = ext.alloc_v4(vs[0])
        out_bytes_k = wire_bytes(kp, ks_, kz); out_bytes_v = wire_bytes(vp, vs_, vz)
        dst = torch.empty_like(ks[0])
        ceil = stats(timer.time(lambda x: dst.copy_(x), ks), 2 * in_bytes)
        row = {"raw_bytes_one_of_KV": in_bytes, "wire_bytes_K": out_bytes_k, "wire_bytes_V": out_bytes_v,
               "compression_ratio_KV": 2 * in_bytes / (out_bytes_k + out_bytes_v), "copy_ceiling": ceil, "quant": {}, "dequant": {}}
        # ---- quantize K+V (both, since the fused kernel does both): bytes = 2*in + out_k + out_v
        qbytes = 2 * in_bytes + out_bytes_k + out_bytes_v
        pairs = list(zip(ks, vs))
        row["quant"]["B0_eager"] = stats(timer.time(lambda kv: (quantize_wire(kv[0], K_SPEC), quantize_wire(kv[1], V_SPEC)), pairs), qbytes)
        if not a.no_compile:
            try:
                torch._dynamo.reset()
                cq_k = torch.compile(lambda x: quantize_wire(x, K_SPEC), mode="max-autotune-no-cudagraphs")
                cq_v = torch.compile(lambda x: quantize_wire(x, V_SPEC), mode="max-autotune-no-cudagraphs")
                t0 = time.time(); cq_k(ks[0]); cq_v(vs[0]); torch.cuda.synchronize(); ct = time.time() - t0
                row["quant"]["B1_compile"] = stats(timer.time(lambda kv: (cq_k(kv[0]), cq_v(kv[1])), pairs), qbytes); row["quant"]["B1_compile"]["compile_s"] = ct
            except Exception as e:
                row["quant"]["B1_compile"] = {"error": repr(e)[:300]}
        for ver in (0, 1, 2):
            fn = lambda kv, ver=ver: mod.quantize_kv4(kv[0], kv[1], kp, ks_, kz, vp, vs_, vz, ver)
            row["quant"][f"v{ver}"] = stats(timer.time(fn, pairs), qbytes)
        # correctness spot check of v2 against eager
        mod.quantize_kv4(ks[0], vs[0], kp, ks_, kz, vp, vs_, vz, 2)
        rp, rs, rz = quantize_wire(ks[0], K_SPEC); row["v2_bitexact_K"] = bool(torch.equal(kp, rp) and torch.equal(ks_, rs) and torch.equal(kz, rz))
        rp, rs, rz = quantize_wire(vs[0], V_SPEC); row["v2_bitexact_V"] = bool(torch.equal(vp, rp) and torch.equal(vs_, rs) and torch.equal(vz, rz))
        # ---- dequantize K+V: bytes = out_k + out_v + 2*in
        qins = [(quantize_wire(k, K_SPEC), quantize_wire(v, V_SPEC)) for k, v in pairs]
        kout = torch.empty_like(ks[0]); vout = torch.empty_like(vs[0])
        row["dequant"]["B0_eager"] = stats(timer.time(lambda q: (dequantize_wire(*q[0], K_SPEC, T), dequantize_wire(*q[1], V_SPEC, T)), qins), qbytes)
        if not a.no_compile:
            try:
                torch._dynamo.reset()
                cd_k = torch.compile(lambda p, s, z: dequantize_wire(p, s, z, K_SPEC, T), mode="max-autotune-no-cudagraphs")
                cd_v = torch.compile(lambda p, s, z: dequantize_wire(p, s, z, V_SPEC, T), mode="max-autotune-no-cudagraphs")
                cd_k(*qins[0][0]); cd_v(*qins[0][1]); torch.cuda.synchronize()
                row["dequant"]["B1_compile"] = stats(timer.time(lambda q: (cd_k(*q[0]), cd_v(*q[1])), qins), qbytes)
            except Exception as e:
                row["dequant"]["B1_compile"] = {"error": repr(e)[:300]}
        for ver in (0, 1, 2):
            fn = lambda q, ver=ver: (mod.dequantize_k4(*q[0], kout, ver), mod.dequantize_v4(*q[1], vout, ver))
            row["dequant"][f"v{ver}"] = stats(timer.time(fn, qins), qbytes)
        for d in ("quant", "dequant"):
            for k_, v_ in row[d].items():
                if "GBps" in v_: v_["pct_of_copy_ceiling"] = 100 * v_["GBps"] / ceil["GBps"]
        out["by_T"][f"T{T}"] = row
        print(f"T={T}: copy ceiling {ceil['GBps']:.0f} GB/s ({ceil['median_us']:.0f} us); ratio {row['compression_ratio_KV']:.2f}; v2 bit-exact K={row['v2_bitexact_K']} V={row['v2_bitexact_V']}")
        for d in ("quant", "dequant"):
            print(f"  {d:8s} " + " | ".join(f"{k_} {v_['median_us']:.0f}us {v_['pct_of_copy_ceiling']:.0f}%" if "GBps" in v_ else f"{k_} ERR" for k_, v_ in row[d].items()))
        del ks, vs, dst, kout, vout, qins; torch.cuda.empty_cache()
    p = pathlib.Path(a.out or f"results/kernels_{tag}{'_smoke' if a.smoke else ''}.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
