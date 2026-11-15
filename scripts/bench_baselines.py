"""R6 (re-opened, DECISIONS.md D36): measure KVPick's shipped v2 kernel against KVServe's
CUDA quantize kernels and KIVI's pack step, on the SAME synthetic Qwen3-4B-shaped K/V
tensors, same bit depth (4-bit codes, levels=16), same GPU, same run.

This is a quantize-only comparison (no transfer, no codec, no decode-attention GEMV).
Granularity and output format differ between the three and are NOT normalised away --
see csrc/baselines/kvserve_kernels.cu and kvpick/baselines/kivi.py for exactly what each
one computes, and DECISIONS.md D36 for why the result is reported this way:
  KVPick  (ours):    K per-channel per-32-TOKEN-group; V per-token per-32-DIM-group; 2 codes/byte.
  KVServe (CUDA):    K per-channel over the WHOLE sequence (no token grouping);
                     V per-token over ALL heads/dims (no dim grouping); 1 B/value (unpacked).
  KIVI    (PyTorch): K per-channel per-32-token-group; V per-token per-32-dim-group (same
                     grouping as ours); packs 8 x 4-bit codes per int32 (same 0.5 B/value
                     density as ours, via a different word size), no custom CUDA kernel.

The cross-implementation-safe number is per-token latency (same token count fed to all
three); GB/s numbers are also reported but are per-implementation bytes (input-only and
input+output), since "bytes written" means different things in each format.

Usage (GPU only; KVServe and KIVI have no CPU path to compare against):
    python scripts/bench_baselines.py                      # T in {512, 2048, 8192}, 36 layers
    python scripts/bench_baselines.py --smoke               # tiny sizes, few runs, few layers
Output: results/baselines_<gpu>.json
"""
from __future__ import annotations
import argparse, json, statistics, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.ref.quant_wire import quantize_wire, wire_bytes
from kvpick.ref.quant_ref import QuantSpec
from kvpick import ext
from kvpick.baselines import kvserve as kvserve_bench
from kvpick.baselines import kivi as kivi_bench

K_SPEC, V_SPEC = QuantSpec(4, "chan", 32), QuantSpec(4, "row", 32)
LEVELS = 16  # 4-bit code range 0..15, matched across all three implementations


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


def stats(ts, T, in_bytes, out_bytes):
    ts = sorted(ts); med = statistics.median(ts)
    return {"median_us": med * 1e6, "p95_us": ts[int(0.95 * (len(ts) - 1))] * 1e6, "min_us": ts[0] * 1e6,
            "n": len(ts), "ns_per_token": med * 1e9 / T,
            "GBps_input_only": in_bytes / med / 1e9, "GBps_input_plus_output": (in_bytes + out_bytes) / med / 1e9,
            "output_bytes": out_bytes}


def bench_layer(timer: Timer, T: int, xs_k: list, xs_v: list) -> dict:
    """One layer's worth of K and V, T tokens, H=8 heads, D=128 dims. Returns per-candidate stats for K and V."""
    H, D = xs_k[0].shape[1], xs_k[0].shape[2]
    in_bytes = T * H * D * 2  # bf16
    row = {"k": {}, "v": {}}

    # ---- KVPick (ours), single-layer call (unsqueeze a length-1 layer axis)
    xs_k4, xs_v4 = [x.unsqueeze(0) for x in xs_k], [x.unsqueeze(0) for x in xs_v]
    kp, ks_, kz = ext.alloc_k4(xs_k4[0]); vp, vs_, vz = ext.alloc_v4(xs_v4[0])
    out_k, out_v = wire_bytes(kp, ks_, kz), wire_bytes(vp, vs_, vz)
    row["k"]["kvpick_v2"] = stats(timer.time(lambda x: ext.kvq().quantize_k4(x, kp, ks_, kz, 2), xs_k4), T, in_bytes, out_k)
    row["v"]["kvpick_v2"] = stats(timer.time(lambda x: ext.kvq().quantize_v4(x, vp, vs_, vz, 2), xs_v4), T, in_bytes, out_v)

    # ---- KVServe CUDA kernels (vendored harness; see module docstring for granularity caveats)
    def kvs_k(x):
        return kvserve_bench.quantize_channel(x, LEVELS)
    def kvs_v(x):
        return kvserve_bench.quantize_token(x, LEVELS)
    p, amn, asc = kvs_k(xs_k[0])
    kvs_out_k = p.numel() * p.element_size() + amn.numel() * 2 + asc.numel() * 2
    p, amn, asc = kvs_v(xs_v[0])
    kvs_out_v = p.numel() * p.element_size() + amn.numel() * 2 + asc.numel() * 2
    row["k"]["kvserve_channel"] = stats(timer.time(kvs_k, xs_k), T, in_bytes, kvs_out_k)
    row["v"]["kvserve_token"] = stats(timer.time(kvs_v, xs_v), T, in_bytes, kvs_out_v)

    # ---- KIVI (plain PyTorch pack; no custom CUDA kernel -- see module docstring)
    def kivi_k(x):
        return kivi_bench.quant_and_pack_k(x, 32, 4)
    def kivi_v(x):
        return kivi_bench.quant_and_pack_v(x, 32, 4)
    code, scale, mn = kivi_k(xs_k[0])
    kivi_out_k = code.numel() * 4 + scale.numel() * 2 + mn.numel() * 2  # int32 payload + fp16 meta
    code, scale, mn = kivi_v(xs_v[0])
    kivi_out_v = code.numel() * 4 + scale.numel() * 2 + mn.numel() * 2
    row["k"]["kivi_pack"] = stats(timer.time(kivi_k, xs_k), T, in_bytes, kivi_out_k)
    row["v"]["kivi_pack"] = stats(timer.time(kivi_v, xs_v), T, in_bytes, kivi_out_v)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="512,2048,8192")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke: a.T, a.runs = "64,256", 5
    assert torch.cuda.is_available(), "KVServe and KIVI baselines are CUDA-only"
    mod = ext.kvq(); assert mod.with_cuda
    gpu = torch.cuda.get_device_name(0); tag = gpu.lower().replace("nvidia ", "").replace(" ", "_")
    timer = Timer(warmup=3 if a.smoke else 20, runs=a.runs)
    out = {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "levels": LEVELS,
           "note": "quantize-only comparison; granularity and output format differ per implementation, see module docstring",
           "by_T": {}}
    for T in [int(t) for t in a.T.split(",")]:
        gen = torch.Generator(device="cuda").manual_seed(0)
        xs_k = [torch.randn(T, 8, 128, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)]
        xs_v = [torch.randn(T, 8, 128, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)]
        row = bench_layer(timer, T, xs_k, xs_v)
        out["by_T"][f"T{T}"] = row
        print(f"T={T}:")
        for part in ("k", "v"):
            print(f"  {part}: " + " | ".join(f"{name} {s['median_us']:.1f}us ({s['ns_per_token']:.1f} ns/tok)" for name, s in row[part].items()))
        del xs_k, xs_v; torch.cuda.empty_cache()
    p = pathlib.Path(a.out or f"results/baselines_{tag}{'_smoke' if a.smoke else ''}.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
