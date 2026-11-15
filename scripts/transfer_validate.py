"""Two-GPU validation of the transfer model (the one piece the feasibility study only computed).

For n_new in {512, 2048, 8192}, format in {raw, kv4}, link in {1, 3, 10, 25} Gbps (rtt 0.5 ms), rep x3:
  serial:    compress on GPU0 (fused kernel, one launch, into ONE flat buffer) -> real GPU0->GPU1 copy paced by the
             link emulator -> decompress on GPU1 -> "KV ready" wall time
  pipelined: per-layer compress -> submit chunk -> receiver waits chunk -> per-layer decompress (36 chunks in flight)
Measured vs predicted (cost model: compress + bytes/link + rtt + decompress) -> % error per point.
--with-model: also measure a REAL recompute (HF prefill on GPU1) for n in {512, 2048} and find the measured crossover
link speed (where kv4 send becomes faster than recompute) against the model's prediction.
Output: results/transfer_validate_<gpu>.json.   --smoke: CPU, pure emulation, tiny sizes (plumbing only).
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick import ext
from kvpick.link.link_emulator import LinkEmulator
from kvpick.cost.model import CostModel

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sync_all():
    for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)


def xcopy(dst: torch.Tensor, src: torch.Tensor, host: torch.Tensor):
    """GPU0 -> pinned host -> GPU1. Used instead of a direct peer copy: on the 2x A40 node the peer path (SYS topology,
    no NVLink) showed a fixed 50-100 ms cost per copy regardless of size, while pinned staging runs at ~20 GB/s (D30)."""
    host.copy_(src, non_blocking=True)
    if src.is_cuda: torch.cuda.synchronize(src.device)
    dst.copy_(host, non_blocking=True)
    if dst.is_cuda: torch.cuda.synchronize(dst.device)


def pinned_like(t: torch.Tensor) -> torch.Tensor:
    return torch.empty(t.shape, dtype=t.dtype, pin_memory=True)


def send_serial(k, v, fmt, link, dev1, mod, stage):
    """Returns (wall_s, breakdown) for one request: compress(GPU0) -> paced copy -> decompress(GPU1).
    stage: dict of preallocated pinned host buffers and GPU1 destinations (allocation is not part of the timed path)."""
    L, T, H, D = k.shape
    sync_all(); t0 = time.perf_counter()
    if fmt == "raw":
        kd, vd, hk, hv = stage["kd"], stage["vd"], stage["hk"], stage["hv"]
        t1 = time.perf_counter()
        link.reset(); d1 = link.submit(k.numel() * 2, lambda: xcopy(kd, k, hk)); d2 = link.submit(v.numel() * 2, lambda: xcopy(vd, v, hv)); link.wait(d2)
        t2 = time.perf_counter(); t3 = t2
    else:
        flat, views = stage["flat"], stage["views"]; ext.quantize_kv4_into_buffer(k, v, flat, views); sync_all()
        t1 = time.perf_counter()
        flat1, hflat = stage["flat1"], stage["hflat"]
        link.reset(); d = link.submit(flat.numel(), lambda: xcopy(flat1, flat, hflat)); link.wait(d)
        t2 = time.perf_counter()
        offs, _ = ext.kv4_buffer_layout(L, T, H, D); Tg = ext.k_groups(T)
        def vw(name, dt, shape): o, sz = offs[name]; return flat1[o:o + sz].view(dt).view(*shape)
        kd, vd = stage["kd"], stage["vd"]
        mod.dequantize_k4(vw("kp", torch.uint8, (L, T, H, D // 2)), vw("ks", torch.float16, (L, Tg, H, D)), vw("kz", torch.float16, (L, Tg, H, D)), kd, 2)
        mod.dequantize_v4(vw("vp", torch.uint8, (L, T, H, D // 2)), vw("vs", torch.float16, (L, T, H, D // 32)), vw("vz", torch.float16, (L, T, H, D // 32)), vd, 2)
        sync_all(); t3 = time.perf_counter()
    return t3 - t0, {"compress_ms": (t1 - t0) * 1e3, "transfer_ms": (t2 - t1) * 1e3, "decompress_ms": (t3 - t2) * 1e3}


def send_pipelined(k, v, fmt, link, dev1, mod, stage):
    """Per-layer chunks: compress layer l, submit; receiver waits chunk l and decompresses it while later layers are in flight."""
    L, T, H, D = k.shape
    sync_all(); t0 = time.perf_counter(); link.reset()
    kd, vd = stage["kd"], stage["vd"]
    deliveries = []; bufs = stage.get("flat1_layers")
    for l in range(L):
        if fmt == "raw":
            deliveries.append((link.submit(2 * T * H * D * 2, lambda l=l: (xcopy(kd[l:l + 1], k[l:l + 1], stage["hk"][l:l + 1]), xcopy(vd[l:l + 1], v[l:l + 1], stage["hv"][l:l + 1]))), None))
        else:
            flat, views = stage["flat_layers"][l]; ext.quantize_kv4_into_buffer(k[l:l + 1], v[l:l + 1], flat, views)
            deliveries.append((link.submit(flat.numel(), lambda l=l, flat=flat: xcopy(bufs[l], flat, stage["hflat_layers"][l])), l))
    offs, _ = ext.kv4_buffer_layout(1, T, H, D); Tg = ext.k_groups(T)
    for d, l in deliveries:
        link.wait(d)
        if fmt != "raw":
            flat1 = bufs[l]
            def vw(name, dt, shape): o, sz = offs[name]; return flat1[o:o + sz].view(dt).view(*shape)
            mod.dequantize_k4(vw("kp", torch.uint8, (1, T, H, D // 2)), vw("ks", torch.float16, (1, Tg, H, D)), vw("kz", torch.float16, (1, Tg, H, D)), kd[l:l + 1], 2)
            mod.dequantize_v4(vw("vp", torch.uint8, (1, T, H, D // 2)), vw("vs", torch.float16, (1, T, H, D // 32)), vw("vz", torch.float16, (1, T, H, D // 32)), vd[l:l + 1], 2)
    sync_all(); return time.perf_counter() - t0


def make_stage(k, dev1, smoke):
    """Preallocate GPU1 destinations, the flat KV4 buffers (whole cache and per layer) and the pinned host staging buffers."""
    L, T, H, D = k.shape; pin = (lambda t: pinned_like(t)) if not smoke else (lambda t: torch.empty_like(t))
    st = {"kd": torch.empty(L, T, H, D, dtype=torch.bfloat16, device=dev1), "vd": torch.empty(L, T, H, D, dtype=torch.bfloat16, device=dev1)}
    st["hk"] = pin(k); st["hv"] = pin(k)
    st["flat"], st["views"] = ext.alloc_kv4_buffer(L, T, H, D, device=k.device); st["flat1"] = torch.empty_like(st["flat"], device=dev1); st["hflat"] = pin(st["flat"])
    st["flat_layers"] = [ext.alloc_kv4_buffer(1, T, H, D, device=k.device) for _ in range(L)]
    st["flat1_layers"] = [torch.empty_like(f, device=dev1) for f, _ in st["flat_layers"]]; st["hflat_layers"] = [pin(f) for f, _ in st["flat_layers"]]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="512,2048,8192"); ap.add_argument("--rates", default="1,3,10,25"); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rtt-ms", type=float, default=0.5); ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--with-model", action="store_true"); ap.add_argument("--cost-tables", default=str(ROOT / "results/cost_tables_a40.json"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cost = CostModel(a.cost_tables)
    if a.smoke:
        print("smoke: plumbing check only (no GPU)"); a.ns, a.rates, a.reps, a.layers = "64", "100", 1, 2
    assert a.smoke or torch.cuda.device_count() >= 2, "needs two GPUs"
    dev0, dev1 = ("cuda:0", "cuda:1") if not a.smoke else ("cpu", "cpu"); mod = ext.kvq(force_cpu=a.smoke)
    gpu = torch.cuda.get_device_name(0) if not a.smoke else "cpu-smoke"; tag = gpu.lower().replace("nvidia ", "").replace(" ", "_")
    out = {"gpu": gpu, "gpu1": torch.cuda.get_device_name(1) if not a.smoke else None, "rtt_ms": a.rtt_ms, "layers": a.layers, "points": [], "note": "measured on real GPU0->pinned host->GPU1 copies paced by the link emulator (peer copy path had a fixed 50-100 ms cost on this node, D30); predicted = cost model (serial)"}
    gen = torch.Generator(device=dev0).manual_seed(0)
    for n in [int(x) for x in a.ns.split(",")]:
        k = torch.randn(a.layers, n, 8, 128, generator=gen, device=dev0).to(torch.bfloat16); v = torch.randn(a.layers, n, 8, 128, generator=gen, device=dev0).to(torch.bfloat16)
        stage = make_stage(k, dev1, a.smoke)
        for fmt in ("raw", "int4_kivi"):
            for rate in [float(x) for x in a.rates.split(",")]:
                link = LinkEmulator(rate, a.rtt_ms / 1e3)
                ser, pipe, brk = [], [], None
                for _ in range(a.reps):
                    w, brk = send_serial(k, v, fmt, link, dev1, mod, stage); ser.append(w * 1e3)
                    pipe.append(send_pipelined(k, v, fmt, link, dev1, mod, stage) * 1e3)
                scale = a.layers / 36            # cost model assumes 36 layers
                pred = cost.t_send_ms(fmt, n, rate, a.rtt_ms) * scale if fmt in cost.formats else None
                pt = {"n_new": n, "format": fmt, "gbps": rate, "serial_ms_median": statistics.median(ser), "pipelined_ms_median": statistics.median(pipe),
                      "serial_breakdown_ms": brk, "predicted_serial_ms": pred,
                      "err_pct_serial": (100 * (statistics.median(ser) - pred) / pred) if pred else None}
                out["points"].append(pt)
                print(f"n={n:5d} {fmt:9s} {rate:5.1f} Gbps: serial {pt['serial_ms_median']:8.1f} ms (pred {pred:8.1f}, {pt['err_pct_serial']:+5.1f}%)  pipelined {pt['pipelined_ms_median']:8.1f} ms  [c {brk['compress_ms']:.1f} / x {brk['transfer_ms']:.1f} / d {brk['decompress_ms']:.1f}]")
        del k, v
        if not a.smoke: torch.cuda.empty_cache()
    if a.with_model and not a.smoke:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).to(dev1).eval()
        out["crossover"] = []
        with torch.no_grad():
            for n in (512, 2048):
                ids = torch.randint(0, 1000, (1, n), device=dev1); ts = []
                for i in range(4):
                    torch.cuda.synchronize(1); t0 = time.perf_counter(); model(input_ids=ids, use_cache=True); torch.cuda.synchronize(1)
                    if i: ts.append((time.perf_counter() - t0) * 1e3)
                rec = statistics.median(ts)
                k = torch.randn(36, n, 8, 128, device=dev0).to(torch.bfloat16); v = torch.randn(36, n, 8, 128, device=dev0).to(torch.bfloat16); stage = make_stage(k, dev1, False)
                meas = {}
                for rate in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
                    link = LinkEmulator(rate, a.rtt_ms / 1e3); meas[rate] = statistics.median([send_serial(k, v, "int4_kivi", link, dev1, mod, stage)[0] * 1e3 for _ in range(3)])
                measured_cross = next((r for r in sorted(meas) if meas[r] < rec), None)
                pred_cross = next((r for r in [x / 10 for x in range(5, 100)] if cost.t_send_ms("int4_kivi", n, r, a.rtt_ms) < cost.t_prefill_ms(n, 0)), None)
                out["crossover"].append({"n_new": n, "measured_recompute_ms": rec, "model_recompute_ms": cost.t_prefill_ms(n, 0), "measured_send_ms_by_gbps": meas,
                                         "measured_crossover_gbps_grid": measured_cross, "predicted_crossover_gbps": pred_cross})
                print(f"crossover n={n}: real recompute {rec:.0f} ms (model {cost.t_prefill_ms(n, 0):.0f}); kv4 send wins from {measured_cross} Gbps measured vs {pred_cross} Gbps predicted")
    p = ROOT / "results" / f"transfer_validate_{tag}{'_smoke' if a.smoke else ''}.json"; p.write_text(json.dumps(out, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
