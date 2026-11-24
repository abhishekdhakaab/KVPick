"""B3: how much does the compress kernel slow a running prefill when they share the GPU? (fast-GPU session, only if time is left)
Prefill chunks run on the default stream; on a second stream the KV4 compress kernel runs in a loop on a tensor sized to cover
`blocks` thread blocks ("few blocks") or the whole cache ("all blocks").  Reports prefill chunk time alone vs with the kernel, and
the compress throughput achieved while overlapping.  Output results/ext/contention_<gpu>.json"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, threading, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick import ext
from kvpick.study import runlog
from kvpick.study.longctx import load_model, chunked_prefill

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--ctx", type=int, default=32768); ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--small-blocks", type=int, default=16); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    info = runlog.start(extra={"model": a.model}); mod = ext.kvq(); assert mod.with_cuda
    model, tok, cfg = load_model(a.model, "cuda"); L = cfg["layers"]; H, D = 8, 128
    ids = torch.randint(1000, 30000, (1, a.ctx), device="cuda"); side = torch.cuda.Stream()
    def side_tensors(T_tokens, layers):
        k = torch.randn(layers, T_tokens, H, D, device="cuda").to(torch.bfloat16); v = torch.randn_like(k, dtype=torch.float32).to(torch.bfloat16)
        flat, views = ext.alloc_kv4_buffer(layers, T_tokens); return k, v, views
    # blocks per launch: K tiles = layers * ceil(T/32) * 2 (8 heads / 4 per block); V blocks = layers*T*8/16  -> tokens giving ~`blocks` K blocks
    small_T = max(32, 32 * (a.small_blocks // (2 * 1) + 1))
    cases = {"all_blocks": side_tensors(4096, L), "few_blocks": side_tensors(small_T, 1)}
    res = {}
    def run_prefill():
        tr = []; chunked_prefill(model, ids, a.chunk, tr); return [t for *_, t in tr]
    run_prefill(); base = [statistics.median(c) for c in zip(*[run_prefill() for _ in range(a.reps)])]
    res["alone_total_ms"] = sum(base) * 1e3
    for name, (k, v, views) in cases.items():
        stop = threading.Event(); count = [0]
        def loop():
            with torch.cuda.stream(side):
                while not stop.is_set():
                    mod.quantize_kv4(k, v, views["kp"], views["ks"], views["kz"], views["vp"], views["vs"], views["vz"], 2); count[0] += 1
                    if count[0] % 8 == 0: side.synchronize()
        th = threading.Thread(target=loop); th.start(); time.sleep(0.2); t0 = time.time()
        runs = [run_prefill() for _ in range(a.reps)]; el = time.time() - t0; stop.set(); th.join(); side.synchronize()
        tot = [sum(r) * 1e3 for r in runs]; res[name] = {"prefill_total_ms": statistics.median(tot), "slowdown_pct": 100 * (statistics.median(tot) / res["alone_total_ms"] - 1),
                                                       "kernel_launches": count[0], "kernel_bytes_in_per_launch": k.numel() * 4,
                                                       "kernel_GBps_input_while_overlapping": count[0] * k.numel() * 4 / el / 1e9}
        print(name, res[name], flush=True)
    out = {"info": runlog.finish(info), "label": "measured", "gpu": info["gpu"], "ctx": a.ctx, "chunk": a.chunk, "result": res}
    tag = info["gpu"].lower().replace("nvidia ", "").replace(" ", "_"); (ROOT / f"results/ext/contention_{tag}.json").write_text(json.dumps(out, indent=1)); print("saved")


if __name__ == "__main__":
    main()
