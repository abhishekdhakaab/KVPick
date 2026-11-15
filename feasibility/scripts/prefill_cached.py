"""E5-lite: costs the simulator will need, measured now so no second rental is required if the project proceeds.
(1) prefill time for n_new tokens with m_cached tokens already in the cache, grid n in {128,512,2048}, m in {0,2048,8192};
(2) decode step time (one token for every sequence) at batch {1,8,16} with 2048-token contexts.
Output: results/E5_prefill_cached.json.  --smoke: tiny random model, small grid."""
from __future__ import annotations
import argparse, json, time, torch
from common import Result, RESULTS, MODEL_ID, SEED
from kv_capture import load_model
from transformers import DynamicCache

@torch.no_grad()
def t_prefill_with_cache(model, n_new, m_cached, reps, dev):
    gen = torch.Generator().manual_seed(SEED); ts = []
    for rep in range(reps + 1):
        cache = None
        if m_cached:
            ids0 = torch.randint(0, 1000, (1, m_cached), generator=gen).to(dev)
            cache = model(input_ids=ids0, use_cache=True).past_key_values
        ids = torch.randint(0, 1000, (1, n_new), generator=gen).to(dev)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); model(input_ids=ids, past_key_values=cache, use_cache=True)
        if dev == "cuda": torch.cuda.synchronize()
        if rep: ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]

@torch.no_grad()
def t_decode_step(model, batch, ctx, steps, dev):
    gen = torch.Generator().manual_seed(SEED)
    ids0 = torch.randint(0, 1000, (batch, ctx), generator=gen).to(dev)
    cache = model(input_ids=ids0, use_cache=True).past_key_values
    nxt = torch.randint(0, 1000, (batch, 1), generator=gen).to(dev); ts = []
    for s in range(steps + 2):
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        if dev == "cuda": torch.cuda.synchronize()
        if s >= 2: ts.append(time.perf_counter() - t0)
        cache = out.past_key_values
    return sorted(ts)[len(ts) // 2]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tiny = a.smoke
    model, tok, cfg = load_model(a.model, dev, tiny); r = Result("E5")
    ns, ms, bs = ([128, 512, 2048], [0, 2048, 8192], [1, 8, 16]) if not a.smoke else ([32, 64], [0, 64], [1, 2])
    pre = {}
    for m in ms:
        for n in ns:
            pre[f"n{n}_m{m}"] = t_prefill_with_cache(model, n, m, a.reps, dev) * 1e3; print(f"prefill n_new={n} m_cached={m}: {pre[f'n{n}_m{m}']:.1f} ms")
    dec = {}
    for b in bs:
        dec[f"batch{b}"] = t_decode_step(model, b, 2048 if not a.smoke else 64, 10, dev) * 1e3; print(f"decode step batch={b} ctx=2048: {dec[f'batch{b}']:.2f} ms")
    r.d["measurements"] = {"model": "tiny" if tiny else a.model, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
                           "prefill_ms": pre, "decode_step_ms": dec}
    print("saved", r.save("INFO", name="E5_prefill_cached"))

if __name__ == "__main__":
    main()
