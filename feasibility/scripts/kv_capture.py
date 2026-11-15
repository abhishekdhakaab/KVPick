"""Capture per-layer K and V from a Hugging Face model prefill, plus per-channel statistics (G1).

Per prompt it saves (optionally) data/kv/<id>.pt = {"k": [L, T, H, D] bf16, "v": [...]} and always accumulates:
  per-layer, per-head, per-channel amax and kurtosis for K and V  -> results/G1_kv_stats.json
  ratio of top-1% channel amax to median channel amax per layer (R2 criterion)

Prompt sources (from manifests/): wikitext calib windows (2048 tokens), hotpotqa calib (truncated to 8192).
--smoke: tiny random-weight Qwen3 config (2 layers, same KV row shape), random token ids, no download.
"""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
import torch
from common import Result, MANIFESTS, MODEL_ID, FEAS, RESULTS, SEED, LOGS


def load_model(model_id: str, device: str, tiny_random: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    if tiny_random:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=1024, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                          num_attention_heads=8, num_key_value_heads=8, head_dim=128, max_position_embeddings=16384)
        torch.manual_seed(SEED)
        model = Qwen3ForCausalLM(cfg).to(device).eval()
        return model, None, cfg
    cfg = AutoConfig.from_pretrained(model_id)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok, cfg


def build_prompts(tok, n_wt: int, n_long: int, smoke: bool):
    """Return list of (id, token_ids) from the calibration manifests."""
    import random
    prompts = []
    if smoke or tok is None:
        rng = torch.Generator().manual_seed(SEED)
        for i in range(n_wt):
            prompts.append((f"rand_{i}", torch.randint(0, 1000, (1, 300), generator=rng)))
        return prompts
    from datasets import load_dataset
    wt = json.load(open(MANIFESTS / "wikitext2_windows.json"))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n".join(ds["text"]), add_special_tokens=False)["input_ids"]
    W = wt["window_tokens"]
    for w in wt["calib_windows"][:n_wt]:
        prompts.append((f"wt_calib_{w}", torch.tensor([ids[w * W:(w + 1) * W]])))
    lb = json.load(open(MANIFESTS / "longbench.json"))
    rows = [json.loads(l) for l in open(FEAS / "external/longbench_data/data/hotpotqa.jsonl")]
    for i in lb["hotpotqa_calib_8192"]["indices"][:n_long]:
        t = tok(rows[i]["context"], add_special_tokens=False)["input_ids"][:8192]
        prompts.append((f"hot_calib_{i}", torch.tensor([t])))
    return prompts


@torch.no_grad()
def prefill_kv(model, input_ids: torch.Tensor):
    """Run prefill; return K, V as [L, T, H, D] (post-RoPE keys, exactly what the cache holds)."""
    out = model(input_ids=input_ids.to(model.device), use_cache=True)
    cache = out.past_key_values
    ks = torch.stack([layer.keys[0].permute(1, 0, 2) for layer in cache.layers])      # [L, T, H, D]
    vs = torch.stack([layer.values[0].permute(1, 0, 2) for layer in cache.layers])
    return ks, vs


def channel_stats(x: torch.Tensor) -> dict:
    """x: [L, T, H, D] -> per (L, H, D) amax and kurtosis over T, plus the R2 ratio per layer."""
    xf = x.float()
    amax = xf.abs().amax(dim=1)                                   # [L, H, D]
    mu = xf.mean(dim=1, keepdim=True)
    var = ((xf - mu) ** 2).mean(dim=1)
    m4 = ((xf - mu) ** 4).mean(dim=1)
    kurt = m4 / var.clamp_min(1e-12) ** 2                          # [L, H, D]  (3 = Gaussian)
    L = x.shape[0]
    ratios, top_channels = [], []
    for l in range(L):
        flat = amax[l].flatten()                                   # 1024 channels per layer
        k = max(1, int(round(0.01 * flat.numel())))
        top = flat.topk(k).values.mean()
        med = flat.median()
        ratios.append(float(top / med))
        top_channels.append([int(i) for i in flat.topk(5).indices])  # (h*D + d)
    return {"amax": amax, "kurtosis": kurt, "top1pct_over_median_per_layer": ratios, "top5_channels_per_layer": top_channels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-wikitext", type=int, default=20)
    ap.add_argument("--n-long", type=int, default=5)
    ap.add_argument("--save-kv-dir", default=None, help="if set, save per-prompt K/V tensors (bf16) here")
    ap.add_argument("--save-subset-layers", default="0,8,17,26,35", help="layers kept when --save-kv-dir is set with --subset")
    ap.add_argument("--subset", action="store_true", help="save only --save-subset-layers (for laptop debugging)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-random", action="store_true", help="random-weight 2-layer Qwen3 (offline smoke)")
    a = ap.parse_args()
    if a.smoke:
        a.n_wikitext, a.n_long = min(a.n_wikitext, 2), min(a.n_long, 1)
    r = Result("G1")
    t0 = time.time()
    model, tok, cfg = load_model(a.model, a.device, a.tiny_random or (a.smoke and a.model == MODEL_ID))
    r.d["measurements"]["model"] = {"id": "tiny-random" if (a.tiny_random or (a.smoke and a.model == MODEL_ID)) else a.model,
                                   "layers": cfg.num_hidden_layers, "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
                                   "load_s": round(time.time() - t0, 1)}
    prompts = build_prompts(tok, a.n_wikitext, a.n_long, a.smoke and tok is None)
    L = cfg.num_hidden_layers
    agg_k_amax, agg_v_amax, agg_k_kurt, agg_v_kurt = [], [], [], []
    per_prompt = []
    if a.save_kv_dir:
        Path(a.save_kv_dir).mkdir(parents=True, exist_ok=True)
    keep = [int(s) for s in a.save_subset_layers.split(",")] if a.subset else None
    for pid, ids in prompts:
        t1 = time.time()
        ks, vs = prefill_kv(model, ids)
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
        sk, sv = channel_stats(ks), channel_stats(vs)
        agg_k_amax.append(sk["amax"].cpu()); agg_v_amax.append(sv["amax"].cpu())
        agg_k_kurt.append(sk["kurtosis"].cpu()); agg_v_kurt.append(sv["kurtosis"].cpu())
        per_prompt.append({"id": pid, "T": int(ids.shape[1]), "prefill_s": round(dt, 3),
                           "k_top1pct_over_median_per_layer": [round(x, 2) for x in sk["top1pct_over_median_per_layer"]],
                           "v_top1pct_over_median_per_layer": [round(x, 2) for x in sv["top1pct_over_median_per_layer"]]})
        if a.save_kv_dir:
            sel = keep if keep else list(range(L))
            torch.save({"id": pid, "T": int(ids.shape[1]), "layers": sel,
                        "k": ks[sel].to(torch.bfloat16).cpu(), "v": vs[sel].to(torch.bfloat16).cpu()},
                       Path(a.save_kv_dir) / f"{pid}.pt")
        del ks, vs
        if a.device == "cuda":
            torch.cuda.empty_cache()
    # aggregate across prompts: max of amax, mean of kurtosis
    k_amax = torch.stack(agg_k_amax).amax(0); v_amax = torch.stack(agg_v_amax).amax(0)
    k_kurt = torch.stack(agg_k_kurt).mean(0); v_kurt = torch.stack(agg_v_kurt).mean(0)
    def layer_ratio(amax):
        out = []
        for l in range(amax.shape[0]):
            flat = amax[l].flatten(); k = max(1, int(round(0.01 * flat.numel())))
            out.append(float(flat.topk(k).values.mean() / flat.median()))
        return out
    kr, vr = layer_ratio(k_amax), layer_ratio(v_amax)
    r.d["measurements"].update({
        "n_prompts": len(prompts), "prompt_lengths": [p["T"] for p in per_prompt],
        "k_top1pct_over_median_per_layer": [round(x, 2) for x in kr],
        "v_top1pct_over_median_per_layer": [round(x, 2) for x in vr],
        "k_ratio_max": max(kr), "k_ratio_max_layer": int(kr.index(max(kr))), "v_ratio_max": max(vr),
        "k_median_kurtosis_per_layer": [round(float(k_kurt[l].median()), 2) for l in range(L)],
        "v_median_kurtosis_per_layer": [round(float(v_kurt[l].median()), 2) for l in range(L)],
        "R2_threshold": 5.0, "R2_pass": max(kr) >= 5.0, "per_prompt": per_prompt,
    })
    torch.save({"k_amax": k_amax, "v_amax": v_amax, "k_kurt": k_kurt, "v_kurt": v_kurt}, RESULTS / "G1_channel_tensors.pt")
    # plot: key channel max for 3 layers
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        layers = sorted(set([0, L // 2, L - 1, int(kr.index(max(kr)))]))[:4]
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3), sharey=False)
        for ax, l in zip(axes if len(layers) > 1 else [axes], layers):
            ax.plot(k_amax[l].flatten().numpy(), lw=0.6)
            ax.set_title(f"layer {l}: key channel amax (ratio {kr[l]:.1f}x)"); ax.set_xlabel("channel (head*128+d)")
        plt.tight_layout(); plt.savefig(RESULTS / "G1_key_channel_amax.png", dpi=120)
        r.note("plot: results/G1_key_channel_amax.png")
    except Exception as e:
        r.note(f"plot skipped: {e!r}")
    status = "INFO" if (a.smoke or a.tiny_random) else ("PASS" if max(kr) >= 5.0 else "FAIL")
    p = r.save(status, name="G1_kv_stats")
    print(json.dumps({k: v for k, v in r.d["measurements"].items() if k != "per_prompt"}, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
