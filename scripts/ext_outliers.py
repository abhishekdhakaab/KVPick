"""Part A3: key-channel outlier analysis for another model (the analysis behind results/feasibility/G1_key_channel_amax.png).
Keys are taken exactly as stored in the cache (after rotary embedding). Per layer: per-(head, channel) amax over all tokens;
ratio = amax of the top 1% of channels / median channel amax (R2 criterion of the feasibility study: >= 5x in most layers).
Prompts: 5 WikiText-2 test windows (2048 tokens) + 2 PG-19 test prefixes (8192 tokens).  Output results/ext/outliers_<model>.json + figures/ext/outliers_<model>.png
--tiny: random tiny model (plumbing)."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import runlog, tasks as T, longctx as LC

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--tiny", action="store_true"); a = ap.parse_args()
    dev = "cpu" if a.tiny else "cuda"; info = runlog.start(extra={"model": a.model}); model, tok, cfg = LC.load_model(a.model, dev, tiny=a.tiny)
    prompts = []
    if a.tiny:
        prompts = [torch.randint(1000, 30000, (1, 512))]
    else:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"); ids = tok("\n".join(ds["text"]), add_special_tokens=False)["input_ids"]
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        prompts += [torch.tensor([bos + ids[i * 2047:(i + 1) * 2047]]) for i in range(5)]
        for text in T.pg19_split("test", 2):
            prompts.append(torch.tensor([T.tokens_for_ppl(tok, text, 8192, bool(bos))]))
    L = len(model.model.layers); amax = None
    for p in prompts:
        _, cache = LC.chunked_prefill(model, p.to(dev)); layers = LC.cache_layers(cache)
        a_ = torch.stack([k.float().abs().amax(dim=(0, 2)) for k, _ in layers]).cpu().numpy()    # [L, H, D]
        amax = a_ if amax is None else np.maximum(amax, a_)
    flat = amax.reshape(L, -1); k = max(1, int(round(0.01 * flat.shape[1])))
    ratios = [float(np.sort(f)[-k:].mean() / np.median(f)) for f in flat]; maxratio = [float(f.max() / np.median(f)) for f in flat]
    out = {"info": runlog.finish(info), "model": a.model, "n_prompts": len(prompts), "layers": L, "top1pct_over_median_per_layer": ratios, "max_over_median_per_layer": maxratio,
           "layers_with_ratio_ge_5": int(sum(r >= 5 for r in ratios)), "median_ratio": float(np.median(ratios)), "max_ratio": float(max(maxratio)),
           "reference_qwen3_4b": "results/feasibility/G1_kv_stats.json"}
    (ROOT / "results/ext").mkdir(exist_ok=True); (ROOT / f"results/ext/outliers_{a.model}{'_tiny' if a.tiny else ''}.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].bar(range(L), ratios); ax[0].axhline(5, color="r", ls="--", label="5x (R2)"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("top-1% channel amax / median"); ax[0].legend()
    l = int(np.argmax(maxratio)); ax[1].bar(range(flat.shape[1]), np.sort(flat[l])[::-1]); ax[1].set_title(f"layer {l}: channel amax (sorted), max/median = {maxratio[l]:.1f}x"); ax[1].set_xlabel("channel rank")
    fig.suptitle(f"{a.model}: post-RoPE key channel outliers"); fig.tight_layout(); (ROOT / "figures/ext").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / f"figures/ext/outliers_{a.model}{'_tiny' if a.tiny else ''}.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k not in ("info", "top1pct_over_median_per_layer", "max_over_median_per_layer")}))


if __name__ == "__main__":
    main()
