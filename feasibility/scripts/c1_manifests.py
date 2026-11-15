"""C1: confirm model/data availability, build fixed subsets (seed 0), record KV-bytes arithmetic.

Outputs: results/C1_model_data.json, manifests/*.json
"""
from __future__ import annotations
import argparse, json, random, zipfile, os
from pathlib import Path
from common import Result, MANIFESTS, MODEL_ID, SEED, FEAS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    r = Result("C1")
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    from datasets import load_dataset
    api = HfApi()

    # ---- model
    info = api.model_info(MODEL_ID, files_metadata=True)
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    L, Hkv, D = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"]
    dtype_bytes = 2  # bf16
    per_layer = 2 * Hkv * D * dtype_bytes            # K and V
    per_token = per_layer * L
    r.d["measurements"]["model"] = {
        "hf_id": MODEL_ID, "license": info.card_data.get("license"), "gated": info.gated,
        "total_bytes": sum((s.size or 0) for s in info.siblings),
        "architecture": cfg["architectures"][0], "num_hidden_layers": L, "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": Hkv, "head_dim": D, "dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
        "max_position_embeddings": cfg["max_position_embeddings"],
        "kv_bytes_per_token_per_layer": per_layer,
        "kv_bytes_per_token": per_token,
        "arithmetic": f"2 (K,V) x {Hkv} kv_heads x {D} head_dim x {dtype_bytes} B (bf16) = {per_layer} B/token/layer; x {L} layers = {per_token} B/token",
        "raw_MB_at_T": {T: round(per_token * T / 1e6, 1) for T in (512, 2048, 8192)},
    }
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rng = random.Random(SEED)
    n_wt_test, n_wt_calib, n_gsm, n_long, n_long_calib = (50, 20, 100, 50, 5) if not args.smoke else (3, 2, 5, 3, 2)

    # ---- WikiText-2 test split -> non-overlapping 2048-token windows
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    W = 2048
    n_windows = len(ids) // W
    windows = list(range(n_windows))
    rng.shuffle(windows)
    wt_test = sorted(windows[:n_wt_test]); wt_calib = sorted(windows[n_wt_test:n_wt_test + n_wt_calib])
    assert not set(wt_test) & set(wt_calib)
    (MANIFESTS / "wikitext2_windows.json").write_text(json.dumps({
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test", "license": "cc-by-sa-3.0 / gfdl", "tokenizer": MODEL_ID,
        "window_tokens": W, "total_tokens": len(ids), "n_windows_available": n_windows, "seed": SEED,
        "test_windows": wt_test, "calib_windows": wt_calib,
        "note": "window i = tokens [i*2048, (i+1)*2048) of the concatenated test split (joined with \\n). test and calib are disjoint."}, indent=1))

    # ---- GSM8K test
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(gsm))); rng.shuffle(idx); gsm_idx = sorted(idx[:n_gsm])
    (MANIFESTS / "gsm8k_test.json").write_text(json.dumps({
        "dataset": "openai/gsm8k:main:test", "license": "mit", "n_total": len(gsm), "seed": SEED, "indices": gsm_idx,
        "example_question_lengths_tokens": [len(tok(gsm[i]["question"])["input_ids"]) for i in gsm_idx[:5]]}, indent=1))

    # ---- LongBench (script-based repo; read data.zip directly)
    r.workaround("THUDM/LongBench is a script dataset that datasets>=4 refuses to load; data.zip is downloaded from the hub and the JSONL files are read directly.")
    zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    dest = FEAS / "external" / "longbench_data"; dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        for n in ("data/qasper.jsonl", "data/hotpotqa.jsonl"):
            if not (dest / n).exists():
                z.extract(n, dest)
    def rows(name):
        return [json.loads(l) for l in open(dest / "data" / f"{name}.jsonl")]
    def tok_len(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])
    # qasper -> quality test set: contexts that fit in 7500 tokens (leaves room for the question + answer)
    qas = rows("qasper")
    qas_len = [(i, tok_len(r_["context"])) for i, r_ in enumerate(qas)]
    eligible = [i for i, n in qas_len if 1024 <= n <= 7500]
    rng.shuffle(eligible); q_idx = sorted(eligible[:n_long])
    # hotpotqa -> calibration prompts of exactly 8192 tokens (truncate)
    hot = rows("hotpotqa")
    hot_len = [(i, tok_len(r_["context"])) for i, r_ in enumerate(hot)]
    elig8k = [i for i, n in hot_len if n >= 8192]
    rng.shuffle(elig8k); h_idx = sorted(elig8k[:n_long_calib])
    (MANIFESTS / "longbench.json").write_text(json.dumps({
        "repo": "THUDM/LongBench (data.zip, sha from hub snapshot)", "license": "dataset card lists none at repo level; per-source licenses apply (Qasper: CC-BY-4.0, HotpotQA: CC-BY-SA-4.0)",
        "seed": SEED,
        "qasper_test": {"indices": q_idx, "ids": [qas[i]["_id"] for i in q_idx], "context_tokens": [dict(qas_len)[i] for i in q_idx],
                         "n_eligible_1024_7500": len(eligible), "n_total": len(qas)},
        "hotpotqa_calib_8192": {"indices": h_idx, "ids": [hot[i]["_id"] for i in h_idx], "context_tokens": [dict(hot_len)[i] for i in h_idx],
                                 "truncate_to": 8192, "n_eligible_ge_8192": len(elig8k), "n_total": len(hot)},
    }, indent=1))
    r.d["measurements"]["data"] = {
        "wikitext2": {"total_tokens": len(ids), "windows_2048": n_windows, "test": len(wt_test), "calib": len(wt_calib)},
        "gsm8k": {"n_total": len(gsm), "selected": len(gsm_idx)},
        "longbench_qasper": {"n_total": len(qas), "eligible": len(eligible), "selected": len(q_idx), "ctx_tokens_min_max": [min(dict(qas_len)[i] for i in q_idx), max(dict(qas_len)[i] for i in q_idx)]},
        "longbench_hotpotqa_8192": {"n_total": len(hot), "eligible_ge_8192": len(elig8k), "selected": len(h_idx)},
    }
    r.note("R1 criterion: model + WikiText-2 + GSM8K + one long-context set available with licenses recorded.")
    p = r.save("PASS", name="C1_model_data")
    print(json.dumps(r.d["measurements"], indent=1)); print("saved", p)

if __name__ == "__main__":
    main()
