"""Quality effect of lossy KV (quantize -> dequantize -> inject back as the cache), with the SHIPPED KERNEL in the loop
for int4_kivi on CUDA (the Python reference for the other formats). Adapted from feasibility G3.

Tasks (sizes from manifests; --smoke shrinks them):
  ppl   WikiText-2 test windows of 2048 tokens: prefix 1536 compressed, last 512 scored by teacher forcing
  gsm8k 100 test questions, 4-shot prefix, greedy generation, last-number match
  qasper 50 LongBench Qasper examples, greedy generation, LongBench F1
Formats: raw (same code path, no quantization), int8_row, int4_row, int4_row_g32, int4_kivi (K chan_g32 + V row_g32).
Reports per-format metric, delta vs raw, paired bootstrap 95% CI (2000 resamples).  R4 uses the best 8-bit / 4-bit ppl delta.
Sanity: the raw-injected path must reproduce the direct forward (asserted in --smoke, logged always).
"""
from __future__ import annotations
import argparse, json, math, re, string, time, collections
from pathlib import Path
import torch
import sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "feasibility" / "scripts"))
from common import Result, MANIFESTS, MODEL_ID, SEED, FEAS
RESULTS = _ROOT / "results"
from kvpick.ref.quant_ref import QuantSpec
from kvpick.ref.quant_wire import roundtrip_wire
from kvpick import ext
USE_KERNEL = {"enabled": True, "checked": False}

FORMATS = {
    "raw": None,
    "int8_row": {"k": QuantSpec(8, "row", None), "v": QuantSpec(8, "row", None)},
    "int4_row": {"k": QuantSpec(4, "row", None), "v": QuantSpec(4, "row", None)},
    "int4_row_g32": {"k": QuantSpec(4, "row", 32), "v": QuantSpec(4, "row", 32)},
    "int4_kivi": {"k": QuantSpec(4, "chan", 32), "v": QuantSpec(4, "row", 32)},
}


# ----------------------------------------------------------------------------- cache plumbing
def cache_to_tensors(cache):
    ks = torch.stack([lay.keys for lay in cache.layers])      # [L, B, H, T, D]
    vs = torch.stack([lay.values for lay in cache.layers])
    return ks, vs


def tensors_to_cache(ks, vs):
    from transformers import DynamicCache
    c = DynamicCache()
    for l in range(ks.shape[0]):
        c.update(ks[l].contiguous(), vs[l].contiguous(), l)
    return c


def apply_format(ks, vs, fmt, starts):
    """Round-trip the real (non-padded) tokens of every batch element through the format. starts[b] = first real token."""
    if fmt is None:
        return ks, vs
    ks, vs = ks.clone(), vs.clone()
    for b in range(ks.shape[1]):
        s = int(starts[b])
        if USE_KERNEL["enabled"] and ks.is_cuda and fmt["k"] == QuantSpec(4, "chan", 32) and fmt["v"] == QuantSpec(4, "row", 32):
            xk = ks[:, b, :, s:, :].permute(0, 2, 1, 3).contiguous(); xv = vs[:, b, :, s:, :].permute(0, 2, 1, 3).contiguous()
            (kp, ksc, kz), (vp, vsc, vz) = ext.quantize_kv4(xk, xv, version=2)
            yk = ext.dequantize_k4(kp, ksc, kz, xk.shape[1], version=2); yv = ext.dequantize_v4(vp, vsc, vz, version=2)
            if not USE_KERNEL["checked"]:                       # once: kernel path == Python reference path, bit for bit
                assert torch.equal(yk, roundtrip_wire(xk, fmt["k"])) and torch.equal(yv, roundtrip_wire(xv, fmt["v"])), "kernel != reference"
                USE_KERNEL["checked"] = True
            ks[:, b, :, s:, :] = yk.permute(0, 2, 1, 3); vs[:, b, :, s:, :] = yv.permute(0, 2, 1, 3)
            continue
        for which, spec in (("k", fmt["k"]), ("v", fmt["v"])):
            src = ks if which == "k" else vs
            x = src[:, b, :, s:, :].permute(0, 2, 1, 3)                 # [L, T, H, D]
            y = roundtrip_wire(x.contiguous(), spec, out_dtype=src.dtype)
            src[:, b, :, s:, :] = y.permute(0, 2, 1, 3)
    return ks, vs


@torch.no_grad()
def prefill(model, input_ids, attention_mask=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1)
    ks, vs = cache_to_tensors(out.past_key_values)
    return out.logits[:, -1], ks, vs


# ----------------------------------------------------------------------------- tasks
@torch.no_grad()
def task_ppl(model, tok, windows_ids, fmts, batch, prefix_len=1536, log=print):
    """Returns {fmt: [mean_nll_per_window]}"""
    res = {f: [] for f in fmts}
    for i in range(0, len(windows_ids), batch):
        ids = torch.tensor(windows_ids[i:i + batch], device=model.device)
        pre, cont = ids[:, :prefix_len], ids[:, prefix_len:]
        last_logit, ks, vs = prefill(model, pre)
        for f in fmts:
            k2, v2 = apply_format(ks, vs, FORMATS[f], [0] * ids.shape[0])
            cache = tensors_to_cache(k2, v2)
            out = model(input_ids=cont, past_key_values=cache, use_cache=True)
            logits = torch.cat([last_logit.unsqueeze(1), out.logits[:, :-1]], dim=1).float()   # predicts cont[0..]
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none")  # [B, 512]
            res[f].extend(nll.mean(1).tolist())
        log(f"  ppl batch {i // batch + 1}/{math.ceil(len(windows_ids) / batch)} done")
    return res


@torch.no_grad()
def generate_from_cache(model, tok, prompts, fmt, max_new_tokens, batch):
    """Left-padded batched prefill of prompt[:-1], apply format, generate greedily. Returns decoded continuations."""
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids, mask = enc.input_ids, enc.attention_mask
        starts = (mask.cumsum(1) == 1).float().argmax(1)                      # first real token per row
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        out = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1], use_cache=True, logits_to_keep=1)
        ks, vs = cache_to_tensors(out.past_key_values)
        k2, v2 = apply_format(ks, vs, FORMATS[fmt], starts.tolist())
        cache = tensors_to_cache(k2, v2)
        gen = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs.extend(tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True))
    return outs


def gsm_extract(text):
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def gsm_prompts(tok, questions, shots):
    sys_msg = "Solve the math problem. Think step by step briefly, then give the final answer on the last line as '#### <number>'."
    fs = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in shots)
    out = []
    for q in questions:
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"{fs}\n\nQuestion: {q}\nAnswer:"}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return out


QASPER_TMPL = ("You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. "
               "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
               "Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
               "If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
               "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:")


def normalize_answer(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def qa_f1(pred, golds):
    best = 0.0
    for g in golds:
        p, gt = normalize_answer(pred).split(), normalize_answer(g).split()
        common = collections.Counter(p) & collections.Counter(gt); ns = sum(common.values())
        if ns == 0:
            continue
        pr, rc = ns / len(p), ns / len(gt); best = max(best, 2 * pr * rc / (pr + rc))
    return best


def bootstrap_delta(a, b, n=2000, seed=0, fn=None):
    """Paired bootstrap CI for mean(fn(b) - fn(a)) (fn=identity) over examples."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    stat = (lambda x, y: float(np.mean(y) - np.mean(x))) if fn is None else fn
    point = stat(a, b); idx = np.arange(len(a)); vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True); vals.append(stat(a[s], b[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5]); return {"point": point, "ci95": [float(lo), float(hi)], "n": int(len(a))}


def ppl_rel_change(nll_raw, nll_fmt):
    import numpy as np
    return float(math.exp(np.mean(nll_fmt) - np.mean(nll_raw)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks", default="ppl,gsm8k,qasper")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--n-ppl", type=int, default=50); ap.add_argument("--n-gsm", type=int, default=100); ap.add_argument("--n-qasper", type=int, default=50)
    ap.add_argument("--batch-ppl", type=int, default=4); ap.add_argument("--batch-gen", type=int, default=10); ap.add_argument("--batch-long", type=int, default=2)
    ap.add_argument("--max-new-gsm", type=int, default=256); ap.add_argument("--max-new-qasper", type=int, default=32)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--no-kernel", action="store_true")
    a = ap.parse_args()
    USE_KERNEL["enabled"] = not a.no_kernel
    if a.smoke:
        a.n_ppl, a.n_gsm, a.n_qasper, a.max_new_gsm, a.max_new_qasper = 2, 3, 1, 48, 12
        a.batch_ppl, a.batch_gen, a.batch_long = 2, 3, 1
    r = Result("quality")
    import common as _common; _common.RESULTS = RESULTS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    fmts = a.formats.split(","); assert fmts[0] == "raw"
    tasks = a.tasks.split(","); M = {"model": a.model, "device": a.device, "dtype": str(dtype), "formats": fmts, "tasks": {}}
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    # sanity: raw injection == direct forward
    ids = torch.randint(100, 5000, (1, 64), device=a.device)
    direct = model(input_ids=ids).logits[:, -1].float()
    last, ks, vs = prefill(model, ids[:, :-1])
    via = model(input_ids=ids[:, -1:], past_key_values=tensors_to_cache(ks, vs)).logits[:, -1].float()
    M["raw_injection_max_abs_logit_diff"] = float((direct - via).abs().max())
    log(f"raw injection sanity: max |logit diff| = {M['raw_injection_max_abs_logit_diff']:.2e}")
    if a.smoke:
        assert M["raw_injection_max_abs_logit_diff"] < 1e-2

    if "ppl" in tasks:
        wt = json.load(open(MANIFESTS / "wikitext2_windows.json"))
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        all_ids = tok("\n".join(ds["text"]), add_special_tokens=False)["input_ids"]; W = wt["window_tokens"]
        windows = [all_ids[w * W:(w + 1) * W] for w in wt["test_windows"][:a.n_ppl]]
        t0 = time.time(); nll = task_ppl(model, tok, windows, fmts, a.batch_ppl, log=log)
        import numpy as np
        T = {"n": len(windows), "seconds": time.time() - t0, "per_format": {}}
        for f in fmts:
            ent = {"ppl": float(math.exp(np.mean(nll[f])))}
            if f != "raw":
                ent["rel_change"] = ppl_rel_change(nll["raw"], nll[f])
                ent["rel_change_ci95"] = bootstrap_delta(nll["raw"], nll[f], fn=lambda x, y: ppl_rel_change(x, y))
            T["per_format"][f] = ent
        M["tasks"]["ppl"] = T; log(f"ppl: " + ", ".join(f"{f} {T['per_format'][f]['ppl']:.3f}" for f in fmts))

    if "gsm8k" in tasks:
        man = json.load(open(MANIFESTS / "gsm8k_test.json"))
        gsm = load_dataset("openai/gsm8k", "main", split="test"); train = load_dataset("openai/gsm8k", "main", split="train")
        shots = [(train[i]["question"], train[i]["answer"]) for i in (0, 1, 2, 3)]
        idx = man["indices"][:a.n_gsm]
        prompts = gsm_prompts(tok, [gsm[i]["question"] for i in idx], shots)
        golds = [float(gsm[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in idx]
        t0 = time.time(); T = {"n": len(idx), "per_format": {}, "shots": 4, "prompt_tokens_mean": float(np.mean([len(tok(p)["input_ids"]) for p in prompts])) if "np" in dir() else None}
        correct = {}
        for f in fmts:
            outs = generate_from_cache(model, tok, prompts, f, a.max_new_gsm, a.batch_gen)
            correct[f] = [float(gsm_extract(o) == g) for o, g in zip(outs, golds)]
            T["per_format"][f] = {"accuracy": sum(correct[f]) / len(idx), "sample_output": outs[0][-200:],
                                  "frac_without_final_marker": sum("####" not in o for o in outs) / len(outs)}
            if f != "raw":
                T["per_format"][f]["delta_ci95"] = bootstrap_delta(correct["raw"], correct[f])
            log(f"gsm8k {f}: acc {T['per_format'][f]['accuracy']:.3f}")
        T["seconds"] = time.time() - t0; M["tasks"]["gsm8k"] = T

    if "qasper" in tasks:
        man = json.load(open(MANIFESTS / "longbench.json"))
        rows = [json.loads(l) for l in open(FEAS / "external/longbench_data/data/qasper.jsonl")]
        idx = man["qasper_test"]["indices"][:a.n_qasper]
        prompts = [QASPER_TMPL.format(context=rows[i]["context"], input=rows[i]["input"]) for i in idx]
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        golds = [rows[i]["answers"] for i in idx]
        t0 = time.time(); T = {"n": len(idx), "per_format": {}}; f1s = {}
        for f in fmts:
            outs = generate_from_cache(model, tok, prompts, f, a.max_new_qasper, a.batch_long)
            f1s[f] = [qa_f1(o.strip().split("\n")[0], g) for o, g in zip(outs, golds)]
            T["per_format"][f] = {"f1": sum(f1s[f]) / len(idx), "sample_output": outs[0][:100]}
            if f != "raw":
                T["per_format"][f]["delta_ci95"] = bootstrap_delta(f1s["raw"], f1s[f])
            log(f"qasper {f}: F1 {T['per_format'][f]['f1']:.3f}")
        T["seconds"] = time.time() - t0; M["tasks"]["qasper"] = T

    # R4 verdict from ppl
    if "ppl" in M["tasks"]:
        pf = M["tasks"]["ppl"]["per_format"]
        best8 = min((pf[f]["rel_change"] for f in fmts if f.startswith("int8")), default=None)
        best4 = min((pf[f]["rel_change"] for f in fmts if f.startswith("int4")), default=None)
        M["R4"] = {"best_8bit_ppl_rel_change": best8, "best_4bit_ppl_rel_change": best4, "thresholds": {"8bit": 0.02, "4bit": 0.05},
                   "R4_pass": (best8 is None or best8 <= 0.02) and (best4 is None or best4 <= 0.05)}
    r.d["measurements"] = M
    status = "INFO" if (a.smoke or a.device != "cuda" or a.model != MODEL_ID) else ("PASS" if M.get("R4", {}).get("R4_pass") else "FAIL")
    M["kernel_in_loop"] = bool(USE_KERNEL["checked"])
    tag = (torch.cuda.get_device_name(0).lower().replace("nvidia ", "").replace(" ", "_") if a.device == "cuda" else a.device)
    r.d["measurements"] = M
    p = r.save(status, name=f"quality_{tag}" + ("_smoke" if a.smoke else ""))
    print(json.dumps({k: v for k, v in M.items() if k != "tasks"}, indent=1))
    for t, T in M["tasks"].items():
        print(t, json.dumps({f: {k: v for k, v in e.items() if k != "sample_output"} for f, e in T["per_format"].items()}, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
