"""Accuracy tasks for the extension study. Each runner yields per-sample records and appends them to JSONL (resumable).
Record: {task, model, fmt, id, ctx, score, output?, nll?, t_prefill_s, ...}.  One prefill per sample, shared by all formats."""
from __future__ import annotations
import json, math, pathlib, random, re, time
import numpy as np
import torch
from kvpick.study import longctx as LC

ROOT = pathlib.Path(__file__).resolve().parents[2]
CITIES = ["Paris", "Lisbon", "Nairobi", "Osaka", "Lima", "Oslo", "Cairo", "Hanoi", "Quito", "Dublin", "Tallinn", "Accra", "Perth", "Kyoto", "Bogota", "Zagreb", "Tunis", "Malmo", "Riga", "Seoul",
          "Hobart", "Cusco", "Doha", "Turin", "Vilnius", "Kandy", "Porto", "Sofia", "Tbilisi", "Muscat", "Ghent", "Bergen", "Sucre", "Aarhus", "Split", "Faro", "Luang", "Medan", "Pune", "Brno"]
DATE = "26 Jul 2024"


class Sink:
    """Append-only JSONL with resume: skip (fmt, id) pairs already present."""
    def __init__(self, root: pathlib.Path, task: str, model: str):
        self.dir = root / model / task; self.dir.mkdir(parents=True, exist_ok=True); self.done = {}
        for f in self.dir.glob("*.jsonl"):
            self.done[f.stem] = {json.loads(l)["id"] for l in open(f) if l.strip()}

    def has(self, fmt, sid): return sid in self.done.get(fmt, ())

    def write(self, fmt, rec):
        with open(self.dir / f"{fmt}.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        self.done.setdefault(fmt, set()).add(rec["id"])

    def pending_fmts(self, fmts, sid): return [f for f in fmts if not self.has(f, sid)]


def chat_ids(tok, user_text_pre: str, user_text_post: str, placeholder="<<BODY>>"):
    """Render the chat template once with a placeholder; return (ids_before, ids_after) token lists around the body."""
    msgs = [{"role": "user", "content": placeholder + user_text_post}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=DATE, enable_thinking=False)
    pre, post = s.split(placeholder)
    return tok(pre, add_special_tokens=False)["input_ids"], tok(post, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------------------------------------------------- NIAH
def pg19_split(split: str, n_books: int):
    """Read ONLY the requested parquet file (datasets.load_dataset would download the 11 GB train split too)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    fn = next(f for f in HfApi().list_repo_files("emozilla/pg19", repo_type="dataset") if f.startswith(f"data/{split}-"))
    tbl = pq.read_table(hf_hub_download("emozilla/pg19", fn, repo_type="dataset"), columns=["text"])
    return [tbl["text"][i].as_py() for i in range(min(n_books, tbl.num_rows))]


def haystack_tokens(tok, cache_path: pathlib.Path, n_books=12):
    if cache_path.exists():
        return torch.load(cache_path)
    toks = []
    for t in pg19_split("validation", n_books):
        toks += tok(t, add_special_tokens=False)["input_ids"]
    arr = torch.tensor(toks, dtype=torch.int32); cache_path.parent.mkdir(parents=True, exist_ok=True); torch.save(arr, cache_path); return arr


def niah_sample(tok, hay: torch.Tensor, N: int, depth_pct: int, trial: int, seed=0):
    rng = random.Random(f"{seed}-{N}-{depth_pct}-{trial}")
    city, number = rng.choice(CITIES), str(rng.randint(1000000, 9999999))
    question = f"\n\nBased only on the text above, what is the special magic number for {city}? Answer with just the number."
    pre, post = chat_ids(tok, "", question)
    needle = tok(f"\nThe special magic number for {city} is {number}.\n", add_special_tokens=False)["input_ids"]
    Lh = N - len(pre) - len(post) - len(needle); assert Lh > 1000, "context too short for template"
    start = rng.randint(0, len(hay) - Lh - 1); body = hay[start:start + Lh].tolist()
    pos = int(round(depth_pct / 100 * Lh)); body = body[:pos] + needle + body[pos:]
    ids = pre + body + post; assert len(ids) == N
    return ids, {"city": city, "number": number}


@torch.no_grad()
def run_niah(model, tok, fmts, sink: Sink, lengths, depths, trials, hay, device, offload_min=20000, limit=None, log=print, max_new=24):
    stop = LC.eos_ids(model, tok); n_done = 0
    for N in lengths:
        for depth in depths:
            for trial in range(trials):
                sid = f"{N}_{depth}_{trial}"; todo = sink.pending_fmts(fmts, sid)
                if not todo: continue
                if limit is not None and n_done >= limit: return
                ids, meta = niah_sample(tok, hay, N, depth, trial)
                t0 = time.time(); x = torch.tensor([ids], device=device)
                _, cache = LC.chunked_prefill(model, x[:, :-1]); layers = LC.cache_layers(cache)
                if N >= offload_min: layers = LC.offload(layers); del cache
                else: layers = [(k.clone(), v.clone()) for k, v in layers]; del cache
                tp = time.time() - t0
                for f in todo:
                    t1 = time.time(); c = LC.restored_cache(layers, f, device)
                    out = LC.greedy(model, c, ids[-1], max_new, stop, device); text = tok.decode(out, skip_special_tokens=True)
                    digits = re.sub(r"[^0-9]", "", text)
                    sink.write(f, {"task": "niah", "id": sid, "ctx": N, "depth": depth, "trial": trial, "fmt": f, "score": float(meta["number"] in digits), "output": text.strip()[:80],
                                   "gold": meta["number"], "t_prefill_s": tp, "t_fmt_s": time.time() - t1})
                    del c
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                n_done += 1; log(f"niah N={N} depth={depth} trial={trial} prefill {tp:.1f}s")


# ------------------------------------------------------------------------------------------------------------- perplexity (PG-19, WikiText-2)
def tokens_for_ppl(tok, text, need, add_bos):
    ids = tok(text[: int(need * 6)], add_special_tokens=False)["input_ids"]
    if add_bos and tok.bos_token_id is not None: ids = [tok.bos_token_id] + ids
    return ids[:need] if len(ids) >= need else None


@torch.no_grad()
def run_pg19(model, tok, fmts, sink: Sink, prefix_len, n_books, device, score_len=512, offload_min=20000, limit=None, log=print):
    books = pg19_split("test", 100)[:n_books]; add_bos = "llama" in tok.name_or_path.lower(); n_done = 0
    for bi, text in enumerate(books):
        sid = f"{prefix_len}_{bi}"; todo = sink.pending_fmts(fmts, sid)
        if not todo: continue
        if limit is not None and n_done >= limit: return
        ids = tokens_for_ppl(tok, text, prefix_len + score_len, add_bos)
        if ids is None:
            for f in todo: sink.write(f, {"task": "pg19", "id": sid, "ctx": prefix_len, "fmt": f, "skipped": "book shorter than prefix+512"})
            continue
        x = torch.tensor([ids], device=device); t0 = time.time()
        last, cache = LC.chunked_prefill(model, x[:, :prefix_len]); layers = LC.cache_layers(cache)
        layers = LC.offload(layers) if prefix_len >= offload_min else [(k.clone(), v.clone()) for k, v in layers]; del cache
        tp = time.time() - t0
        for f in todo:
            c = LC.restored_cache(layers, f, device); nll = LC.continuation_nll(model, c, last, x[:, prefix_len:])
            sink.write(f, {"task": "pg19", "id": sid, "ctx": prefix_len, "fmt": f, "nll": nll, "t_prefill_s": tp}); del c
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        n_done += 1; log(f"pg19 prefix={prefix_len} book={bi} prefill {tp:.1f}s")


@torch.no_grad()
def run_wikitext(model, tok, fmts, sink: Sink, device, window=2048, prefix=1536, batch=4, limit=None, log=print):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    all_ids = tok("\n".join(ds["text"]), add_special_tokens=False)["input_ids"]
    bos = [tok.bos_token_id] if ("llama" in tok.name_or_path.lower() and tok.bos_token_id is not None) else []
    W = window - len(bos); nwin = len(all_ids) // W; wins = [bos + all_ids[i * W:(i + 1) * W] for i in range(nwin)]
    if limit is not None: wins = wins[:limit]
    for b0 in range(0, len(wins), batch):
        idx = list(range(b0, min(b0 + batch, len(wins)))); todo = {i: sink.pending_fmts(fmts, f"w{i}") for i in idx}
        if not any(todo.values()): continue
        x = torch.tensor([wins[i] for i in idx], device=device)
        last, cache = LC.chunked_prefill(model, x[:, :prefix], chunk=prefix); layers = LC.cache_layers(cache)
        for f in fmts:
            if all(f not in todo[i] for i in idx): continue
            c = LC.restored_cache(layers, f, device); cont = x[:, prefix:]
            out = model(input_ids=cont, past_key_values=c, use_cache=True)
            logits = torch.cat([last.unsqueeze(1), out.logits[:, :-1]], dim=1).float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none").mean(1).tolist(); del c
            for j, i in enumerate(idx):
                if f in todo[i]: sink.write(f, {"task": "wikitext", "id": f"w{i}", "ctx": prefix, "fmt": f, "nll": nll[j]})
        log(f"wikitext windows {idx[0]}..{idx[-1]} of {len(wins)}")


# ------------------------------------------------------------------------------------------------------------- GSM8K
def gsm_extract(text):
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"-?[\d,]*\.?\d+", text)
    try: return float(m[-1].replace(",", "")) if m else None
    except ValueError: return None


@torch.no_grad()
def run_gsm8k(model, tok, fmts, sink: Sink, device, batch=16, max_new=256, limit=None, log=print):
    from datasets import load_dataset
    from kvpick.study.formats import roundtrip_cache_layers
    test = load_dataset("openai/gsm8k", "main", split="test"); train = load_dataset("openai/gsm8k", "main", split="train")
    shots = "\n\n".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}" for i in range(4))
    sysm = "Solve the math problem. Think step by step briefly, then give the final answer on the last line as '#### <number>'."
    N = len(test) if limit is None else limit; tok.padding_side = "left"
    for b0 in range(0, N, batch):
        idx = list(range(b0, min(b0 + batch, N))); todo = {i: sink.pending_fmts(fmts, f"g{i}") for i in idx}
        if not any(todo.values()): continue
        prompts = [tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": f"{shots}\n\nQuestion: {test[i]['question']}\nAnswer:"}],
                                           tokenize=False, add_generation_prompt=True, date_string=DATE, enable_thinking=False) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device); ids, mask = enc.input_ids, enc.attention_mask
        starts = (mask.cumsum(1) == 1).float().argmax(1).tolist(); pos = (mask.cumsum(1) - 1).clamp_min(0)
        with torch.no_grad():
            out = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1], use_cache=True, logits_to_keep=1)
            layers = LC.cache_layers(out.past_key_values)
            for f in fmts:
                if all(f not in todo[i] for i in idx): continue
                c = LC.to_cache(roundtrip_cache_layers(layers, f, device, starts=starts))
                gen = model.generate(input_ids=ids, attention_mask=mask, past_key_values=c, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
                for j, i in enumerate(idx):
                    if f not in todo[i]: continue
                    gold = float(test[i]["answer"].split("####")[-1].strip().replace(",", ""))
                    sink.write(f, {"task": "gsm8k", "id": f"g{i}", "ctx": int(mask[j].sum()), "fmt": f, "score": float(gsm_extract(texts[j]) == gold), "output": texts[j][-160:]})
                del c
        log(f"gsm8k {idx[0]}..{idx[-1]} of {N}")


# ------------------------------------------------------------------------------------------------------------- RULER (pre-generated with the official generators)
@torch.no_grad()
def run_ruler(model, tok, fmts, sink: Sink, jsonl_path, ctx, device, offload_min=20000, limit=None, log=print, max_new=64):
    """jsonl from the official RULER data generator: {index, input, outputs:[...]}. Score = mean of 'string_match_all' (fraction of gold strings found)."""
    stop = LC.eos_ids(model, tok); rows = [json.loads(l) for l in open(jsonl_path)][:limit]; name = pathlib.Path(jsonl_path).stem
    for r in rows:
        sid = f"{name}_{ctx}_{r['index']}"; todo = sink.pending_fmts(fmts, sid)
        if not todo: continue
        s = tok.apply_chat_template([{"role": "user", "content": r["input"]}], tokenize=False, add_generation_prompt=True, date_string=DATE)
        ids = tok(s, add_special_tokens=False)["input_ids"]; x = torch.tensor([ids], device=device)
        _, cache = LC.chunked_prefill(model, x[:, :-1]); layers = LC.cache_layers(cache)
        layers = LC.offload(layers) if len(ids) >= offload_min else [(k.clone(), v.clone()) for k, v in layers]; del cache
        for f in todo:
            c = LC.restored_cache(layers, f, device); text = tok.decode(LC.greedy(model, c, ids[-1], max_new, stop, device), skip_special_tokens=True)
            sc = float(np.mean([g.lower() in text.lower() for g in r["outputs"]]))
            sink.write(f, {"task": f"ruler_{name}", "id": sid, "ctx": ctx, "fmt": f, "score": sc, "output": text[:80]}); del c
        log(f"ruler {name} {r['index']}")
