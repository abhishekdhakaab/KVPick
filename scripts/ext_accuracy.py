"""Part C runner. One task per call; resumable (per-sample JSONL under results/ext/acc/<model>/<task>/<fmt>.jsonl).
  python scripts/ext_accuracy.py --model llama-3.1-8b --task niah --lengths 8192,32768,131072 --formats raw,int8_row,kv4 [--limit 20]
Tasks: niah, pg19 (--prefix 8192|32768|131072), wikitext, gsm8k, ruler (--ruler-jsonl).  --limit N = pilot (N samples), records seconds/sample.
--tiny: random tiny model + Qwen tokenizer, CPU, for plumbing tests only."""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import runlog, tasks as T, longctx as LC

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--task", required=True, choices=["niah", "pg19", "wikitext", "gsm8k", "ruler"])
    ap.add_argument("--formats", default="raw,int8_row,kv4"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lengths", default="8192,32768,131072"); ap.add_argument("--depths", default="0,10,20,30,40,50,60,70,80,90,100"); ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--prefix", type=int, default=8192); ap.add_argument("--books", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--ruler-jsonl", default=None); ap.add_argument("--ruler-ctx", type=int, default=32768)
    ap.add_argument("--tiny", action="store_true"); ap.add_argument("--outroot", default=str(ROOT / "results/ext/acc")); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cpu" if a.tiny else "cuda"; fmts = a.formats.split(","); assert fmts[0] == "raw", "baseline first (paired)"
    info = runlog.start(extra={"task": a.task, "model": a.model, "formats": fmts, "limit": a.limit})
    model, tok, cfg = LC.load_model(a.model, dev, tiny=a.tiny); info["config_check"] = cfg
    task_name = a.task + (f"_{a.prefix}" if a.task == "pg19" else "") + a.tag
    sink = T.Sink(pathlib.Path(a.outroot + ("_tiny" if a.tiny else "")), task_name, a.model); t0 = time.time()
    if a.task == "niah":
        hay = torch.randint(1000, 30000, (300000,), dtype=torch.int32) if a.tiny else T.haystack_tokens(tok, ROOT / "data" / f"haystack_{a.model}.pt", n_books=12)
        T.run_niah(model, tok, fmts, sink, [int(x) for x in a.lengths.split(",")], [int(x) for x in a.depths.split(",")], a.trials, hay, dev, limit=a.limit)
    elif a.task == "pg19": T.run_pg19(model, tok, fmts, sink, a.prefix, a.books, dev, limit=a.limit)
    elif a.task == "wikitext": T.run_wikitext(model, tok, fmts, sink, dev, limit=a.limit)
    elif a.task == "gsm8k": T.run_gsm8k(model, tok, fmts, sink, dev, batch=a.batch, limit=a.limit)
    elif a.task == "ruler": T.run_ruler(model, tok, fmts, sink, a.ruler_jsonl, a.ruler_ctx, dev, limit=a.limit)
    info["task_seconds"] = time.time() - t0
    n = max((len(v) for v in sink.done.values()), default=0)
    pilot = {"task": task_name, "model": a.model, "formats": fmts, "samples_total_in_sink": n, "limit": a.limit, "seconds": info["task_seconds"],
             "seconds_per_sample": info["task_seconds"] / max(1, a.limit or n)}
    runlog.finish(info); (sink.dir / f"run_{int(time.time())}.json").write_text(json.dumps({"info": info, "pilot": pilot}, indent=1)); print(json.dumps(pilot))


if __name__ == "__main__":
    main()
