"""Rule: run the baseline (no compression) twice on a small slice before anything else; if the runs differ, fix the cause first.
Also checks that the restored-raw path equals a plain forward (first greedy token). Output results/ext/determinism_<model>.json"""
from __future__ import annotations
import argparse, json, pathlib, shutil, sys, tempfile
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.study import runlog, tasks as T, longctx as LC
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--gsm", type=int, default=32); ap.add_argument("--niah-len", type=int, default=8192); a = ap.parse_args()
    dev = "cpu" if a.tiny else "cuda"; info = runlog.start(extra={"model": a.model}); model, tok, cfg = LC.load_model(a.model, dev, tiny=a.tiny); res = {}
    hay = torch.randint(1000, 30000, (300000,), dtype=torch.int32) if a.tiny else T.haystack_tokens(tok, ROOT / "data" / f"haystack_{a.model}.pt")
    for task in ("gsm8k", "niah"):
        outs = []
        for run in (1, 2):
            d = pathlib.Path(tempfile.mkdtemp()); sink = T.Sink(d, task, a.model)
            if task == "gsm8k": T.run_gsm8k(model, tok, ["raw"], sink, dev, batch=16, limit=a.gsm, log=lambda *_: None)
            else: T.run_niah(model, tok, ["raw"], sink, [a.niah_len], [0, 50, 100], 1, hay, dev, log=lambda *_: None)
            outs.append({json.loads(l)["id"]: json.loads(l) for l in open(sink.dir / "raw.jsonl")}); shutil.rmtree(d)
        same = all(outs[0][i].get("output") == outs[1][i].get("output") and outs[0][i].get("score") == outs[1][i].get("score") for i in outs[0])
        res[task] = {"n": len(outs[0]), "identical": bool(same), "n_different": int(sum(outs[0][i].get("output") != outs[1][i].get("output") for i in outs[0]))}
        print(task, res[task], flush=True)
    ids = torch.randint(1000, 30000, (1, 700), device=dev)
    with torch.no_grad(): direct = int(model(input_ids=ids).logits[0, -1].argmax())
    _, cache = LC.chunked_prefill(model, ids[:, :-1], chunk=256); via = LC.greedy(model, LC.restored_cache(LC.cache_layers(cache), "raw", dev), int(ids[0, -1]), 1, set(), dev)[0]
    res["raw_injection_first_token_equal"] = bool(direct == via)
    res["PASS"] = bool(all(v["identical"] for k, v in res.items() if isinstance(v, dict)) and res["raw_injection_first_token_equal"])
    (ROOT / f"results/ext/determinism_{a.model}{'_tiny' if a.tiny else ''}.json").write_text(json.dumps({"info": runlog.finish(info), "result": res}, indent=1)); print(json.dumps(res))
    sys.exit(0 if res["PASS"] else 3)


if __name__ == "__main__":
    main()
