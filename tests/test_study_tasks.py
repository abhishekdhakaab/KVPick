import json, pathlib, pytest, torch
from kvpick.study import tasks as T


def test_sink_resume(tmp_path):
    s = T.Sink(tmp_path, "t", "m"); s.write("raw", {"id": "a", "score": 1.0}); s.write("kv4", {"id": "a", "score": 0.0})
    s2 = T.Sink(tmp_path, "t", "m"); assert s2.has("raw", "a") and not s2.has("raw", "b") and s2.pending_fmts(["raw", "kv4", "fp8"], "a") == ["fp8"]


def test_gsm_extract():
    assert T.gsm_extract("so 3+4=7\n#### 1,234") == 1234.0 and T.gsm_extract("the answer is 42.") == 42.0 and T.gsm_extract("no digits") is None


def tok_or_skip():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception:
        pytest.skip("tokenizer not available offline")


@pytest.mark.parametrize("N,depth", [(2048, 0), (2048, 50), (4096, 100)])
def test_niah_sample_exact_length_and_needle_position(N, depth):
    tok = tok_or_skip(); hay = torch.randint(1000, 30000, (50000,), dtype=torch.int32)
    ids, meta = T.niah_sample(tok, hay, N, depth, 0)
    assert len(ids) == N and meta["number"] in tok.decode(ids)
    again, meta2 = T.niah_sample(tok, hay, N, depth, 0); assert ids == again and meta == meta2            # deterministic
    other, meta3 = T.niah_sample(tok, hay, N, depth, 1); assert meta3["number"] != meta["number"] or other != ids
    needle_pos = tok.decode(ids).index(meta["number"]) / len(tok.decode(ids))
    if depth == 0: assert needle_pos < 0.1
    if depth == 100: assert needle_pos > 0.85
