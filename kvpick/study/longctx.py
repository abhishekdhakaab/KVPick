"""Long-context plumbing shared by the prefill benchmark (B2) and the accuracy tasks (Part C).

Protocol (C0): prefill the prompt in chunks with the baseline model, take the KV cache, round-trip it through the format
under test, put it back as the cache and continue (scoring / greedy generation) with a native-precision cache for new tokens.
The baseline uses the same code path (format "raw").  Logits are computed only for the last position of every chunk.
"""
from __future__ import annotations
import time
import torch

CHUNK = 4096


def load_model(key_or_hf: str, device: str = "cuda", dtype=torch.bfloat16, tiny: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from kvpick.study.models import MODELS, verify_config
    hf = MODELS[key_or_hf]["hf"] if key_or_hf in MODELS else key_or_hf
    if tiny:                                                           # random weights, same cache shape per layer, no weight download
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=152000, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=128, max_position_embeddings=65536)
        torch.manual_seed(0); model = Qwen3ForCausalLM(cfg).to(device).eval()
        tok = AutoTokenizer.from_pretrained(MODELS["qwen3-4b"]["hf"])
        return model, tok, None
    cfg = AutoConfig.from_pretrained(hf)
    info = verify_config(key_or_hf, cfg) if key_or_hf in MODELS else None
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, info


def eos_ids(model, tok):
    g = getattr(model, "generation_config", None); e = getattr(g, "eos_token_id", None) if g else None
    ids = set([e] if isinstance(e, int) else (e or []))
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


@torch.no_grad()
def chunked_prefill(model, ids: torch.Tensor, chunk: int = CHUNK, trace: list | None = None):
    """ids [1, N] on the model device. Returns (last_logits [1, V], cache). trace gets (start_pos, n_tokens, seconds) per chunk."""
    from transformers import DynamicCache
    cache = DynamicCache(); logits = None; cuda = ids.is_cuda
    for s in range(0, ids.shape[1], chunk):
        x = ids[:, s:s + chunk]
        if trace is not None:
            if cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1]
        if trace is not None:
            if cuda: torch.cuda.synchronize()
            trace.append((s, int(x.shape[1]), time.perf_counter() - t0))
    return logits, cache


def cache_layers(cache):
    return [(l.keys, l.values) for l in cache.layers]


def to_cache(layers):
    """layers may be a list or a generator; each layer is moved into the cache and the local reference dropped immediately."""
    from transformers import DynamicCache
    c = DynamicCache()
    for i, (k, v) in enumerate(layers):
        c.update(k.contiguous(), v.contiguous(), i); del k, v
    return c


def offload(layers):
    """Move the raw cache to (pinned, if possible) host memory so only one cache lives on the GPU at a time."""
    out = []
    for k, v in layers:
        try:
            hk = torch.empty(k.shape, dtype=k.dtype, pin_memory=True); hv = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
        except Exception:
            hk = torch.empty(k.shape, dtype=k.dtype); hv = torch.empty(v.shape, dtype=v.dtype)
        hk.copy_(k); hv.copy_(v); out.append((hk, hv))
    return out


def restored_cache(layers, fmt: str, device, starts=None, use_kernel=True):
    """Round-trip the cache through `fmt` layer by layer (restoring from host if offloaded) and return a DynamicCache."""
    from kvpick.study.formats import iter_roundtrip_layers
    return to_cache(iter_roundtrip_layers(layers, fmt, device, use_kernel=use_kernel, starts=starts))


@torch.no_grad()
def greedy(model, cache, last_id: int, max_new: int, stop_ids: set, device) -> list[int]:
    """Greedy decode from a cache that holds everything BEFORE last_id. Generated tokens extend the cache natively."""
    cur = torch.tensor([[last_id]], device=device); out_ids = []
    for _ in range(max_new):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = int(out.logits[0, -1].argmax()); cache = out.past_key_values
        if nxt in stop_ids:
            break
        out_ids.append(nxt); cur = torch.tensor([[nxt]], device=device)
    return out_ids


@torch.no_grad()
def continuation_nll(model, cache, last_logits, cont: torch.Tensor) -> float:
    """Mean NLL of cont [1, S] given a cache holding the prefix; last_logits [1,V] (from the prefix prefill) predicts cont[0]."""
    out = model(input_ids=cont, past_key_values=cache, use_cache=True)
    logits = torch.cat([last_logits.unsqueeze(1), out.logits[:, :-1]], dim=1).float()
    return float(torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="mean"))
