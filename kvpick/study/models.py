"""Model table for the extension study. Values are checked against each model's HF config by verify_config()."""
from __future__ import annotations

# params: parameter count used for the FLOPs model. q_heads*head_dim: attention width used for the attention-FLOPs term.
MODELS = {
    "qwen3-4b": dict(hf="Qwen/Qwen3-4B-Instruct-2507", layers=36, kv_heads=8, head_dim=128, q_heads=32, params=4.02e9, native_ctx=32768, runnable=True),
    "llama-3.1-8b": dict(hf="meta-llama/Llama-3.1-8B-Instruct", layers=32, kv_heads=8, head_dim=128, q_heads=32, params=8.03e9, native_ctx=131072, runnable=True),
    "llama-3.1-70b": dict(hf="meta-llama/Llama-3.1-70B-Instruct", layers=80, kv_heads=8, head_dim=128, q_heads=64, params=70.6e9, native_ctx=131072, runnable=False),
}
RAW_BYTES_PER_TOKEN_PER_LAYER = 2 * 8 * 128 * 2          # K+V, 8 heads, 128 dims, 2 bytes
FMT_BYTES_PER_TOKEN_PER_LAYER = {
    "raw": 4096,
    "int8_row": 2112,                                   # 2 x (1024 payload + 8 rows x (2 B scale + 2 B zero) = 32)
    "fp8": 2048,                                        # 1 byte per value; one fp32 scale per tensor per layer is negligible
    "kv4": 1280,
    "int4_g32": 1280,
    "lossless": 4096 / 1.33,                            # modeled, from the SplitZip paper's design (ratio 1.33)
}


def kv_bytes_per_token(model: str, fmt: str) -> float:
    return MODELS[model]["layers"] * FMT_BYTES_PER_TOKEN_PER_LAYER[fmt]


def flops_prefill(model: str, n_new: int, m_cached: int = 0) -> float:
    """Matrix part 2*P*n plus causal attention 4*L*(q_heads*head_dim)*n*(m + n/2)."""
    c = MODELS[model]
    return 2 * c["params"] * n_new + 4 * c["layers"] * c["q_heads"] * c["head_dim"] * n_new * (m_cached + n_new / 2)


def verify_config(model: str, cfg) -> dict:
    """cfg: HF config (object or dict). Returns the checked values; raises if the per-layer KV shape differs from the table."""
    g = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    c = MODELS[model]
    hd = g("head_dim") or g("hidden_size") // g("num_attention_heads")
    got = dict(layers=g("num_hidden_layers"), kv_heads=g("num_key_value_heads"), head_dim=hd, q_heads=g("num_attention_heads"), native_ctx_hf=g("max_position_embeddings"))
    for k in ("layers", "kv_heads", "head_dim", "q_heads"):
        if got[k] != c[k]:
            raise ValueError(f"{model}: config {k}={got[k]} but table says {c[k]}")
    got["raw_bytes_per_token"] = got["layers"] * 2 * got["kv_heads"] * got["head_dim"] * 2
    return got
