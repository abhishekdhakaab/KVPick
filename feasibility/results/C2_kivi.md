# C2.2 KIVI (github.com/jy-yuan/KIVI) — read-only review, 2026-10-16

Clone: `feasibility/external/kivi` (shallow). License: MIT. Paper: arXiv 2402.02750 (ICML 2024).

## Scheme (from `quant/new_pack.py` and `models/llama_kivi.py`)
- Tensor layout inside KIVI: `[B, nh, T, D]`.
- **Keys: per-channel.** `quant_and_pack_kcache(k, group_size, bits)` reshapes T into groups of `group_size` tokens and takes `min`/`max` over the **token axis inside each group**, separately for every (head, channel d). So one (scale, zero) pair per (head, d, token-group). `scale = (max - min) / (2^bits - 1)`, `q = clamp(round((x - min) / scale), 0, 2^bits - 1)`. Asymmetric, zero point = min.
- **Values: per-token.** `quant_and_pack_vcache(v, group_size, bits)` groups along the **head_dim axis** (`D // group_size` groups per token-head row) and takes min/max within each group. One (scale, zero) pair per (token, head, dim-group).
- Defaults in `example.py`: `k_bits = v_bits = 2`, `group_size = 32`, `residual_length = 32`.
- Scales and zero points are stored in the activation dtype (fp16).
- Packing: `pack_tensor` packs `32 // bits` values into one int32 along the quantized axis (keys packed along T, values along D), little-endian within the word. A Triton kernel `_pack_along_last_dim` does the value packing; key packing is plain PyTorch.
- Dequant: `data.to(fp16) * scale + mn`.

## Residual window
- The most recent `residual_length` tokens of K and V are kept in **full precision** (`key_states_full`, `value_states_full`). Once the full buffer reaches `residual_length` tokens, that block is quantized as a whole group and appended to the quantized store. This is why `residual_length % group_size == 0` is asserted: per-channel key groups need complete token groups.
- During prefill, KIVI quantizes `T - (T % residual_length)` tokens and leaves the tail full precision (`llama_kivi.py:193-217`).

## What this means for KVPick
- For a one-shot transfer of a finished prefill, the residual window is a design choice: the tail of `T mod 32` tokens can either be sent raw (KIVI behaviour) or quantized as a partial group. The reference supports partial groups; the byte accounting counts a raw tail if `--kivi-tail` is set. Default: quantize partial groups (simpler wire format); recorded here.
- Per-channel keys with token groups of 32 require **a transpose of the access pattern** relative to per-row quantization: the reduction runs down 32 tokens for a fixed (h, d) rather than across 128 dims of one row. Memory-wise, 32 consecutive tokens of one head are 32 rows of 256 B with stride `H*256` B, so a warp-per-(head, token-group) tile of 32x128 bf16 = 8 KB fits easily in registers/shared memory. This is the kernel design G2 is deciding between.
- KIVI's CUDA code (`quant/csrc/gemv_cuda.cu`) is a fused **dequant+GEMV for decode attention**, not a standalone pack kernel. Not reusable as a quantize baseline.
