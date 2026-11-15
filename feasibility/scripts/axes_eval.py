"""G2: which quantization axis? Compare key/value grouping variants at 8 and 4 bits on captured (or synthetic) KV.

Inputs: --kv-dir with files from kv_capture.py ({"k": [L,T,H,D] bf16, "v": ...}) or --synthetic (Gaussian with
injected outlier key channels, for smoke).  For every (key variant, value variant, bits):
  rel L2 error of K and of V, attention-output relative error (8 queries per layer and head, fp32),
  bytes per token per layer (K+V, incl. fp16 scale/zero), compression ratio vs raw bf16 (4096 B).
Also K-only and V-only attention errors (the other tensor exact) so R3 can be judged on keys alone.
Output: results/G2_axes.json and a markdown table printed to stdout.
"""
from __future__ import annotations
import argparse, json, math, glob
import torch
from common import Result, RESULTS, SEED
from quant_ref import QuantSpec, roundtrip, kv_bytes_per_token_per_layer, RAW_BYTES_PER_TOKEN_PER_LAYER

KEY_VARIANTS = {"row": ("row", None), "row_g32": ("row", 32), "chan_g32": ("chan", 32)}
VAL_VARIANTS = {"row": ("row", None), "row_g32": ("row", 32)}


def attn_out(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q [H, Q, D], k/v [T, H, D] -> out [H, Q, D] in fp32 (no mask: queries are 'last token')."""
    kk = k.permute(1, 0, 2).float(); vv = v.permute(1, 0, 2).float()          # [H, T, D]
    logits = torch.einsum("hqd,htd->hqt", q.float(), kk) / math.sqrt(k.shape[-1])
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hqt,htd->hqd", p, vv)


def make_queries(k: torch.Tensor, n: int, gen: torch.Generator) -> dict:
    """Two query sets per layer: rows sampled from K (realistic logit scale) and Gaussian scaled to K's row RMS."""
    T, H, D = k.shape
    idx = torch.randint(0, T, (n,), generator=gen).to(k.device)
    from_k = k[idx].permute(1, 0, 2).float()                                    # [H, n, D]
    rms = k.float().pow(2).mean(dim=(0, 2), keepdim=True).sqrt().permute(1, 0, 2)  # [H,1,1]
    gauss = torch.randn(H, n, D, generator=gen).to(k.device) * rms
    return {"k_rows": from_k, "gauss": gauss}


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def evaluate(ks: torch.Tensor, vs: torch.Tensor, bits_list=(8, 4), n_q=8, symmetric=False) -> dict:
    """ks, vs: [L, T, H, D]. Returns per-variant aggregates (mean over layers)."""
    L = ks.shape[0]
    gen = torch.Generator().manual_seed(SEED)
    out = {}
    for bits in bits_list:
        kspecs = {n: QuantSpec(bits, ax, g, symmetric) for n, (ax, g) in KEY_VARIANTS.items()}
        vspecs = {n: QuantSpec(bits, ax, g, symmetric) for n, (ax, g) in VAL_VARIANTS.items()}
        acc = {}
        for l in range(L):
            k, v = ks[l].float(), vs[l].float()
            qs = make_queries(k, n_q, gen)
            ref = {name: attn_out(q, k, v) for name, q in qs.items()}
            dk = {n: roundtrip(k, s) for n, s in kspecs.items()}
            dv = {n: roundtrip(v, s) for n, s in vspecs.items()}
            for kn in kspecs:
                e = acc.setdefault(f"K:{kn}", {"rel_l2_k": [], "attn_k_rows": [], "attn_gauss": []})
                e["rel_l2_k"].append(rel(dk[kn], k))
                e["attn_k_rows"].append(rel(attn_out(qs["k_rows"], dk[kn], v), ref["k_rows"]))
                e["attn_gauss"].append(rel(attn_out(qs["gauss"], dk[kn], v), ref["gauss"]))
            for vn in vspecs:
                e = acc.setdefault(f"V:{vn}", {"rel_l2_v": [], "attn_k_rows": [], "attn_gauss": []})
                e["rel_l2_v"].append(rel(dv[vn], v))
                e["attn_k_rows"].append(rel(attn_out(qs["k_rows"], k, dv[vn]), ref["k_rows"]))
                e["attn_gauss"].append(rel(attn_out(qs["gauss"], k, dv[vn]), ref["gauss"]))
            for kn in kspecs:
                for vn in vspecs:
                    e = acc.setdefault(f"KV:{kn}+{vn}", {"attn_k_rows": [], "attn_gauss": []})
                    e["attn_k_rows"].append(rel(attn_out(qs["k_rows"], dk[kn], dv[vn]), ref["k_rows"]))
                    e["attn_gauss"].append(rel(attn_out(qs["gauss"], dk[kn], dv[vn]), ref["gauss"]))
        T = ks.shape[1]
        res = {}
        for name, e in acc.items():
            row = {m: {"mean": sum(x) / len(x), "max": max(x), "per_layer": [round(x_, 5) for x_ in x]} for m, x in e.items()}
            if name.startswith("KV:"):
                kn, vn = name[3:].split("+")
                bpt = kv_bytes_per_token_per_layer(kspecs[kn], vspecs[vn], T=T)
                row["bytes_per_token_per_layer"] = bpt
                row["compression_ratio"] = RAW_BYTES_PER_TOKEN_PER_LAYER / bpt
            res[name] = row
        out[f"{bits}bit"] = res
    return out


def r3_verdict(out: dict) -> dict:
    """R3: at 4 bits, per-channel keys give >= 20% lower attention-output error than per-row keys (K-only, k_rows queries)."""
    e_row = out["4bit"]["K:row"]["attn_k_rows"]["mean"]
    e_chan = out["4bit"]["K:chan_g32"]["attn_k_rows"]["mean"]
    e_rowg = out["4bit"]["K:row_g32"]["attn_k_rows"]["mean"]
    return {"attn_err_4b_key_row": e_row, "attn_err_4b_key_row_g32": e_rowg, "attn_err_4b_key_chan_g32": e_chan,
            "improvement_vs_row_pct": 100 * (1 - e_chan / e_row), "improvement_vs_row_g32_pct": 100 * (1 - e_chan / e_rowg),
            "R3_pass_vs_row": e_chan <= 0.8 * e_row, "threshold_pct": 20}


def table(out: dict) -> str:
    lines = ["| bits | variant | relL2 K | relL2 V | attn err (K-row q) | attn err (gauss q) | B/tok/layer | ratio |", "|---|---|---|---|---|---|---|---|"]
    for bits, res in out.items():
        for name, row in res.items():
            lines.append("| %s | %s | %s | %s | %.4f | %.4f | %s | %s |" % (
                bits, name, "%.4f" % row["rel_l2_k"]["mean"] if "rel_l2_k" in row else "",
                "%.4f" % row["rel_l2_v"]["mean"] if "rel_l2_v" in row else "",
                row["attn_k_rows"]["mean"], row["attn_gauss"]["mean"],
                "%.0f" % row["bytes_per_token_per_layer"] if "bytes_per_token_per_layer" in row else "",
                "%.2f" % row["compression_ratio"] if "compression_ratio" in row else ""))
    return "\n".join(lines)


def synthetic(L=2, T=256, H=8, D=128):
    """Gaussian KV with the KIVI-style key pathology: a few fixed channels with a large consistent offset
    (channel 3: +20, channel 77: -15, both with 2x spread), plus one zero-mean high-variance channel (50: x10)."""
    gen = torch.Generator().manual_seed(SEED)
    ks = torch.randn(L, T, H, D, generator=gen); vs = torch.randn(L, T, H, D, generator=gen)
    ks[..., 3] = 20 + 2 * ks[..., 3]; ks[..., 77] = -15 + 2 * ks[..., 77]; ks[..., 50] *= 10
    return ks.to(torch.bfloat16), vs.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dir", default=None)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--layers", default=None, help="comma list of layer positions to evaluate (default all)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    r = Result("G2")
    if a.smoke or a.synthetic or not a.kv_dir:
        ks, vs = synthetic()
        r.note("synthetic Gaussian KV with 2 injected outlier key channels (x30); numbers are NOT model measurements")
        files = ["synthetic"]
        agg = evaluate(ks, vs)
    else:
        files = sorted(glob.glob(f"{a.kv_dir}/*.pt"))[:a.max_files]
        per_file = []
        for f in files:
            d = torch.load(f)
            ks, vs = d["k"].to(a.device), d["v"].to(a.device)
            if a.layers:
                sel = [int(x) for x in a.layers.split(",")]; ks, vs = ks[sel], vs[sel]
            per_file.append(evaluate(ks, vs, symmetric=a.symmetric))
        # average the scalar means over files
        agg = {}
        for bits in per_file[0]:
            agg[bits] = {}
            for name in per_file[0][bits]:
                agg[bits][name] = {}
                for m in per_file[0][bits][name]:
                    vals = [pf[bits][name][m] for pf in per_file]
                    if isinstance(vals[0], dict):
                        agg[bits][name][m] = {"mean": sum(v["mean"] for v in vals) / len(vals), "max": max(v["max"] for v in vals)}
                    else:
                        agg[bits][name][m] = vals[0]
        r.d["measurements"]["per_file_ids"] = [f.split("/")[-1] for f in files]
    r.d["measurements"]["variants"] = agg
    r.d["measurements"]["R3"] = r3_verdict(agg)
    r.d["measurements"]["query_sets"] = {"k_rows": "8 rows sampled from this layer's K (realistic logit scale)", "gauss": "N(0, rms(K)) per head"}
    status = "INFO" if (a.smoke or a.synthetic or not a.kv_dir) else ("PASS" if r.d["measurements"]["R3"]["R3_pass_vs_row"] else "FAIL")
    p = r.save(status, name="G2_axes")
    print(table(agg)); print(json.dumps(r.d["measurements"]["R3"], indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
