"""Print the README tables from results/*.json so every number in the README is script-produced. Usage: python scripts/make_tables.py > docs/tables.md"""
from __future__ import annotations
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results"; FE = R / "feasibility"
def load(p): return json.load(open(p))
out = []
# ---- kernels
k = load(R / "kernels_a40.json")
out.append(f"### Kernel benchmark ({k['gpu']}, torch {k['torch']}, CUDA {k['cuda']}; 36-layer cache, K and V; median of 100 runs, L2 flushed)\n")
out.append("| T | copy ceiling | metric | eager PyTorch | torch.compile (max-autotune) | v0 naive | v1 tiled | v2 fused |"); out.append("|---|---|---|---|---|---|---|---|")
for Tk, row in k["by_T"].items():
    c = row["copy_ceiling"]
    for d, lab in (("quant", "compress"), ("dequant", "decompress")):
        cells = [f"{row[d][x]['median_us']:.0f} us ({row[d][x]['pct_of_copy_ceiling']:.0f}%)" for x in ("B0_eager", "B1_compile", "v0", "v1", "v2")]
        out.append(f"| {Tk[1:]} | {c['GBps']:.0f} GB/s, {c['median_us']:.0f} us | {lab} | " + " | ".join(cells) + " |")
t8 = k["by_T"]["T8192"]
out.append(f"\nAt T=8192: compress v2 is {t8['quant']['B1_compile']['median_us']/t8['quant']['v2']['median_us']:.1f}x faster than torch.compile and {t8['quant']['B0_eager']['median_us']/t8['quant']['v2']['median_us']:.0f}x faster than eager; compression ratio {t8['compression_ratio_KV']:.2f}x; v2 bit-exact vs reference: K={t8['v2_bitexact_K']}, V={t8['v2_bitexact_V']}.\n")
# ---- quality
qp = R / "quality_a40.json"
if qp.exists():
    q = load(qp)["measurements"]; T = q["tasks"]
    out.append(f"### Quality vs RAW ({q['model']}, bf16, {q['device']}; kernel in the loop for KV4: {q.get('kernel_in_loop')}; 95% paired bootstrap CIs)\n")
    out.append(f"| format | ratio | WikiText-2 ppl (n={T['ppl']['n']}) | ppl change | GSM8K acc (n={T['gsm8k']['n']}) | change (pts) | Qasper F1 (n={T['qasper']['n']}) | change (pts) | quality class |"); out.append("|---|---|---|---|---|---|---|---|---|")
    ratio = {"raw": "1.0x", "int8_row": "1.94x", "int4_row": "3.76x", "int4_row_g32": "3.2x", "int4_kivi": "3.2x"}
    cls = load(R / "cost_tables_a40.json")["quality"]
    for f in q["formats"]:
        pp, gs, qa = T["ppl"]["per_format"][f], T["gsm8k"]["per_format"][f], T["qasper"]["per_format"][f]
        ci = lambda e, key, sc: (f"{e[key]['point']*sc:+.2f} [{e[key]['ci95'][0]*sc:+.2f}, {e[key]['ci95'][1]*sc:+.2f}]" if key in e else "")
        out.append(f"| {f} | {ratio.get(f,'')} | {pp['ppl']:.3f} | {ci(pp,'rel_change_ci95',100)}{'%' if f!='raw' else ''} | {gs['accuracy']:.2f} | {ci(gs,'delta_ci95',100)} | {qa['f1']:.3f} | {ci(qa,'delta_ci95',100)} | {cls.get(f,{}).get('class','')} |")
    out.append("")
# ---- transfer validation
tv = load(R / "transfer_validate_a40.json"); pts = tv["points"]; errs = [abs(p["err_pct_serial_final_model"]) for p in pts]
out.append(f"### Transfer model vs two-GPU measurement (2x {tv['gpu']}, {tv['layers']} layers, rtt {tv['rtt_ms']} ms, pinned-staged copies; {len(pts)} points, max |err| {max(errs):.1f}%, mean {sum(errs)/len(errs):.2f}%)\n")
out.append("| n new tokens | format | link Gbps | measured serial ms | predicted ms | error | pipelined (36 chunks) ms |"); out.append("|---|---|---|---|---|---|---|")
for p in pts:
    out.append(f"| {p['n_new']} | {p['format']} | {p['gbps']:.0f} | {p['serial_ms_median']:.1f} | {p['predicted_serial_ms_final_model']:.1f} | {p['err_pct_serial_final_model']:+.1f}% | {p['pipelined_ms_median']:.1f} |")
out.append("")
if "crossover" in tv:
    out.append("### Recompute vs KV4 send, measured end to end on two GPUs\n"); out.append("| n new tokens | real recompute on GPU 1 (ms) | model (ms) | measured crossover (Gbps) | predicted (Gbps) |"); out.append("|---|---|---|---|---|")
    for x in tv["crossover"]:
        out.append(f"| {x['n_new']} | {x['measured_recompute_ms']:.0f} | {x['model_recompute_ms']:.0f} | {x['measured_crossover_gbps_grid']} | {x['predicted_crossover_gbps']} |")
    out.append("")
# ---- decision map crossovers (normal budget, idle decode)
dm = load(R / "decision_map.json")
out.append("### Planner crossovers (cost model; decode GPU idle)\n"); out.append("| budget | cached tokens | new tokens | fastest action by link speed |"); out.append("|---|---|---|---|")
for row in dm["crossovers"]:
    if row["decode_batch"] == 0 and row["budget"] in ("normal", "lenient"):
        out.append(f"| {row['budget']} | {row['m_cached']} | {row['n_new']} | " + "; ".join(f">= {g} Gbps: {a}" for g, a in row["action_from_gbps"]) + " |")
out.append("")
pc = load(R / "policy_compare.json")
out.append(f"### Planner vs fixed policies (synthetic multi-round workload, {pc['n_requests']} requests, cost model)\n"); out.append("| link Gbps | planner mean ms | best fixed policy | its mean ms | planner gain (mean) | planner action mix |"); out.append("|---|---|---|---|---|---|")
for g in pc["links_gbps"]:
    e = pc["improvement_vs_best_fixed_pct"][str(g)]
    out.append(f"| {g} | {pc['policies']['planner'][str(g)]['mean_ms']:.1f} | {e['best_fixed']} | {pc['policies'][e['best_fixed']][str(g)]['mean_ms']:.1f} | {e['mean']:+.1f}% | " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(e['planner_action_mix'].items())) + " |")
out.append("")
# ---- feasibility-era facts used in the README
g1 = load(FE / "G1_kv_stats.json")["measurements"]; g2 = load(FE / "G2_axes.json")["measurements"]; g4b = load(FE / "G4b_compile_modes.json")["measurements"]
out.append("### Facts from the feasibility study (Qwen3-4B, A40)\n")
out.append(f"- Key outlier channels: top-1% channel amax / median = {g1['k_ratio_max']:.1f}x at layer {g1['k_ratio_max_layer']}; >= 5x in {sum(x>=5 for x in g1['k_top1pct_over_median_per_layer'])}/36 layers ({g1['n_prompts']} prompts).")
r3 = g2["R3"]; out.append(f"- 4-bit key attention-output error: per-row {r3['attn_err_4b_key_row']:.3f}, per-row g32 {r3['attn_err_4b_key_row_g32']:.3f}, per-channel g32 {r3['attn_err_4b_key_chan_g32']:.3f} ({r3['improvement_vs_row_pct']:.0f}% lower than per-row).")
out.append(f"- Total paid GPU time: feasibility ~50 min 1x A40; kernel debugging ~55 min 1x RTX A5000; results ~45 min 2x A40.")
(ROOT / "docs").mkdir(exist_ok=True); (ROOT / "docs/tables.md").write_text("\n".join(out)); print("\n".join(out)[:3000]); print("...\nwrote docs/tables.md")
