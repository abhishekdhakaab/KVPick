"""Re-score results/transfer_validate_a40.json against the FINAL cost model (our A40 kernel times) and store the fields
predicted_serial_ms_final_model / err_pct_serial_final_model. The run-time prediction (torch.compile placeholders) is kept."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kvpick.cost.model import CostModel
R = pathlib.Path(__file__).resolve().parents[1] / "results"; p = R / "transfer_validate_a40.json"
tv = json.load(open(p)); cm = CostModel(); errs = []
for pt in tv["points"]:
    pred = cm.t_send_ms(pt["format"], pt["n_new"], pt["gbps"], tv["rtt_ms"]); pt["predicted_serial_ms_final_model"] = pred
    pt["err_pct_serial_final_model"] = 100 * (pt["serial_ms_median"] - pred) / pred; errs.append(abs(pt["err_pct_serial_final_model"]))
tv["final_model_note"] = "err_pct_serial_final_model uses cost_tables_a40.json with our A40 kernel times; err_pct_serial used torch.compile placeholders at run time"
tv["final_model_summary"] = {"max_abs_err_pct": max(errs), "mean_abs_err_pct": sum(errs) / len(errs), "n_points": len(errs)}
json.dump(tv, open(p, "w"), indent=1); print("rescored:", tv["final_model_summary"])
