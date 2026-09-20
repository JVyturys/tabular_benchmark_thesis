"""
src.production.from_final_panel.models.rerun_ftt

Re-run the ceiling-bound trial at a raised ceiling, preserving its seed and id.
"""
import json, torch, numpy as np, config as con
from datetime import datetime
from stageguard import Gatekeeper
from src.production.from_final_panel.models.run_ftt import build_model, train_and_curve

TRIAL_ID = 12
NEW_MAX_EPOCHS = 800
SOURCE_LOG = "search_log_20260910_160947.jsonl"
filepath = con.FTT_VAL_TRIALS / SOURCE_LOG

target_params = None

with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  
            trial_data = json.loads(line)
        
            if trial_data["trial"] == TRIAL_ID:
                target_params = trial_data["params"]
                old_curve = trial_data["curve"]
                break

if target_params is None:
    raise ValueError(f"Trial {TRIAL_ID} could not be found in {SOURCE_LOG}.")

gk = Gatekeeper(model="nICL")
X_fit, y_fit = gk.stage_one_data()
X_val, y_val = gk.stage_two_data()

torch.manual_seed(con.SEED+TRIAL_ID)
torch.cuda.manual_seed(con.SEED+TRIAL_ID)

trial_start = datetime.now()
trial_timestamp = trial_start.strftime("%Y-%m-%d %H:%M:%S")

model = build_model(X_fit=X_fit, params=target_params)
score, best_epoch, curve = train_and_curve(model, 
                                        X_fit=X_fit, y_fit=y_fit,
                                        X_val=X_val, y_val=y_val,
                                        max_epochs=NEW_MAX_EPOCHS, params=target_params)
trial_duration = (datetime.now() - trial_start).total_seconds()/60 

# checking previous' run per-epoch overlap
overlap_len = len(old_curve)
new_curve_overlap = curve[:overlap_len]
abs_diffs = np.abs(np.array(old_curve) - np.array(new_curve_overlap))
max_abs_diff = np.max(abs_diffs)

print(f"Reproducibility Check (Trial {TRIAL_ID}):")
print(f" -> Max absolute difference over first {overlap_len} epochs: {max_abs_diff}")

# log rereun
def _numpy_converter(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

trial_results = {
    "trial": TRIAL_ID, 
    "score": score, 
    "MAX_EPOCHS": NEW_MAX_EPOCHS,  
    "best_epoch": best_epoch, 
    "truncated curve flag": len(curve) == NEW_MAX_EPOCHS, 
    "curve": curve, 
    "params": target_params,
    "trial_duration": f"{round(trial_duration, 2)}min",
    "trial_timestamp": trial_timestamp,
} 

# generate a distinct filename 
new_filename = f"rerun_trial_{TRIAL_ID}_ceiling_{NEW_MAX_EPOCHS}_{trial_start.strftime('%Y%m%d_%H%M%S')}.jsonl"

with open(con.FTT_VAL_TRIALS / new_filename, "w", encoding="utf-8") as f:
    f.write(json.dumps(trial_results, default=_numpy_converter) + "\n")
    
print(f" [>] rerun completed and saved to {new_filename}")