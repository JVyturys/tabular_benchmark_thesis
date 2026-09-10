"""
src.production.from_final_panel.models.run_ftt

FT-Transformer, non-ICL anchor. The epoch count is a
capacity parameter selected by retrospective early stopping on pooled validation
RMSE and the selected count is carried to the stage-3 refit unchanged.
"""
import config as con, utils as ut, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch, rtdl, time, yaml, subprocess
import json
from stageguard import Gatekeeper
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import root_mean_squared_error as rmse
from datetime import datetime
from scipy.stats import loguniform, uniform, randint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 200   
BATCH_SIZE = 256   
SEARCH_SPACE = {
    # architecture Defaults
    'n_blocks': randint(1, 7),  # scipy's randint is exclusive at the upper bound (1 to 6)
    'd_token': [64, 96, 128, 192, 256, 320, 384, 512],
    'ffn_d_hidden_multiplier': uniform(0.66, 2.0),  # loc=0.66, scale=2.0 spans [0.66, 2.66]
    'attention_dropout': uniform(0.0, 0.5),
    'ffn_dropout': uniform(0.0, 0.5),
    'residual_dropout': uniform(0.0, 0.5),
    
    # optimization defaults
    'lr': loguniform(1e-5, 1e-3),
    'weight_decay': loguniform(1e-6, 1e-3)
}

def build_model(X_fit, params: dict) -> torch.nn.Module:
    """Return an FT-Transformer on DEVICE configured with `params`.
    Fixed, non-searched settings."""
    arch_params = {k: params[k] for k in ['d_token', 'n_blocks', 'attention_dropout', 'ffn_dropout', 'residual_dropout']}
    arch_params['ffn_d_hidden'] = int(params['d_token'] * params['ffn_d_hidden_multiplier'])
    return rtdl.FTTransformer.make_baseline(n_num_features=X_fit.shape[1],
                                        cat_cardinalities=None, d_out=1,
                                          **arch_params).to(DEVICE)

def train_and_curve(model, X_fit, y_fit, X_val, y_val, max_epochs, params) -> tuple[float, int, list]:
    """Train `model` for `max_epochs`, scoring pooled stage-2 RMSE after each epoch.
    Returns (best_score, best_epoch, curve). No weights are updated on val."""

    # slice optimization relevant parameters 
    opt_params = {k: params[k] for k in ['lr', 'weight_decay']}

    # initiate optimizer
    optimizer = optim.AdamW(model.optimization_param_groups(), **opt_params)

    # transform partitions into torch friendly tensors
    y_fit_t = y_fit.to_frame('y_fit')
    y_fit_t = torch.tensor(y_fit_t.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    X_fit_t = torch.tensor(X_fit.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)

    # define validation containers
    curve = []
    best_score = float('inf')
    best_epoch = 0

    # train network
    for epoch in range(max_epochs):
        ## train
        model.train()
        perm = torch.randperm(X_fit_t.shape[0], device=DEVICE)
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_fit_t[idx], y_fit_t[idx]
            optimizer.zero_grad()
            yb_pred = model(xb, None)
            loss = F.mse_loss(yb_pred, yb) 
            loss.backward()
            optimizer.step()

        ## evaluate    
        model.eval()
        with torch.no_grad(): # deactivate gradient tracking
            y_val_pred = model(X_val_t, None)
        y_val_pred = y_val_pred.cpu().numpy().flatten()
        y_val_pred = pd.Series(y_val_pred, index=y_val.index)
        metrics = ut.pooled_metrics(y_true=y_val, y_pred=y_val_pred)
        score = metrics[0]
        curve.append(score)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1 # +1 because range(max_epoch) is zero based 

    return best_score, best_epoch, curve

def search(X_fit, y_fit, X_val, y_val, n_iter, seed=con.SEED) -> tuple[dict, int, list]:
    """Score n_iter sampled configurations. Returns (winning_params, winning_epoch, trial_log).
    Each trial is seeded on its own index so a crashed run resumes exactly."""

    # define helper function for JSON logs that converts np.floats to python native dtypes 
    def _numpy_converter(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # intiate sampler
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=seed)

    # set up placeholders
    search_log = []
    winning_score = np.inf
    winning_params = None
    winning_epoch = None

    # create timestamp
    time_now = datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S")

    # start model selection
    for i, params in enumerate(sampler):
        # start timer 
        trial_start = datetime.now()
        trial_timestamp = trial_start.strftime("%Y-%m-%d %H:%M:%S")
        # set seed
        torch.manual_seed(con.SEED+i)
        torch.cuda.manual_seed(con.SEED+i)
        # initiate and train model
        model = build_model(X_fit=X_fit, params=params)
        score, best_epoch, curve = train_and_curve(model, 
                                                X_fit=X_fit, y_fit=y_fit,
                                                X_val=X_val, y_val=y_val,
                                                max_epochs=MAX_EPOCHS, params=params)
        trial_duration = (datetime.now() - trial_start).total_seconds()/60 
        # save results
        trial_results = {"trial": i, "score":score, "best_epoch":best_epoch, "curve":curve, "params":params,
                         "trial_duration":f"{round(trial_duration, 2)}min","trial_timestamp":trial_timestamp} 
        search_log.append(trial_results)
        with open(con.FTT_VAL_TRIALS / f"search_log_{timestamp}.jsonl", "a") as f:
            f.write(json.dumps(trial_results, default=_numpy_converter) + "\n")

        # check for rmse minimum
        if score < winning_score:
            winning_score = score
            winning_epoch = best_epoch
            winning_params = params

    return winning_params, winning_epoch, search_log

def run_ftt(condition: str = "tuned", n_iter: int = 30) -> None:
    """Hoisted stage 1+2 -> search -> fresh model trained on stage 3 for
    winning_epoch -> score stage 4 -> persist predictions, manifest, trial log."""
    # TODO train, score and log

### --- baseline run
test_gate = Gatekeeper(model='nICL')
X_fit, y_fit = test_gate.stage_one_data()
X_val, y_val = test_gate.stage_two_data()

baseline_params = {
    # architecture 
    'd_token': 192,
    'n_blocks': 3,
    'ffn_d_hidden': 256,
    'attention_dropout': 0.2,
    'ffn_dropout': 0.1,
    'residual_dropout': 0.0,
    
    # optimization
    'lr': 1e-4,
    'weight_decay': 1e-5
}
ft_baseline = build_model(X_fit=X_fit, params=baseline_params)
best_score, best_epoch, curve = train_and_curve(ft_baseline,
                                                X_fit=X_fit,
                                                y_fit=y_fit,
                                                X_val=X_val,
                                                y_val=y_val,
                                                max_epochs=MAX_EPOCHS, 
                                                params=baseline_params)

plt.style.use('seaborn-v0_8-whitegrid')

color_line = "#1f4e79"
epochs = np.arange(1, len(curve) + 1)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    epochs, 
    curve, 
    color=color_line, 
    linewidth=2.5,
    label="Validation RMSE"
)

ax.set_title("Model RMSE over Epochs", fontweight="bold", pad=15, fontsize=14)
ax.set_xlabel("Epoch", labelpad=10)
ax.set_ylabel("RMSE Score", labelpad=10)

ax.grid(True, linestyle="--", alpha=0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

ax.legend(frameon=True, fancybox=True, shadow=False, borderpad=1)

plt.tight_layout()
plt.savefig(con.VIZ_RMSE, dpi=600, bbox_inches='tight')
plt.show()