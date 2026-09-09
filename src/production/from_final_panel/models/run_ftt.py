"""
src.production.from_final_panel.models.run_ftt

FT-Transformer, non-ICL anchor. The epoch count is a
capacity parameter selected by retrospective early stopping on pooled validation
RMSE and the selected count is carried to the stage-3 refit unchanged.
"""
import config as con, utils as ut, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import matplotlib.pyplot as plt
import torch, rtdl, time, yaml, subprocess
from stageguard import Gatekeeper
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import root_mean_squared_error as rmse
from datetime import datetime




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 200   # set after inspecting one val curve
BATCH_SIZE = 256   # REMINDER: not searched, explain why in write up 


def build_model(X_fit, params: dict) -> torch.nn.Module:
    """Return an FT-Transformer on DEVICE configured with `params`.
    Fixed, non-searched settings."""
    arch_params = {k: params[k] for k in ['d_token', 'n_blocks', 'ffn_d_hidden', 'attention_dropout', 'ffn_dropout', 'residual_dropout']} 
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
    best_epoch = -1

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
            best_epoch = epoch

    return best_score, best_epoch, curve

def search(X_fit, y_fit, X_val, y_val, n_iter, seed=con.SEED) -> tuple[dict, int, list]:
    """Score n_iter sampled configurations. Returns (params, epoch, trial_log).
    Log every configuration, its best score and its best epoch."""
    # TODO implement hyperparameter tuning

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