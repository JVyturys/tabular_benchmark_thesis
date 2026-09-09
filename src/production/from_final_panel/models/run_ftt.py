"""
src.production.from_final_panel.models.run_ftt

FT-Transformer, non-ICL anchor. The epoch count is a
capacity parameter selected by retrospective early stopping on pooled validation
RMSE and the selected count is carried to the stage-3 refit unchanged.
"""
import config as con, utils as ut, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import torch, rtdl, time, yaml, subprocess
from stageguard import Gatekeeper
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import root_mean_squared_error as rmse
from datetime import datetime
from rtdl_revisiting_models import FTTransformer



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = None   # set after inspecting one val curve
BATCH_SIZE = None   # fixed, not searched state the reason


def build_model(X_fit, params: dict) -> torch.nn.Module:
    """Return an FT-Transformer on DEVICE configured with `params`.
    Fixed, non-searched settings."""
    arch_params = {k: params[k] for k in ['d_token', 'n_blocks', 'ffn_d_hidden', 'attention_dropout', 'ffn_dropout', 'residual_dropout']} 
    return FTTransformer.make_baseline(
    n_num_features=X_fit.shape[1], cat_cardinalities=None, d_out=1, **arch_params).to(DEVICE)

def train_and_curve(model, X_fit, y_fit, X_val, y_val, max_epochs, params) -> tuple[float, int, list]:
    """Train `model` for `max_epochs`, scoring pooled stage-2 RMSE after each epoch.
    Returns (best_score, best_epoch, curve). No weights are updated on val."""

    # slice optimization relevant parameters 
    opt_params = {k: params[k] for k in ['lr', 'weight_decay']}

    # initiate optimizer
    optimizer = optim.AdamW(model.optimization_param_groups(), **opt_params)

    # transform partitions into torch friendly tensors
    y_fit_t = y_fit.to_frame('y_fit').copy()
    y_fit_t = torch.tensor(y_fit.values, dtype=torch.float32, device=DEVICE)
    X_fit_t = torch.tensor(X_fit.values, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32, device=DEVICE)

    # define validation containers
    curve = []
    best_score = float('inf')
    best_epoch = -1

    # train network
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(X_fit.shape_t[0], device=DEVICE)
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_fit_t[idx], y_fit_t[idx]
            ## train
            optimizer.zero_grad()
            yb_pred = model(xb, None)
            loss = F.mse_loss(yb_pred, yb) 
            loss.backward()
            optimizer.step()

            ## evaluate    
            model.eval()
            with torch.no_grad(): # deactivate gradient tracking
                y_val_pred = model(X_val_t, None)
            y_val_pred = y_val_pred.cpu().numpy().flatten() # format back to to 1D-series
            score = ut.pooled_metrics(y_true=y_val, y_pred=y_val_pred)
            curve.append(score)
            if score < best_score:
                best_score = score
                best_epoch = epoch

    return best_score, best_epoch, curve

                
        
    # TODO optimizer, loss, batching, per-epoch val forward pass, curve
    # TODO score via ut.pooled_metrics

def search(X_fit, y_fit, X_val, y_val, n_iter, seed=con.SEED) -> tuple[dict, int, list]:
    """Score n_iter sampled configurations. Returns (params, epoch, trial_log).
    Log every configuration, its best score and its best epoch."""
    # TODO implement hyperparameter tuning

def run_ftt(condition: str = "tuned", n_iter: int = 30) -> None:
    """Hoisted stage 1+2 -> search -> fresh model trained on stage 3 for
    winning_epoch -> score stage 4 -> persist predictions, manifest, trial log."""
    # TODO train, score and log