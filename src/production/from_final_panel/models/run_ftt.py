"""
src.production.from_final_panel.models.run_ftt

FT-Transformer, non-ICL anchor. The epoch count is a
capacity parameter selected by retrospective early stopping on pooled validation
RMSE and the selected count is carried to the stage-3 refit unchanged.
"""
import config as con, utils as ut, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim
import torch, rtdl, time, yaml, subprocess
from stageguard import Gatekeeper
from sklearn.model_selection import ParameterSampler
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
    opt_params = {k: params[k] for k in ['lr', 'weight_decay']}
    optimizer = optim.AdamW(model.optimization_param_groups(), weight_decay=opt_params['weight_decay'])
    
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