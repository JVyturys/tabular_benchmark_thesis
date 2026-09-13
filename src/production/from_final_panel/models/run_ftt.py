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
MAX_EPOCHS = 800
PATIENCE = 50   
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

def _train_one_epoch(model, X_t, y_t, optimizer) -> None:
    """One pass over the training rows in shuffled batches of BATCH_SIZE."""
    ## train
    model.train()
    perm = torch.randperm(X_t.shape[0], device=DEVICE)
    for start in range(0, len(perm), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        xb, yb = X_t[idx], y_t[idx]
        optimizer.zero_grad()
        yb_pred = model(xb, None)
        loss = F.mse_loss(yb_pred, yb) 
        loss.backward()
        optimizer.step()

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
    patience_counter = 0

    # train network
    for epoch in range(max_epochs):
        # check if if loop ran out of patience
        if patience_counter >= PATIENCE: break

        ## train
        _train_one_epoch(model, X_fit_t, y_fit_t, optimizer)

        ## evaluate    
        model.eval()
        with torch.no_grad(): # deactivate gradient tracking
            y_val_pred = model(X_val_t, None)
        y_val_pred = y_val_pred.cpu().numpy().flatten()
        y_val_pred = pd.Series(y_val_pred, index=y_val.index)
        metrics = ut.pooled_metrics(y_true=y_val, y_pred=y_val_pred)
        rmse = np.sqrt(metrics[0]/(len(y_val)))
        curve.append(rmse)
        if rmse < best_score:
            patience_counter = 0 # reset patience
            best_score = rmse
            best_epoch = epoch + 1 # +1 because range(max_epoch) is zero based 
        else:
            patience_counter += 1     

    return best_score, best_epoch, curve

def search(X_fit, y_fit, X_val, y_val, n_iter, max_epochs,seed=con.SEED) -> tuple[list, str]:
    """Score n_iter sampled configurations. Returns (trial_log, filename).
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

    # create timestamp
    time_now = datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S")

    # start model selection
    for i, params in enumerate(sampler):
        hit_ceiling = False # idicate if training curve was possibly truncated; boolean
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
        
        if len(curve) == MAX_EPOCHS:
            hit_ceiling = True

        # save results
        trial_results = {"trial": i, "score":score, "MAX_EPOCHS": MAX_EPOCHS, "best_epoch":best_epoch, "truncated curve flag":hit_ceiling, "curve":curve, "params":params,
                         "trial_duration":f"{round(trial_duration, 2)}min","trial_timestamp":trial_timestamp} 
        search_log.append(trial_results)
        filename = f"search_log_{timestamp}.jsonl"
        with open(con.FTT_VAL_TRIALS / filename, "a") as f:
            f.write(json.dumps(trial_results, default=_numpy_converter) + "\n")

    return search_log, filename

def _select_winner(trials: list[dict]) -> dict:
    """Pick the lowest-scoring trial from validation log."""

    # check data integrity
    required_keys = {"trial","score", "best_epoch","curve","params","trial_duration","trial_timestamp", "truncated curve flag"}
    for trial in trials:
        assert required_keys.issubset(trial.keys()), f"Corrupted trial log: missing required keys in {trial.keys()}"
        
    return min(trials, key=lambda x: x["score"])

def _load_trial_logs(*filenames: str) -> list[dict]:
    """Parse JSONL trial logs. On a duplicate `trial` id, the later file wins,
    so a corrected rerun supersedes its original without the original being edited."""
    trials_by_id = {}
    
    for filename in filenames:
        filepath = con.FTT_VAL_TRIALS / filename
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trial_data = json.loads(line)
                    trials_by_id[trial_data["trial"]] = trial_data
    return list(trials_by_id.values())


def run_ftt(condition: str = "udepl", configuration: str = 'tuned', n_iter: int = 30, resume_from: list[str] = None) -> None:
    """Hoisted stage 1+2 -> search (or resume) -> select -> fresh model trained on
    stage 3 for winning_epoch -> score stage 4 -> persist.
    `resume_from`: JSONL filenames under con.FTT_VAL_TRIALS. When given, the search
    is skipped and those logs supply the trials. n_iter still sets the refit seed."""
    model_tag = "ftt"
    start_total = time.perf_counter()
    gk = Gatekeeper(model="nICL")
    X_fit, y_fit = gk.stage_one_data()
    X_val, y_val = gk.stage_two_data()

    if not resume_from:
        print(f'      [>>>] starting model selection for {model_tag}...')
        start_tuning = time.perf_counter()
        search_log, filename = search(X_fit, y_fit, X_val, y_val, n_iter, MAX_EPOCHS, con.SEED)
        filenames = [filename]
        end_tuning = time.perf_counter() - start_tuning
        print(f'      [>>>] model selection succesfull; duration: {round(end_tuning/60,2)}min')
    
    else:
        print(f'      [>>>] loading tuned {model_tag} model from JSON logs ...')
        search_log = _load_trial_logs(*resume_from)
        filenames = list(resume_from)
        print('      [>>>] configuration loaded ...')

    winner = _select_winner(search_log)
    winning_params = winner["params"]
    winning_epoch  = winner["best_epoch"]

    if winner["truncated curve flag"]:
        raise RuntimeError("winning epoch hit MAX_EPOCH ceiling, possibly truncated learning curve...")

    # refit on train = fit+val partition with winning_epoch
    print(f'      [>>>] fitting {model_tag} model on train partition...')
    start_training = time.perf_counter()
    ## get refit data partition
    X_tr, y_tr = gk.stage_three_data()

    ## set seed
    torch.manual_seed(con.SEED+n_iter)
    torch.cuda.manual_seed(con.SEED+n_iter)

    print(f'          [>] intializing model with tuned parameters...')
    model = build_model(X_fit, winning_params)

    ## transform partitions to torch friendly tensors
    y_tr_t = y_tr.to_frame('y_tr')
    y_tr_t = torch.tensor(y_tr_t.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    X_tr_t = torch.tensor(X_tr.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)

    ## slice optimization relevant parameters 
    opt_params = {k: winning_params[k] for k in ['lr', 'weight_decay']}
    optimizer = optim.AdamW(model.optimization_param_groups(), **opt_params)

    ## refit on new partition and tuned parametrization 
    print(f'          [>] fitting on training (fit+val) partition...')
    for epoch in range(winning_epoch):
        _train_one_epoch(model, X_tr_t, y_tr_t, optimizer)

    end_training = time.perf_counter() - start_training
    print(f'      [>>>] fitting train partition successfull; duration {round(end_training/60,2)}min')

    # perform scoring
    print(f'      [>>>] predicting on test partition ...')
    start_testing = time.perf_counter()
    ## get testing data
    X_test, y_test, geo_id = gk.stage_four_data()
    X_test_t = torch.tensor(X_test.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    ## evaluate    
    model.eval()
    preds_list = []
    with torch.no_grad(): # deactivate gradient tracking
        for start in range(0, len(X_test_t), BATCH_SIZE): # forward in batches
            xb = X_test_t[start:start + BATCH_SIZE]
            batch_pred = model(xb, None)
            preds_list.append(batch_pred)
    y_pred = torch.cat(preds_list).cpu().numpy().flatten()
    y_pred = pd.Series(y_pred, index=y_test.index)
    end_testing = time.perf_counter() - start_testing
    print(f'      [>>>] predictions successfully calculated; duration {round(end_testing/60,2)}min')

    # calculate metrics 
    print(f'      [>>>] calculating scores & saving+logging results to drive ...')
    metrics_per_region = ut.per_region_metrics(y_true=y_test, y_pred=y_pred, geoID=geo_id)
    metrics_pooled = ut.pooled_metrics(y_true=y_test, y_pred=y_pred)
    metrics_average = ut.macro_average_metrics(metrics_per_region, [*con.TIER1_REGS])
    ut.assert_ss_res_decomposition(metrics_per_region, metrics_pooled)
    df_region_report, pooled_r2, pooled_rmse, average_rmse, average_r2, average_rmse_sq,regional_bias_gap, regional_bias_gap_rmse = ut.report_metrics(metrics_per_region, metrics_pooled, metrics_average, [*con.TIER1_REGS])

    # save results
    orgpermid_year = pd.read_parquet(con.PANEL, columns=['orgpermid', 'year']).iloc[X_test.index] 
    results = y_pred.to_frame('y_pred').join(y_test)
    results = results.join(orgpermid_year)
    results = results.join(geo_id)
    results.to_parquet(con.PRED_DIR/f'predictions_{model_tag}_{condition}__n_iter_{n_iter}__seed_{con.SEED}.parquet')

    # stop time counter
    total_time = time.perf_counter() - start_total

    # log metrics 
    ## definer helper for git hash 
    def get_git_revision_hash(short: bool = True) -> str:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
        
    ## log parameters & results
    manifest_dict = {
        "meta":{
            "model":model_tag,
            "condition":condition,
            'configuration': configuration,
            "tuning iterations": n_iter,
            "hyperparameters": winning_params,
            "epochs trained" : winning_epoch,
            "winning validation RMSE":winner["score"],
            "winning trial":winner["trial"],
            "MAX_EPOCHS": winner.get("MAX_EPOCHS", MAX_EPOCHS),
            "PATIENCE": PATIENCE,
            "BATCH_SIZE": BATCH_SIZE,
            "parameter search log": filenames,
            "total processing time": total_time,
            "fit partition":X_fit.shape,
            "val partition":X_val.shape,
            "train partition":X_tr.shape,
            "test partition":X_test.shape,
            "git_commit": get_git_revision_hash(short=True),
            "timestamp": datetime.now().isoformat(),
            "used seed": con.SEED

        },
        "metrics": {
            "pooled_RMSE":pooled_rmse,
            "pooled R2":pooled_r2,
            "average_RMSE":average_rmse,
            "average R2 (global denominator)":average_r2,
            "average_rmse_sq":average_rmse_sq,
            "regional bias gap":regional_bias_gap,
            "regional bias gap rmse":regional_bias_gap_rmse
        },

        "region metrics": {
            "shape": list(df_region_report.shape),
            "columns": list(df_region_report.columns),
            "results": df_region_report.to_dict(orient="index"),
            },
    }
    ## change yaml settings to process NumPy Scalars
    class LogDumper(yaml.SafeDumper):
        '''custom Dumper that tells PyYAML to serialize NumPy scalars as standard numbers and tuples as regular YAML lists'''
        pass

    #### Use add_representer for exact types like tuple
    LogDumper.add_representer(
        tuple,
        lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data),
    )

    ### Use add_multi_representer for abstract base classes and subclasses
    LogDumper.add_multi_representer(
        np.floating,
        lambda dumper, data: dumper.represent_float(float(data)),
    )

    LogDumper.add_multi_representer(
        np.integer,
        lambda dumper, data: dumper.represent_int(int(data)),
    )

    LogDumper.add_multi_representer(
        np.ndarray,
        lambda dumper, data: dumper.represent_list(data.tolist()),
    )

    LogDumper.add_multi_representer(
        np.bool_,
        lambda dumper, data: dumper.represent_bool(bool(data)),
    )

    with open(con.FTT_VAL_TRIALS / f"results_{model_tag}_{configuration}_{condition}_n_iter_{n_iter}_seed_{con.SEED}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest_dict, f, Dumper=LogDumper, sort_keys=False, default_flow_style=False)

    print(f'\n[°°°]{model_tag.upper()} regressor sucessfully tested and results saved - elapsed time: {round(total_time/60, 2)}min [°°°]')

