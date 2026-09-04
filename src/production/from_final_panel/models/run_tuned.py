"""
src.production.from_final_panel.models.run_tuned
"""
import config as con, utils as ut, pandas as pd, numpy as np
import time
import yaml
import subprocess
from stageguard import Gatekeeper
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform, uniform
from datetime import datetime


SEARCH_SPACES = {
    "rf":  {'max_features':(0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 1.0),
            'min_samples_leaf':(1, 2, 5, 10, 20),
            'max_depth':(None, 10, 20, 30)}, 
            
    "xgb": {"learning_rate":loguniform(0.01, 0.3), 'n_estimators':(200, 400, 800, 1500), 'max_depth':(3, 4, 6, 8, 10),
            'min_child_weight':loguniform(1, 20), 'subsample': uniform(0.6, 0.4), 'colsample_bytree':uniform(0.5, 0.5),
            'reg_lambda':loguniform(0.1, 20)}	
}

def build_estimator(model_tag: str, params: dict):
    """Return an unfitted estimator for `model_tag` configured with `params`.
    Fixed, non-searched settings live here — not in SEARCH_SPACES."""
    if model_tag == 'rf':
        model = RandomForestRegressor(n_estimators = 500,
                                        random_state = con.SEED,
                                        n_jobs = -1,
                                        **params)
        return model
    elif model_tag=='xgb':
        model= XGBRegressor(random_state=con.SEED, **params)
        return model
    else:
        raise ValueError(f"expected model 'rf' or 'xgb', saw {model_tag}")

def search(model_tag, X_fit, y_fit, X_val, y_val, n_iter, seed=con.SEED) -> tuple[dict, list]:
    """Score `n_iter` sampled configurations on stage-2 data.
    Returns (winning_params, trial_log). trial_log records every sampled
    configuration and its stage-2 score, the spread provides
    evidence about differential tunability."""
    sampler = ParameterSampler(SEARCH_SPACES[model_tag], n_iter=n_iter, random_state=seed)

    # initialize placeholders
    search_log = []
    winning_score = np.inf
    winning_params = None

    # perform tuning
    for params in sampler:
        model = build_estimator(model_tag=model_tag, params=params)
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_val)
        y_pred = pd.Series(y_pred, index=y_val.index)
        metrics = ut.pooled_metrics(y_true=y_val, y_pred=y_pred)
        score = metrics[0]

        # log current configuration results
        search_log.append([model_tag, params, score])

        # check for optimum
        if score < winning_score:
            winning_score = score
            winning_params = params
        else:
            pass

    return winning_params, search_log

def run_tuned(model_tag: str, condition: str = "tuned", n_iter: int = 30) -> None:
    """Hoisted stage 1+2 → search → refit on stage 3 → score stage 4 → persist."""
    # start time counter 
    start_total = time.perf_counter()
    gk = Gatekeeper(model="nICL")

    # initiallize data pipeline
    X_fit, y_fit = gk.stage_one_data()
    X_val, y_val = gk.stage_two_data()

    # perform hyperparameter tuning
    print(f'      [>>>] starting model selection for {model_tag}...')
    start_tuning = time.perf_counter()
    params, log = search(model_tag, X_fit, y_fit, X_val, y_val, n_iter)
    end_tuning = time.perf_counter() -start_tuning
    model = build_estimator(model_tag=model_tag, params=params)
    print(f'      [>>>] model selection succesfull; duration: {round(end_tuning,2)}s')

    # fit on training partition
    print(f'      [>>>] fitting {model_tag} model on train partition...')
    start_training = time.perf_counter()
    X_tr, y_tr = gk.stage_three_data()
    model.fit(X_tr, y_tr)
    end_training = time.perf_counter() - start_training
    print(f'      [>>>] fitting train partition successfull; duration {round(end_training,2)}s')

    # score
    print(f'      [>>>] calculating test scores ...')
    start_testing = time.perf_counter()
    X_test, y_test, geo_id = gk.stage_four_data()
    y_pred = model.predict(X_test)
    end_testing = time.perf_counter() - start_testing
    print(f'      [>>>] test scores successfully calculated; duration {round(end_testing, 2)}s')
    y_pred = pd.Series(y_pred, index=y_test.index)

    # calculate metrics 
    print(f'      [>>>] saving&logging results to drive ...')
    metrics_per_region = ut.per_region_metrics(y_true=y_test, y_pred=y_pred, geoID=geo_id)
    metrics_pooled = ut.pooled_metrics(y_true=y_test, y_pred=y_pred)
    metrics_average = ut.macro_average_metrics(metrics_per_region)
    ut.assert_ss_res_decomposition(metrics_per_region, metrics_pooled)
    df_region_report, pooled_metrics_tupel, pooled_rmse_100, macro_average_metrics, macro_average_metrics_100, macro_average_metrics_q_100, regional_bias_gap, regional_bias_gap_rmse = ut.report_metrics(metrics_per_region, metrics_pooled, metrics_average, [*con.TIER1_REGS])

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
            "tuning iterations": n_iter,
            "hyperparameters": model.get_params(),
            "parameter search log": log,
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
            "pooled metrics": pooled_metrics_tupel,
            "pooled RMSE *100": pooled_rmse_100,
            "macro average": macro_average_metrics,
            "macro average rmse *100": macro_average_metrics_100,
            "macro average q": macro_average_metrics_q_100,
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

    with open(con.PRED_DIR_MAN / f"results_{model_tag}_{condition}_n_iter_{n_iter}_seed_{con.SEED}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest_dict, f, Dumper=LogDumper, sort_keys=False, default_flow_style=False)

    print(f'\n[°°°]{model_tag.upper()} regressor sucessfully tested and results saved - elapsed time: {round(total_time/60, 2)}min [°°°]')

# run_tuned("rf", condition= "tuned", n_iter=30)
run_tuned("xgb", condition= "tuned", n_iter=30)