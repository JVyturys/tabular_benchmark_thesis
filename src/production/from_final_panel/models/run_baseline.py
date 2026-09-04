'''
src.production.from_final_panel.models.run_baseline
'''
import pandas as pd, config as con, utils as ut, numpy as np
from stageguard import Gatekeeper
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from pathlib import Path
import time
import yaml

import subprocess


def run_baseline(model_tag: str, condition: str) -> None:
    """Walk the nICL stage path once and persist predictions at test grain.

    writes results/predictions/{model_tag}__{condition}__seed{SEED}.parquet
    at (orgpermid, year) grain. No stage skipped: 1 -> 2 -> 3 -> 4.
    """

    # start time counter 
    start_total = time.perf_counter()

    # initialize gatekeeper
    gk = Gatekeeper(model='nICL')
    
    # initiallize data pipeline
    X_fit, y_fit = gk.stage_one_data()
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=con.SEED, n_jobs=-1) 
    rf_baseline.fit(X_fit, y_fit)

    X_val, _ = gk.stage_two_data()

    X_tr, y_tr = gk.stage_three_data()
    rf_baseline.fit(X_tr, y_tr)

    X_test, y_test, geo_id = gk.stage_four_data()
    y_pred = rf_baseline.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)

    # calculate metrics 
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
    results.to_parquet(con.PRED_DIR/f'predictions_{model_tag}__{condition}__seed_{con.SEED}.parquet')

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
            "hyperparameters": rf_baseline.get_params(),
            "total processing time": total_time,
            "fit partition":X_fit.shape,
            "val partition":X_val.shape,
            "train partition":X_tr.shape,
            "test partition":X_test.shape,
            "git_commit": get_git_revision_hash(short=True),
            "timestamp": datetime.now().isoformat(),

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

    with open(con.PRED_DIR_MAN / f"results_{model_tag}_{condition}_seed_{con.SEED}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest_dict, f, Dumper=LogDumper, sort_keys=False, default_flow_style=False)

run_baseline('rf', 'baseline')

