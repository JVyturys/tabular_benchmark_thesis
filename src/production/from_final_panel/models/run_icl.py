##################################################
'''
src.production.from_final_panel.models.run_icl

input:      panel.parquet, split.parquet, ref_geo_table.parquet,
            pre_processing_constants.parquet
purpose:    score the in-context roster (TabPFN-3, TabICL) on the stage-4 test
            partition under a named condition
output:     results/predictions/predictions_{model_tag}__{configuration}__{condition}__seed_{SEED}.parquet
            model_scores/icl_scores/results_{model_tag}__{configuration}__{condition}__seed_{SEED}.yaml
'''
##################################################
import config as con, utils as ut, pandas as pd, numpy as np
import torch, time, yaml, subprocess
from datetime import datetime
from stageguard import Gatekeeper
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


QUERY_BATCH = {"tabpfn3": 256, "tabicl": 256}   
ICL_N_ESTIMATORS = {"tabpfn3": 8, "tabicl": 8}
IGNORE_PRETRAINING_LIMITS = True


def _resolve_keys(gk: Gatekeeper, row_index: pd.Index) -> pd.DataFrame:
    """Resolve stage-slice row labels to (orgpermid, year).
    """
    ref = gk._preprocessed_data.loc[row_index, 'orgpermid']
    assert ref.index.equals(row_index), "label-based key resolution lost the stage row index"
    assert ref.notna().all(), "NaN orgpermid in the preprocessed frame"

    panel_keys = pd.read_parquet(con.PANEL, columns=['orgpermid', 'year'])
    assert row_index.max() < len(panel_keys), f"row label {row_index.max()} exceeds panel length {len(panel_keys)}"
    positional = panel_keys.iloc[row_index]
    assert positional.index.equals(row_index), "panel row labels are not positional"

    mismatches = int((positional['orgpermid'].to_numpy() != ref.to_numpy()).sum())
    assert mismatches == 0, f"{mismatches} rows where the panel position and the gatekeeper frame disagree on orgpermid - a merge changed the row count"
    return positional


def _context_row_mask(gk: Gatekeeper, X_ctx: pd.DataFrame, context_entities) -> np.ndarray:
    """Boolean mask over X_ctx rows: True where the row's entity is retained.
    """
    entities = pd.Index(pd.unique(pd.Series(list(context_entities))))
    row_entities = _resolve_keys(gk, X_ctx.index)['orgpermid']

    missing = entities.difference(pd.Index(row_entities.unique()))
    assert len(missing) == 0, f"{len(missing)} draw entities absent from the stage-3 context, e.g. {list(missing[:5])}"

    mask = row_entities.isin(entities).to_numpy()
    assert mask.sum() > 0, "depletion draw leaves an empty context"
    assert mask.sum() < len(mask), "depletion draw retains the full context, condition is not a depletion"

    print(f"    [+++] context depletion applied - entities {len(entities)}/{row_entities.nunique()}, rows {mask.sum()}/{len(mask)}")
    return mask

def build_model(model_tag: str, seed: int = con.SEED):
    """Return an unfitted in-context estimator. Model class declared once, here.

    Both estimators take the context through .fit(), which stores it rather than
    updating weights. Device, ensemble size and the pretraining-limit commitment
    are fixed here so no caller can vary them silently.
    """
    if model_tag == 'tabpfn3':
        return TabPFNRegressor(
            n_estimators=ICL_N_ESTIMATORS[model_tag],
            device=DEVICE,
            ignore_pretraining_limits=IGNORE_PRETRAINING_LIMITS,
            inference_precision="auto",
            fit_mode="fit_preprocessors",
            random_state=seed,
        )
    elif model_tag == 'tabicl':
        return TabICLRegressor(
            n_estimators=ICL_N_ESTIMATORS[model_tag],
            device=DEVICE,
            random_state=seed,
        )
    else:
        raise ValueError(f"expected model 'tabpfn3' or 'tabicl', saw {model_tag}")

def predict_chunked(model, X_test: np.ndarray, *, batch: int) -> np.ndarray:
    """Score X_test in chunks.
    """
    n_rows = X_test.shape[0]
    assert batch > 0, f"query batch must be positive, saw {batch}"

    bounds = [(start, min(start + batch, n_rows)) for start in range(0, n_rows, batch)]

    assert bounds[0][0] == 0 and bounds[-1][1] == n_rows, "chunk bounds do not span the test partition"
    assert all(prv[1] == nxt[0] for prv, nxt in zip(bounds, bounds[1:])), "chunk bounds overlap or leave a gap"
    assert sum(stop - start for start, stop in bounds) == n_rows, "chunk lengths do not sum to the test partition"

    chunks = []
    for i, (start, stop) in enumerate(bounds, start=1):
        chunk_pred = np.asarray(model.predict(X_test[start:stop])).reshape(-1)
        assert chunk_pred.shape[0] == stop - start, f"chunk {i} returned {chunk_pred.shape[0]} predictions for {stop - start} rows"
        chunks.append(chunk_pred)
        if i % 20 == 0 or i == len(bounds):
            print(f"      [---] scored chunk {i}/{len(bounds)} - rows {stop}/{n_rows}")

    y_pred = np.concatenate(chunks, axis=0)

    assert y_pred.shape == (n_rows,), f"reassembled predictions {y_pred.shape}, expected ({n_rows},)"
    assert np.isfinite(y_pred).all(), f"{(~np.isfinite(y_pred)).sum()} non-finite predictions"
    return y_pred

def run_icl(model_tag: str, *, condition: str = 'udepl', configuration: str = 'default',seed: int = con.SEED, context_entities=None) -> None:
    """One scoring pass.
    context_entities=None -> full train+val context.
    context_entities=<frozen entity id set> -> a depletion condition.
    """
    assert model_tag in QUERY_BATCH, f"expected model 'tabpfn3' or 'tabicl', saw {model_tag}"
    start_total = time.perf_counter()

    print(f"\n[°°°] in-context pass - model {model_tag}, condition {condition}, seed {seed} [°°°]\n")

    gk = Gatekeeper(model='ICL')
    X_ctx, y_ctx = gk.stage_three_data()
    n_ctx_full = len(X_ctx)

    if context_entities is not None:
        mask = _context_row_mask(gk, X_ctx, context_entities)
        X_ctx, y_ctx = X_ctx.loc[mask], y_ctx.loc[mask]
    else:
        print(f"    [+++] full context retained - rows {n_ctx_full}")

    n_ctx = len(X_ctx)
    assert n_ctx == len(y_ctx), "context X and y row count mismatch after depletion"
    assert (context_entities is None) == (n_ctx == n_ctx_full), "condition and context size disagree"


    print(f"    [+++] storing context in {model_tag} - rows {n_ctx}, features {X_ctx.shape[1]}")
    start_fit = time.perf_counter()
    model = build_model(model_tag, seed=seed)
    model.fit(X_ctx, y_ctx)
    fit_time = time.perf_counter() - start_fit
    print(f"    [+++] context stored - duration {round(fit_time / 60, 2)}min")

    max_samples = getattr(getattr(model, 'inference_config_', None), 'MAX_NUMBER_OF_SAMPLES', None)
    print(f"    [+++] checkpoint sample cap {max_samples}, context {n_ctx} - within limits: {None if max_samples is None else n_ctx <= max_samples}")

    X_test, y_test, geo_id = gk.stage_four_data()

    assert X_test.index.equals(y_test.index), "test features and target are not aligned before coercion"
    assert X_test.index.equals(geo_id.index), "test features and geoID are not aligned before coercion"

    start_score = time.perf_counter()
    y_pred = predict_chunked(model, X_test.to_numpy(dtype='float32'),
                             batch=QUERY_BATCH[model_tag])
    score_time = time.perf_counter() - start_score
    print(f"    [+++] test partition scored - duration {round(score_time / 60, 2)}min")

    y_pred = pd.Series(y_pred, index=y_test.index)
    assert y_pred.index.equals(geo_id.index), "prediction index does not match geoID index"

    print(f"    [+++] calculating scores ...")
    metrics_per_region = ut.per_region_metrics(y_true=y_test, y_pred=y_pred, geoID=geo_id)
    metrics_pooled = ut.pooled_metrics(y_true=y_test, y_pred=y_pred)
    metrics_average = ut.macro_average_metrics(metrics_per_region, [*con.TIER1_REGS])
    ut.assert_ss_res_decomposition(metrics_per_region, metrics_pooled)
    df_region_report, pooled_r2, pooled_rmse, average_rmse, average_r2, average_rmse_sq, regional_bias_gap, regional_bias_gap_rmse = ut.report_metrics(
        metrics_per_region, metrics_pooled, metrics_average, [*con.TIER1_REGS])
    print(f"    [+++] pooled RMSE {round(pooled_rmse, 5)}, pooled R2 {round(pooled_r2, 5)}, macro RMSE {round(average_rmse, 5)}")

    print(f"    [+++] saving predictions and manifest ...")
    con.PRED_DIR.mkdir(parents=True, exist_ok=True)
    con.ICL_SCORES.mkdir(parents=True, exist_ok=True)
    keys = _resolve_keys(gk, X_test.index)
    results = y_pred.to_frame('y_pred').join(y_test)
    results = results.join(keys)
    results = results.join(geo_id)
    assert len(results) == len(y_test), "prediction table row count changed on join"
    assert results.notna().all().all(), "NaN in the persisted prediction table"
    results.to_parquet(con.PRED_DIR / f'predictions_{model_tag}__{condition}__seed_{seed}.parquet')

    total_time = time.perf_counter() - start_total

    def get_git_revision_hash(short: bool = True) -> str:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    manifest_dict = {
        "meta": {
            "model": model_tag,
            "condition": condition,
            "hyperparameters": model.get_params(),
            "device": DEVICE,
            "ignore_pretraining_limits": IGNORE_PRETRAINING_LIMITS,
            "checkpoint_max_samples": max_samples,
            "query_batch": QUERY_BATCH[model_tag],
            "context rows": n_ctx,
            "context rows full": n_ctx_full,
            "context entities": None if context_entities is None else len(pd.unique(pd.Series(list(context_entities)))),
            "context storage time": fit_time,
            "scoring time": score_time,
            "total processing time": total_time,
            "test partition": X_test.shape,
            "git_commit": get_git_revision_hash(short=True),
            "timestamp": datetime.now().isoformat(),
            "used seed": seed
        },
        "metrics": {
            "pooled_RMSE": pooled_rmse,
            "pooled R2": pooled_r2,
            "average_RMSE": average_rmse,
            "average R2 (global denominator)": average_r2,
            "average_rmse_sq": average_rmse_sq,
            "regional bias gap": regional_bias_gap,
            "regional bias gap rmse": regional_bias_gap_rmse
        },

        "region metrics": {
            "shape": list(df_region_report.shape),
            "columns": list(df_region_report.columns),
            "results": df_region_report.to_dict(orient="index"),
            },
    }

    class LogDumper(yaml.SafeDumper):
        '''custom Dumper that tells PyYAML to serialize NumPy scalars as standard numbers and tuples as regular YAML lists'''
        pass

    LogDumper.add_representer(
        tuple,
        lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data),
    )

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

    with open(con.ICL_SCORES / f"results_{model_tag}__{configuration}__{condition}__seed_{seed}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest_dict, f, Dumper=LogDumper, sort_keys=False, default_flow_style=False)

    print(f'\n[°°°]{model_tag.upper()} in-context pass complete, condition {condition} - elapsed time: {round(total_time/60, 2)}min [°°°]')