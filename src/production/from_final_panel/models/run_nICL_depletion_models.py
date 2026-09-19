##################################################
'''
src.production.from_final_panel.models.run_nICL_depletion_models

input:      panel.parquet, split.parquet, ref_geo_table.parquet,
            pre_processing_constants.parquet, depletion_draws.parquet,
            depletion_counts.parquet, the tuned undepl manifests of rf/xgb/ftt,
            the icl depletion manifests of the same condition
purpose:    refit the fitted roster (RF, XGB, FT-T) under each frozen depletion
            condition with locked hyperparameters and score stage 4
output:     results/predictions/predictions_{model_tag}__tuned__{condition}__seed_{SEED}.parquet
            model_scores/rf_xgb_scores/results_{model_tag}__tuned__{condition}__seed_{SEED}.yaml
            model_scores/ftt_scores/results_ftt__tuned__{condition}__seed_{SEED}.yaml
'''
##################################################
import config as con, utils as ut, pandas as pd, numpy as np
import time, math, json, yaml, torch
import torch.optim as optim
from datetime import datetime
from stageguard import Gatekeeper
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.production.from_final_panel.models.run_ftt import (
    build_model, train_and_curve, _train_one_epoch, DEVICE, MAX_EPOCHS, PATIENCE, BATCH_SIZE)

FIELD_SEPARATOR: str = '__'
CONFIGURATION: str = 'tuned'
SOURCE_N_ITER: int = 30
ICL_MODELS: tuple[str, ...] = ('tabicl', 'tabpfn3')
ICL_CONFIGURATION: str = 'default'
FITTED_MODELS: tuple[str, ...] = ('rf', 'xgb')
ESTIMATORS: dict = {'rf': RandomForestRegressor, 'xgb': XGBRegressor}
MODEL_ORDER: tuple[str, ...] = ('rf', 'xgb', 'ftt')
LIBRARIES: dict[str, tuple[str, ...]] = {
    'rf': ('numpy', 'pandas', 'scikit-learn'),
    'xgb': ('numpy', 'pandas', 'xgboost'),
    'ftt': ('numpy', 'pandas', 'torch', 'rtdl-revisiting-models'),
}


class CeilingReached(RuntimeError):
    pass


def _condition_tag(level: int, draw: int) -> str:
    tag = f'depl_L{int(level)}_d{int(draw)}'
    assert FIELD_SEPARATOR not in tag, f"tag {tag} contains {FIELD_SEPARATOR!r}"
    return tag


def _read_manifest_meta(path) -> dict:
    assert path.exists(), f"manifest of record absent: {path.name}"
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return yaml.safe_load(text)['meta']


def load_draw(level: int, draw: int) -> tuple[pd.Index, dict]:
    """Frozen entity ids retained at (level, draw), plus the condition record. Never regenerates."""
    tag = _condition_tag(level, draw)
    draws = pd.read_parquet(con.DEPLETION_DRAWS)
    counts = pd.read_parquet(con.DEPLETION_COUNTS)
    assert draws['orgpermid'].dtype == np.int64, f"orgpermid is {draws['orgpermid'].dtype}, want int64"

    block = draws.loc[(draws['level'] == level) & (draws['draw'] == draw)]
    row = counts.loc[(counts['level'] == level) & (counts['draw'] == draw)]
    assert len(block) > 0, f"{tag} absent from {con.DEPLETION_DRAWS.name}"
    assert len(row) == 1, f"{tag}: {len(row)} rows in {con.DEPLETION_COUNTS.name}, want 1"
    row = row.iloc[0]

    entities = pd.Index(np.sort(block['orgpermid'].to_numpy()))
    assert entities.is_unique, f"{tag}: duplicate entity in the draw"
    assert len(entities) == int(row['context_entities_retained']), \
        f"{tag}: {len(entities)} entities vs counts {int(row['context_entities_retained'])}"
    assert int(row['seed']) == con.SEED, f"{tag}: drawn at seed {int(row['seed'])}, run is at {con.SEED}"
    assert int(row['context_rows_full']) == con.CONTEXT_ROWS, \
        f"{tag}: drawn against {int(row['context_rows_full'])} rows, bank is {con.CONTEXT_ROWS}"

    carried = draws.loc[~draws['is_anchor']].groupby(['level', 'draw'])['orgpermid'].apply(
        lambda s: tuple(np.sort(s.to_numpy())))
    n_conditions = len(draws[['level', 'draw']].drop_duplicates())
    assert len(carried) == n_conditions, f"{n_conditions - len(carried)} condition(s) carry no non-anchor block"
    assert carried.nunique() == 1, f"carried block differs across conditions ({carried.nunique()} variants)"
    n_carried = len(carried.iloc[0])
    assert n_carried == int(row['context_entities_full']) - int(row['anchor_entities_full']), \
        f"{tag}: carried block {n_carried}, roster minus anchor is " \
        f"{int(row['context_entities_full']) - int(row['anchor_entities_full'])}"
    assert int((~block['is_anchor']).sum()) == n_carried, f"{tag}: condition does not carry the full block"

    for icl_model in ICL_MODELS:
        stem = FIELD_SEPARATOR.join([icl_model, ICL_CONFIGURATION, tag, f'seed_{con.SEED}'])
        meta = _read_manifest_meta(con.ICL_SCORES / f"results_{stem}.yaml")
        assert meta['condition'] == tag, f"{icl_model} manifest records {meta['condition']}, name claims {tag}"
        assert int(meta['context rows']) == int(row['context_rows_retained']), \
            f"{tag}: {icl_model} consumed {meta['context rows']} rows, counts record {int(row['context_rows_retained'])}"
        assert int(meta['context entities']) == len(entities), \
            f"{tag}: {icl_model} consumed {meta['context entities']} entities, draw holds {len(entities)}"

    record = {
        'level': int(level), 'draw': int(draw), 'tag': tag,
        'anchor_entities': pd.Index(np.sort(block.loc[block['is_anchor'], 'orgpermid'].to_numpy())),
        'context_rows_retained': int(row['context_rows_retained']),
        'context_entities_retained': int(row['context_entities_retained']),
        'anchor_rows_retained': int(row['anchor_rows_retained']),
        'anchor_entities_full': int(row['anchor_entities_full']),
        'context_entities_full': int(row['context_entities_full']),
        'carried_entities': n_carried,
        'draw_build_commit': str(row['git_commit']),
    }
    print(f"    [+++] draw {tag} loaded - entities {len(entities)}, anchor entities {len(record['anchor_entities'])}, "
          f"rows on record {record['context_rows_retained']}, carried {n_carried}, agrees with {', '.join(ICL_MODELS)}")
    return entities, record


def _load_locked(model_tag: str) -> dict:
    """Hyperparameters of the tuned undepl run of record, cross-checked against its own search evidence."""
    if model_tag in FITTED_MODELS:
        path = con.PRED_DIR_MAN / f"results_{model_tag}_{CONFIGURATION}_undepl_n_iter_{SOURCE_N_ITER}_seed_{con.SEED}.yaml"
    elif model_tag == 'ftt':
        path = con.FTT_VAL_TRIALS / f"results_ftt_{CONFIGURATION}_undepl_n_iter_{SOURCE_N_ITER}_seed_{con.SEED}.yaml"
    else:
        raise ValueError(f"expected model 'rf', 'xgb' or 'ftt', saw {model_tag}")

    meta = _read_manifest_meta(path)
    assert meta['model'] == model_tag, f"{path.name} records model {meta['model']}"
    assert meta['condition'] == 'undepl', f"{path.name} records condition {meta['condition']}"
    assert meta['configuration'] == CONFIGURATION, f"{path.name} records configuration {meta['configuration']}"
    assert int(meta['used seed']) == con.SEED, f"{path.name} ran at seed {meta['used seed']}, run is at {con.SEED}"
    assert int(meta['tuning iterations']) == SOURCE_N_ITER, f"{path.name} records {meta['tuning iterations']} iterations"
    params = meta['hyperparameters']

    locked = {'params': params, 'source': path.name, 'source_commit': meta['git_commit'],
              'n_iter': int(meta['tuning iterations'])}

    if model_tag in FITTED_MODELS:
        log = meta['parameter search log']
        assert len(log) == SOURCE_N_ITER, f"{path.name}: {len(log)} logged trials, want {SOURCE_N_ITER}"
        winner = min(log, key=lambda trial: trial[2])
        drift = {k: (v, params.get(k)) for k, v in winner[1].items() if params.get(k) != v}
        assert not drift, f"{path.name}: locked params disagree with the search winner - {drift}"
        assert params['random_state'] == con.SEED, f"{path.name}: estimator seed {params['random_state']}"

    else:
        for name, value in (('MAX_EPOCHS', MAX_EPOCHS), ('PATIENCE', PATIENCE), ('BATCH_SIZE', BATCH_SIZE)):
            assert int(meta[name]) == value, f"{path.name}: {name} {meta[name]}, run_ftt now has {value}"
        trials = {}
        for filename in meta['parameter search log']:
            with open(con.FTT_VAL_TRIALS / filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trial = json.loads(line)
                        trials[trial['trial']] = trial
        winner = trials.get(int(meta['winning trial']))
        assert winner is not None, f"{path.name}: winning trial {meta['winning trial']} absent from its logs"
        assert winner['params'] == params, f"{path.name}: locked params disagree with trial {winner['trial']}"
        assert int(winner['best_epoch']) == int(meta['epochs trained']), \
            f"{path.name}: trained {meta['epochs trained']} epochs, trial log says {winner['best_epoch']}"
        assert not winner['truncated curve flag'], f"{path.name}: winning trial was ceiling-truncated"
        locked.update({'winning_trial': int(meta['winning trial']), 'undepl_epochs': int(meta['epochs trained'])})

    print(f"    [+++] locked {model_tag} params loaded - source {path.name} @ {locked['source_commit']}")
    return locked


def depleted_labels(gk: Gatekeeper, X_stage: pd.DataFrame, entities: pd.Index, record: dict) -> pd.Index:
    """Labels of stage-3 rows whose entity is retained."""
    assert len(X_stage) == con.CONTEXT_ROWS, f"stage slice holds {len(X_stage)} rows, the stage-3 bank is {con.CONTEXT_ROWS}"
    row_entities = ut.resolve_keys(gk, X_stage.index)['orgpermid']
    stage_entities = pd.Index(row_entities.unique())
    assert len(stage_entities) == record['context_entities_full'], \
        f"{record['tag']}: stage 3 resolves {len(stage_entities)} entities, draws were built against {record['context_entities_full']}"

    missing = entities.difference(stage_entities)
    assert len(missing) == 0, f"{record['tag']}: {len(missing)} draw entities absent from stage 3, e.g. {list(missing[:5])}"

    kept = X_stage.index[row_entities.isin(entities).to_numpy()]
    assert 0 < len(kept) < len(X_stage), f"{record['tag']}: {len(kept)}/{len(X_stage)} rows kept, not a depletion"
    assert len(kept) == record['context_rows_retained'], \
        f"{record['tag']}: {len(kept)} rows realised, {record['context_rows_retained']} on record"

    dropped = stage_entities.difference(entities)
    n_anchor_dropped = record['anchor_entities_full'] - len(record['anchor_entities'])
    assert len(dropped) == n_anchor_dropped, \
        f"{record['tag']}: {len(dropped)} entities dropped, anchor lost {n_anchor_dropped}"

    print(f"    [+++] stage-3 depletion applied - rows {len(kept)}/{len(X_stage)}, entities dropped {len(dropped)} (all anchor)")
    return kept


def split_depleted(kept: pd.Index, X_fit: pd.DataFrame, X_val: pd.DataFrame, X_train: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Fit/val labels of the retained rows, derived from the stage-3 mask, in stage order."""
    assert X_fit.index.intersection(X_val.index).empty, "fit and val share row labels"
    assert X_fit.index.append(X_val.index).sort_values().equals(X_train.index.sort_values()), \
        "stage 3 is not the union of stage 1 and stage 2"

    kept_fit = X_fit.index[X_fit.index.isin(kept)]
    kept_val = X_val.index[X_val.index.isin(kept)]
    assert len(kept_fit) > 0 and len(kept_val) > 0, f"empty slice - fit {len(kept_fit)}, val {len(kept_val)}"
    assert kept_fit.append(kept_val).sort_values().equals(kept.sort_values()), \
        "depleted fit and val do not partition the depleted stage 3"

    print(f"    [+++] fit/val derived - fit {len(kept_fit)}/{len(X_fit)}, val {len(kept_val)}/{len(X_val)}")
    return kept_fit, kept_val


def anchor_rows(gk: Gatekeeper, labels: pd.Index, record: dict) -> int:
    """Rows among `labels` whose entity belongs to the anchor."""
    row_entities = ut.resolve_keys(gk, labels)['orgpermid']
    return int(row_entities.isin(record['anchor_entities']).sum())


def steps_per_epoch(n_rows: int) -> int:
    return math.ceil(n_rows / BATCH_SIZE)


class LogDumper(yaml.SafeDumper):
    pass

LogDumper.add_representer(tuple, lambda d, x: d.represent_sequence('tag:yaml.org,2002:seq', x))
LogDumper.add_multi_representer(np.floating, lambda d, x: d.represent_float(float(x)))
LogDumper.add_multi_representer(np.integer, lambda d, x: d.represent_int(int(x)))
LogDumper.add_multi_representer(np.ndarray, lambda d, x: d.represent_list(x.tolist()))
LogDumper.add_multi_representer(np.bool_, lambda d, x: d.represent_bool(bool(x)))


def _condition_meta(model_tag: str, record: dict, locked: dict, seed: int,
                    gk: Gatekeeper, kept: pd.Index, kept_fit: pd.Index, kept_val: pd.Index) -> dict:
    """Condition record from what this run sliced; the draw-time counts only serve as the check."""
    n_anchor = anchor_rows(gk, kept, record)
    assert n_anchor == record['anchor_rows_retained'], \
        f"{record['tag']}: run fits on {n_anchor} anchor rows, draw recorded {record['anchor_rows_retained']}"
    n_anchor_val = anchor_rows(gk, kept_val, record)
    return {
        'model': model_tag,
        'condition': record['tag'],
        'configuration': CONFIGURATION,
        'level': record['level'],
        'draw': record['draw'],
        'hyperparameters': locked['params'],
        'locked source': locked['source'],
        'locked source commit': locked['source_commit'],
        'draw build commit': record['draw_build_commit'],
        'context rows full': con.CONTEXT_ROWS,
        'context rows': len(kept),
        'context entities': record['context_entities_retained'],
        'carried entities': record['carried_entities'],
        'anchor rows': n_anchor,
        'anchor share': n_anchor / len(kept),
        'fit rows': len(kept_fit),
        'val rows': len(kept_val),
        'anchor val rows': n_anchor_val,
        'anchor val share': n_anchor_val / len(kept_val),
        'library versions': ut.library_versions(LIBRARIES[model_tag]),
        'used seed': seed,
    }


def _artifact_paths(model_tag: str, tag: str, seed: int) -> tuple:
    manifest_dir = con.FTT_VAL_TRIALS if model_tag == 'ftt' else con.PRED_DIR_MAN
    stem = FIELD_SEPARATOR.join([model_tag, CONFIGURATION, tag, f'seed_{seed}'])
    return manifest_dir / f'results_{stem}.yaml', con.PRED_DIR / f'predictions_{stem}.parquet'


def _score_and_persist(gk: Gatekeeper, predict, meta: dict, start_total: float) -> None:
    """Stage-4 scoring, metrics, prediction table and manifest. `predict` maps X_test to a 1-d array."""
    manifest_path, pred_path = _artifact_paths(meta['model'], meta['condition'], meta['used seed'])
    manifest_dir = manifest_path.parent
    assert not manifest_path.exists(), f"{manifest_path.name} already on record - remove it deliberately to rerun"

    X_test, y_test, geo_id = gk.stage_four_data()
    assert X_test.index.equals(geo_id.index), "test features and geoID are not aligned"
    keys = ut.resolve_keys(gk, X_test.index)

    start_score = time.perf_counter()
    y_pred = np.asarray(predict(X_test), dtype='float64').ravel()
    score_time = time.perf_counter() - start_score
    assert len(y_pred) == len(y_test), f"{len(y_pred)} predictions for {len(y_test)} test rows"
    assert np.isfinite(y_pred).all(), f"{int((~np.isfinite(y_pred)).sum())} non-finite predictions"
    y_pred = pd.Series(y_pred, index=y_test.index)
    print(f"    [+++] test partition scored - duration {round(score_time / 60, 2)}min")

    metrics_per_region = ut.per_region_metrics(y_true=y_test, y_pred=y_pred, geoID=geo_id)
    metrics_pooled = ut.pooled_metrics(y_true=y_test, y_pred=y_pred)
    metrics_average = ut.macro_average_metrics(metrics_per_region, [*con.TIER1_REGS])
    ut.assert_ss_res_decomposition(metrics_per_region, metrics_pooled)
    df_region_report, pooled_r2, pooled_rmse, average_rmse, average_r2, average_rmse_sq, regional_bias_gap, regional_bias_gap_rmse = \
        ut.report_metrics(metrics_per_region, metrics_pooled, metrics_average, [*con.TIER1_REGS])

    con.PRED_DIR.mkdir(parents=True, exist_ok=True)
    results = y_pred.to_frame('y_pred').join(y_test).join(keys).join(geo_id)
    assert len(results) == len(y_test), "prediction table row count changed on join"
    plumbing_cols = [c for c in results.columns if c != 'year']
    assert results[plumbing_cols].notna().all().all(), "NaN in the persisted prediction table"
    assert results['year'].notna().all(), f"{int(results['year'].isna().sum())} test rows carry no panel year"
    results.to_parquet(pred_path)

    manifest = {
        'meta': {**meta,
                 'test partition': X_test.shape,
                 'scoring time': score_time,
                 'total processing time': time.perf_counter() - start_total,
                 'timestamp': datetime.now().isoformat()},
        'metrics': {
            'pooled_RMSE': pooled_rmse,
            'pooled R2': pooled_r2,
            'average_RMSE': average_rmse,
            'average R2 (global denominator)': average_r2,
            'average_rmse_sq': average_rmse_sq,
            'regional bias gap': regional_bias_gap,
            'regional bias gap rmse': regional_bias_gap_rmse},
        'region metrics': {
            'shape': list(df_region_report.shape),
            'columns': list(df_region_report.columns),
            'results': df_region_report.to_dict(orient='index')},
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, Dumper=LogDumper, sort_keys=False, default_flow_style=False)
    print(f"    [+++] persisted - {pred_path.name}, {manifest_path.name}")
    print(f"    [---] pooled RMSE {round(pooled_rmse, 5)}, average RMSE {round(average_rmse, 5)}, bias gap {round(regional_bias_gap, 5)}")


def run_fitted_depleted(model_tag: str, *, level: int, draw: int, seed: int = con.SEED) -> None:
    """RF / XGB under depletion. Locked params, no search."""
    assert model_tag in FITTED_MODELS, f"expected one of {FITTED_MODELS}, saw {model_tag}"
    assert seed == con.SEED, f"locked params carry random_state={con.SEED}; seed {seed} would be another configuration"
    git = ut.git_state()
    start_total = time.perf_counter()
    print(f"[°°°] {model_tag} depletion run - L{level} d{draw}, seed {seed}")

    locked = _load_locked(model_tag)
    entities, record = load_draw(level, draw)

    gk = Gatekeeper(model="nICL")
    X_fit, _ = gk.stage_one_data()
    X_val, _ = gk.stage_two_data()
    X_tr, y_tr = gk.stage_three_data()
    kept = depleted_labels(gk, X_tr, entities, record)
    kept_fit, kept_val = split_depleted(kept, X_fit, X_val, X_tr)

    model = ESTIMATORS[model_tag](**locked['params'])
    drift = set(model.get_params()) ^ set(locked['params'])
    assert not drift, f"installed {model_tag} estimator and the dump disagree on parameter names: {sorted(drift)}"

    start_fit = time.perf_counter()
    model.fit(X_tr.loc[kept], y_tr.loc[kept])
    fit_time = time.perf_counter() - start_fit
    print(f"    [+++] {model_tag} fitted on {len(kept)} rows - duration {round(fit_time / 60, 2)}min")

    meta = {**_condition_meta(model_tag, record, locked, seed, gk, kept, kept_fit, kept_val),
            'train partition': X_tr.loc[kept].shape,
            'fit time': fit_time,
            **git}
    _score_and_persist(gk, model.predict, meta, start_total)


def _ftt_predictor(model):
    def predict(X_test: pd.DataFrame) -> np.ndarray:
        X_t = torch.tensor(X_test.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            batches = [model(X_t[start:start + BATCH_SIZE], None) for start in range(0, len(X_t), BATCH_SIZE)]
        return torch.cat(batches).cpu().numpy().flatten()
    return predict


def run_ftt_depleted(*, level: int, draw: int, seed: int = con.SEED, max_epochs: int = MAX_EPOCHS) -> None:
    """FT-T under depletion. Epochs are an outcome: early stopping on this condition's own val curve, then refit on fit+val."""
    model_tag = 'ftt'
    assert seed == con.SEED, f"seeds derive from the locked run at {con.SEED}; seed {seed} would be another configuration"
    assert max_epochs >= MAX_EPOCHS, f"max_epochs {max_epochs} below the ceiling of record {MAX_EPOCHS}"
    git = ut.git_state()
    start_total = time.perf_counter()
    print(f"[°°°] {model_tag} depletion run - L{level} d{draw}, seed {seed}, ceiling {max_epochs}")

    locked = _load_locked(model_tag)
    params = locked['params']
    entities, record = load_draw(level, draw)

    gk = Gatekeeper(model="nICL")
    X_fit, y_fit = gk.stage_one_data()
    X_val, y_val = gk.stage_two_data()
    X_tr, y_tr = gk.stage_three_data()
    kept = depleted_labels(gk, X_tr, entities, record)
    kept_fit, kept_val = split_depleted(kept, X_fit, X_val, X_tr)
    X_fit_d, y_fit_d = X_fit.loc[kept_fit], y_fit.loc[kept_fit]
    X_val_d, y_val_d = X_val.loc[kept_val], y_val.loc[kept_val]
    X_tr_d, y_tr_d = X_tr.loc[kept], y_tr.loc[kept]

    es_seed = seed + locked['winning_trial']
    torch.manual_seed(es_seed)
    torch.cuda.manual_seed(es_seed)
    start_es = time.perf_counter()
    model = build_model(X_fit_d, params)
    best_score, best_epoch, curve = train_and_curve(model, X_fit_d, y_fit_d, X_val_d, y_val_d, max_epochs, params)
    es_time = time.perf_counter() - start_es
    if len(curve) >= max_epochs:
        raise CeilingReached(f"{record['tag']}: val curve reached the ceiling {max_epochs} (best epoch {best_epoch})")
    assert 1 <= best_epoch <= len(curve) and curve[best_epoch - 1] == best_score, \
        f"{record['tag']}: best epoch {best_epoch} inconsistent with a curve of {len(curve)} epochs"
    print(f"    [+++] early stopping on depleted fit/val - best epoch {best_epoch}/{len(curve)}, "
          f"val RMSE {round(best_score, 5)}, duration {round(es_time / 60, 2)}min")

    refit_seed = seed + locked['n_iter']
    torch.manual_seed(refit_seed)
    torch.cuda.manual_seed(refit_seed)
    start_refit = time.perf_counter()
    model = build_model(X_fit_d, params)
    optimizer = optim.AdamW(model.make_parameter_groups(), lr=params['lr'], weight_decay=params['weight_decay'])
    X_tr_t = torch.tensor(X_tr_d.to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr_d.to_frame().to_numpy(dtype='float32'), dtype=torch.float32, device=DEVICE)
    for epoch in range(best_epoch):
        _train_one_epoch(model, X_tr_t, y_tr_t, optimizer)
    refit_time = time.perf_counter() - start_refit
    print(f"    [+++] refit on depleted fit+val - {best_epoch} epochs on {len(kept)} rows, duration {round(refit_time / 60, 2)}min")

    fit_steps = steps_per_epoch(len(kept_fit))
    meta = {**_condition_meta(model_tag, record, locked, seed, gk, kept, kept_fit, kept_val),
            'train partition': X_tr_d.shape,
            'early stopping seed': es_seed,
            'refit seed': refit_seed,
            'MAX_EPOCHS': max_epochs,
            'PATIENCE': PATIENCE,
            'BATCH_SIZE': BATCH_SIZE,
            'best epoch': best_epoch,
            'epochs run': len(curve),
            'best validation RMSE': best_score,
            'truncated curve flag': False,
            'steps per epoch fit': fit_steps,
            'best steps': best_epoch * fit_steps,
            'refit steps': best_epoch * steps_per_epoch(len(kept)),
            'undepl epochs': locked['undepl_epochs'],
            'undepl steps': locked['undepl_epochs'] * steps_per_epoch(len(X_fit)),
            'validation curve': curve,
            'early stopping time': es_time,
            'refit time': refit_time,
            **git}
    _score_and_persist(gk, _ftt_predictor(model), meta, start_total)


def _on_record(model_tag: str, tag: str, seed: int) -> bool:
    manifest_path, pred_path = _artifact_paths(model_tag, tag, seed)
    if not manifest_path.exists():
        return False
    meta = _read_manifest_meta(manifest_path)
    assert meta['model'] == model_tag and meta['condition'] == tag, \
        f"{manifest_path.name} records {meta['model']}/{meta['condition']}"
    assert pred_path.exists(), f"{manifest_path.name} on record without {pred_path.name}"
    return True


def run_grid(model_tags: tuple[str, ...] = MODEL_ORDER, seed: int = con.SEED) -> None:
    """Every frozen condition for every fitted model, model by model; conditions on record are skipped."""
    start_total = time.perf_counter()
    counts = pd.read_parquet(con.DEPLETION_COUNTS)
    conditions = sorted({(int(level), int(draw)) for level, draw in counts[['level', 'draw']].to_numpy()})
    assert len(conditions) == len(counts), f"{len(counts)} count rows for {len(conditions)} conditions"
    assert set(model_tags) <= set(MODEL_ORDER), f"expected models from {MODEL_ORDER}, saw {model_tags}"
    print(f"[°°°] nICL depletion grid - {len(conditions)} conditions x {', '.join(model_tags)}, seed {seed}")

    ran, skipped, ceiling = [], [], []
    for model_tag in model_tags:
        for level, draw in conditions:
            tag = _condition_tag(level, draw)
            if _on_record(model_tag, tag, seed):
                skipped.append((model_tag, tag))
                print(f"    [+++] skip {model_tag} {tag} - on record")
                continue
            if model_tag == 'ftt':
                try:
                    run_ftt_depleted(level=level, draw=draw, seed=seed)
                except CeilingReached as err:
                    ceiling.append((level, draw, str(err)))
                    print(f"    [+++] CEILING {err} - not scored, continuing")
                    continue
            else:
                run_fitted_depleted(model_tag, level=level, draw=draw, seed=seed)
            assert _on_record(model_tag, tag, seed), f"{model_tag} {tag} returned without a complete record"
            ran.append((model_tag, tag))
            print(f"    [---] {len(ran)} run, {len(skipped)} skipped, elapsed {round((time.perf_counter() - start_total) / 3600, 2)}h")

    print(f"[°°°] grid finished - {len(ran)} run, {len(skipped)} skipped, {len(ceiling)} at the ceiling, "
          f"elapsed {round((time.perf_counter() - start_total) / 3600, 2)}h")
    if ceiling:
        for level, draw, message in ceiling:
            print(f"    [---] rerun with a raised max_epochs: run_ftt_depleted(level={level}, draw={draw}, max_epochs=...)")
        raise RuntimeError(f"{len(ceiling)} ftt condition(s) reached the ceiling and are not on record")


if __name__ == '__main__':
    run_grid()