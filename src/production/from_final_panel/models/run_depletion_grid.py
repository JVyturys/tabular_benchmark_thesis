##################################################
'''
src.production.from_final_panel.models.run_depletion_grid

input:      depletion_draws.parquet, depletion_counts.parquet, plus every input
            run_icl consumes (panel.parquet, split.parquet, ref_geo_table.parquet,
            pre_processing_constants.parquet)
purpose:    enumerate the frozen depletion grid and score the in-context arm.
output:     per condition, run_icl's own artifacts -
              results/predictions/predictions_{model_tag}__default__depl_L{level}_d{draw}__seed_{SEED}.parquet
              model_scores/icl_scores/results_{model_tag}__default__depl_L{level}_d{draw}__seed_{SEED}.yaml
            model_scores/icl_scores/grid_log_{timestamp}.jsonl 

'''
##################################################
import config as con, pandas as pd, numpy as np
import os, socket, json, time, traceback, yaml
import pyarrow.parquet as pq
from datetime import datetime
from src.production.from_final_panel.models.run_icl import run_icl
from src.production.from_final_panel.build_depletion_draws import DEPLETION_GRID

MODEL_ORDER: tuple[str, ...] = ('tabicl', 'tabpfn3')
CONFIGURATION: str = 'default'
FIELD_SEPARATOR: str = '__'
PRIOR_CONDITION_HOURS: dict[str, float] = {'tabicl': 0.75, 'tabpfn3': 1.5}


def _condition_tag(level: int, draw: int) -> str:
    """'depl_L{level}_d{draw}' - must match the manifest and filename convention."""
    tag = f'depl_L{int(level)}_d{int(draw)}'
    assert FIELD_SEPARATOR not in tag, f"tag {tag} contains {FIELD_SEPARATOR!r}"
    return tag


def _load_conditions() -> pd.DataFrame:
    """Read DEPLETION_DRAWS; return one row per (level, draw) with its entity array.
    """
    print(f"\n[°°°] loading frozen depletion grid [°°°]\n")

    draws = pd.read_parquet(con.DEPLETION_DRAWS)
    counts = pd.read_parquet(con.DEPLETION_COUNTS)

    assert draws['orgpermid'].dtype == np.int64, f"orgpermid is {draws['orgpermid'].dtype}, want int64"
    assert not draws.duplicated(subset=['level', 'draw', 'orgpermid']).any(), "duplicate (level, draw, orgpermid)"

    drawn_conditions = {(int(level), int(draw)) for level, draw in draws[['level', 'draw']].drop_duplicates().to_numpy()}
    counted_conditions = {(int(level), int(draw)) for level, draw in counts[['level', 'draw']].to_numpy()}
    assert drawn_conditions == counted_conditions, \
        f"artifacts disagree: +{sorted(drawn_conditions - counted_conditions)} -{sorted(counted_conditions - drawn_conditions)}"

    declared = {(int(level), draw) for level, repeats in DEPLETION_GRID.items() for draw in range(repeats)}
    assert drawn_conditions == declared, \
        f"stale artifact: written-not-declared {sorted(drawn_conditions - declared)}, declared-not-written {sorted(declared - drawn_conditions)}"
    print(f"    [+++] grid matched - {len(declared)} conditions, levels {sorted(DEPLETION_GRID)}")

    carried = draws.loc[~draws['is_anchor']].groupby(['level', 'draw'])['orgpermid'].apply(
        lambda s: tuple(np.sort(s.to_numpy())))
    assert carried.nunique() == 1, f"carried block differs across conditions ({carried.nunique()} variants)"
    print(f"    [+++] carried block identical - {len(carried.iloc[0])} entities")

    records = []
    for (level, draw), block in draws.groupby(['level', 'draw'], sort=True):
        entities = np.sort(block['orgpermid'].to_numpy()).astype(np.int64)
        row = counts.loc[(counts['level'] == level) & (counts['draw'] == draw)]
        assert len(row) == 1, f"({level},{draw}): {len(row)} counts rows, want 1"
        row = row.iloc[0]

        assert pd.Index(entities).is_unique, f"({level},{draw}): duplicate entity"
        assert len(entities) == int(row['context_entities_retained']), \
            f"({level},{draw}): {len(entities)} entities vs counts {int(row['context_entities_retained'])}"
        assert int(row['context_entities_retained']) < int(row['context_entities_full']), \
            f"({level},{draw}): full entity set, not a depletion"
        assert int(row['context_rows_retained']) < int(row['context_rows_full']), \
            f"({level},{draw}): full row set, not a depletion"
        assert int(row['context_rows_full']) == con.CONTEXT_ROWS, \
            f"({level},{draw}): drawn against {int(row['context_rows_full'])} rows, bank is {con.CONTEXT_ROWS}"
        assert int(row['seed']) == con.SEED, \
            f"({level},{draw}): drawn at seed {int(row['seed'])}, grid runs at {con.SEED}"

        records.append({
            'level': int(level), 'draw': int(draw), 'tag': _condition_tag(level, draw),
            'entities': entities,
            'context_rows_retained': int(row['context_rows_retained']),
            'context_entities_retained': int(row['context_entities_retained']),
            'anchor_rows_retained': int(row['anchor_rows_retained']),
            'build_commit': str(row['git_commit']),
        })

    conditions = pd.DataFrame(records).sort_values(['level', 'draw']).reset_index(drop=True)
    assert conditions['tag'].is_unique, "tag collision"
    assert len(conditions) == sum(DEPLETION_GRID.values()), \
        f"{len(conditions)} conditions, grid declares {sum(DEPLETION_GRID.values())}"

    print(f"    [+++] ordered level ascending - {conditions['tag'].iloc[0]} .. {conditions['tag'].iloc[-1]}")
    print(f"      [---] context rows {conditions['context_rows_retained'].min()}-{conditions['context_rows_retained'].max()}, build {conditions['build_commit'].iloc[0]}")
    return conditions


def _artifact_paths(model_tag: str, condition: str, configuration: str, seed: int) -> tuple:

    stem = f"{model_tag}{FIELD_SEPARATOR}{configuration}{FIELD_SEPARATOR}{condition}{FIELD_SEPARATOR}seed_{seed}"
    return con.ICL_SCORES / f"results_{stem}.yaml", con.PRED_DIR / f"predictions_{stem}.parquet"


def _is_complete(model_tag: str, condition: str, configuration: str, seed: int, verbose: bool = True) -> bool:
    """True if this condition's manifest already exists and parses.

    """
    manifest_path, prediction_path = _artifact_paths(model_tag, condition, configuration, seed)

    def note(message: str) -> bool:
        if verbose:
            print(f"      [---] incomplete {model_tag} {condition} - {message}")
        return False

    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError:
        return note("manifest unparseable")

    meta = (manifest or {}).get('meta', {})
    if meta.get('model') != model_tag or meta.get('condition') != condition:
        return note(f"manifest records {meta.get('model')}/{meta.get('condition')}, name claims {model_tag}/{condition}")

    score = (manifest or {}).get('metrics', {}).get('pooled_RMSE')
    if score is None or not np.isfinite(score):
        return note("no finite pooled_RMSE")

    if not prediction_path.exists():
        return note("prediction parquet absent")

    if pq.ParquetFile(prediction_path).metadata.num_rows == 0:
        return note("prediction parquet empty")

    return True


def _manifest_duration(model_tag: str, condition: str, configuration: str, seed: int) -> float:
    """Wall-clock seconds recorded by a completed condition, for the estimate."""
    manifest_path, _ = _artifact_paths(model_tag, condition, configuration, seed)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return float(yaml.safe_load(f)['meta']['total processing time'])


def _completion_frame(conditions: pd.DataFrame, model_tags: tuple, seed: int,
                      verbose: bool = False) -> pd.DataFrame:
    """Completion and recorded duration for every model x condition pair.

    """
    rows = []
    for condition in conditions.itertuples(index=False):
        for model_tag in model_tags:
            complete = _is_complete(model_tag, condition.tag, CONFIGURATION, seed, verbose=verbose)
            duration = (_manifest_duration(model_tag, condition.tag, CONFIGURATION, seed)
                        if complete else float('nan'))
            rows.append({'model': model_tag, 'level': int(condition.level), 'draw': int(condition.draw),
                         'tag': condition.tag, 'complete': complete, 'duration_s': duration})
    return pd.DataFrame(rows)


def _estimate_hours(pending: set, durations: dict, model_tags: tuple) -> tuple:
    """Remaining hours, per model, from that model's own completed durations.


    """
    total, bases = 0.0, []
    for model_tag in model_tags:
        remaining = sum(1 for model, _ in pending if model == model_tag)
        observed = durations.get(model_tag, [])
        per_hour = float(np.mean(observed)) / 3600 if observed else PRIOR_CONDITION_HOURS[model_tag]
        total += remaining * per_hour
        bases.append(f"{model_tag} {round(per_hour, 2)}h x{remaining} "
                     f"({'obs ' + str(len(observed)) if observed else 'prior'})")
    return total, bases


def _resume_banner(status: pd.DataFrame, conditions: pd.DataFrame, model_tags: tuple,
                   durations: dict, seed: int) -> None:
    """What will be skipped, what will run, and the estimate against the budget."""
    n_done, n_todo = int(status['complete'].sum()), int((~status['complete']).sum())
    pending = {(row['model'], row['tag']) for _, row in status.loc[~status['complete']].iterrows()}
    estimate_h, bases = _estimate_hours(pending, durations, model_tags)

    print(f"\n[°°°] resume scan - {len(model_tags)}x{len(conditions)} pairs [°°°]\n")
    print(f"    [+++] complete {n_done}/{len(status)}, to run {n_todo}")
    print(f"    [+++] estimate {round(estimate_h, 1)}h / budget {con.DEPLETION_BUDGET_H}h - "
          f"{'within' if estimate_h <= con.DEPLETION_BUDGET_H else 'OVER'}")
    print(f"      [---] {', '.join(bases)}")

    if n_done:
        done = status.loc[status['complete'], 'tag'].unique()
        print(f"      [---] skipping {n_done} pair(s), {done[0]} .. {done[-1]}")

    expected = {_artifact_paths(m, t, CONFIGURATION, seed)[0].name
                for t in conditions['tag'] for m in model_tags}
    expected |= {_artifact_paths(m, 'undepl', CONFIGURATION, seed)[0].name for m in model_tags}
    orphans = sorted({path.name for path in con.ICL_SCORES.glob('results_*.yaml')} - expected)
    if orphans:
        print(f"      [---] WARNING {len(orphans)} orphan manifest(s) in {con.ICL_SCORES.name}: {', '.join(orphans)}")


def _claim_path(model_tag: str, condition: str, configuration: str, seed: int):
    """Sibling of the manifest, distinct prefix."""
    stem = f"{model_tag}{FIELD_SEPARATOR}{configuration}{FIELD_SEPARATOR}{condition}{FIELD_SEPARATOR}seed_{seed}"
    return con.ICL_SCORES / f"claim_{stem}.json"


def _claim_condition(model_tag: str, condition: str, configuration: str, seed: int) -> bool:
    """Atomically claim a condition. False if another process holds it.
    """
    path = _claim_path(model_tag, condition, configuration, seed)
    payload = json.dumps({'pid': os.getpid(), 'host': socket.gethostname(),
                          'model': model_tag, 'condition': condition,
                          'claimed_at': datetime.now().isoformat()})
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                held = json.load(f)
        except (json.JSONDecodeError, OSError):
            held = {}
        print(f"    [+++] skip {model_tag} {condition} - held by pid {held.get('pid', '?')}@{held.get('host', '?')} since {held.get('claimed_at', 'unknown')}")
        print(f"      [---] if stale, remove by hand: rm {path}")
        return False
    with os.fdopen(handle, 'w') as f:
        f.write(payload)
    return True


def _release_condition(model_tag: str, condition: str, configuration: str, seed: int) -> None:
    _claim_path(model_tag, condition, configuration, seed).unlink(missing_ok=True)


def _append_log(log_path, model_tag: str, condition, outcome: dict) -> None:
    """One JSONL line per attempt."""
    line = {
        'timestamp': datetime.now().isoformat(),
        'model': model_tag,
        'level': int(condition.level),
        'draw': int(condition.draw),
        'tag': condition.tag,
        'context_rows_retained': int(condition.context_rows_retained),
        'context_entities_retained': int(condition.context_entities_retained),
        'build_commit': condition.build_commit,
        'configuration': CONFIGURATION,
        **outcome,
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(line) + '\n')


def run_grid(model_tags: tuple = MODEL_ORDER, seed: int = con.SEED) -> None:
    """Score every condition for every model, skipping completed ones."""
    start_total = time.perf_counter()
    print(f"\n[°°°] depletion grid - {list(model_tags)}, {CONFIGURATION}, seed {seed} [°°°]\n")

    conditions = _load_conditions()

    con.ICL_SCORES.mkdir(parents=True, exist_ok=True)
    con.PRED_DIR.mkdir(parents=True, exist_ok=True)

    status = _completion_frame(conditions, model_tags, seed, verbose=True)
    durations = {m: status.loc[(status['model'] == m) & status['complete'], 'duration_s'].tolist()
                 for m in model_tags}
    _resume_banner(status, conditions, model_tags, durations, seed)

    pending = {(row['model'], row['tag']) for _, row in status.loc[~status['complete']].iterrows()}
    log_path = con.ICL_SCORES / f"grid_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"\n    [+++] log {log_path.name}\n")

    failures, claimed_elsewhere, consecutive_failures, broke_early = [], [], 0, False

    for condition in conditions.itertuples(index=False):
        if broke_early:
            break
        for model_tag in model_tags:
            tag = condition.tag

            if _is_complete(model_tag, tag, CONFIGURATION, seed):
                print(f"    [+++] skip {model_tag} {tag} - complete")
                continue

            if not _claim_condition(model_tag, tag, CONFIGURATION, seed):
                claimed_elsewhere.append((model_tag, tag))
                pending.discard((model_tag, tag))
                continue

            print(f"\n    [+++] run {model_tag} {tag} - rows {condition.context_rows_retained}, "
                  f"entities {condition.context_entities_retained}, {len(pending)} outstanding")

            start_condition = time.perf_counter()
            try:
                run_icl(model_tag, condition=tag, configuration=CONFIGURATION,
                        seed=seed, context_entities=condition.entities)
                elapsed = time.perf_counter() - start_condition

                assert _is_complete(model_tag, tag, CONFIGURATION, seed), \
                    f"{model_tag} {tag} returned clean but fails the completion check"

                manifest_path, _ = _artifact_paths(model_tag, tag, CONFIGURATION, seed)
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    runner_commit = yaml.safe_load(f)['meta'].get('git_commit', 'unknown')

                durations[model_tag].append(elapsed)
                pending.discard((model_tag, tag))
                consecutive_failures = 0
                outcome = {'status': 'complete', 'duration_s': round(elapsed, 1), 'runner_commit': runner_commit}
                print(f"    [+++] done {model_tag} {tag} - {round(elapsed / 60, 1)}min")

            except AssertionError:
                print(f"\n[°°°] ABORT - assertion on {model_tag} {tag}, contract breach not a fault [°°°]\n")
                _append_log(log_path, model_tag, condition,
                            {'status': 'assertion_abort',
                             'duration_s': round(time.perf_counter() - start_condition, 1),
                             'traceback': traceback.format_exc()})
                raise

            except Exception as err:
                elapsed = time.perf_counter() - start_condition
                consecutive_failures += 1
                failures.append((model_tag, tag, f"{type(err).__name__}: {err}"))
                pending.discard((model_tag, tag))
                outcome = {'status': 'failed', 'duration_s': round(elapsed, 1),
                           'error': f"{type(err).__name__}: {err}", 'traceback': traceback.format_exc()}
                print(f"    [+++] FAIL {model_tag} {tag} after {round(elapsed / 60, 1)}min - {type(err).__name__}: {err}")
                print(f"      [---] consecutive {consecutive_failures}/{con.GRID_MAX_CONSECUTIVE_FAILURES}")

            finally:
                _release_condition(model_tag, tag, CONFIGURATION, seed)

            _append_log(log_path, model_tag, condition, outcome)

            if consecutive_failures >= con.GRID_MAX_CONSECUTIVE_FAILURES:
                print(f"\n[°°°] BREAKER - {consecutive_failures} consecutive failures, stopping [°°°]\n")
                broke_early = True
                break

            estimate_h, _ = _estimate_hours(pending, durations, model_tags)
            print(f"      [---] elapsed {round((time.perf_counter() - start_total) / 3600, 2)}h, "
                  f"est {round(estimate_h, 1)}h left ({len(failures)} failed, {len(claimed_elsewhere)} claimed, excluded)")

    final = _completion_frame(conditions, model_tags, seed)
    incomplete = final.loc[~final['complete'], ['model', 'tag']]

    print(f"\n[°°°] grid finished - {int(final['complete'].sum())}/{len(final)} complete, "
          f"elapsed {round((time.perf_counter() - start_total) / 3600, 2)}h [°°°]\n")
    for model_tag in model_tags:
        observed = durations[model_tag]
        if observed:
            print(f"    [+++] {model_tag} - {len(observed)} on record, mean {round(float(np.mean(observed)) / 60, 1)}min")

    if len(incomplete):
        for _, row in incomplete.iterrows():
            print(f"      [---] incomplete {row['model']} {row['tag']}")
        for model_tag, tag, err in failures:
            print(f"      [---] failed {model_tag} {tag} - {err}")
        for model_tag, tag in claimed_elsewhere:
            print(f"      [---] claimed elsewhere {model_tag} {tag}")
        raise RuntimeError(f"{len(incomplete)}/{len(final)} incomplete - partial grid, re-run to resume")

    print(f"    [+++] all conditions verified complete")


if __name__ == '__main__':
    run_grid()