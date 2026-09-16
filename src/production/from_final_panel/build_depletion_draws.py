##################################################
'''
src.production.from_final_panel.build_depletion_draws

input:      panel.parquet, ref_geo_table.parquet, split.parquet
purpose:    generate frozen, seeded depletion draws for the anchor region.
            Written once, before any model runs: every model at a given
            (level, draw) receives the identical entity set, so draw luck
            cannot be confounded with architecture. Draws are org-wise and
            stratified on entity mean target in N_STRATA quantile bins; the
            entity that crosses a stratum's row target is kept, so a draw
            overshoots by construction and the curve is read against realised
            N, not against the target.
output:     depletion_draws.parquet
            depletion_counts.parquet

'''
##################################################
import config as con, pandas as pd, numpy as np
import subprocess
from datetime import datetime
from scipy.stats import wasserstein_distance

DEPLETION_GRID: dict[int, int] = {8000: 2, 4000: 3, 2000: 3, 1000: 5, 350: 5}
N_STRATA: int = 4
ANCHOR_REGION: int = 100089


def _load_context_frame() -> pd.DataFrame:
    """Row-grain train+val context, reproducing the Gatekeeper's stage-3 slice.
    """
    print(f"\n[°°°] assembling stage-3 context frame [°°°]\n")

    panel = pd.read_parquet(con.PANEL, columns=['orgpermid', 'esg_combined_score'])
    geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
    split = pd.read_parquet(con.SPLIT)
    n_panel = len(panel)

    for name, frame in (('panel', panel), ('ref_geo_table', geo), ('split', split)):
        assert frame['orgpermid'].dtype.kind == 'i', \
            f"{name} carries orgpermid as {frame['orgpermid'].dtype} - a non-integer entity id matches nothing downstream"
    assert geo['lvl3permid'].dtype.kind == 'i', \
        f"ref_geo_table carries lvl3permid as {geo['lvl3permid'].dtype}, tier lists are integer"
    assert geo['orgpermid'].is_unique, "ref_geo_table duplicates orgpermid - the left merge is not row-preserving"
    assert split['orgpermid'].is_unique, "split table duplicates orgpermid - the left merge is not row-preserving"

    context = panel.merge(geo, on='orgpermid', how='left', validate='many_to_one')
    context = context.merge(split, on='orgpermid', how='left', validate='many_to_one')
    assert len(context) == n_panel, f"panel grew or shrank on merge - {n_panel} rows in, {len(context)} out"

    regions = [*con.TIER1_REGS, *con.TIER2_REGS]
    context = context.loc[context['lvl3permid'].isin(regions)]
    print(f"    [+++] tier-3 and unreferenced rows excluded - {n_panel - len(context)} of {n_panel}")

    assert context['partition'].notna().all(), \
        f"{int(context['partition'].isna().sum())} in-tier rows carry no partition - the split does not cover the context"

    context = context.loc[context['partition'].isin(['fit', 'val'])]
    context = context.drop(columns=['partition'])

    assert context['orgpermid'].dtype.kind == 'i', \
        f"orgpermid degraded to {context['orgpermid'].dtype} across the merges"
    assert context['lvl3permid'].dtype.kind == 'i', \
        f"lvl3permid degraded to {context['lvl3permid'].dtype} across the merges"
    assert context['esg_combined_score'].notna().all(), \
        f"{int(context['esg_combined_score'].isna().sum())} context rows carry no target - data fault, not plumbing"
    assert context['lvl3permid'].nunique() == len(regions), \
        f"context covers {context['lvl3permid'].nunique()} regions, tier lists declare {len(regions)}"
    assert ANCHOR_REGION in set(context['lvl3permid']), f"anchor region {ANCHOR_REGION} absent from the context"

    print(f"    [+++] context frame assembled - rows {len(context)}, entities {context['orgpermid'].nunique()}, regions {context['lvl3permid'].nunique()}")
    return context


def _entity_mean_target(panel: pd.DataFrame, region: int) -> pd.Series:
    """orgpermid -> mean target, over that region's train+val rows only."""
    slice_ = panel.loc[panel['lvl3permid'] == region]
    assert len(slice_) > 0, f"region {region} contributes no train+val rows"

    entity_means = slice_.groupby('orgpermid')['esg_combined_score'].mean()

    assert entity_means.index.is_unique, "groupby produced a duplicated orgpermid"
    assert entity_means.index.dtype.kind == 'i', f"entity index is {entity_means.index.dtype}, expected an integer id"
    assert entity_means.notna().all(), "NaN entity mean - a group of NaN-free rows cannot average to NaN"
    assert len(entity_means) == slice_['orgpermid'].nunique(), \
        f"{len(entity_means)} means for {slice_['orgpermid'].nunique()} distinct entities"
    assert entity_means.between(0, 1).all(), \
        f"entity mean outside the target's native [0,1] - observed [{entity_means.min()}, {entity_means.max()}]"

    print(f"    [+++] anchor entity means - entities {len(entity_means)}, rows {len(slice_)}, mean {round(float(entity_means.mean()), 5)}")
    return entity_means


def _assign_strata(entity_means: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """orgpermid -> stratum label. Bin edges fixed once, reused at every level."""
    strata, edges = pd.qcut(entity_means, q=N_STRATA, labels=False, retbins=True, duplicates='raise')
    strata = strata.astype('int64')

    assert strata.notna().all(), f"{int(strata.isna().sum())} entities left unlabelled by qcut"
    assert strata.index.equals(entity_means.index), "stratum labels lost the entity index"
    observed = sorted(strata.unique())
    assert observed == list(range(N_STRATA)), f"expected strata {list(range(N_STRATA))}, realised {observed}"
    sizes = strata.value_counts().sort_index()
    assert int(sizes.sum()) == len(entity_means), "stratum sizes do not sum to the anchor entity set"
    assert (sizes > 0).all(), f"empty stratum - sizes {sizes.to_dict()}"

    print(f"    [+++] strata assigned - edges {[round(float(e), 5) for e in edges]}")
    print(f"      [---] stratum sizes {sizes.to_dict()}")
    return strata, edges


def _draw_one(entity_means: pd.Series, strata: pd.Series, rows_per_entity: pd.Series,
              target_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Return the anchor entities RETAINED at this level.

    Allocates the row target across strata in proportion to each stratum's row
    mass, then admits whole entities until the target is met or passed; the
    entity that crosses it is kept.
    """
    assert strata.index.equals(entity_means.index), "strata and entity means disagree on the entity set"
    assert rows_per_entity.index.equals(entity_means.index), "row counts and entity means disagree on the entity set"

    rows_total = int(rows_per_entity.sum())
    assert 0 < target_rows < rows_total, \
        f"target {target_rows} is not a depletion of a {rows_total}-row anchor region"

    retained = []
    for stratum in range(N_STRATA):
        members = strata.index[strata == stratum]
        rows_s = rows_per_entity.loc[members]
        stratum_rows = int(rows_s.sum())

        target_s = target_rows * stratum_rows / rows_total
        assert target_s < stratum_rows, \
            f"stratum {stratum} asked for {target_s} of {stratum_rows} rows - allocation is not a depletion"

        order = rng.permutation(len(members))
        ordered = rows_s.iloc[order]
        cumulative = ordered.to_numpy().cumsum()

        crossing = int(np.searchsorted(cumulative, target_s, side='left'))
        admitted = ordered.index[:crossing + 1]

        realised_s = int(cumulative[crossing])
        overshoot_s = realised_s - target_s
        assert realised_s >= target_s, f"stratum {stratum} stopped short - {realised_s} rows against {target_s}"
        assert overshoot_s < int(rows_s.max()), \
            f"stratum {stratum} overshot by {overshoot_s}, more than its largest entity ({int(rows_s.max())} rows)"
        assert len(admitted) <= len(members), f"stratum {stratum} admitted more entities than it holds"

        print(f"      [---] stratum {stratum} - target {round(target_s, 1)}, realised {realised_s} rows, entities {len(admitted)}/{len(members)}")
        retained.append(admitted.to_numpy())

    drawn = np.concatenate(retained)
    assert drawn.dtype.kind == 'i', f"draw produced entity ids as {drawn.dtype}, expected an integer type"
    assert pd.Index(drawn).is_unique, "an entity was admitted by two strata"
    assert pd.Index(drawn).isin(entity_means.index).all(), "draw admitted an entity outside the anchor set"
    return drawn


def _verify(panel: pd.DataFrame, retained_entities: np.ndarray, region: int) -> dict:
    """Row-level distribution checks of the retained slice against the full
    region. Location alone is insufficient on a bounded target, so spread is
    checked independently. Bounds are tripwires, not a rejection rule."""
    region_rows = panel.loc[panel['lvl3permid'] == region]
    retained_rows = region_rows.loc[region_rows['orgpermid'].isin(retained_entities)]

    assert len(retained_rows) > 0, "retained slice is empty"
    assert len(retained_rows) < len(region_rows), "retained slice is the whole region - not a depletion"
    assert retained_rows['orgpermid'].nunique() == len(retained_entities), \
        f"{retained_rows['orgpermid'].nunique()} entities materialised for {len(retained_entities)} drawn"

    kept_counts = retained_rows.groupby('orgpermid').size()
    full_counts = region_rows.groupby('orgpermid').size().reindex(kept_counts.index)
    assert kept_counts.equals(full_counts), \
        "an admitted entity carries fewer rows than it holds in the region - org-wise admission violated"

    full_y = region_rows['esg_combined_score'].to_numpy()
    kept_y = retained_rows['esg_combined_score'].to_numpy()

    quantiles = [0.25, 0.50, 0.75]
    full_q = np.quantile(full_y, quantiles)
    kept_q = np.quantile(kept_y, quantiles)

    checks = {
        'anchor_rows_full': int(len(region_rows)),
        'anchor_entities_full': int(region_rows['orgpermid'].nunique()),
        'anchor_rows_retained': int(len(retained_rows)),
        'anchor_entities_retained': int(len(retained_entities)),
        'anchor_row_share': float(len(retained_rows) / len(region_rows)),

        'target_mean_full': float(full_y.mean()),
        'target_mean_retained': float(kept_y.mean()),
        'target_mean_delta': float(kept_y.mean() - full_y.mean()),

        'target_std_full': float(full_y.std(ddof=0)),
        'target_std_retained': float(kept_y.std(ddof=0)),
        'target_std_ratio': float(kept_y.std(ddof=0) / full_y.std(ddof=0)),
        'target_iqr_full': float(full_q[2] - full_q[0]),
        'target_iqr_retained': float(kept_q[2] - kept_q[0]),

        'target_q25_full': float(full_q[0]), 'target_q25_retained': float(kept_q[0]),
        'target_q50_full': float(full_q[1]), 'target_q50_retained': float(kept_q[1]),
        'target_q75_full': float(full_q[2]), 'target_q75_retained': float(kept_q[2]),
        'target_min_full': float(full_y.min()), 'target_min_retained': float(kept_y.min()),
        'target_max_full': float(full_y.max()), 'target_max_retained': float(kept_y.max()),
        'wasserstein_1': float(wasserstein_distance(full_y, kept_y)),
    }

    standard_error = checks['target_std_full'] / np.sqrt(checks['anchor_entities_retained'])
    checks['mean_delta_se_units'] = float(checks['target_mean_delta'] / standard_error)

    assert all(np.isfinite(v) for v in checks.values()), "non-finite value in the verification record"

    assert checks['target_min_retained'] >= checks['target_min_full'] and \
           checks['target_max_retained'] <= checks['target_max_full'], \
        f"retained target range [{checks['target_min_retained']}, {checks['target_max_retained']}] escapes the " \
        f"region's [{checks['target_min_full']}, {checks['target_max_full']}] - the slice is not drawn from this region"

    assert abs(checks['mean_delta_se_units']) < con.TRIPWIRE_MEAN_SE_FACTOR, \
        f"retained mean sits {round(checks['mean_delta_se_units'], 2)} standard errors off the region - " \
        f"tripwire is {con.TRIPWIRE_MEAN_SE_FACTOR} SE and unreachable by sampling, so this is an implementation fault"

    std_lo, std_hi = con.TRIPWIRE_STD_RATIO
    assert std_lo < checks['target_std_ratio'] < std_hi, \
        f"retained spread is {round(checks['target_std_ratio'], 3)} of the region's, outside the tripwire " \
        f"({std_lo}, {std_hi}) - the draw has collapsed onto part of the target range"

    print(f"    [+++] verified - rows {checks['anchor_rows_retained']}/{checks['anchor_rows_full']} "
          f"({round(checks['anchor_row_share'], 4)}), mean delta {round(checks['target_mean_delta'], 5)} "
          f"({round(checks['mean_delta_se_units'], 2)} SE), std ratio {round(checks['target_std_ratio'], 4)}, "
          f"W1 {round(checks['wasserstein_1'], 5)}")
    return checks


def _git_revision_hash(short: bool = True) -> str:
    """Short or full HEAD hash, 'unknown' outside a work tree."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_draws(seed: int = con.SEED) -> None:
    """Enumerate the grid, draw, verify, and write both artifacts."""
    print(f"\n[°°°] building depletion draws - anchor {ANCHOR_REGION}, strata {N_STRATA}, seed {seed} [°°°]\n")

    context = _load_context_frame()

    assert len(context) == con.CONTEXT_ROWS, \
        f"context frame holds {len(context)} rows, the runs of record were scored on {con.CONTEXT_ROWS}. " \
        f"Either an upstream filter, tier list or split rule moved, or the two derivations of the context " \
        f"have diverged - check_context_frame_agreement tells you which. Do not draw against an unbanked " \
        f"context: establish agreement, rebank con.CONTEXT_ROWS, rerun the curve"
    print(f"    [+++] context row count matches the bank - {con.CONTEXT_ROWS}")

    stage3_entities = set(context['orgpermid'].unique())
    anchor_rows = context.loc[context['lvl3permid'] == ANCHOR_REGION]
    anchor_entities = set(anchor_rows['orgpermid'].unique())
    non_anchor_entities = np.array(sorted(stage3_entities - anchor_entities), dtype=np.int64)
    rows_per_entity = anchor_rows.groupby('orgpermid').size()

    print(f"    [+++] anchor {ANCHOR_REGION} - rows {len(anchor_rows)}, entities {len(anchor_entities)}")
    print(f"    [+++] carried unchanged - entities {len(non_anchor_entities)}, rows {len(context) - len(anchor_rows)}")

    entity_means = _entity_mean_target(anchor_rows, ANCHOR_REGION)
    assert set(entity_means.index) == anchor_entities, "entity means and the anchor row slice disagree on the entity set"
    assert rows_per_entity.reindex(entity_means.index).notna().all(), "an anchor entity has no row count"
    rows_per_entity = rows_per_entity.reindex(entity_means.index)

    strata, edges = _assign_strata(entity_means)

    draw_frames, count_records = [], []
    git_commit, timestamp = _git_revision_hash(short=True), datetime.now().isoformat()

    for level, repeats in DEPLETION_GRID.items():
        for draw in range(repeats):
            entropy = [seed, level, draw]
            print(f"\n    [+++] condition level {level} draw {draw} - entropy {entropy}")
            rng = np.random.default_rng(entropy)

            drawn = _draw_one(entity_means, strata, rows_per_entity, level, rng)
            dropped_anchor = anchor_entities - set(drawn)

            assert set(drawn) < anchor_entities, "retained anchor set is not a strict subset of the anchor region"
            assert len(dropped_anchor) > 0, f"level {level} draw {draw} retains every anchor entity - not a depletion"

            drawn_strata = strata.loc[drawn]
            covered = sorted(int(label) for label in drawn_strata.unique())
            assert covered == list(range(N_STRATA)), \
                f"draw covers strata {covered}, not {list(range(N_STRATA))} - the retained slice has " \
                f"collapsed onto part of the target range, which the allocation rule cannot do"
            entities_per_stratum = drawn_strata.value_counts().sort_index().to_dict()

            checks = _verify(context, drawn, ANCHOR_REGION)

            written = np.concatenate([non_anchor_entities, np.sort(drawn)]).astype(np.int64)

            assert set(written) | set(dropped_anchor) == stage3_entities, \
                "context entities appeared or vanished: the only permitted difference " \
                "between a condition and the full context is the drawn anchor entities"
            assert written.dtype == np.int64, f"written entity ids are {written.dtype}, a non-int64 id matches nothing in run_icl"
            assert pd.Index(written).is_unique, "duplicate orgpermid within a condition"
            assert set(written).issubset(stage3_entities), "a written entity is absent from the stage-3 context"

            context_rows = len(context) - int(anchor_rows['orgpermid'].isin(dropped_anchor).sum())
            assert 0 < context_rows < len(context), \
                f"condition retains {context_rows} of {len(context)} context rows - run_icl's depletion check would fire"

            draw_frames.append(pd.DataFrame({
                'level': np.int64(level),
                'draw': np.int64(draw),
                'orgpermid': written,
                'is_anchor': np.isin(written, np.sort(drawn)),
            }))
            count_records.append({
                'level': int(level), 'draw': int(draw), 'target_rows': int(level),
                **checks,
                'anchor_rows_dropped': int(checks['anchor_rows_full'] - checks['anchor_rows_retained']),
                'anchor_entities_dropped': int(len(dropped_anchor)),
                'overshoot_rows': int(checks['anchor_rows_retained'] - level),
                'context_rows_retained': int(context_rows),
                'context_rows_full': int(len(context)),
                'context_entities_retained': int(len(written)),
                'context_entities_full': int(len(stage3_entities)),
                'entities_per_stratum': str(entities_per_stratum),
                'rng_entropy': str(entropy),
                'seed': int(seed), 'n_strata': int(N_STRATA),
                'stratum_edges': str([round(float(e), 8) for e in edges]),
                'anchor_region': int(ANCHOR_REGION),
                'tripwire_mean_se_factor': int(con.TRIPWIRE_MEAN_SE_FACTOR),
                'tripwire_std_ratio': str(list(con.TRIPWIRE_STD_RATIO)),
                'git_commit': git_commit, 'timestamp': timestamp,
            })

    draws = pd.concat(draw_frames, ignore_index=True)
    counts = pd.DataFrame(count_records)

    carried = draws.loc[~draws['is_anchor']].groupby(['level', 'draw'])['orgpermid'].apply(
        lambda s: tuple(np.sort(s.to_numpy())))
    assert carried.nunique() == 1, "the non-anchor entity block differs between conditions"

    assert draws['orgpermid'].dtype == np.int64, f"persisted orgpermid is {draws['orgpermid'].dtype}, expected int64"
    assert len(draws) == int(counts['context_entities_retained'].sum()), \
        f"draws table holds {len(draws)} rows against {int(counts['context_entities_retained'].sum())} counted entities"
    assert len(counts) == sum(DEPLETION_GRID.values()), \
        f"{len(counts)} conditions written, grid declares {sum(DEPLETION_GRID.values())}"
    assert not draws.duplicated(subset=['level', 'draw', 'orgpermid']).any(), "duplicate (level, draw, orgpermid)"
    assert (counts['overshoot_rows'] >= 0).all(), "a condition fell short of its row target"

    probe = counts.iloc[-1]
    replay = _draw_one(entity_means, strata, rows_per_entity, int(probe['target_rows']),
                       np.random.default_rng([seed, int(probe['level']), int(probe['draw'])]))
    original = draws.loc[(draws['level'] == probe['level']) & (draws['draw'] == probe['draw']) & draws['is_anchor'], 'orgpermid']
    assert np.array_equal(np.sort(replay), np.sort(original.to_numpy())), \
        "replaying a condition from its entropy produced a different entity set"
    print(f"\n    [+++] replay check passed - level {int(probe['level'])} draw {int(probe['draw'])}")

    draws = draws.sort_values(['level', 'draw', 'orgpermid']).reset_index(drop=True)
    con.DEPLETION_DRAWS.parent.mkdir(parents=True, exist_ok=True)
    draws.to_parquet(con.DEPLETION_DRAWS)
    counts.to_parquet(con.DEPLETION_COUNTS)

    print(f"\n    [+++] written - {con.DEPLETION_DRAWS.name} rows {len(draws)}, {con.DEPLETION_COUNTS.name} rows {len(counts)}")
    print(counts[['level', 'draw', 'anchor_rows_retained', 'overshoot_rows', 'anchor_entities_retained',
                  'target_mean_delta', 'mean_delta_se_units', 'target_std_ratio', 'wasserstein_1']].to_string(index=False))
    print(f"\n[°°°] depletion draws complete - commit {git_commit} [°°°]")


if __name__ == '__main__':
    build_draws()