##################################################
'''
src.production.descriptives.build_desc_panel

input:      panel.parquet, ref_geo_table.parquet, pre_processing_constants.parquet
purpose:    pre-split description of the final panel: regional balance, temporal
            coverage, panel balance, target shape, feature missingness
output:     desc_region_balance.csv, desc_year_coverage.csv, desc_obs_per_entity.csv,
            desc_target_by_region.csv, desc_target_by_year.csv, desc_feature_missingness.csv,
            coverage_region_year.png, obs_per_entity.png

'''
##################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import wasserstein_distance

import config as con

ID_COLS = ['orgpermid', 'year', 'esg_combined_score']
TIERS = {'tier1': con.TIER1_REGS, 'tier2': con.TIER2_REGS, 'tier3': con.TIER3_REGS}


def _tier(region: int) -> str:
    return next(t for t, regs in TIERS.items() if region in regs)


def _label(region: int) -> str:
    return f"{region}*" if region in con.TIER2_REGS else str(region)


def _style(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')


def _save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {path.name}")


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(con.PANEL)
    geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
    for frame, col in ((panel, 'orgpermid'), (panel, 'year'), (geo, 'orgpermid'), (geo, 'lvl3permid')):
        assert frame[col].notna().all(), f"null {col}"
        cast = frame[col].astype('int64')
        assert (cast.astype(frame[col].dtype) == frame[col]).all(), f"{col} ({frame[col].dtype}) does not cast to int64 losslessly"
        frame[col] = cast

    assert not panel.duplicated(['orgpermid', 'year']).any(), "duplicate (orgpermid, year) in panel"
    y = panel['esg_combined_score']
    assert y.notna().all() and y.between(0, 1).all(), "target null or outside [0,1]"
    assert geo['orgpermid'].is_unique, "ref_geo_table duplicates orgpermid"

    n = len(panel)
    panel = panel.merge(geo, on='orgpermid', how='left', validate='many_to_one')
    assert len(panel) == n, "panel grew or shrank on the geo merge"
    assert panel['lvl3permid'].notna().all(), "panel row without a region"
    untiered = set(panel['lvl3permid']) - {r for regs in TIERS.values() for r in regs}
    assert not untiered, f"regions without a tier: {sorted(untiered)}"
    panel['tier'] = panel['lvl3permid'].map(_tier)

    features = [c for c in panel.columns if c not in ID_COLS + ['lvl3permid', 'tier']]
    print(f"    [+++] panel loaded - rows {len(panel)}, entities {panel['orgpermid'].nunique()}, "
          f"regions {panel['lvl3permid'].nunique()}, features {len(features)}")
    return panel, features


def region_balance(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.groupby('lvl3permid').agg(tier=('tier', 'first'), entities=('orgpermid', 'nunique'),
                                          rows=('orgpermid', 'size'))
    assert out['rows'].sum() == len(panel), "region rows do not sum to panel rows"
    assert out['entities'].sum() == panel['orgpermid'].nunique(), "an entity spans two regions"
    out['rows_per_entity'] = out['rows'] / out['entities']
    out['row_share'] = out['rows'] / out['rows'].sum()
    out['entity_share'] = out['entities'] / out['entities'].sum()
    return out.sort_values('rows', ascending=False).reset_index()


def year_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    years = np.arange(panel['year'].min(), panel['year'].max() + 1)
    first = panel.groupby('orgpermid')['year'].min().rename('first_year')
    frame = panel.merge(first, on='orgpermid', validate='many_to_one')
    frame['is_entry'] = frame['year'].eq(frame['first_year'])

    def _agg(g: pd.DataFrame) -> pd.DataFrame:
        a = g.groupby('year').agg(rows=('orgpermid', 'size'), entities=('orgpermid', 'nunique'),
                                  new_entrants=('is_entry', 'sum'))
        return a.reindex(years, fill_value=0).rename_axis('year')

    glob = _agg(frame).reset_index().assign(scope='all', lvl3permid=pd.NA, tier=pd.NA)
    regs = frame.groupby('lvl3permid').apply(_agg, include_groups=False).reset_index().assign(scope='region')
    regs['tier'] = regs['lvl3permid'].map(_tier)

    assert regs.groupby('year')['rows'].sum().reindex(years).eq(glob.set_index('year')['rows']).all(), \
        "region-year rows do not sum to global year rows"
    assert glob['new_entrants'].sum() == panel['orgpermid'].nunique(), "global new entrants != entities"
    per_reg = regs.groupby('lvl3permid')['new_entrants'].sum()
    assert per_reg.eq(panel.groupby('lvl3permid')['orgpermid'].nunique().reindex(per_reg.index)).all(), \
        "regional new entrants != regional entities"

    out = pd.concat([glob, regs], ignore_index=True)
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    return out[['scope', 'lvl3permid', 'tier', 'year', 'rows', 'entities', 'new_entrants']]


def obs_per_entity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = int(panel['year'].max() - panel['year'].min() + 1)
    ent = panel.groupby('orgpermid').agg(n_obs=('year', 'size'), first=('year', 'min'), last=('year', 'max'),
                                         lvl3permid=('lvl3permid', 'first'))
    ent['gapped'] = (ent['last'] - ent['first'] + 1) > ent['n_obs']

    def _stats(g: pd.DataFrame) -> pd.Series:
        q = g['n_obs'].quantile([0.25, 0.5, 0.75])
        return pd.Series({'entities': len(g), 'mean': g['n_obs'].mean(), 'min': g['n_obs'].min(),
                          'q25': q[0.25], 'median': q[0.5], 'q75': q[0.75], 'max': g['n_obs'].max(),
                          'full_window': int(g['n_obs'].eq(window).sum()), 'gapped': int(g['gapped'].sum())})

    out = pd.concat([_stats(ent).to_frame().T.assign(scope='all', lvl3permid=pd.NA),
                     ent.groupby('lvl3permid').apply(_stats, include_groups=False).reset_index().assign(scope='region')], ignore_index=True)
    out = out.astype({c: 'int64' for c in ('entities', 'min', 'max', 'full_window', 'gapped')})
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    out['window_years'] = window
    assert out.loc[out['scope'].eq('region'), 'entities'].sum() == len(ent)
    return out[['scope', 'lvl3permid', 'entities', 'mean', 'min', 'q25', 'median', 'q75', 'max',
                'full_window', 'gapped', 'window_years']], ent


def _target_stats(y: pd.Series) -> dict:
    q = y.quantile([0.25, 0.5, 0.75])
    return {'rows': len(y), 'mean': y.mean(), 'std': y.std(), 'min': y.min(), 'q25': q[0.25],
            'median': q[0.5], 'q75': q[0.75], 'max': y.max()}


def target_by_region(panel: pd.DataFrame) -> pd.DataFrame:
    ref = panel.loc[panel['lvl3permid'].eq(con.ANCHOR_REGION), 'esg_combined_score'].to_numpy()
    assert len(ref) > 0, f"anchor {con.ANCHOR_REGION} absent from panel"
    records = [{'scope': 'all', 'lvl3permid': pd.NA, 'tier': pd.NA, **_target_stats(panel['esg_combined_score']),
                'entities': panel['orgpermid'].nunique(),
                'std_entity_means': panel.groupby('orgpermid')['esg_combined_score'].mean().std(),
                'wasserstein_anchor': wasserstein_distance(ref, panel['esg_combined_score'])}]
    for region, g in panel.groupby('lvl3permid'):
        y = g['esg_combined_score']
        records.append({'scope': 'region', 'lvl3permid': region, 'tier': _tier(region), **_target_stats(y),
                        'entities': g['orgpermid'].nunique(),
                        'std_entity_means': g.groupby('orgpermid')['esg_combined_score'].mean().std(),
                        'wasserstein_anchor': wasserstein_distance(ref, y)})
    out = pd.DataFrame(records)
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    assert out.loc[out['scope'].eq('region'), 'rows'].sum() == len(panel)
    return out.sort_values(['scope', 'rows'], ascending=[True, False]).reset_index(drop=True)


def target_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([{'year': yr, **_target_stats(g['esg_combined_score'])} for yr, g in panel.groupby('year')])
    assert out['rows'].sum() == len(panel)
    return out


def feature_missingness(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    retained = set(pd.read_parquet(con.PRE_PROS_CONTS, columns=['variable'])['variable'])
    cutoff, degvar = set(con.CUTOFF_VARS), set(con.DEGVAR_VARS)
    assert not (retained & cutoff or retained & degvar or cutoff & degvar), "feature status sets overlap"
    assert retained | cutoff | degvar == set(features), \
        f"status sets and panel features differ - status-only {sorted((retained | cutoff | degvar) - set(features))}, " \
        f"panel-only {sorted(set(features) - (retained | cutoff | degvar))}"

    status = {**{f: 'retained' for f in retained}, **{f: 'cutoff' for f in cutoff}, **{f: 'degvar' for f in degvar}}
    nan = panel[features].isna()
    out = pd.DataFrame({'variable': features, 'status': [status[f] for f in features],
                        'nan_share_all': nan.mean().to_numpy()})
    per_region = nan.groupby(panel['lvl3permid']).mean().T
    per_region.columns = [f"nan_share_{r}" for r in per_region.columns]
    out = out.merge(per_region, left_on='variable', right_index=True, how='left', validate='one_to_one')
    assert len(out) == len(features) and out.notna().all().all()
    return out.sort_values('nan_share_all').reset_index(drop=True)


def plot_coverage(coverage: pd.DataFrame, balance: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    order = balance.loc[balance['tier'].isin(['tier1', 'tier2']), 'lvl3permid'].tolist()
    grid = (coverage.loc[coverage['scope'].eq('region')]
            .pivot(index='lvl3permid', columns='year', values='rows').loc[order])
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.2, 1, 256)))
    sns.heatmap(grid, mask=grid.eq(0), norm=LogNorm(vmin=1, vmax=grid.to_numpy().max()), cmap=cmap,
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Observations (log scale, blank = 0)'}, ax=ax)
    ax.grid(False)
    ax.set_yticklabels([_label(r) for r in order], rotation=0, fontsize=10)
    ax.set_title("Observations per Region and Year\nTier-1 and Tier-2 (*) Regions", pad=15, fontweight='bold')
    ax.set_xlabel("Year", labelpad=10)
    ax.set_ylabel("Level 3 PermID")
    _save(fig, con.VIZ_DESC_COVERAGE)


def plot_obs_per_entity(ent: pd.DataFrame, window: int) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    counts = ent['n_obs'].value_counts().reindex(range(1, window + 1), fill_value=0)
    median = ent['n_obs'].median()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts.index, counts.to_numpy(), color="#1f4e79", width=0.75, label="Entities")
    ax.axvline(median, color="#d9534f", linestyle=":", linewidth=1.5, label=f"Median: {median:g}")
    ax.set_xticks(range(1, window + 1))
    ax.set_title(f"Panel Balance\nObservations per Entity (n = {len(ent):,})".replace(",", "."), pad=15, fontweight='bold')
    ax.set_xlabel("Observed years per entity", labelpad=10)
    ax.set_ylabel("Entities")
    ax.legend(frameon=False)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    _style(ax)
    _save(fig, con.VIZ_DESC_OBS_PER_ENTITY)


if __name__ == '__main__':
    print(f"\n[°°°] building panel descriptives [°°°]\n")
    panel, features = load_panel()

    balance = region_balance(panel)
    coverage = year_coverage(panel)
    obs, ent = obs_per_entity(panel)
    target_region = target_by_region(panel)
    target_year = target_by_year(panel)
    missing = feature_missingness(panel, features)
    print(f"    [+++] tables built")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in ((balance, con.TAB_DESC_REGION_BALANCE), (coverage, con.TAB_DESC_YEAR_COVERAGE),
                        (obs, con.TAB_DESC_OBS_PER_ENTITY), (target_region, con.TAB_DESC_TARGET_REGION),
                        (target_year, con.TAB_DESC_TARGET_YEAR), (missing, con.TAB_DESC_FEATURE_NAN)):
        frame.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {frame.shape[0]} rows x {frame.shape[1]} cols")

    plot_coverage(coverage, balance)
    plot_obs_per_entity(ent, int(obs['window_years'].iloc[0]))

    print(f"\n    [+++] region balance\n{balance.to_string(index=False)}")
    print(f"\n[°°°] panel descriptives complete [°°°]")