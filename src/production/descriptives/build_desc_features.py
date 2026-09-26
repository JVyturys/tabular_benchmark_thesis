##################################################
'''
src.production.descriptives.build_desc_features

input:      ref_ws_variables.parquet, panel.parquet, split.parquet, ref_geo_table.parquet,
            pre_processing_constants.parquet
purpose:    feature funnel from the Worldscope candidate set to the model features,
            reproduction of the fit-partition preprocessing constants, and the
            median-imputation burden per region and partition
output:     desc_feature_funnel.csv, desc_feature_constants.csv, desc_imputation_by_region.csv,
            imputation_by_region.png

'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import config as con
import utils as ut

PARTITIONS = ['fit', 'val', 'test']
PART_COLORS = {'train': "#1f4e79", 'test': "#a6a6a6"}
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


def _int_col(frame: pd.DataFrame, col: str, name: str) -> None:
    assert frame[col].notna().all(), f"{name}: null {col}"
    cast = frame[col].astype('int64')
    assert (cast.astype(frame[col].dtype) == frame[col]).all(), f"{name}: {col} ({frame[col].dtype}) does not cast to int64 losslessly"
    frame[col] = cast


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, set]]:
    wsvar = pd.read_parquet(con.REF_WSVAR, columns=['number', 'name'])
    _int_col(wsvar, 'number', 'ref_ws_variables')
    assert wsvar['number'].is_unique, "ref_ws_variables duplicates an item number"
    wsvar['variable'] = 'item' + wsvar['number'].astype(str)

    panel = pd.read_parquet(con.PANEL)
    geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
    split = pd.read_parquet(con.SPLIT, columns=['orgpermid', 'partition'])
    for frame, name, cols in ((panel, 'panel', ['orgpermid']), (geo, 'ref_geo_table', ['orgpermid', 'lvl3permid']),
                              (split, 'split', ['orgpermid'])):
        for col in cols:
            _int_col(frame, col, name)
    assert geo['orgpermid'].is_unique and split['orgpermid'].is_unique, "geo or split duplicates orgpermid"

    features = [c for c in panel.columns if c not in ('orgpermid', 'year', 'esg_combined_score')]
    assert set(features) <= set(wsvar['variable']), \
        f"panel features outside the candidate set: {sorted(set(features) - set(wsvar['variable']))}"

    status = {'retained': set(pd.read_parquet(con.PRE_PROS_CONTS, columns=['variable'])['variable']),
              'cutoff': set(con.CUTOFF_VARS), 'degvar': set(con.DEGVAR_VARS)}
    sets = list(status.values())
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:]), "feature status sets overlap"
    assert set().union(*sets) == set(features), "status sets do not exactly cover the panel features"

    n = len(panel)
    panel = (panel.merge(geo, on='orgpermid', how='left', validate='many_to_one')
                  .merge(split, on='orgpermid', how='left', validate='many_to_one'))
    assert len(panel) == n, "panel grew or shrank on merge"
    assert panel['lvl3permid'].notna().all(), "panel row without a region"
    in_tier = panel['lvl3permid'].isin([*con.TIER1_REGS, *con.TIER2_REGS])
    assert panel.loc[in_tier, 'partition'].isin(PARTITIONS).all(), "in-tier row without a valid partition"
    assert panel.loc[~in_tier, 'partition'].isna().all(), "tier-3 row carries a partition"
    tiered = panel.loc[in_tier].copy()
    tiered['tier'] = tiered['lvl3permid'].map(_tier)
    print(f"    [+++] inputs loaded - candidates {len(wsvar)}, panel features {len(features)}, in-tier rows {len(tiered)}")
    return wsvar, tiered, features, status


def reproduce_constants(tiered: pd.DataFrame, features: list[str], status: dict[str, set]) -> pd.DataFrame:
    fit = tiered.loc[tiered['partition'].eq('fit'), features]
    vl, deg = ut.variance_loss(fit)
    assert set(deg) == status['degvar'], \
        f"recomputed degenerate set differs - extra {sorted(set(deg) - status['degvar'])}, missing {sorted(status['degvar'] - set(deg))}"

    vl = vl.set_index('variable')
    max_kept = vl.loc[sorted(status['retained']), 'variance_loss'].max()
    min_cut = vl.loc[sorted(status['cutoff']), 'variance_loss'].min()
    assert max_kept < min_cut, f"VL does not separate retained (max {max_kept}) from cutoff (min {min_cut})"

    stored = pd.read_parquet(con.PRE_PROS_CONTS).set_index('variable')
    rec = vl.loc[stored.index]
    for col, ref in (('mean', rec['mean']), ('median', rec['median']), ('std', np.sqrt(rec['variance']))):
        assert np.allclose(stored[col], ref, rtol=con.INVARIANT_RTOL, atol=0), \
            f"stored {col} not reproduced on the fit partition - constants and split have drifted apart"
    print(f"    [+++] constants reproduced on fit - {len(stored)} features, VL bound retained {max_kept:.5f} < cutoff {min_cut:.5f}")
    return vl.assign(max_vl_retained=max_kept, min_vl_cutoff=min_cut)


def feature_funnel(wsvar: pd.DataFrame, features: list[str], status: dict[str, set], vl: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([
        {'step': 'candidates', 'role': 'stage', 'n_features': len(wsvar), 'vl_bound': np.nan},
        {'step': 'not_in_panel', 'role': 'dropped', 'n_features': len(wsvar) - len(features), 'vl_bound': np.nan},
        {'step': 'panel_features', 'role': 'stage', 'n_features': len(features), 'vl_bound': np.nan},
        {'step': 'degenerate_variance', 'role': 'dropped', 'n_features': len(status['degvar']), 'vl_bound': np.nan},
        {'step': 'vl_cutoff', 'role': 'dropped', 'n_features': len(status['cutoff']), 'vl_bound': vl['min_vl_cutoff'].iloc[0]},
        {'step': 'retained', 'role': 'stage', 'n_features': len(status['retained']), 'vl_bound': vl['max_vl_retained'].iloc[0]},
    ])
    assert out.loc[0, 'n_features'] - out.loc[1, 'n_features'] == out.loc[2, 'n_features']
    assert out.loc[2, 'n_features'] - out.loc[3:4, 'n_features'].sum() == out.loc[5, 'n_features']
    return out


def feature_constants(wsvar: pd.DataFrame, tiered: pd.DataFrame, status: dict[str, set], vl: pd.DataFrame) -> pd.DataFrame:
    kept = sorted(status['retained'])
    stored = pd.read_parquet(con.PRE_PROS_CONTS).set_index('variable').loc[kept, ['mean', 'median', 'std']]
    out = stored.join(vl.loc[kept, ['variance_loss', 'n_observed']])
    for p, mask in (('fit', tiered['partition'].eq('fit')), ('val', tiered['partition'].eq('val')),
                    ('train', tiered['partition'].isin(['fit', 'val'])), ('test', tiered['partition'].eq('test'))):
        out[f'nan_share_{p}'] = tiered.loc[mask, kept].isna().mean()
    out = out.join(wsvar.set_index('variable')['name'], how='left')
    assert len(out) == len(kept) and out['name'].notna().all()
    out = out.rename_axis('variable').reset_index()
    return out[['variable', 'name'] + [c for c in out.columns if c not in ('variable', 'name')]].sort_values('variance_loss')


def imputation_table(tiered: pd.DataFrame, status: dict[str, set]) -> pd.DataFrame:
    kept = sorted(status['retained'])
    frame = tiered[['lvl3permid', 'tier', 'partition']].copy()
    frame['row_imputed_share'] = tiered[kept].isna().mean(axis=1)
    train = frame.loc[frame['partition'].isin(['fit', 'val'])].assign(partition='train')
    long = pd.concat([frame, train], ignore_index=True)

    def _stats(g: pd.DataFrame) -> pd.Series:
        s = g['row_imputed_share']
        return pd.Series({'rows': len(g), 'imputed_cell_share': s.mean(), 'row_q25': s.quantile(0.25),
                          'row_median': s.median(), 'row_q75': s.quantile(0.75), 'row_max': s.max(),
                          'rows_fully_observed': int(s.eq(0).sum())})

    glob = long.groupby('partition').apply(_stats, include_groups=False).reset_index().assign(scope='all', lvl3permid=pd.NA)
    reg = long.groupby(['lvl3permid', 'partition']).apply(_stats, include_groups=False).reset_index().assign(scope='region')
    for p, g in reg.groupby('partition'):
        assert g['rows'].sum() == glob.loc[glob['partition'].eq(p), 'rows'].iloc[0], f"region rows do not sum for {p}"
    out = pd.concat([glob, reg], ignore_index=True)
    out = out.astype({'rows': 'int64', 'rows_fully_observed': 'int64'})
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    out['tier'] = out['lvl3permid'].map(lambda r: pd.NA if pd.isna(r) else _tier(int(r)))
    return out[['scope', 'lvl3permid', 'tier', 'partition', 'rows', 'imputed_cell_share', 'row_q25', 'row_median',
                'row_q75', 'row_max', 'rows_fully_observed']]


def plot_imputation(imp: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    reg = imp.loc[imp['scope'].eq('region') & imp['partition'].isin(['train', 'test'])]
    wide = reg.pivot(index='lvl3permid', columns='partition', values='imputed_cell_share')
    n_train = reg.loc[reg['partition'].eq('train')].set_index('lvl3permid')['rows']
    wide = wide.loc[n_train.sort_values(ascending=True).index]
    y = np.arange(len(wide))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y + h / 2, wide['train'], height=h, color=PART_COLORS['train'], label='train (fit + val)')
    ax.barh(y - h / 2, wide['test'], height=h, color=PART_COLORS['test'], label='test')
    glob = imp.loc[imp['scope'].eq('all') & imp['partition'].eq('train'), 'imputed_cell_share'].iloc[0]
    ax.axvline(glob, color="#d9534f", linestyle=":", linewidth=1.5, label=f"train, all regions: {glob:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_label(r)}  (n={n_train[r]:,})".replace(",", ".") for r in wide.index])
    ax.set_title("Median-Imputed Cells per Region\nShare of Retained Feature Cells", pad=15, fontweight='bold')
    ax.set_xlabel("Imputed cell share", labelpad=10)
    ax.set_ylabel("Level 3 PermID (train rows)")
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.grid(False, axis='y')
    _style(ax)
    _save(fig, con.VIZ_DESC_IMPUTATION)


if __name__ == '__main__':
    print(f"\n[°°°] building feature descriptives [°°°]\n")
    wsvar, tiered, features, status = load_inputs()

    vl = reproduce_constants(tiered, features, status)
    funnel = feature_funnel(wsvar, features, status, vl)
    constants = feature_constants(wsvar, tiered, status, vl)
    imputation = imputation_table(tiered, status)
    print(f"    [+++] tables built")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    for tab, path in ((funnel, con.TAB_DESC_FEATURE_FUNNEL), (constants, con.TAB_DESC_FEATURE_CONSTANTS),
                      (imputation, con.TAB_DESC_IMPUTATION)):
        tab.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_imputation(imputation)

    print(f"\n    [+++] feature funnel\n{funnel.to_string(index=False)}")
    print(f"\n[°°°] feature descriptives complete [°°°]")