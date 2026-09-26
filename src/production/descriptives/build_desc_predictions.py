##################################################
'''
src.production.descriptives.build_desc_predictions

input:      results_*.yaml via assemble_results.load_manifests, depletion_counts.parquet,
            predictions_*.parquet in PRED_DIR (one per manifest, same stem)
purpose:    describe the stage-4 predictions of every run of record per region:
            prediction level and spread against the target, residual bias
            (y_pred - y_true, positive = over-prediction) and residual spread,
            per draw and aggregated per depletion level
output:     desc_prediction_by_region.csv, desc_prediction_levels.csv,
            residual_bias_by_region.png, prediction_dispersion.png

'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config as con
from src.production.from_final_panel.assemble_results import (load_manifests, attach_realised_n, KEY,
                                                              MODEL_COLORS, MODEL_LABELS)

UNDEPL = 'undepl'
COLS = ['y_pred', 'esg_combined_score', 'orgpermid', 'year', 'lvl3permid']
TIERS = {'tier1': con.TIER1_REGS, 'tier2': con.TIER2_REGS}
STATS = ['y_true_mean', 'y_true_std', 'y_pred_mean', 'y_pred_std', 'y_pred_min', 'y_pred_max',
         'dispersion_ratio', 'mean_residual', 'residual_std', 'corr']


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


def _pred_path(manifest_name: str):
    assert manifest_name.startswith('results_') and manifest_name.endswith('.yaml'), f"unexpected manifest name {manifest_name}"
    return con.PRED_DIR / f"predictions_{manifest_name[len('results_'):-len('.yaml')]}.parquet"


def _stats(g: pd.DataFrame) -> dict:
    y, p = g['esg_combined_score'].to_numpy(), g['y_pred'].to_numpy()
    r = p - y
    return {'n': len(g), 'y_true_mean': y.mean(), 'y_true_std': y.std(), 'y_pred_mean': p.mean(), 'y_pred_std': p.std(),
            'y_pred_min': p.min(), 'y_pred_max': p.max(), 'dispersion_ratio': p.std() / y.std(),
            'mean_residual': r.mean(), 'residual_std': r.std(), 'corr': np.corrcoef(y, p)[0, 1],
            'ssr': float(np.sum(r ** 2))}


def load_predictions(runs: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    paths = {row['file']: _pred_path(row['file']) for _, row in runs.iterrows()}
    missing = sorted(p.name for p in paths.values() if not p.exists())
    assert not missing, f"{len(missing)} of {len(paths)} prediction files missing in {con.PRED_DIR}:\n  " + "\n  ".join(missing)

    in_tier = set([*con.TIER1_REGS, *con.TIER2_REGS])
    ref, records = None, []
    for _, run in runs.iterrows():
        pred = pd.read_parquet(paths[run['file']])
        name = paths[run['file']].name
        assert set(COLS) <= set(pred.columns), f"{name}: columns {list(pred.columns)}"
        pred = pred[COLS]
        assert pred.notna().all().all(), f"{name}: NaN in prediction table"
        assert not pred.duplicated(['orgpermid', 'year']).any(), f"{name}: duplicate (orgpermid, year)"
        assert set(pred['lvl3permid']) == in_tier, f"{name}: regions {sorted(set(pred['lvl3permid']))}"

        pred = pred.sort_values(['orgpermid', 'year']).reset_index(drop=True)
        if ref is None:
            ref = pred[['orgpermid', 'year', 'esg_combined_score', 'lvl3permid']]
        else:
            for col in ('orgpermid', 'year', 'lvl3permid', 'esg_combined_score'):
                assert np.array_equal(pred[col].to_numpy(), ref[col].to_numpy()), f"{name}: {col} differs from the first test set"

        key = {k: run[k] for k in KEY}
        man = regions.loc[(regions[KEY] == pd.Series(key)).all(axis=1)].set_index('region')
        assert set(man.index) == set(con.TIER1_REGS), f"{name}: manifest regions {sorted(man.index)}"

        t1 = pred['lvl3permid'].isin(con.TIER1_REGS)
        records.append({**key, 'scope': 'all', 'lvl3permid': pd.NA, **_stats(pred)})
        records.append({**key, 'scope': 'tier1', 'lvl3permid': pd.NA, **_stats(pred.loc[t1])})
        for region, g in pred.groupby('lvl3permid'):
            s = _stats(g)
            if region in con.TIER1_REGS:
                m = man.loc[region]
                assert s['n'] == int(m['n_r']), f"{name} {region}: n {s['n']} vs manifest {m['n_r']}"
                assert np.isclose(s['ssr'], m['ssr_r'], rtol=con.INVARIANT_RTOL, atol=0), \
                    f"{name} {region}: SSR {s['ssr']} vs manifest {m['ssr_r']} - file and manifest are not one run"
                assert np.isclose(s['ssr'], s['n'] * (s['mean_residual'] ** 2 + s['residual_std'] ** 2),
                                  rtol=con.INVARIANT_RTOL, atol=0), f"{name} {region}: bias-variance decomposition fails"
            records.append({**key, 'scope': 'region', 'lvl3permid': region, **s})
    print(f"    [+++] {len(runs)} prediction files validated against their manifests - one test set of {len(ref)} rows")

    out = pd.DataFrame(records).drop(columns='ssr')
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    out['tier'] = out['lvl3permid'].map(lambda r: pd.NA if pd.isna(r) else next(t for t, regs in TIERS.items() if r in regs))
    return out


def level_table(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ['model', 'configuration', 'scope', 'lvl3permid', 'level']
    out = pred.groupby(keys, dropna=False).agg(
        n_draws=('condition', 'size'), anchor_rows_mean=('anchor_rows_retained', 'mean'),
        **{f"{c}_{s}": (c, s) for c in ['y_pred_mean', 'dispersion_ratio', 'mean_residual', 'residual_std', 'corr']
           for s in ('mean', 'min', 'max')}).reset_index()
    assert out['n_draws'].sum() == len(pred)
    return out.sort_values(['model', 'configuration', 'scope', 'lvl3permid', 'level'],
                           na_position='first').reset_index(drop=True)


def _curves(pred: pd.DataFrame) -> pd.DataFrame:
    has_curve = pred.loc[pred['condition'].ne(UNDEPL), ['model', 'configuration']].drop_duplicates()
    assert not has_curve['model'].duplicated().any(), "a model carries two depletion curves"
    return pred.merge(has_curve, on=['model', 'configuration'], how='inner')


def plot_residual_bias(pred: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    und = _curves(pred)
    und = und.loc[und['condition'].eq(UNDEPL) & und['scope'].eq('region') & und['tier'].eq('tier1')]
    order = und.groupby('lvl3permid')['n'].first().sort_values(ascending=True)
    models = [m for m in MODEL_COLORS if m in set(und['model'])]
    y = np.arange(len(order))
    step = 0.7 / len(models)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, m in enumerate(models):
        s = und.loc[und['model'].eq(m)].set_index('lvl3permid').loc[order.index]
        ax.scatter(s['mean_residual'], y - 0.35 + (i + 0.5) * step, color=MODEL_COLORS[m], s=24, zorder=3,
                   label=MODEL_LABELS[m])
    ax.axvline(0, color="#333333", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  (n={n:,})".replace(",", ".") for r, n in order.items()])
    ax.set_title("Mean Residual per Tier-1 Region, Undepleted\n(prediction - target; positive = over-prediction)",
                 pad=15, fontweight='bold')
    ax.set_xlabel("Mean residual (ESG score, native [0,1])", labelpad=10)
    ax.set_ylabel("Level 3 PermID (test rows)")
    ax.legend(frameon=False, ncol=len(models), loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.grid(False, axis='y')
    _style(ax)
    _save(fig, con.VIZ_DESC_RESID_BIAS)


def plot_dispersion(pred: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    cur = _curves(pred)
    cur = cur.loc[cur['scope'].eq('region') & cur['tier'].eq('tier1')]
    anchor = cur.loc[cur['lvl3permid'].eq(con.ANCHOR_REGION)]
    comp = (cur.loc[cur['lvl3permid'].ne(con.ANCHOR_REGION)]
               .groupby(KEY + ['level', 'anchor_rows_retained'], dropna=False)['dispersion_ratio'].mean().reset_index())
    assert len(comp) == len(anchor)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, frame, title in ((axes[0], anchor, f"Anchor {con.ANCHOR_REGION}"),
                             (axes[1], comp, "Comparators (unweighted mean, 12 Tier-1 regions)")):
        for m in [m for m in MODEL_COLORS if m in set(frame['model'])]:
            f = frame.loc[frame['model'].eq(m)]
            is_u = f['condition'].eq(UNDEPL)
            means = f.groupby('level', dropna=False).agg(x=('anchor_rows_retained', 'mean'),
                                                         y=('dispersion_ratio', 'mean')).sort_values('x')
            ax.plot(means['x'], means['y'], color=MODEL_COLORS[m], lw=1.6, marker='o', ms=4, label=MODEL_LABELS[m], zorder=3)
            ax.scatter(f.loc[~is_u, 'anchor_rows_retained'], f.loc[~is_u, 'dispersion_ratio'], color=MODEL_COLORS[m],
                       s=14, alpha=0.4, edgecolor='none', zorder=2)
            ax.scatter(f.loc[is_u, 'anchor_rows_retained'], f.loc[is_u, 'dispersion_ratio'], marker='D', s=40,
                       facecolor='white', edgecolor=MODEL_COLORS[m], lw=1.4, zorder=4)
        ax.set_xscale('log')
        ticks = sorted(frame['level'].dropna().astype(int).unique()) + [int(frame['anchor_rows_retained'].max())]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(title, pad=10, fontweight='bold')
        ax.set_xlabel("Realised anchor rows, train+val (log scale)", labelpad=10)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    axes[0].set_ylabel("Dispersion ratio  std(prediction) / std(target)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45),
                plt.Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec="#888888")]
    labels += ["individual draw", "undepleted (full anchor)"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Prediction Dispersion under Anchor Depletion", fontweight='bold')
    _save(fig, con.VIZ_DESC_DISPERSION)


if __name__ == '__main__':
    print(f"\n[°°°] building prediction descriptives [°°°]\n")
    runs, regions = load_manifests()
    print(f"    [+++] manifests loaded - {len(runs)} runs")

    pred = attach_realised_n(load_predictions(runs, regions))
    assert pred['anchor_rows_retained'].notna().all(), "run without a realised anchor N"
    levels = level_table(pred)
    print(f"    [+++] tables built - {len(pred)} run x scope rows")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    order = KEY + ['level', 'draw', 'anchor_rows_retained', 'anchor_entities_retained', 'scope', 'lvl3permid', 'tier', 'n'] + STATS
    pred = pred[order].sort_values(['model', 'configuration', 'level', 'draw', 'scope', 'lvl3permid'], na_position='first')
    for tab, path in ((pred, con.TAB_DESC_PRED_REGION), (levels, con.TAB_DESC_PRED_LEVELS)):
        tab.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_residual_bias(pred)
    plot_dispersion(pred)

    und = pred.loc[pred['condition'].eq(UNDEPL) & pred['scope'].eq('tier1')]
    print(f"\n    [+++] undepleted, tier-1 pooled\n{und[['model', 'configuration', 'n', 'y_pred_mean', 'dispersion_ratio', 'mean_residual', 'corr']].to_string(index=False)}")
    print(f"\n[°°°] prediction descriptives complete [°°°]")