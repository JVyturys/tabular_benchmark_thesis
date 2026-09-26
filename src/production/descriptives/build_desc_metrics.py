##################################################
'''
src.production.descriptives.build_desc_metrics

input:      results_*.yaml via assemble_results.load_manifests, depletion_counts.parquet
purpose:    describe the reported performance metrics of every run of record: all seven
            aggregate metrics and the tier-1 region metrics per depletion level, within-condition
            model ranks, and region skill against region size (undepleted)
output:     desc_metric_levels.csv, desc_metric_region_levels.csv, desc_metric_ranks.csv,
            desc_metric_region_size.csv, metric_curves.png, region_delta_r_sq_deepest.png,
            region_r_sq_vs_size.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config as con
from src.production.from_final_panel.assemble_results import (load_manifests, attach_realised_n, KEY, CURVE_KEYS,
                                                              MODEL_COLORS, MODEL_LABELS)

UNDEPL = 'undepl'
METRICS = {'pooled R2': 'pooled_r_sq', 'average R2 (global denominator)': 'macro_r_sq',
           'regional bias gap': 'gap_r_sq', 'pooled_RMSE': 'pooled_rmse', 'average_RMSE': 'macro_rmse',
           'average_rmse_sq': 'macro_rmse_q', 'regional bias gap rmse': 'gap_rmse'}
HIGHER_BETTER = {'pooled_r_sq', 'macro_r_sq'}
LABELS = {'pooled_r_sq': 'Pooled R²', 'macro_r_sq': 'Macro R²', 'gap_r_sq': 'Gap R² (pooled - macro)',
          'pooled_rmse': 'Pooled RMSE', 'macro_rmse_q': 'Macro RMSE (quadratic)', 'gap_rmse': 'Gap RMSE (macro_q - pooled)'}
REGION_METRICS = ['r_sq', 'rmse_r', 'r_sq_reg']


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


def _log_x(ax, levels: pd.Series, full: int) -> None:
    ax.set_xscale('log')
    ax.set_xticks(sorted(levels.dropna().astype(int).unique()) + [int(full)])
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, regions = load_manifests()
    runs = runs.rename(columns=METRICS)
    assert set(METRICS.values()) <= set(runs.columns), "manifest metric keys changed"

    reg = regions.loc[regions['region'].isin(con.TIER1_REGS)]
    per_run = reg.groupby(KEY).apply(lambda g: pd.Series({
        'macro_r_sq': g['r_sq'].mean(), 'macro_rmse': g['rmse_r'].mean(),
        'macro_rmse_q': np.sqrt((g['rmse_r'] ** 2).mean()),
        'pooled_rmse': np.sqrt(g['ssr_r'].sum() / g['n_r'].sum()),
        'pooled_r_sq': 1 - g['ssr_r'].sum() / (g['n_r'].sum() * ((g['ssr_r'] / g['n_r']) / (1 - g['r_sq'])).iloc[0]),
    }), include_groups=False)
    chk = runs.set_index(KEY).join(per_run, rsuffix='_re', how='left', validate='one_to_one')
    assert len(chk) == len(runs)
    for m in per_run.columns:
        assert np.allclose(chk[m], chk[f"{m}_re"], rtol=con.INVARIANT_RTOL, atol=0), f"{m} does not follow from the region metrics"
    assert np.allclose(chk['gap_r_sq'], chk['pooled_r_sq'] - chk['macro_r_sq'], rtol=con.INVARIANT_RTOL, atol=0), "gap R2 identity"
    assert np.allclose(chk['gap_rmse'], chk['macro_rmse_q'] - chk['pooled_rmse'], rtol=con.INVARIANT_RTOL, atol=0), "gap RMSE identity"
    print(f"    [+++] report_metrics identities hold on all {len(runs)} manifests")

    runs = attach_realised_n(runs.drop(columns='file'))
    regions = attach_realised_n(regions)
    assert runs['anchor_rows_retained'].notna().all() and regions['anchor_rows_retained'].notna().all()
    return runs, regions


def _with_undepl(frame: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    base = frame.loc[frame['condition'].eq(UNDEPL), keys + cols].rename(columns={c: f"{c}_undepl" for c in cols})
    assert not base.duplicated(keys).any()
    out = frame.merge(base, on=keys, how='left', validate='many_to_one')
    assert len(out) == len(frame) and out[[f"{c}_undepl" for c in cols]].notna().all().all()
    for c in cols:
        out[f"{c}_delta"] = out[c] - out[f"{c}_undepl"]
    return out


def level_table(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS.values())
    frame = _with_undepl(runs, CURVE_KEYS, metrics)
    out = frame.groupby(CURVE_KEYS + ['level'], dropna=False).agg(
        n_draws=('condition', 'size'), anchor_rows_mean=('anchor_rows_retained', 'mean'),
        **{f"{m}_{s}": (m, s) for m in metrics for s in ('mean', 'min', 'max')},
        **{f"{m}_delta_mean": (f"{m}_delta", 'mean') for m in metrics}).reset_index()
    assert out['n_draws'].sum() == len(runs)
    return out.sort_values(CURVE_KEYS + ['anchor_rows_mean'], ascending=[True, True, False]).reset_index(drop=True)


def region_level_table(regions: pd.DataFrame) -> pd.DataFrame:
    keys = CURVE_KEYS + ['region']
    frame = _with_undepl(regions, keys, REGION_METRICS)
    out = frame.groupby(keys + ['level'], dropna=False).agg(
        n_draws=('condition', 'size'), n_r=('n_r', 'first'), anchor_rows_mean=('anchor_rows_retained', 'mean'),
        **{f"{m}_{s}": (m, s) for m in REGION_METRICS for s in ('mean', 'min', 'max')},
        **{f"{m}_delta_mean": (f"{m}_delta", 'mean') for m in REGION_METRICS}).reset_index()
    assert out['n_draws'].sum() == len(regions)
    out['is_anchor'] = out['region'].eq(con.ANCHOR_REGION)
    return out.sort_values(keys + ['anchor_rows_mean'], ascending=[True, True, True, False]).reset_index(drop=True)


def _curve_runs(runs: pd.DataFrame) -> pd.DataFrame:
    has_curve = runs.loc[runs['condition'].ne(UNDEPL), CURVE_KEYS].drop_duplicates()
    assert not has_curve['model'].duplicated().any(), "a model carries two depletion curves"
    return runs.merge(has_curve, on=CURVE_KEYS, how='inner')


def rank_table(runs: pd.DataFrame) -> pd.DataFrame:
    cur = _curve_runs(runs)
    n_models = cur['model'].nunique()
    assert cur.groupby('condition')['model'].nunique().eq(n_models).all(), "a condition lacks a curve model"
    parts = []
    for m in METRICS.values():
        ranked = cur.assign(rank=cur.groupby('condition')[m].rank(ascending=m not in HIGHER_BETTER, method='min'))
        parts.append(ranked.groupby(['level', 'model'], dropna=False).agg(
            n_conditions=('condition', 'size'), mean_rank=('rank', 'mean'),
            best_share=('rank', lambda r: float((r == 1).mean()))).reset_index().assign(metric=m))
    out = pd.concat(parts, ignore_index=True)
    return out[['metric', 'level', 'model', 'n_conditions', 'mean_rank', 'best_share']].sort_values(
        ['metric', 'level', 'mean_rank'], na_position='first').reset_index(drop=True)


def region_size_table(regions: pd.DataFrame) -> pd.DataFrame:
    und = regions.loc[regions['condition'].eq(UNDEPL), CURVE_KEYS + ['region', 'n_r', 'r_sq', 'rmse_r']].copy()
    rho = und.groupby(CURVE_KEYS).apply(lambda g: pd.Series({
        'spearman_n_r_sq': g['n_r'].corr(g['r_sq'], method='spearman'),
        'spearman_n_rmse': g['n_r'].corr(g['rmse_r'], method='spearman')}), include_groups=False).reset_index()
    und['r_sq_rank'] = und.groupby(CURVE_KEYS)['r_sq'].rank(ascending=False, method='min')
    und['n_rank'] = und.groupby(CURVE_KEYS)['n_r'].rank(ascending=False, method='min')
    out = und.merge(rho, on=CURVE_KEYS, how='left', validate='many_to_one')
    assert len(out) == len(und)
    return out.sort_values(CURVE_KEYS + ['n_r'], ascending=[True, True, False]).reset_index(drop=True)


def plot_metric_curves(runs: pd.DataFrame, levels: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    cur = _curve_runs(runs)
    lev = levels.merge(cur[CURVE_KEYS].drop_duplicates(), on=CURVE_KEYS, how='inner')
    full = int(cur['anchor_rows_retained'].max())
    panels = ['pooled_r_sq', 'macro_r_sq', 'gap_r_sq', 'pooled_rmse', 'macro_rmse_q', 'gap_rmse']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.flat, panels):
        for model in [k for k in MODEL_COLORS if k in set(cur['model'])]:
            p, s = cur.loc[cur['model'].eq(model)], lev.loc[lev['model'].eq(model)].sort_values('anchor_rows_mean')
            is_u = p['condition'].eq(UNDEPL)
            ax.plot(s['anchor_rows_mean'], s[f"{m}_mean"], color=MODEL_COLORS[model], lw=1.5, marker='o', ms=3.5,
                    label=MODEL_LABELS[model], zorder=3)
            ax.scatter(p.loc[~is_u, 'anchor_rows_retained'], p.loc[~is_u, m], color=MODEL_COLORS[model], s=12,
                       alpha=0.4, edgecolor='none', zorder=2)
            ax.scatter(p.loc[is_u, 'anchor_rows_retained'], p.loc[is_u, m], marker='D', s=34, facecolor='white',
                       edgecolor=MODEL_COLORS[model], lw=1.3, zorder=4)
        if m.startswith('gap'):
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
        _log_x(ax, cur['level'], full)
        ax.set_title(LABELS[m], fontweight='bold', pad=8)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Realised anchor rows, train+val (log scale)", labelpad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45),
                plt.Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec="#888888")]
    labels += ["individual draw", "undepleted (full anchor)"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Tier-1 Performance Metrics under Anchor Depletion ({con.ANCHOR_REGION})", fontweight='bold', fontsize=14)
    _save(fig, con.VIZ_DESC_METRIC_CURVES)


def plot_region_delta(region_levels: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    rl = region_levels.merge(region_levels.loc[region_levels['level'].notna(), CURVE_KEYS].drop_duplicates(),
                             on=CURVE_KEYS, how='inner')
    deepest = rl['level'].min()
    d = rl.loc[rl['level'].eq(deepest)]
    order = d.groupby('region')['n_r'].first().sort_values(ascending=True)
    models = [m for m in MODEL_COLORS if m in set(d['model'])]
    y, step = np.arange(len(order)), 0.7 / len(models)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, m in enumerate(models):
        s = d.loc[d['model'].eq(m)].set_index('region').loc[order.index]
        ax.scatter(s['r_sq_delta_mean'], y - 0.35 + (i + 0.5) * step, color=MODEL_COLORS[m], s=24, zorder=3,
                   label=MODEL_LABELS[m])
    ax.axvline(0, color="#333333", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  (n={n:,})".replace(",", ".") for r, n in order.items()])
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith(str(con.ANCHOR_REGION)):
            lab.set_color("#d9534f")
            lab.set_fontweight('bold')
    ax.set_title(f"Change in Region R² at the Deepest Level (L{int(deepest)}, mean over draws)\n"
                 f"depleted - undepleted; anchor {con.ANCHOR_REGION} in red", pad=15, fontweight='bold')
    ax.set_xlabel("Δ R² (global denominator)", labelpad=10)
    ax.set_ylabel("Level 3 PermID (test rows)")
    ax.legend(frameon=False, ncol=len(models), loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.grid(False, axis='y')
    _style(ax)
    _save(fig, con.VIZ_DESC_REGION_DELTA)


def plot_region_size(size: pd.DataFrame, runs: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    s = size.merge(_curve_runs(runs)[CURVE_KEYS].drop_duplicates(), on=CURVE_KEYS, how='inner')
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in [k for k in MODEL_COLORS if k in set(s['model'])]:
        g = s.loc[s['model'].eq(m)].sort_values('n_r')
        rho = g['spearman_n_r_sq'].iloc[0]
        ax.scatter(g['n_r'], g['r_sq'], color=MODEL_COLORS[m], s=26, zorder=3, label=f"{MODEL_LABELS[m]} (ρ = {rho:+.2f})")
    for region in s['region'].unique():
        g = s.loc[s['region'].eq(region)]
        ax.text(g['n_r'].iloc[0], g['r_sq'].max(), f" {region}", fontsize=7.5, rotation=90, va='bottom', ha='center',
                color="#d9534f" if region in (con.ANCHOR_REGION, con.EQUAL_N_REGION) else "#888888")
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title("Region R² against Region Size, Undepleted\nρ: Spearman rank correlation, descriptive",
                 pad=15, fontweight='bold')
    ax.set_xlabel("Region test rows (log scale)", labelpad=10)
    ax.set_ylabel("R² (global denominator)")
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    _style(ax)
    _save(fig, con.VIZ_DESC_REGION_SIZE)


if __name__ == '__main__':
    print(f"\n[°°°] building performance-metric descriptives [°°°]\n")
    runs, regions = load_runs()

    levels = level_table(runs)
    region_levels = region_level_table(regions)
    ranks = rank_table(runs)
    size = region_size_table(regions)
    print(f"    [+++] tables built")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    for tab, path in ((levels, con.TAB_DESC_METRIC_LEVELS), (region_levels, con.TAB_DESC_METRIC_REGION_LEVELS),
                      (ranks, con.TAB_DESC_METRIC_RANKS), (size, con.TAB_DESC_METRIC_REGION_SIZE)):
        tab.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_metric_curves(runs, levels)
    plot_region_delta(region_levels)
    plot_region_size(size, runs)

    print(f"\n    [+++] size monotonicity (undepleted)\n"
          f"{size.groupby(CURVE_KEYS)[['spearman_n_r_sq', 'spearman_n_rmse']].first().to_string()}")
    print(f"\n[°°°] performance-metric descriptives complete [°°°]")