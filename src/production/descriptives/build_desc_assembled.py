##################################################
'''
src.production.descriptives.build_desc_assembled

input:      aggregate_undepl.csv, per_region_undepl.csv, did_r_sq.csv, gap_curve.csv,
            headline_changes.csv (assemble_results), results_*.yaml, depletion_counts.parquet
purpose:    figures on the assembled results: pooled vs macro R2 per model, per-region R2,
            share of the regional bias gap removed by depletion, the difference-in-differences
            decomposition, and the headline changes at the deepest level
output:     pooled_vs_macro_r_sq.png, region_r_sq_heatmap.png, gap_share_removed.png,
            did_decomposition.png, headline_changes.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import config as con
from src.production.from_final_panel.assemble_results import (load_manifests, attach_realised_n, did_table, gap_curve,
                                                              headline_changes, CURVE_KEYS, REGION_METRICS,
                                                              MODEL_COLORS, MODEL_LABELS)

HEADLINE_LABELS = {'pooled_RMSE': 'Pooled RMSE', 'pooled R2': 'Pooled R²', 'average_RMSE': 'Macro RMSE',
                   'average R2 (global denominator)': 'Macro R²', 'average_rmse_sq': 'Macro RMSE (q)',
                   'regional bias gap': 'Gap R²', 'regional bias gap rmse': 'Gap RMSE'}


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


def _run_label(model: str, configuration: str, multi: set) -> str:
    return f"{MODEL_LABELS[model]} ({configuration})" if model in multi else MODEL_LABELS[model]


def _assert_current(disk: pd.DataFrame, fresh: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    assert set(disk.columns) == set(fresh.columns), \
        f"{name}: columns differ - disk-only {sorted(set(disk.columns) - set(fresh.columns))}, fresh-only {sorted(set(fresh.columns) - set(disk.columns))}"
    assert len(disk) == len(fresh), f"{name}: {len(disk)} rows on disk, {len(fresh)} from the current manifests - rerun assemble_results"
    d = disk.sort_values(keys, na_position='first').reset_index(drop=True)
    f = fresh[disk.columns].sort_values(keys, na_position='first').reset_index(drop=True)
    for c in disk.columns:
        a, b = d[c], f[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a.astype(float), b.astype(float), rtol=con.INVARIANT_RTOL, atol=0, equal_nan=True), \
                f"{name}: column {c} differs from the current manifests - rerun assemble_results"
        else:
            assert (a.astype(str) == b.astype(str)).all(), f"{name}: column {c} differs from the current manifests - rerun assemble_results"
    return d


def load_tables() -> dict[str, pd.DataFrame]:
    runs, regions = load_manifests()
    runs = attach_realised_n(runs)
    did = attach_realised_n(did_table(regions))
    _, summary = gap_curve(runs)
    headline = headline_changes(runs)
    is_undepl = runs['condition'].eq('undepl')
    fresh = {
        'aggregate': (runs.loc[is_undepl].drop(columns=['condition', 'level', 'draw']), CURVE_KEYS, con.TAB_AGGREGATE),
        'per_region': (regions.loc[regions['condition'].eq('undepl'), CURVE_KEYS + ['region'] + REGION_METRICS],
                       CURVE_KEYS + ['region'], con.TAB_PER_REGION),
        'did': (did, ['model', 'configuration', 'condition'], con.TAB_DID),
        'gap_curve': (summary, CURVE_KEYS + ['level'], con.TAB_GAP_CURVE),
        'headline': (headline, CURVE_KEYS, con.TAB_HEADLINE),
    }
    tables = {}
    for name, (frame, keys, path) in fresh.items():
        tables[name] = _assert_current(pd.read_csv(path), frame, keys, path.name)
    print(f"    [+++] assemble_results tables current - {', '.join(f'{k} {len(v)}' for k, v in tables.items())}")

    n_r = regions.loc[regions['condition'].eq('undepl')].groupby('region')['n_r'].first()
    assert tables['per_region'].groupby(CURVE_KEYS).size().eq(len(con.TIER1_REGS)).all(), "per-region table is not 13 regions per run"
    assert tables['headline']['level'].nunique() == 1, "headline rows sit at different levels"
    return tables | {'n_r': n_r}


def plot_pooled_macro(agg: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    multi = set(agg.loc[agg.duplicated('model', keep=False), 'model'])
    a = agg.sort_values('pooled R2').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, r in a.iterrows():
        c = MODEL_COLORS[r['model']]
        pooled, macro = r['pooled R2'], r['average R2 (global denominator)']
        ax.plot([macro, pooled], [i, i], color=c, lw=2, alpha=0.6, zorder=2)
        ax.scatter(pooled, i, color=c, s=60, zorder=3)
        ax.scatter(macro, i, facecolor='white', edgecolor=c, lw=1.6, s=60, zorder=3)
        ax.text(max(pooled, macro) + 0.004, i, f"gap {r['regional bias gap']:+.4f}", va='center', fontsize=9, color="#555555")
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([_run_label(m, c, multi) for m, c in zip(a['model'], a['configuration'])])
    ax.set_xlim(right=ax.get_xlim()[1] + 0.03)
    handles = [plt.Line2D([], [], ls='none', marker='o', ms=8, color="#888888"),
               plt.Line2D([], [], ls='none', marker='o', ms=8, mfc='white', mec="#888888")]
    ax.legend(handles, ['pooled R² (Tier-1)', 'macro R² (13 regions, equal weight)'], frameon=False,
              ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Pooled vs Macro R², Undepleted\nregional bias gap = pooled - macro", pad=15, fontweight='bold')
    ax.set_xlabel("R² (global denominator)", labelpad=10)
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.grid(False, axis='y')
    _style(ax)
    _save(fig, con.VIZ_DESC_POOLED_MACRO)


def plot_region_heatmap(per_region: pd.DataFrame, n_r: pd.Series) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    multi = set(per_region.groupby('model')['configuration'].nunique().loc[lambda s: s > 1].index)
    p = per_region.assign(run=[_run_label(m, c, multi) for m, c in zip(per_region['model'], per_region['configuration'])])
    grid = p.pivot(index='run', columns='region', values='r_sq')
    grid = grid[n_r.sort_values(ascending=False).index]
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(grid, annot=True, fmt=".2f", annot_kws={'fontsize': 8},
                cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=min(grid.min().min(), -0.01), vmax=grid.max().max()),
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'R² (global denominator)'}, ax=ax)
    ax.grid(False)
    ax.set_xticklabels([f"{r}\nn={n_r[r]:,}".replace(",", ".") for r in grid.columns], rotation=0, fontsize=8.5)
    for lab in ax.get_xticklabels():
        if lab.get_text().startswith(str(con.ANCHOR_REGION)):
            lab.set_color("#d9534f")
            lab.set_fontweight('bold')
    ax.set_title("Tier-1 Region R², Undepleted\nregions by test rows (descending), anchor in red", pad=15, fontweight='bold')
    ax.set_xlabel("Level 3 PermID", labelpad=10)
    ax.set_ylabel("")
    _save(fig, con.VIZ_DESC_REGION_HEATMAP)


def plot_gap_share(summary: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, g, title in ((axes[0], 'gap_r_sq', 'R² form'), (axes[1], 'gap_rmse', 'RMSE form')):
        for m in [k for k in MODEL_COLORS if k in set(summary['model'])]:
            s = summary.loc[summary['model'].eq(m)].sort_values('anchor_rows_mean')
            col = f"share_removed_{g}"
            ax.fill_between(s['anchor_rows_mean'], s[f"{col}_min"], s[f"{col}_max"], color=MODEL_COLORS[m], alpha=0.15, lw=0)
            ax.plot(s['anchor_rows_mean'], s[f"{col}_mean"], color=MODEL_COLORS[m], lw=1.6, marker='o', ms=4,
                    label=MODEL_LABELS[m])
        ax.axhline(0, color="#333333", lw=0.9)
        ax.axhline(1, color="#d9534f", lw=1.2, ls=':')
        ax.set_xscale('log')
        ax.set_xticks(sorted(summary['level'].dropna().astype(int).unique()) + [int(summary['anchor_rows_mean'].max())])
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel("Mean realised anchor rows, train+val (log scale)", labelpad=10)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    axes[0].set_ylabel("Share of undepleted gap removed")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.2), plt.Line2D([], [], color="#d9534f", ls=':')]
    labels += ["min-max over draws", "gap fully closed"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Share of the Regional Bias Gap Removed by Anchor Depletion", fontweight='bold', fontsize=13)
    _save(fig, con.VIZ_DESC_GAP_SHARE)


def plot_did(did: pd.DataFrame, full_rows: int) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    panels = (('d_anchor', f"Anchor {con.ANCHOR_REGION}"), ('d_comp', "Comparators (mean of 12)"), ('did', "DiD = anchor - comparators"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        for m in [k for k in MODEL_COLORS if k in set(did['model'])]:
            d = did.loc[did['model'].eq(m)]
            means = d.groupby('level').agg(x=('anchor_rows_retained', 'mean'), y=(col, 'mean')).sort_values('x')
            ax.scatter(d['anchor_rows_retained'], d[col], color=MODEL_COLORS[m], s=12, alpha=0.4, edgecolor='none', zorder=2)
            ax.plot(means['x'], means['y'], color=MODEL_COLORS[m], lw=1.6, marker='o', ms=4, label=MODEL_LABELS[m], zorder=3)
        ax.axhline(0, color="#333333", lw=0.9)
        ax.set_xscale('log')
        ax.set_xticks(sorted(did['level'].dropna().astype(int).unique()) + [full_rows])
        ax.set_xlim(right=full_rows * 1.15)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel("Realised anchor rows, train+val (log scale)", labelpad=10)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    axes[0].set_ylabel("R² loss: undepleted - depleted\n(positive = R² fell)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45)]
    labels += ["individual draw"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Difference-in-Differences Decomposition of the R² Change", fontweight='bold', fontsize=13)
    _save(fig, con.VIZ_DESC_DID)


def plot_headline(headline: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = [c for c in HEADLINE_LABELS if c in headline.columns]
    grid = headline.set_index('model')[cols].rename(columns=HEADLINE_LABELS)
    grid.index = [MODEL_LABELS[m] for m in grid.index]
    lim = float(np.nanmax(np.abs(grid.to_numpy())))
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(grid, annot=grid.map(lambda v: f"{v:+.1%}"), fmt="", annot_kws={'fontsize': 9}, cmap='RdBu_r',
                norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'relative change', 'format': mticker.PercentFormatter(1.0)}, ax=ax)
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    level, n_draws = int(headline['level'].iloc[0]), int(headline['n_draws'].iloc[0])
    ax.set_title(f"Relative Change at the Deepest Level (L{level}, mean of {n_draws} draws) vs Undepleted\n"
                 f"sign follows the metric: RMSE up = worse, R² up = better; gap columns = minus the share removed",
                 pad=15, fontweight='bold')
    ax.set_ylabel("")
    _save(fig, con.VIZ_DESC_HEADLINE)


if __name__ == '__main__':
    print(f"\n[°°°] building figures from the assembled results [°°°]\n")
    t = load_tables()

    plot_pooled_macro(t['aggregate'])
    plot_region_heatmap(t['per_region'], t['n_r'])
    plot_gap_share(t['gap_curve'])
    plot_did(t['did'], int(t['aggregate']['anchor_rows_retained'].max()))
    plot_headline(t['headline'])

    print(f"\n[°°°] assembled-result figures complete [°°°]")