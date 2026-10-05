##################################################
'''
src.production.descriptives.build_desc_assembled

input:      aggregate_undepl.csv, per_region_undepl.csv, did_r_sq.csv, gap_curve.csv,
            headline_changes.csv, gap_contributions.csv (assemble_results), results_*.yaml,
            depletion_counts.parquet
purpose:    pooled vs macro coefficient (R2_P, R2_M, gap G) per model, per-region R2_glob, the share S of
            the gap removed by depletion, the difference-in-differences decomposition (d_A, d_comp,
            D_A, D_N), the relative change of the gap metrics R2_P, R2_M, G at the deepest level, and
            the regional contributions k_r * Delta R2_glob / Delta G to the gap change at the deepest level
output:     pooled_vs_macro_r_sq.png, region_r_sq_heatmap.png, gap_share_removed.png,
            did_decomposition.png, headline_changes.png, gap_contributions.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import textwrap
from matplotlib.colors import TwoSlopeNorm
import config as con
from src.production.from_final_panel.assemble_results import (load_manifests, attach_realised_n, did_table, gap_curve,
                                                              headline_changes, gap_contributions, CURVE_KEYS, KEY,
                                                              REGION_METRICS, MODEL_COLORS, MODEL_LABELS)
from src.production.descriptives.build_desc_metrics import _region_panels

HEADLINE_LABELS = {'pooled R2': r'$R^2_P$', 'average R2 (global denominator)': r'$R^2_M$',
                   'regional bias gap': r'$G = R^2_P - R^2_M$'}
R2_GLOB = r'$R^{2,\mathrm{glob}}_r$'
X_ANCHOR = "Realised anchor training rows\n(fit + validation, log scale)"

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
    return f"{MODEL_LABELS[model]} ({'untuned' if configuration == 'baseline' else configuration})" if model in multi else MODEL_LABELS[model]

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
        'contrib': (attach_realised_n(gap_contributions(runs, regions)), KEY + ['region'], con.TAB_GAP_CONTRIB),
    }
    tables = {}
    for name, (frame, keys, path) in fresh.items():
        tables[name] = _assert_current(pd.read_csv(path), frame, keys, path.name)
    print(f"    [+++] assemble_results tables current - {', '.join(f'{k} {len(v)}' for k, v in tables.items())}")

    n_r = regions.loc[regions['condition'].eq('undepl')].groupby('region')['n_r'].first()
    assert tables['per_region'].groupby(CURVE_KEYS).size().eq(len(con.TIER1_REGS)).all(), "per-region table is not 13 regions per run"
    assert tables['headline']['level'].nunique() == 1, "headline rows sit at different levels"
    return tables | {'n_r': n_r}

def _base_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE

def plot_pooled_macro(agg: pd.DataFrame) -> None:
    _base_style()
    multi = set(agg.loc[agg.duplicated('model', keep=False), 'model'])
    a = agg.sort_values('pooled R2').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, r in a.iterrows():
        c = MODEL_COLORS[r['model']]
        pooled, macro = r['pooled R2'], r['average R2 (global denominator)']
        ax.plot([macro, pooled], [i, i], color=c, lw=2, alpha=0.6, zorder=2)
        ax.scatter(pooled, i, color=c, s=60, zorder=3)
        ax.scatter(macro, i, facecolor='white', edgecolor=c, lw=1.6, s=60, zorder=3)
        ax.text(max(pooled, macro) + 0.004, i, f"$G$ = {r['regional bias gap']:+.4f}", va='center',
                fontsize=9 * con.FONT_SCALE, color="#555555")
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([_run_label(m, c, multi) for m, c in zip(a['model'], a['configuration'])])
    ax.set_xlim(right=ax.get_xlim()[1] + 0.03)
    handles = [plt.Line2D([], [], ls='none', marker='o', ms=8, color="#888888"),
               plt.Line2D([], [], ls='none', marker='o', ms=8, mfc='white', mec="#888888")]
    ax.legend(handles, [r'$R^2_P$ (row-weighted, Tier-1)', rf'$R^2_M$ ({len(con.TIER1_REGS)} regions, equal weight)'],
              frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Pooled vs Macro Coefficient, Undepleted\n" r"performance gap $G = R^2_P - R^2_M$",
                 pad=15, fontweight='bold')
    ax.set_xlabel(r"$R^2_P$, $R^2_M$", labelpad=10)
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.grid(False, axis='y')
    _style(ax)
    _save(fig, con.VIZ_DESC_POOLED_MACRO)

def plot_region_heatmap(per_region: pd.DataFrame, n_r: pd.Series) -> None:
    _base_style()
    multi = set(per_region.groupby('model')['configuration'].nunique().loc[lambda s: s > 1].index)
    p = per_region.assign(run=[_run_label(m, c, multi) for m, c in zip(per_region['model'], per_region['configuration'])])
    # regions as rows so the full names read horizontally; rows by test rows (descending), runs by mean R2
    grid = p.pivot(index='region', columns='run', values='r_sq')
    grid = grid.loc[n_r.loc[grid.index].sort_values(ascending=False).index]
    grid = grid[grid.mean(axis=0).sort_values(ascending=False).index]
    assert grid.notna().all().all(), "missing region x run R2"
    fig, ax = plt.subplots(figsize=(max(8, 3 + 1.5 * grid.shape[1]), 7))
    sns.heatmap(grid, annot=True, fmt=".2f", annot_kws={'fontsize': 8 * con.FONT_SCALE},
                cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=min(grid.min().min(), -0.01), vmax=grid.max().max()),
                linewidths=0.5, linecolor='white', cbar_kws={'label': R2_GLOB}, ax=ax)
    ax.grid(False)
    ax.set_yticklabels([f"{con.REGION_LABELS[r]}  (n={n_r[r]:,})" for r in grid.index], rotation=0)
    ax.set_xticklabels(grid.columns, rotation=0)
    for lab, r in zip(ax.get_yticklabels(), grid.index):
        if r == con.ANCHOR_REGION:
            lab.set_color("#d9534f")
            lab.set_fontweight('bold')
    ax.set_title(f"Tier-1 {R2_GLOB}, Undepleted\nregions by test rows (descending), anchor in red", pad=15, fontweight='bold')
    ax.set_ylabel(r"Region (test rows $n_r$)")
    ax.set_xlabel("")
    _save(fig, con.VIZ_DESC_REGION_HEATMAP)

def plot_gap_share(summary: pd.DataFrame) -> None:
    _base_style()
    col = "share_removed_gap_r_sq"      # S = (G_u - G_c) / G_u; the RMSE-form share is not defined in the methodology
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in [k for k in MODEL_COLORS if k in set(summary['model'])]:
        s = summary.loc[summary['model'].eq(m)].sort_values('anchor_rows_mean')
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
    ax.set_xlabel("Mean realised anchor training rows (fit + validation, log scale)", labelpad=10)
    ax.set_ylabel(r"Share of the gap removed $S = (G_u - G_c)\,/\,G_u$")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.2), plt.Line2D([], [], color="#d9534f", ls=':')]
    labels += ["min–max over draws", r"gap fully closed ($S = 1$)"]
    ax.legend(handles, labels, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Share of the Performance Gap Removed by Anchor Depletion", pad=15, fontweight='bold')
    _save(fig, con.VIZ_DESC_GAP_SHARE)

def plot_did(did: pd.DataFrame, full_rows: int) -> None:
    _base_style()
    n_comp = len(con.TIER1_REGS) - 1
    panels = (('d_anchor', rf"$d_A$: anchor ({con.REGION_LABELS[con.ANCHOR_REGION]})"),
              ('d_comp', rf"$d_{{\mathrm{{comp}}}}$: unweighted mean of the other {n_comp}"),
              ('did', r"$D_A = d_A - d_{\mathrm{comp}}$"),
              ('did_equal_n', rf"$D_N = d_A - d_N$ ($N$: {con.REGION_LABELS[con.EQUAL_N_REGION]})"))
    assert did[[c for c, _ in panels]].notna().all().all(), "missing DiD term"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes.flat, panels):
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
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel(X_ANCHOR, labelpad=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss in " + R2_GLOB + r": undepleted $-$ depleted" + "\n(positive = accuracy fell)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45)]
    labels += ["individual draw"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Difference-in-Differences Decomposition of the Change in " + R2_GLOB,
                 fontweight='bold', fontsize=13 * con.FONT_SCALE)
    _save(fig, con.VIZ_DESC_DID)

def plot_headline(headline: pd.DataFrame) -> None:
    _base_style()
    missing = set(HEADLINE_LABELS) - set(headline.columns)
    assert not missing, f"headline table lacks gap metrics {sorted(missing)}"
    grid = headline.set_index('model')[list(HEADLINE_LABELS)].rename(columns=HEADLINE_LABELS)
    grid.index = [MODEL_LABELS[m] for m in grid.index]
    lim = float(np.nanmax(np.abs(grid.to_numpy())))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(grid, annot=grid.map(lambda v: f"{v:+.1%}"), fmt="", annot_kws={'fontsize': 9 * con.FONT_SCALE},
                cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'relative change', 'format': mticker.PercentFormatter(1.0)}, ax=ax)
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    level, n_draws = int(headline['level'].iloc[0]), int(headline['n_draws'].iloc[0])
    ax.set_title(f"Relative Change of the Gap Metrics at L{level} (mean of {n_draws} draws)\n"
                 r"(depleted $-$ undepleted) / |undepleted|; for $G$ this equals $-S$",
                 pad=15, fontweight='bold')
    ax.set_ylabel("")
    _save(fig, con.VIZ_DESC_HEADLINE)

def plot_gap_contributions(contrib: pd.DataFrame, n_r: pd.Series) -> None:
    _base_style()
    deepest = contrib['level'].min()
    d = contrib.loc[contrib['level'].eq(deepest)]
    n_draws = d.groupby(CURVE_KEYS)['draw'].nunique()
    assert n_draws.nunique() == 1, "deepest level holds different draw counts per model"
    models = [m for m in MODEL_COLORS if m in set(d['model'])]
    assert d.groupby('model')['configuration'].nunique().eq(1).all(), "more than one depletion curve per model"

    # level statistics over draws: mean share as bar, min-max as whisker
    stats = d.groupby(['model', 'region'])['share'].agg(['mean', 'min', 'max'])
    order = n_r.loc[con.TIER1_REGS].sort_values(ascending=True)
    mean = stats['mean'].unstack('model').loc[order.index, models]
    lo, hi = stats['min'].unstack('model').loc[order.index, models], stats['max'].unstack('model').loc[order.index, models]
    dg = d.groupby('model')['delta_gap'].mean()

    fig, axes = _region_panels(mean, order, highlight={con.ANCHOR_REGION}, kind='bar')
    y = np.arange(len(order))
    left, right = min(lo.to_numpy().min(), 0.0), max(hi.to_numpy().max(), 0.0)
    pad = 0.08 * (right - left)
    for ax, m in zip(axes, models):
        ax.hlines(y, lo[m], hi[m], color="#333333", lw=1.0, zorder=5)
        ax.set_xlim(left - pad, right + pad)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{MODEL_LABELS[m]}\n" + r"$\Delta G$" + f" = {dg[m]:+.3f}", fontweight='bold', pad=8)
    axes[0].set_ylabel(r"Region (test rows $n_r$)")
    fig.supxlabel(r"Share of the gap change $k_r\,\Delta R^{2,\mathrm{glob}}_r\,/\,\Delta G$")
    fig.suptitle(f"Regional Contributions to the Change of the Performance Gap at L{int(deepest)} "
                 f"(mean over {int(n_draws.iloc[0])} draws)\n"
                 "bars: mean share, lines: min–max over draws; shares sum to 100% per model; anchor in red",
                 fontweight='bold')
    _save(fig, con.VIZ_DESC_GAP_CONTRIB)


if __name__ == '__main__':
    print(f"\n[°°°] building figures from the assembled results [°°°]\n")
    t = load_tables()

    plot_pooled_macro(t['aggregate'])
    plot_region_heatmap(t['per_region'], t['n_r'])
    plot_gap_share(t['gap_curve'])
    plot_did(t['did'], int(t['aggregate']['anchor_rows_retained'].max()))
    plot_headline(t['headline'])
    plot_gap_contributions(t['contrib'], t['n_r'])

    print(f"\n[°°°] assembled-result figures complete [°°°]")