##################################################
'''
src.production.descriptives.build_desc_metrics

input:      results_*.yaml via assemble_results.load_manifests, depletion_counts.parquet
purpose:    describe the reported performance metrics of every run of record: all seven
            aggregate metrics and the tier-1 region metrics per depletion level, within-condition
            model ranks, region skill against region size (undepleted), the anchor learning curve
            (RMSE_r over realised anchor rows, log-log, with comparators) and the two-scale identity
            R2_glob = 1 - (sigma2_r / sigma2_y)(1 - R2_reg) of methodology Section 3.4
output:     desc_metric_levels.csv, desc_metric_region_levels.csv, desc_metric_ranks.csv,
            desc_metric_region_size.csv, metric_curves.png, region_delta_r_sq_deepest.png,
            region_r_sq_vs_size.png, anchor_learning_curve.png, r_sq_scale_identity.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib as mpl
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
LABELS = {'pooled_r_sq': r'Pooled coefficient $R^2_P$', 'macro_r_sq': r'Macro coefficient $R^2_M$',
          'gap_r_sq': r'Performance gap $G = R^2_P - R^2_M$'}
R2_GLOB = r'$R^{2,\mathrm{glob}}_r$'
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

def _region_panels(values: pd.DataFrame, n_r: pd.Series, highlight: set, kind: str):
    """One panel per model (columns of values), one mark per region (rows, bottom -> top), shared x range.
    kind 'bar': signed quantity, bars from zero on a zero-centred range; kind 'dot': level quantity,
    dots on a range zoomed to the data (a bar would need the zero baseline). Regions in highlight get a
    red row band across all panels and a red tick label."""
    assert kind in {'bar', 'dot'}, f"kind {kind}"
    assert values.index.equals(n_r.index), "values and n_r are not aligned on the same regions"
    assert values.notna().all().all(), "missing region x model value"
    y, v = np.arange(len(values)), values.to_numpy()
    if kind == 'bar':
        lim = np.abs(v).max() * 1.1
        xlim = (-lim, lim)
    else:
        pad = 0.1 * (v.max() - v.min())
        xlim = (v.min() - pad, v.max() + pad)
    fig, axes = plt.subplots(1, values.shape[1], figsize=(16, 7), sharey=True, squeeze=False)
    axes = axes[0]
    rows = list(values.index)
    for ax, m in zip(axes, values.columns):
        for r in highlight:
            ax.axhspan(rows.index(r) - 0.5, rows.index(r) + 0.5, color="#fbe3e2", zorder=0)
        if kind == 'bar':
            ax.barh(y, values[m], color=MODEL_COLORS[m], height=0.65, zorder=3)
            ax.axvline(0, color="#333333", lw=0.9, zorder=4)
        else:
            ax.hlines(y, *xlim, color="#e6e6e6", lw=0.8, zorder=1)
            ax.scatter(values[m], y, color=MODEL_COLORS[m], s=40, zorder=3)
        ax.set_xlim(xlim)
        ax.grid(True, axis='x', linestyle="--", alpha=0.5)
        ax.grid(False, axis='y')
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{con.REGION_LABELS[r]}  (n={n_r[r]:,})" for r in rows])
    axes[0].set_ylim(-0.5, len(rows) - 0.5)
    for lab, r in zip(axes[0].get_yticklabels(), rows):
        if r in highlight:
            lab.set_color("#d9534f")
            lab.set_fontweight('bold')
    return fig, axes

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
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    cur = _curve_runs(runs)
    lev = levels.merge(cur[CURVE_KEYS].drop_duplicates(), on=CURVE_KEYS, how='inner')
    full = int(cur['anchor_rows_retained'].max())
    panels = ['pooled_r_sq', 'macro_r_sq', 'gap_r_sq']      
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
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
    for ax in axes:
        ax.set_xlabel("Realised anchor training rows\n(fit + validation, log scale)", labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45),
                plt.Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec="#888888")]
    labels += ["individual draw", "undepleted (full anchor)"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Tier-1 Performance Metrics under Anchor Depletion ({con.REGION_LABELS[con.ANCHOR_REGION]})",
                 fontweight='bold', fontsize=14 * con.FONT_SCALE)
    _save(fig, con.VIZ_DESC_METRIC_CURVES)

def plot_region_delta(region_levels: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    rl = region_levels.merge(region_levels.loc[region_levels['level'].notna(), CURVE_KEYS].drop_duplicates(),
                             on=CURVE_KEYS, how='inner')
    deepest = rl['level'].min()
    d = rl.loc[rl['level'].eq(deepest)]
    order = d.groupby('region')['n_r'].first().sort_values(ascending=True)
    models = [m for m in MODEL_COLORS if m in set(d['model'])]
    values = -d.pivot(index='region', columns='model', values='r_sq_delta_mean').loc[order.index, models]
    fig, axes = _region_panels(values, order, highlight={con.ANCHOR_REGION}, kind='bar')
    for ax, m in zip(axes, models):
        ax.set_title(MODEL_LABELS[m], fontweight='bold', pad=8)
    axes[0].set_ylabel(r"Region (test rows $n_r$)")
    fig.supxlabel(r"$\Delta R^{2,\mathrm{glob}}_r$ (undepleted $-$ depleted)")
    fig.suptitle(f"Change in {R2_GLOB} at the Deepest Level (L{int(deepest)}, mean over draws)\n"
                 r"$\Delta R^{2,\mathrm{glob}}_r = R^{2,\mathrm{glob}}_{r,u} - R^{2,\mathrm{glob}}_{r,c}$"
                 "; positive = loss of accuracy; anchor in red", fontweight='bold')
    _save(fig, con.VIZ_DESC_REGION_DELTA)

def plot_region_size(size: pd.DataFrame, runs: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    s = size.merge(_curve_runs(runs)[CURVE_KEYS].drop_duplicates(), on=CURVE_KEYS, how='inner')
    assert s.groupby('region')['n_r'].nunique().eq(1).all(), "a region carries two test-row counts"
    order = s.groupby('region')['n_r'].first().sort_values(ascending=True)
    models = [k for k in MODEL_COLORS if k in set(s['model'])]
    values = s.pivot(index='region', columns='model', values='r_sq').loc[order.index, models]
    rho = s.groupby('model')['spearman_n_r_sq'].first()
    fig, axes = _region_panels(values, order, highlight={con.ANCHOR_REGION, con.EQUAL_N_REGION}, kind='dot')
    for ax, m in zip(axes, models):
        ax.set_title(f"{MODEL_LABELS[m]}\nρ = {rho[m]:+.2f}", fontweight='bold', pad=8)
    axes[0].set_ylabel(r"Region, ordered by test rows $n_r$")
    fig.supxlabel(R2_GLOB)
    fig.suptitle(f"{R2_GLOB} against Region Size, Undepleted\n"
                 r"ρ: Spearman rank correlation of $n_r$ and " + R2_GLOB + " (descriptive); anchor and equal-N region in red",
                 fontweight='bold')
    _save(fig, con.VIZ_DESC_REGION_SIZE)

def _loglog_slope(x: pd.Series, y: pd.Series) -> float:
    """OLS slope of log10 y on log10 x - the power-law exponent, descriptive."""
    return float(np.polyfit(np.log10(x.astype(float)), np.log10(y.astype(float)), 1)[0])


def plot_anchor_learning_curve(runs: pd.DataFrame, regions: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    reg = regions.merge(_curve_runs(runs)[CURVE_KEYS].drop_duplicates(), on=CURVE_KEYS, how='inner')
    comparators = [r for r in con.TIER1_REGS if r != con.ANCHOR_REGION]
    anchor = reg.loc[reg['region'].eq(con.ANCHOR_REGION)]
    comp = (reg.loc[reg['region'].isin(comparators)]
            .groupby(KEY + ['level', 'anchor_rows_retained'], dropna=False)['rmse_r'].agg(['mean', 'size']).reset_index())
    assert comp['size'].eq(len(comparators)).all(), "comparator mean over an incomplete region set"
    comp = comp.rename(columns={'mean': 'rmse_r'})
    assert len(comp) == len(anchor), "anchor and comparator conditions differ"
    full = int(anchor['anchor_rows_retained'].max())
    levels = anchor['level'].dropna()
    models = [m for m in MODEL_COLORS if m in set(anchor['model'])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panels = ((axes[0], anchor, rf"Anchor: {con.REGION_LABELS[con.ANCHOR_REGION]}", (0.03, 0.03, 'left', 'bottom')),
              (axes[1], comp, f"Comparators: unweighted mean of the other {len(comparators)} Tier-1 regions",
               (0.97, 0.97, 'right', 'top')))
    for ax, frame, title, (bx, by, bha, bva) in panels:
        slopes = []
        for m in models:
            f = frame.loc[frame['model'].eq(m)]
            dep = f.loc[f['condition'].ne(UNDEPL)]
            und = f.loc[f['condition'].eq(UNDEPL)]
            means = dep.groupby('level').agg(x=('anchor_rows_retained', 'mean'), y=('rmse_r', 'mean'))
            curve = pd.concat([means, und[['anchor_rows_retained', 'rmse_r']].set_axis(['x', 'y'], axis=1)]).sort_values('x')
            ax.scatter(dep['anchor_rows_retained'], dep['rmse_r'], color=MODEL_COLORS[m], s=14, alpha=0.4, edgecolor='none', zorder=2)
            ax.plot(curve['x'], curve['y'], color=MODEL_COLORS[m], lw=1.6, marker='o', ms=4, label=MODEL_LABELS[m], zorder=3)
            ax.scatter(und['anchor_rows_retained'], und['rmse_r'], marker='D', s=42, facecolor='white',
                       edgecolor=MODEL_COLORS[m], lw=1.4, zorder=4)
            slopes.append(f"{MODEL_LABELS[m]:<15}{_loglog_slope(f['anchor_rows_retained'], f['rmse_r']):+.3f}")
        ax.text(bx, by, "log–log slope\n" + "\n".join(slopes), transform=ax.transAxes, ha=bha, va=bva,
                family='DejaVu Sans Mono', fontsize=9 * con.FONT_SCALE, color="#333333", linespacing=1.35,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#dddddd', alpha=0.9), zorder=6)
        _log_x(ax, levels, full)
        ax.set_yscale('log')
        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel("Realised anchor training rows\n(fit + validation, log scale)", labelpad=8)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        ax.grid(False, axis='x')
        _style(ax)
    lo, hi = axes[0].get_ylim()
    ticks = np.unique(np.round(np.geomspace(lo * 1.01, hi * 0.99, 6), 3))
    axes[0].yaxis.set_major_locator(mticker.FixedLocator(ticks))
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3f}"))
    axes[0].yaxis.set_minor_locator(mticker.NullLocator())
    axes[0].set_ylabel(r"$\mathrm{RMSE}_r$ (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45),
                plt.Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec="#888888")]
    labels += ["individual draw", "undepleted (full anchor)"]
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Anchor Learning Curve under Depletion (log–log)\n"
                 "slope: OLS fit of log RMSE on log rows over all conditions, descriptive", fontweight='bold')
    _save(fig, con.VIZ_DESC_ANCHOR_CURVE)


def plot_scale_identity(regions: pd.DataFrame, runs: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    und = regions.loc[regions['condition'].eq(UNDEPL)].merge(_curve_runs(runs)[CURVE_KEYS].drop_duplicates(),
                                                              on=CURVE_KEYS, how='inner')
    und = und.assign(ratio=(1 - und['r_sq']) / (1 - und['r_sq_reg']))
    ratio = und.groupby('region')['ratio'].first()
    assert np.allclose(und['ratio'], und['region'].map(ratio), rtol=con.INVARIANT_RTOL, atol=0), "variance ratio not run-invariant"
    models = [m for m in MODEL_COLORS if m in set(und['model'])]

    fig, ax = plt.subplots(figsize=(10, 8))
    x_lo, x_hi = und['r_sq_reg'].min() - 0.05, und['r_sq_reg'].max() + 0.05
    xs = np.array([x_lo, x_hi])
    right = {r: 1 - ratio[r] * (1 - x_hi) for r in ratio.index}           
    y_lo = und['r_sq'].min() - 0.05
    y_hi = max(und['r_sq'].max(), max(right.values())) + 0.05
    ax.plot(xs, xs, color="#333333", ls='--', lw=1.1, zorder=2)
    gap_min = (y_hi - y_lo) * 0.032
    placed = []
    for r, y_end in sorted(right.items(), key=lambda kv: kv[1]):
        y_lab = max(y_end, placed[-1] + gap_min) if placed else y_end
        placed.append(y_lab)
        anchor = r == con.ANCHOR_REGION
        col = "#d9534f" if anchor else "#9a9a9a"
        ax.plot(xs, 1 - ratio[r] * (1 - xs), color=col, lw=1.4 if anchor else 0.8, alpha=1 if anchor else 0.8, zorder=1)
        ax.annotate(f"{con.REGION_LABELS[r]} ({ratio[r]:.2f})", xy=(x_hi, y_end), xytext=(x_hi + 0.06 * (x_hi - x_lo), y_lab),
                    textcoords='data', annotation_clip=False, ha='left', va='center', fontsize=8.5 * con.FONT_SCALE,
                    color=col, fontweight='bold' if anchor else 'normal',
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6, shrinkA=2, shrinkB=0, relpos=(0, 0.5)))
    for m in models:
        g = und.loc[und['model'].eq(m)]
        ax.scatter(g['r_sq_reg'], g['r_sq'], color=MODEL_COLORS[m], s=30, zorder=3, label=MODEL_LABELS[m])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, max(y_hi, placed[-1] + gap_min))
    ax.set_xlabel(r"$R^{2,\mathrm{reg}}_r$ (regional denominator $\sigma_r^2$)", labelpad=10)
    ax.set_ylabel(r"$R^{2,\mathrm{glob}}_r$ (global denominator $\sigma_y^2$)")
    models_leg = ax.legend(frameon=False, ncol=len(models), loc='upper center', bbox_to_anchor=(0.5, -0.11))
    ax.add_artist(models_leg)
    ax.legend([plt.Line2D([], [], color="#9a9a9a", lw=0.9), plt.Line2D([], [], color="#333333", ls='--', lw=1.1)],
              [r"region line $R^{2,\mathrm{glob}}_r = 1 - (\sigma_r^2/\sigma_y^2)(1 - R^{2,\mathrm{reg}}_r)$",
               r"$\sigma_r^2 = \sigma_y^2$ (scales coincide)"],
              frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    ax.set_title("Two Scales of the Regional Coefficient, Undepleted\n"
                 r"one line per region, labelled with $\sigma_r^2/\sigma_y^2$ (fixed by the test targets); anchor in red",
                 pad=15, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.4)
    _style(ax)
    _save(fig, con.VIZ_DESC_SCALE_IDENTITY)

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
    plot_anchor_learning_curve(runs, regions)
    plot_scale_identity(regions, runs)

    print(f"\n    [+++] size monotonicity (undepleted)\n"
          f"{size.groupby(CURVE_KEYS)[['spearman_n_r_sq', 'spearman_n_rmse']].first().to_string()}")
    print(f"\n[°°°] performance-metric descriptives complete [°°°]")