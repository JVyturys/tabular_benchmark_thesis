##################################################
'''
src.production.descriptives.build_desc_draw_distributions

input:      depletion_draws.parquet, depletion_counts.parquet
            (+ panel.parquet, ref_geo_table.parquet, split.parquet via build_depletion_draws._load_context_frame)
purpose:    small-multiple overview of the retained anchor target distribution: one row per depletion
            level, one panel per draw, plus a level-average panel; every panel is set against the full
            anchor and annotated with the row-level mean shift, std ratio and Wasserstein distance
output:     depletion_draw_distributions.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import config as con
from src.production.from_final_panel.build_depletion_draws import _load_context_frame, DEPLETION_GRID, ANCHOR_REGION

BINS = np.linspace(0, 1, 26)
C_DRAW, C_BAND, C_FULL = "#1f4e79", "#6fa8dc", "#cccccc"
STATS = [('target_mean_delta', 'Δ mean', '{:+.4f}'), ('target_std_ratio', 'σ ratio', '{:.3f}'),
         ('wasserstein_1', 'W₁', '{:.4f}')]


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _fmt(fmt: str, value: float) -> str:
    return fmt.format(0.0 if abs(value) < 5e-5 else value)


def _density(y: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(y, bins=BINS, density=True)
    return h


def collect() -> tuple[np.ndarray, pd.DataFrame, dict]:
    assert ANCHOR_REGION == con.ANCHOR_REGION, f"draw build anchor {ANCHOR_REGION} differs from con.ANCHOR_REGION"
    context = _load_context_frame()
    assert len(context) == con.CONTEXT_ROWS, f"context holds {len(context)} rows, con.CONTEXT_ROWS is {con.CONTEXT_ROWS}"
    anchor = context.loc[context['lvl3permid'].eq(ANCHOR_REGION), ['orgpermid', 'esg_combined_score']]
    full_y = anchor['esg_combined_score'].to_numpy()

    draws = pd.read_parquet(con.DEPLETION_DRAWS)
    counts = pd.read_parquet(con.DEPLETION_COUNTS).set_index(['level', 'draw'])
    grid = {(lvl, d) for lvl, reps in DEPLETION_GRID.items() for d in range(reps)}
    assert set(counts.index) == grid, "counts do not match DEPLETION_GRID"
    assert set(map(tuple, draws[['level', 'draw']].drop_duplicates().to_numpy())) == grid, "draws do not match DEPLETION_GRID"

    retained = {}
    for (lvl, d), g in draws.loc[draws['is_anchor']].groupby(['level', 'draw']):
        y = anchor.loc[anchor['orgpermid'].isin(g['orgpermid']), 'esg_combined_score'].to_numpy()
        c = counts.loc[(lvl, d)]
        assert len(y) == c['anchor_rows_retained'], f"L{lvl} d{d}: {len(y)} rows vs counts {c['anchor_rows_retained']}"
        mine = {'target_mean_delta': y.mean() - full_y.mean(), 'target_std_ratio': y.std(ddof=0) / full_y.std(ddof=0),
                'wasserstein_1': wasserstein_distance(full_y, y)}
        for k, val in mine.items():
            assert np.isclose(val, c[k], rtol=con.INVARIANT_RTOL, atol=1e-15), f"L{lvl} d{d}: {k} {val} vs counts {c[k]}"
        retained[(lvl, d)] = y
    print(f"    [+++] {len(retained)} draws rebuilt and reconciled with depletion_counts")
    return full_y, counts.reset_index(), retained


def _stats_box(ax, lines: list[str]) -> None:
    ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
            family='DejaVu Sans Mono', color="#333333", linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.9))


def _panel(ax, full_h: np.ndarray) -> None:
    ax.stairs(full_h, BINS, fill=True, color=C_FULL, alpha=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.grid(True, axis='x', linestyle="--", alpha=0.4)
    ax.grid(False, axis='y')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')


def plot(full_y: np.ndarray, counts: pd.DataFrame, retained: dict) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    levels = sorted(DEPLETION_GRID, reverse=True)
    n_cols = max(DEPLETION_GRID.values())
    full_h = _density(full_y)
    fig, axes = plt.subplots(len(levels), n_cols + 1, figsize=(19, 14), sharex=True, sharey=True,
                             gridspec_kw={'width_ratios': [1] * n_cols + [1.15], 'wspace': 0.1, 'hspace': 0.32})
    y_max = max(full_h.max(), max(_density(y).max() for y in retained.values()))

    for i, lvl in enumerate(levels):
        c_lvl = counts.loc[counts['level'].eq(lvl)].sort_values('draw')
        dens = []
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(c_lvl):
                ax.set_axis_off()
                continue
            c = c_lvl.iloc[j]
            h = _density(retained[(lvl, int(c['draw']))])
            dens.append(h)
            _panel(ax, full_h)
            ax.stairs(h, BINS, color=C_DRAW, lw=1.6, zorder=3)
            ax.set_title(f"draw {int(c['draw']) + 1} · {_n(c['anchor_rows_retained'])} rows", fontsize=10.5, pad=4)
            _stats_box(ax, [f"{lab:<7} {_fmt(fmt, c[k])}" for k, lab, fmt in STATS])

        ax = axes[i, n_cols]
        dens = np.vstack(dens)
        _panel(ax, full_h)
        ax.set_facecolor("#f3f6fa")
        ax.fill_between(BINS[:-1], dens.min(axis=0), dens.max(axis=0), step='post', color=C_BAND, alpha=0.45, lw=0, zorder=2)
        ax.stairs(dens.mean(axis=0), BINS, color=C_DRAW, lw=2.2, zorder=3)
        ax.set_title(f"level mean · {len(c_lvl)} draws", fontsize=10.5, pad=4, fontweight='bold', color=C_DRAW)
        _stats_box(ax, [f"{lab:<7} {_fmt(fmt, c_lvl[k].mean())}" for k, lab, fmt in STATS])

        axes[i, 0].set_ylabel(f"L{_n(lvl)}", rotation=0, fontsize=14, fontweight='bold', color=C_DRAW,
                              labelpad=38, va='center')

    axes[0, 0].set_ylim(0, y_max * 1.55)
    for ax in axes[-1]:
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.tick_params(axis='x', labelsize=10)
    axes[-1, n_cols // 2].set_xlabel("ESG Combined Score (row level)", fontsize=12, labelpad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_FULL, alpha=0.8), plt.Line2D([], [], color=C_DRAW, lw=1.8),
               plt.Rectangle((0, 0), 1, 1, color=C_BAND, alpha=0.45)]
    labels = [f"full anchor ({_n(len(full_y))} rows)", "retained rows of the draw (level mean in the last column)",
              "min–max over a level's draws"]
    fig.legend(handles, labels, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.035), fontsize=12)
    fig.text(0.5, 0.012, "Δ mean: retained − full mean · σ ratio: retained / full std · W₁: Wasserstein distance to the full anchor",
             ha='center', fontsize=11, color="#555555")
    fig.suptitle(f"Retained Anchor Target Distribution per Depletion Draw ({ANCHOR_REGION})", fontsize=17,
                 fontweight='bold', y=0.955)

    con.VIZ_DESC_DRAW_DISTR.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(con.VIZ_DESC_DRAW_DISTR, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {con.VIZ_DESC_DRAW_DISTR.name}")


if __name__ == '__main__':
    print(f"\n[°°°] building draw distribution overview - anchor {con.ANCHOR_REGION} [°°°]\n")
    full_y, counts, retained = collect()
    plot(full_y, counts, retained)
    print("\n[°°°] draw distribution overview complete [°°°]")