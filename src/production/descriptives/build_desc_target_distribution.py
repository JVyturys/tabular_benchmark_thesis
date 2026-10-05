##################################################
'''
src.production.descriptives.build_desc_target_distribution

input:      panel.parquet, ref_geo_table.parquet (via build_desc_panel.load_panel)
purpose:    one figure on the target across the Tier-1 regions: the ESG Combined Score
            distribution per region (left) and its Wasserstein distance W1 to the anchor
            (right), regions as rows ordered by panel rows; replaces the two plots of
            src.exploration.final_panel.determine_target_distribution
output:     target_distribution_by_region.png

'''
##################################################
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config as con
from src.production.descriptives.build_desc_panel import load_panel, target_by_region

C_BAR = "#1f4e79"
C_REF = "#d9534f"


def _style(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')


def build_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, _ = load_panel()
    t1 = panel.loc[panel['lvl3permid'].isin(con.TIER1_REGS), ['orgpermid', 'lvl3permid', 'esg_combined_score']]
    stats = target_by_region(t1)
    stats = stats.loc[stats['scope'].eq('region')].set_index('lvl3permid')
    stats.index = stats.index.astype('int64')
    stats = stats.sort_values('rows', ascending=True)          

    assert stats['rows'].idxmax() == con.ANCHOR_REGION, \
        f"largest Tier-1 region is {stats['rows'].idxmax()}, not the anchor {con.ANCHOR_REGION} - W1 reference unclear"
    assert set(stats.index) == set(con.TIER1_REGS), f"Tier-1 regions missing: {sorted(set(con.TIER1_REGS) - set(stats.index))}"
    assert stats['rows'].sum() == len(t1), "region rows do not sum to the Tier-1 panel rows"
    assert np.isclose(stats.loc[con.ANCHOR_REGION, 'wasserstein_anchor'], 0.0, rtol=0, atol=1e-12), "anchor W1 to itself is not 0"
    print(f"    [+++] target by region - {len(stats)} Tier-1 regions, {len(t1):,} rows, "
          f"W1 reference {con.REGION_LABELS[con.ANCHOR_REGION]}")
    return t1, stats


def plot_target_distribution(t1: pd.DataFrame, stats: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = mpl.rcParamsDefault['font.size'] * con.FONT_SCALE
    regions = list(stats.index)
    y = np.arange(len(regions))
    anchor_y = regions.index(con.ANCHOR_REGION)
    norm = LogNorm(vmin=stats['rows'].min(), vmax=stats['rows'].max())
    cmap = plt.get_cmap("Blues")

    fig, (ax_box, ax_wd) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, gridspec_kw={'width_ratios': [1.6, 1]})
    for ax in (ax_box, ax_wd):
        ax.axhspan(anchor_y - 0.5, anchor_y + 0.5, color="#fbe3e2", zorder=0)

    values = [t1.loc[t1['lvl3permid'].eq(r), 'esg_combined_score'].to_numpy() for r in regions]
    bp = ax_box.boxplot(values, positions=y, orientation='horizontal', widths=0.6, patch_artist=True,
                        medianprops=dict(color="#222222", lw=1.4), whiskerprops=dict(color="#333333"),
                        capprops=dict(color="#333333"), boxprops=dict(edgecolor="#333333", lw=1.1),
                        flierprops=dict(marker='o', ms=2, mfc="#888888", mec='none', alpha=0.5), zorder=3)
    for patch, r in zip(bp['boxes'], regions):
        patch.set_facecolor(cmap(0.35 + 0.65 * norm(stats.loc[r, 'rows'])))
    ax_box.axvline(stats.loc[con.ANCHOR_REGION, 'median'], color=C_REF, ls=":", lw=1.3, zorder=2)
    ax_box.set_xlim(0, 1)
    ax_box.set_xlabel("ESG Combined Score", labelpad=10)
    ax_box.set_title("Distribution (dotted: anchor median)", fontweight='bold', pad=8)

    ax_wd.barh(y, stats['wasserstein_anchor'], color=C_BAR, height=0.6, zorder=3)
    ax_wd.text(0.002, anchor_y, "reference", va='center', ha='left', color=C_REF, fontstyle='italic', zorder=4)
    ax_wd.set_xlim(0, stats['wasserstein_anchor'].max() * 1.15)
    ax_wd.set_xlabel(rf"$W_1$ to {con.REGION_LABELS[con.ANCHOR_REGION]}", labelpad=10)
    ax_wd.set_title(r"Wasserstein distance $W_1$ to the anchor", fontweight='bold', pad=8)

    for ax in (ax_box, ax_wd):
        ax.grid(True, axis='x', linestyle="--", alpha=0.5)
        ax.grid(False, axis='y')
        _style(ax)
    ax_box.set_yticks(y)
    ax_box.set_yticklabels([f"{con.REGION_LABELS[r]}  (n={n:,})" for r, n in stats['rows'].items()])
    ax_box.set_ylim(-0.5, len(regions) - 0.5)
    for lab, r in zip(ax_box.get_yticklabels(), regions):
        if r == con.ANCHOR_REGION:
            lab.set_color(C_REF)
            lab.set_fontweight('bold')
    ax_box.set_ylabel("Region (panel rows)")
    fig.suptitle(f"ESG Combined Score per Tier-1 Region\n"
                 f"all panel rows; anchor ({con.REGION_LABELS[con.ANCHOR_REGION]}) in red", fontweight='bold')

    con.VIZ_DESC_TARGET_DIST.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(con.VIZ_DESC_TARGET_DIST, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {con.VIZ_DESC_TARGET_DIST.name}")


if __name__ == '__main__':
    print(f"\n[°°°] building target distribution figure - anchor {con.ANCHOR_REGION} [°°°]\n")
    t1, stats = build_inputs()
    plot_target_distribution(t1, stats)
    print(f"\n[°°°] target distribution figure complete [°°°]")