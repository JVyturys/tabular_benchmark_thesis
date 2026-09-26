##################################################
'''
src.production.descriptives.build_desc_strata

input:      panel.parquet, ref_geo_table.parquet, split.parquet
            (via build_depletion_draws: _load_context_frame, _entity_mean_target, _assign_strata)
purpose:    distribution of the anchor's entity mean target over its train+val (fit + val) rows,
            with the quantile edges of the depletion strata; each row carries its entity's mean,
            so the bars show row mass, the outline shows entity counts
output:     desc_depletion_strata.csv - stratum, edge_low, edge_high, entities, rows, shares,
                                        mean of entity means
            depletion_strata.png
'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config as con
from src.production.from_final_panel.build_depletion_draws import (_load_context_frame, _entity_mean_target,
                                                                   _assign_strata, ANCHOR_REGION, N_STRATA)

N_BINS = 50


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def build_strata() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    assert ANCHOR_REGION == con.ANCHOR_REGION, f"draw build anchor {ANCHOR_REGION} differs from con.ANCHOR_REGION"
    context = _load_context_frame()
    assert len(context) == con.CONTEXT_ROWS, f"context holds {len(context)} rows, con.CONTEXT_ROWS is {con.CONTEXT_ROWS}"

    anchor_rows = context.loc[context['lvl3permid'].eq(ANCHOR_REGION)]
    means = _entity_mean_target(anchor_rows, ANCHOR_REGION)
    strata, edges = _assign_strata(means)
    rows = anchor_rows.groupby('orgpermid').size()
    assert len(edges) == N_STRATA + 1
    assert set(rows.index) == set(means.index) == set(strata.index), "anchor entity sets disagree"

    ent = pd.DataFrame({'entity_mean': means, 'stratum': strata, 'rows': rows.reindex(means.index)})
    lo, hi = edges[ent['stratum']], edges[ent['stratum'] + 1]
    assert ((ent['entity_mean'] >= lo) & (ent['entity_mean'] <= hi)).all(), "entity mean outside its stratum's edges"

    tab = ent.groupby('stratum').agg(entities=('rows', 'size'), rows=('rows', 'sum'),
                                     mean_entity_mean=('entity_mean', 'mean'),
                                     mean_rows_per_entity=('rows', 'mean')).reset_index()
    tab.insert(1, 'edge_low', edges[:-1])
    tab.insert(2, 'edge_high', edges[1:])
    tab['entity_share'] = tab['entities'] / tab['entities'].sum()
    tab['row_share'] = tab['rows'] / tab['rows'].sum()
    assert tab['entities'].sum() == len(ent) and tab['rows'].sum() == len(anchor_rows)
    print(f"    [+++] strata rebuilt - anchor {ANCHOR_REGION}, entities {len(ent)}, rows {len(anchor_rows)}")
    return ent, edges, tab


def plot_strata(ent: pd.DataFrame, edges: np.ndarray, tab: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    bins = np.linspace(0, 1, N_BINS + 1)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.4 + 0.55 * s / max(N_STRATA - 1, 1)) for s in range(N_STRATA)]
    total_rows, total_ents = ent['rows'].sum(), len(ent)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = np.zeros(N_BINS)
    for s in range(N_STRATA):
        g = ent.loc[ent['stratum'].eq(s)]
        h, _ = np.histogram(g['entity_mean'], bins=bins, weights=g['rows'] / total_rows)
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align='edge', color=colors[s],
               edgecolor='white', linewidth=0.5, label=f"stratum {s + 1}")
        bottom += h
    ent_h, _ = np.histogram(ent['entity_mean'], bins=bins, weights=np.full(total_ents, 1 / total_ents))
    assert np.isclose(bottom.sum(), 1) and np.isclose(ent_h.sum(), 1)
    ax.stairs(ent_h, bins, color="#333333", lw=1.4, label="entity share (unweighted)", zorder=4)

    for e in edges[1:-1]:
        ax.axvline(e, color="#d9534f", ls=":", lw=1.8, zorder=5)
    y_top = max(bottom.max(), ent_h.max()) * 1.32
    ax.set_ylim(0, y_top)
    for _, r in tab.iterrows():
        x = (r['edge_low'] + r['edge_high']) / 2
        ax.text(x, y_top * 0.97, f"S{int(r['stratum']) + 1}\n{_n(r['entities'])} ent. · {r['row_share']:.0%} rows",
                ha='center', va='top', fontsize=10, color="#333333", linespacing=1.4)
    for e in edges[1:-1]:
        ax.text(e + 0.006, y_top * 0.8, f"{e:.3f}", ha='left', va='center', fontsize=10, color="#d9534f",
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'), zorder=6)

    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"Anchor Entity Mean Target and Depletion Strata ({ANCHOR_REGION})\n"
                 f"Train+val rows, n = {_n(total_rows)} rows · {_n(total_ents)} entities", pad=15, fontweight='bold')
    ax.set_xlabel("Entity mean ESG Combined Score (train+val)", labelpad=10)
    ax.set_ylabel("Share per bin")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#d9534f", ls=":", lw=1.8))
    labels.append("stratum edge (entity quartile)")
    ax.legend(handles, labels, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=10)
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    con.VIZ_DESC_STRATA.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(con.VIZ_DESC_STRATA, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {con.VIZ_DESC_STRATA.name}")


if __name__ == '__main__':
    print(f"\n[°°°] building depletion strata plot - anchor {con.ANCHOR_REGION}, strata {N_STRATA} [°°°]\n")
    ent, edges, tab = build_strata()

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(con.TAB_DESC_STRATA, index=False)
    print(f"      [---] written {con.TAB_DESC_STRATA.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_strata(ent, edges, tab)
    print(f"\n    [+++] strata\n{tab.to_string(index=False)}")
    print("\n[°°°] depletion strata plot complete [°°°]")