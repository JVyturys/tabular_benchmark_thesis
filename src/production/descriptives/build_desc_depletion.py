##################################################
'''
src.production.descriptives.build_desc_depletion

input:      depletion_draws.parquet, depletion_counts.parquet, split.parquet,
            plus the stage-3 context inputs of build_depletion_draws._load_context_frame
            (panel.parquet, ref_geo_table.parquet, split.parquet)
purpose:    describe every depletion condition's training context: anchor size,
            overshoot, anchor pool share, anchor fit/val split, regional composition,
            and the retained anchor target distribution
output:     desc_depletion_conditions.csv, desc_depletion_composition.csv, desc_depletion_levels.csv,
            depletion_pool_share.png, depletion_target.png

'''
##################################################
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import config as con
from src.production.from_final_panel.build_depletion_draws import _load_context_frame, ANCHOR_REGION as DRAW_ANCHOR

UNDEPL = 'undepl'
TIERS = {'tier1': con.TIER1_REGS, 'tier2': con.TIER2_REGS, 'tier3': con.TIER3_REGS}


def _tag(level: int, draw: int) -> str:
    return f"depl_L{int(level)}_d{int(draw)}"


def _tier(region: int) -> str:
    return next(t for t, regs in TIERS.items() if region in regs)


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert DRAW_ANCHOR == con.ANCHOR_REGION, f"draw build anchor {DRAW_ANCHOR} differs from con.ANCHOR_REGION {con.ANCHOR_REGION}"
    context = _load_context_frame()
    assert len(context) == con.CONTEXT_ROWS, f"context holds {len(context)} rows, con.CONTEXT_ROWS is {con.CONTEXT_ROWS}"

    split = pd.read_parquet(con.SPLIT, columns=['orgpermid', 'partition'])
    split['orgpermid'] = split['orgpermid'].astype('int64')
    assert split['orgpermid'].is_unique, "split duplicates orgpermid"
    context['partition'] = context['orgpermid'].map(split.set_index('orgpermid')['partition'])
    assert context['partition'].isin(['fit', 'val']).all(), "context row outside fit/val"

    draws = pd.read_parquet(con.DEPLETION_DRAWS)
    counts = pd.read_parquet(con.DEPLETION_COUNTS)
    assert not counts.duplicated(['level', 'draw']).any(), "duplicate (level, draw) in depletion_counts"
    assert not draws.duplicated(['level', 'draw', 'orgpermid']).any(), "duplicate (level, draw, orgpermid) in draws"
    assert set(map(tuple, draws[['level', 'draw']].drop_duplicates().to_numpy())) == \
           set(map(tuple, counts[['level', 'draw']].to_numpy())), "draws and counts disagree on the condition set"
    assert counts['anchor_region'].eq(con.ANCHOR_REGION).all(), "counts built for a different anchor"
    print(f"    [+++] inputs loaded - context rows {len(context)}, conditions {len(counts)}, draw rows {len(draws)}")
    return context, draws, counts


def condition_frames(context: pd.DataFrame, draws: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ctx_ents = set(context['orgpermid'])
    anchor_ents = set(context.loc[context['lvl3permid'].eq(con.ANCHOR_REGION), 'orgpermid'])
    frames = {UNDEPL: context}
    for (level, draw), d in draws.groupby(['level', 'draw']):
        ents = set(d['orgpermid'])
        assert ents <= ctx_ents, f"{_tag(level, draw)}: drawn entity outside the context"
        assert set(d.loc[d['is_anchor'], 'orgpermid']) == ents & anchor_ents, f"{_tag(level, draw)}: is_anchor mislabels entities"
        assert set(d.loc[~d['is_anchor'], 'orgpermid']) == ctx_ents - anchor_ents, f"{_tag(level, draw)}: non-anchor block incomplete"
        frames[_tag(level, draw)] = context.loc[context['orgpermid'].isin(ents)]
    return frames


def condition_table(frames: dict[str, pd.DataFrame], counts: pd.DataFrame) -> pd.DataFrame:
    records = []
    for tag, f in frames.items():
        a = f.loc[f['lvl3permid'].eq(con.ANCHOR_REGION)]
        y = a['esg_combined_score']
        records.append({
            'condition': tag, 'context_rows': len(f), 'context_entities': f['orgpermid'].nunique(),
            'context_rows_fit': int(f['partition'].eq('fit').sum()), 'context_rows_val': int(f['partition'].eq('val').sum()),
            'anchor_rows': len(a), 'anchor_entities': a['orgpermid'].nunique(),
            'anchor_rows_fit': int(a['partition'].eq('fit').sum()), 'anchor_rows_val': int(a['partition'].eq('val').sum()),
            'anchor_target_mean': y.mean(), 'anchor_target_std': y.std(ddof=0),
        })
    out = pd.DataFrame(records)
    out['anchor_pool_share'] = out['anchor_rows'] / out['context_rows']
    out['anchor_val_share'] = out['anchor_rows_val'] / out['anchor_rows']
    out['anchor_share_of_val'] = out['anchor_rows_val'] / out['context_rows_val']

    c = counts.assign(condition=[_tag(l, d) for l, d in zip(counts['level'], counts['draw'])])
    c = c[['condition', 'level', 'draw', 'target_rows', 'overshoot_rows', 'anchor_rows_retained', 'anchor_entities_retained',
           'context_rows_retained', 'context_entities_retained', 'target_mean_retained', 'target_std_retained',
           'target_mean_delta', 'mean_delta_se_units', 'target_std_ratio', 'wasserstein_1']]
    n = len(out)
    out = out.merge(c, on='condition', how='left', validate='one_to_one')
    assert len(out) == n

    dep = out.loc[out['condition'].ne(UNDEPL)]
    for mine, theirs in (('anchor_rows', 'anchor_rows_retained'), ('anchor_entities', 'anchor_entities_retained'),
                         ('context_rows', 'context_rows_retained'), ('context_entities', 'context_entities_retained')):
        assert dep[mine].eq(dep[theirs]).all(), f"recomputed {mine} differs from depletion_counts"
    assert dep['overshoot_rows'].eq(dep['anchor_rows'] - dep['target_rows']).all(), "overshoot does not reconcile"
    for mine, theirs in (('anchor_target_mean', 'target_mean_retained'), ('anchor_target_std', 'target_std_retained')):
        assert np.allclose(dep[mine], dep[theirs], rtol=con.INVARIANT_RTOL, atol=0), f"recomputed {mine} differs from depletion_counts"

    out = out.drop(columns=['anchor_rows_retained', 'anchor_entities_retained', 'context_rows_retained',
                            'context_entities_retained', 'target_mean_retained', 'target_std_retained'])
    out['overshoot_share'] = out['overshoot_rows'] / out['target_rows']
    out[['level', 'draw', 'target_rows', 'overshoot_rows']] = out[['level', 'draw', 'target_rows', 'overshoot_rows']].astype('Int64')
    return out.sort_values(['level', 'draw'], ascending=[False, True], na_position='first').reset_index(drop=True)


def composition_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for tag, f in frames.items():
        g = f.groupby('lvl3permid').agg(rows=('orgpermid', 'size'), entities=('orgpermid', 'nunique')).reset_index()
        assert g['rows'].sum() == len(f), f"{tag}: region rows do not sum to context rows"
        parts.append(g.assign(condition=tag, pool_share=g['rows'] / len(f)))
    out = pd.concat(parts, ignore_index=True)
    out['tier'] = out['lvl3permid'].map(_tier)

    base = out.loc[out['condition'].eq(UNDEPL)].set_index('lvl3permid')
    out = out.join(base[['rows', 'pool_share']].rename(columns={'rows': 'rows_undepl', 'pool_share': 'pool_share_undepl'}),
                   on='lvl3permid', how='left')
    non_anchor = out['lvl3permid'].ne(con.ANCHOR_REGION)
    assert out.loc[non_anchor, 'rows'].eq(out.loc[non_anchor, 'rows_undepl']).all(), "a non-anchor region lost rows"
    out['pool_share_delta'] = out['pool_share'] - out['pool_share_undepl']
    return out[['condition', 'lvl3permid', 'tier', 'rows', 'entities', 'pool_share', 'pool_share_undepl', 'pool_share_delta']]


def level_table(cond: pd.DataFrame) -> pd.DataFrame:
    cols = ['anchor_rows', 'anchor_entities', 'overshoot_rows', 'overshoot_share', 'anchor_pool_share', 'anchor_val_share',
            'context_rows', 'target_mean_delta', 'mean_delta_se_units', 'target_std_ratio', 'wasserstein_1']
    dep = cond.loc[cond['condition'].ne(UNDEPL)]
    out = dep.groupby('level').agg(n_draws=('draw', 'size'),
                                   **{f"{c}_{s}": (c, s) for c in cols for s in ('mean', 'min', 'max')}).reset_index()
    assert out['n_draws'].sum() == len(dep)
    return out.sort_values('level', ascending=False).reset_index(drop=True)


def plot_pool_share(cond: pd.DataFrame, comp: pd.DataFrame, levels: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    base = comp.loc[comp['condition'].eq(UNDEPL) & comp['tier'].eq('tier1') & comp['lvl3permid'].ne(con.ANCHOR_REGION)]
    for i, (_, r) in enumerate(base.sort_values('rows').iterrows()):
        eq = r['lvl3permid'] == con.EQUAL_N_REGION
        ax.axvline(r['rows'], color="#d9534f" if eq else "#cccccc", linestyle=":", linewidth=1.3 if eq else 0.9, zorder=1)
        ax.text(r['rows'], 1.0 if i % 2 == 0 else 0.86, str(int(r['lvl3permid'])), transform=ax.get_xaxis_transform(),
                rotation=90, ha='right', va='top', fontsize=7.5, color="#d9534f" if eq else "#888888")
    dep = cond.loc[cond['condition'].ne(UNDEPL)]
    und = cond.loc[cond['condition'].eq(UNDEPL)]
    ax.scatter(dep['anchor_rows'], dep['anchor_pool_share'], color="#1f4e79", s=16, alpha=0.45, edgecolor='none', zorder=2,
               label='individual draw')
    ax.plot(levels['anchor_rows_mean'], levels['anchor_pool_share_mean'], color="#1f4e79", lw=1.6, marker='o', ms=4,
            zorder=3, label='level mean')
    ax.scatter(und['anchor_rows'], und['anchor_pool_share'], marker='D', s=42, facecolor='white', edgecolor="#1f4e79",
               lw=1.4, zorder=4, label='undepleted (full anchor)')
    ax.set_xscale('log')
    ax.set_xticks(sorted(levels['level'].astype(int)) + [int(und['anchor_rows'].iloc[0])])
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"Anchor Share of the Training Pool under Depletion ({con.ANCHOR_REGION})", pad=15, fontweight='bold')
    ax.set_xlabel("Realised anchor rows, train+val (log scale); dotted: Tier-1 regions' train+val rows", labelpad=10)
    ax.set_ylabel("Anchor rows / context rows")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    _style(ax)
    _save(fig, con.VIZ_DESC_POOL_SHARE)


def plot_target(frames: dict[str, pd.DataFrame], cond: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    order = cond['condition'].tolist()
    long = pd.concat([f.loc[f['lvl3permid'].eq(con.ANCHOR_REGION), ['esg_combined_score']].assign(condition=t)
                      for t, f in frames.items()], ignore_index=True)
    n = cond.set_index('condition')['anchor_rows']
    norm = LogNorm(vmin=n.min(), vmax=n.max())
    cmap = plt.get_cmap("Blues")
    palette = {t: cmap(0.35 + 0.65 * norm(n[t])) for t in order}

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=long, x='condition', y='esg_combined_score', order=order, hue='condition', palette=palette,
                legend=False, ax=ax, linewidth=1.2, fliersize=2)
    full_median = long.loc[long['condition'].eq(UNDEPL), 'esg_combined_score'].median()
    ax.axhline(full_median, color="#d9534f", linestyle=":", linewidth=1.3, zorder=0)
    y_max = long['esg_combined_score'].max()
    for i, t in enumerate(order):
        ax.text(i, y_max * 1.02, f"n={n[t]:,}".replace(",", "."), ha='left', va='bottom', rotation=45,
                fontsize=8, color="#555555", fontstyle='italic')
    ax.set_ylim(top=y_max * 1.15)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([t.replace('depl_', '') for t in order], rotation=45, ha='right', fontsize=9)
    ax.set_title(f"Anchor Target Distribution per Depletion Condition ({con.ANCHOR_REGION})\n"
                 f"Retained train+val rows; dotted: undepleted median", pad=15, fontweight='bold')
    ax.set_xlabel("Condition", labelpad=10)
    ax.set_ylabel("ESG Combined Score")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    _style(ax)
    _save(fig, con.VIZ_DESC_DEPL_TARGET)


if __name__ == '__main__':
    print(f"\n[°°°] building depletion descriptives - anchor {con.ANCHOR_REGION} [°°°]\n")
    context, draws, counts = load_inputs()

    frames = condition_frames(context, draws)
    cond = condition_table(frames, counts)
    comp = composition_table(frames)
    levels = level_table(cond)
    print(f"    [+++] tables built - {len(frames)} conditions incl. {UNDEPL}")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    for tab, path in ((cond, con.TAB_DESC_DEPL_CONDITIONS), (comp, con.TAB_DESC_DEPL_COMPOSITION),
                      (levels, con.TAB_DESC_DEPL_LEVELS)):
        tab.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_pool_share(cond, comp, levels)
    plot_target(frames, cond)

    print(f"\n    [+++] conditions\n{cond[['condition', 'anchor_rows', 'overshoot_rows', 'anchor_pool_share', 'anchor_val_share', 'context_rows']].to_string(index=False)}")
    print(f"\n[°°°] depletion descriptives complete [°°°]")