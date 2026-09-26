##################################################
'''
src.production.descriptives.build_desc_depletion_design

input:      depletion_counts.parquet, desc_depletion_conditions.csv, desc_depletion_composition.csv
            (build_desc_depletion), results_*.yaml (all manifests of record)
purpose:    design schematic of the anchor-depletion experiment; every number on the figure
            is computed from the artifacts and every "held fixed" claim that the manifests
            can prove is checked before it is drawn
output:     desc_depletion_design_facts.csv - fact, value, source
            depletion_design_schematic.png
'''
##################################################
import yaml
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory
import config as con
from src.production.from_final_panel.build_depletion_draws import DEPLETION_GRID, N_STRATA, ANCHOR_REGION as DRAW_ANCHOR
from src.production.from_final_panel.assemble_results import SCORE_DIRS, load_manifests

UNDEPL = 'undepl'
FITTED = ('rf', 'xgb', 'ftt')
C_MAIN, C_REF, C_TXT = "#1f4e79", "#d9534f", "#222222"
FACE_MAIN, FACE_REF, FACE_GREY = "#f3f6fa", "#fdf3f2", "#f7f7f7"


def _tag(level: int, draw: int) -> str:
    return f"depl_L{int(level)}_d{int(draw)}"


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _relation(v: dict, short: bool) -> str:
    rel, eq = v['bottom_vs_equal_n'], v['equal_n_region']
    if rel == 'brackets':
        return f"brackets {eq}" if short else "brackets it"
    gap, share = abs(v['bottom_mean_minus_equal_n_rows']), abs(v['bottom_mean_minus_equal_n_share'])
    if short:
        return f"all draws {rel} {eq} ({_n(v['equal_n_rows_tv'])})"
    return f"all draws {rel} it, mean {gap:.0f} rows ({share:.1%}) {'short' if rel == 'below' else 'over'}"


def load_meta() -> list[dict]:
    metas = []
    for d in SCORE_DIRS:
        for path in sorted(d.glob("results_*.yaml")):
            metas.append({**yaml.safe_load(path.read_text(encoding="utf-8"))['meta'], 'file': path.name})
    return metas


def collect_facts() -> tuple[dict, pd.DataFrame]:
    facts = {}

    def put(key, value, source):
        facts[key] = (value, source)
        return value

    assert DRAW_ANCHOR == con.ANCHOR_REGION, f"draw build anchor {DRAW_ANCHOR} differs from con.ANCHOR_REGION"
    counts = pd.read_parquet(con.DEPLETION_COUNTS)
    grid = {(lvl, d) for lvl, reps in DEPLETION_GRID.items() for d in range(reps)}
    assert set(zip(counts['level'], counts['draw'])) == grid and len(counts) == len(grid), "counts do not match DEPLETION_GRID"
    assert counts['n_strata'].eq(N_STRATA).all() and counts['anchor_region'].eq(con.ANCHOR_REGION).all()
    assert counts['anchor_rows_full'].nunique() == 1 and counts['anchor_entities_full'].nunique() == 1
    counts = counts.assign(condition=[_tag(l, d) for l, d in zip(counts['level'], counts['draw'])])

    cond = pd.read_csv(con.TAB_DESC_DEPL_CONDITIONS)
    comp = pd.read_csv(con.TAB_DESC_DEPL_COMPOSITION)
    c = cond[['condition', 'context_rows', 'anchor_rows', 'anchor_rows_val', 'context_rows_val']].merge(
        counts, on='condition', how='inner', validate='one_to_one')
    assert len(c) == len(counts), "depletion conditions table lacks a condition - rerun build_desc_depletion"
    assert c['anchor_rows'].eq(c['anchor_rows_retained']).all() and c['context_rows'].eq(c['context_rows_retained']).all(), \
        "depletion conditions table disagrees with depletion_counts - rerun build_desc_depletion"
    und = cond.loc[cond['condition'].eq(UNDEPL)].iloc[0]
    assert und['context_rows'] == con.CONTEXT_ROWS and und['anchor_rows'] == counts['anchor_rows_full'].iloc[0]
    non_anchor = (cond['context_rows'] - cond['anchor_rows']).unique()
    assert len(non_anchor) == 1, "non-anchor context rows vary across conditions"

    base = comp.loc[comp['condition'].eq(UNDEPL) & comp['lvl3permid'].isin(con.TIER1_REGS)].set_index('lvl3permid')
    runs, regions = load_manifests()
    assert regions.groupby('region')['n_r'].nunique().eq(1).all(), "test rows per region differ between runs"
    n_test = regions.groupby('region')['n_r'].first()
    assert n_test.idxmax() == con.ANCHOR_REGION and base['rows'].idxmax() == con.ANCHOR_REGION, "anchor is not the largest tier-1 region"
    assert n_test.idxmin() == con.EQUAL_N_REGION and base['rows'].idxmin() == con.EQUAL_N_REGION, "equal-N region is not the smallest tier-1 region"

    put('anchor_region', con.ANCHOR_REGION, 'config.ANCHOR_REGION')
    put('anchor_rows_tv', int(counts['anchor_rows_full'].iloc[0]), 'depletion_counts.anchor_rows_full')
    put('anchor_entities_tv', int(counts['anchor_entities_full'].iloc[0]), 'depletion_counts.anchor_entities_full')
    put('anchor_test_rows', int(n_test[con.ANCHOR_REGION]), 'manifests n_r')
    put('tier1_test_rows', int(n_test.sum()), 'manifests sum n_r')
    put('anchor_test_share', n_test[con.ANCHOR_REGION] / n_test.sum(), 'manifests n_r')
    put('equal_n_region', con.EQUAL_N_REGION, 'config.EQUAL_N_REGION')
    eq_rows = put('equal_n_rows_tv', int(base.loc[con.EQUAL_N_REGION, 'rows']), 'desc_depletion_composition undepl')
    put('equal_n_entities_tv', int(base.loc[con.EQUAL_N_REGION, 'entities']), 'desc_depletion_composition undepl')
    put('equal_n_test_rows', int(n_test[con.EQUAL_N_REGION]), 'manifests n_r')
    put('context_rows_full', int(und['context_rows']), 'desc_depletion_conditions undepl')
    put('non_anchor_context_rows', int(non_anchor[0]), 'desc_depletion_conditions context - anchor')

    levels = sorted(DEPLETION_GRID, reverse=True)
    put('levels', "/".join(str(l) for l in levels), 'DEPLETION_GRID')
    put('repeats', "/".join(str(DEPLETION_GRID[l]) for l in levels), 'DEPLETION_GRID')
    put('n_conditions', len(counts), 'DEPLETION_GRID')
    bottom = min(levels)
    bot = counts.loc[counts['level'].eq(bottom), 'anchor_rows_retained']
    relation = 'brackets' if bot.min() <= eq_rows <= bot.max() else ('below' if bot.max() < eq_rows else 'above')
    put('bottom_level', bottom, 'DEPLETION_GRID')
    put('bottom_realised_min', int(bot.min()), 'depletion_counts')
    put('bottom_realised_max', int(bot.max()), 'depletion_counts')
    put('bottom_realised_mean', bot.mean(), 'depletion_counts')
    put('bottom_vs_equal_n', relation, 'derived')
    put('bottom_mean_minus_equal_n_rows', bot.mean() - eq_rows, 'derived')
    put('bottom_mean_minus_equal_n_share', (bot.mean() - eq_rows) / eq_rows, 'derived')

    put('n_strata', N_STRATA, 'build_depletion_draws.N_STRATA')
    over = put('overshoot_mean_rows', counts['overshoot_rows'].mean(), 'depletion_counts.overshoot_rows')
    ent_rows = put('anchor_rows_per_entity', counts['anchor_rows_full'].iloc[0] / counts['anchor_entities_full'].iloc[0], 'depletion_counts')
    put('overshoot_per_stratum_rows', over / N_STRATA, 'derived')
    put('overshoot_per_stratum_entities', over / N_STRATA / ent_rows, 'derived')

    metas = load_meta()
    assert len(metas) == len(runs)
    by_file = {m['file']: m for m in metas}
    by_cond = counts.set_index('condition')
    depl = [m for m in metas if m['condition'] != UNDEPL]
    for m in depl:
        ref = by_cond.loc[m['condition']]
        assert m['context rows'] == ref['context_rows_retained'] and m['context entities'] == ref['context_entities_retained'], \
            f"{m['file']}: context differs from the frozen draw"
    put('manifests_frozen_context', f"{len(depl)}/{len(depl)}", 'manifests context rows/entities vs depletion_counts')

    partitions = {tuple(m['test partition']) for m in metas}
    assert len(partitions) == 1, f"test partitions differ: {partitions}"
    test_rows, test_feats = next(iter(partitions))
    put('test_rows', int(test_rows), 'manifests test partition')
    put('test_features', int(test_feats), 'manifests test partition')
    put('manifests_test_set', f"{len(metas)}/{len(metas)}", 'manifests test partition + n_r')

    for m in depl:
        src = by_file[m['locked source']] if 'locked source' in m else next(
            u for u in metas if u['condition'] == UNDEPL and u['model'] == m['model'] and u['configuration'] == m['configuration'])
        assert m['hyperparameters'] == src['hyperparameters'], f"{m['file']}: hyperparameters differ from {src['file']}"
    put('manifests_locked_hp', f"{len(depl)}/{len(depl)}", 'manifests hyperparameters vs undepleted configuration')

    commits = {m['draw build commit'] for m in metas if 'draw build commit' in m}
    assert len(commits) == 1, f"draw-build commits {commits}"
    put('draw_build_commit', commits.pop(), 'manifests draw build commit')

    cidx = cond.set_index('condition')
    fitted = [m for m in depl if m['model'] in FITTED]
    for m in fitted:
        assert m['anchor val rows'] == cidx.loc[m['condition'], 'anchor_rows_val'], f"{m['file']}: anchor val rows"
        assert m['val rows'] == cidx.loc[m['condition'], 'context_rows_val'], f"{m['file']}: val rows"
    und_val = {m['val partition'][0] for m in metas if m['condition'] == UNDEPL and 'val partition' in m}
    assert und_val == {und['context_rows_val']}, f"undepleted val partition {und_val} vs {und['context_rows_val']}"
    bot_c = c.loc[c['level'].eq(bottom)]
    put('anchor_val_rows_undepl', int(und['anchor_rows_val']), 'desc_depletion_conditions')
    put('anchor_val_rows_bottom_min', int(bot_c['anchor_rows_val'].min()), 'desc_depletion_conditions')
    put('anchor_val_rows_bottom_max', int(bot_c['anchor_rows_val'].max()), 'desc_depletion_conditions')
    put('val_rows_undepl', int(und['context_rows_val']), 'desc_depletion_conditions')
    put('val_rows_bottom_min', int(bot_c['context_rows_val'].min()), 'desc_depletion_conditions')
    put('val_rows_bottom_max', int(bot_c['context_rows_val'].max()), 'desc_depletion_conditions')

    ftt = pd.DataFrame([{'level': m['level'], 'epochs': m['epochs run'], 'steps': m['refit steps'], 'undepl_steps': m['undepl steps']}
                        for m in depl if m['model'] == 'ftt'])
    assert len(ftt) == len(counts) and ftt['undepl_steps'].nunique() == 1
    ftt_und = [m for m in metas if m['model'] == 'ftt' and m['condition'] == UNDEPL]
    assert len(ftt_und) == 1
    put('ftt_undepl_epochs', int(ftt_und[0]['epochs trained']), 'ftt undepleted manifest')
    put('ftt_undepl_steps', int(ftt['undepl_steps'].iloc[0]), 'ftt depleted manifests undepl steps')
    for lvl in (max(levels), bottom):
        s = ftt.loc[ftt['level'].eq(lvl)]
        put(f'ftt_steps_L{lvl}_min', int(s['steps'].min()), 'ftt manifests refit steps')
        put(f'ftt_steps_L{lvl}_max', int(s['steps'].max()), 'ftt manifests refit steps')

    print(f"    [+++] {len(facts)} facts collected, all design checks passed")
    per_level = c[['level', 'draw', 'anchor_rows_retained', 'target_mean_retained']]
    return facts, per_level


FONT = 'DejaVu Sans'


def _box(ax, x: float, y: float, w: float, h: float, title: str, lines: list[str], edge: str, face: str,
         title_fs: float = 17, fs: float = 14.5) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=face, ec=edge, lw=2))
    ax.text(x + w / 2, y + h - 1.1, title, ha='center', va='top', fontsize=title_fs, fontweight='bold', color=edge,
            fontfamily=FONT)
    ax.text(x + w / 2, y + h - 4.6, "\n".join(lines), ha='center', va='top', fontsize=fs, color=C_TXT,
            linespacing=1.55, fontfamily=FONT, multialignment='center')


def _arrow(ax, start: tuple, end: tuple) -> None:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color="#8a8a8a", lw=2.2, mutation_scale=24, shrinkA=0, shrinkB=0))


def _list_box(ax, title: str, lines: list[str], edge: str, face: str, footer: str | None = None) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.98, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=face, ec=edge, lw=2))
    ax.annotate(title, xy=(0.05, 1.0), xytext=(0, -14), textcoords='offset points', va='top', fontsize=17,
                fontweight='bold', color=edge, fontfamily=FONT)
    ax.annotate("\n".join(lines), xy=(0.05, 1.0), xytext=(0, -50), textcoords='offset points', va='top',
                fontsize=14.5, color=C_TXT, linespacing=1.6, fontfamily=FONT)
    if footer:
        ax.text(0.05, 0.04, footer, va='bottom', fontsize=11.5, color="#666666", fontstyle='italic', fontfamily=FONT)


def plot_flow(ax, v: dict) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    rel = {'brackets': "brackets it", 'below': "all draws below", 'above': "all draws above"}[v['bottom_vs_equal_n']]
    _box(ax, 0, 17, 18, 15, "Anchor · East Asia", [
        f"{_n(v['anchor_rows_tv'])} rows", f"{_n(v['anchor_entities_tv'])} entities",
        f"{v['anchor_test_share']:.0%} of Tier-1 test"], C_MAIN, FACE_MAIN)
    _box(ax, 0, 0, 18, 15, f"Equal-N · {v['equal_n_region']}", [
        f"{_n(v['equal_n_rows_tv'])} rows",
        f"L{_n(v['bottom_level'])}: {_n(v['bottom_realised_min'])}–{_n(v['bottom_realised_max'])}", rel], C_REF, FACE_REF)
    _box(ax, 22, 6, 19, 20, "Entity-level draw", [
        "unit: entity", f"{v['n_strata']} stratified bins", "∝ row mass per bin",], C_MAIN, FACE_MAIN)
    _box(ax, 45, 6, 17, 20, "Frozen draws", [
        f"{v['n_conditions']} conditions", f"draws {v['repeats']}", "identical context",
        f"✓ {v['manifests_frozen_context']}"], C_MAIN, FACE_MAIN)
    _box(ax, 66, 17, 17, 15, "Fitted arm", ["RF · XGBoost · FT-T", "refit on", "depleted data"], "#555555", FACE_GREY)
    _box(ax, 66, 0, 17, 15, "In-context arm", ["TabICL · TabPFN-3", "context removed", "no refit"], "#555555", FACE_GREY)
    _box(ax, 87, 6, 13, 20, "One test set", [f"{_n(v['test_rows'])} rows", f"✓ {v['manifests_test_set']} runs"],
         C_MAIN, FACE_MAIN)
    _arrow(ax, (18, 24.5), (22, 18))
    _arrow(ax, (41, 16), (45, 16))
    _arrow(ax, (62, 18), (66, 24.5))
    _arrow(ax, (62, 14), (66, 7.5))
    _arrow(ax, (83, 24.5), (87, 18))
    _arrow(ax, (83, 7.5), (87, 14))


def plot_grid(ax, v: dict, per_level: pd.DataFrame) -> None:
    levels = sorted(DEPLETION_GRID, reverse=True)
    ypos = {lvl: i for i, lvl in enumerate(levels)}
    full_y = -1
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.axvline(v['equal_n_rows_tv'], color=C_REF, ls=':', lw=2, zorder=1)
    ax.text(v['equal_n_rows_tv'] * 1.08, full_y, f"{v['equal_n_region']} ({_n(v['equal_n_rows_tv'])})", color=C_REF,
            fontsize=13, va='center', fontfamily=FONT)
    for lvl in levels:
        g = per_level.loc[per_level['level'].eq(lvl)]
        y = ypos[lvl]
        ax.scatter([lvl], [y], marker='|', s=700, color="#a70000", lw=2, zorder=2)
        jitter = np.linspace(-0.2, 0.2, len(g)) if len(g) > 1 else np.zeros(1)
        ax.scatter(g['anchor_rows_retained'], y + jitter, s=55, color=C_MAIN, zorder=3, edgecolor='white', lw=0.6)
        spread = g['target_mean_retained'].max() - g['target_mean_retained'].min()
        ax.text(0.97, y, f"{spread:.3f}", transform=tr, va='center', ha='right', fontsize=13, color="#555555", fontfamily=FONT)
    ax.scatter([v['anchor_rows_tv']], [full_y], marker='D', s=90, facecolor='white', edgecolor=C_MAIN, lw=2, zorder=3)
    ax.text(0.97, full_y, "spread of\ntarget mean", transform=tr, va='center', ha='right', fontsize=11.5, color="#555555",
            fontstyle='italic', fontfamily=FONT)
    ax.set_yticks([full_y] + [ypos[l] for l in levels])
    ax.set_yticklabels(["full anchor"] + [f"L{_n(l)}  ×{DEPLETION_GRID[l]}" for l in levels], fontsize=14, fontfamily=FONT)
    ax.set_ylim(len(levels) - 0.4, full_y - 0.7)
    ax.set_xscale('log')
    ax.set_xticks(levels + [v['anchor_rows_tv']])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _n(x)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis='x', labelsize=13)
    ax.set_xlim(min(levels) * 0.75, v['anchor_rows_tv'] * 4.5)
    ax.set_xlabel("realised anchor rows    | target   ● draw", fontsize=13.5, labelpad=8, fontfamily=FONT)
    ax.set_title("Depletion grid · more draws where N is small", fontsize=17, fontweight='bold', color=C_MAIN,
                 loc='left', pad=12, fontfamily=FONT)
    ax.grid(True, axis='x', linestyle="--", alpha=0.4)
    ax.grid(False, axis='y')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#cccccc')


def plot_schematic(f: dict, per_level: pd.DataFrame) -> None:
    v = {k: val for k, (val, _) in f.items()}
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 14.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.42, 2.5, 2.6], width_ratios=[1.35, 1.0], hspace=0.18, wspace=0.1)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0.05), 1, 0.9, boxstyle="round,pad=0,rounding_size=0.02", fc=C_MAIN, ec=C_MAIN))
    ax.text(0.5, 0.5, "Depletion Experiment - Varying the Anchor's Training N", ha='center', va='center', fontsize=24,
            fontweight='bold', color='white', fontfamily=FONT)

    plot_flow(fig.add_subplot(gs[1, :]), v)
    plot_grid(fig.add_subplot(gs[2, 0]), v, per_level)

    right = gs[2, 1].subgridspec(2, 1, hspace=0.12, height_ratios=[1.25, 1.0])
    bottom = v['bottom_level']
    _list_box(fig.add_subplot(right[0]), "Held fixed", [
        "✓  test set",
        "✓  non-anchor regions",
        "✓  hyperparameters",
        "✓  preprocessing constants",
    ], C_MAIN, FACE_MAIN, footer="")
    _list_box(fig.add_subplot(right[1]), "Varied across depletion levels", [
        "validation partition shrinks with fit",
        "FT-T epochs are re-derived",
    ], C_REF, FACE_REF)

    con.VIZ_DESC_DEPL_DESIGN.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(con.VIZ_DESC_DEPL_DESIGN, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {con.VIZ_DESC_DEPL_DESIGN.name}")


if __name__ == '__main__':
    print(f"\n[°°°] building depletion design schematic - anchor {con.ANCHOR_REGION} [°°°]\n")
    facts, per_level = collect_facts()

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{'fact': k, 'value': val, 'source': src} for k, (val, src) in facts.items()])
    table.to_csv(con.TAB_DESC_DEPL_DESIGN, index=False)
    print(f"      [---] written {con.TAB_DESC_DEPL_DESIGN.name} - {len(table)} facts")

    plot_schematic(facts, per_level)
    print(f"\n{table.to_string(index=False)}")
    print("\n[°°°] depletion design schematic complete [°°°]")