##################################################
'''
src.exploration.raw_data.determine_esg_coverage_asymmetry

input:      raw_esg_scores.parquet, raw_ws_universe.parquet, ref_ws_nation_xwalk.parquet
purpose:    make the region dependence of ESG-rating coverage visible:
            coverage = rated firm-years / Worldscope universe firm-years per lvl3
            region (Worldscope nation crosswalk), pooled over the window and per
            year; diagnose how far each region's coverage is interpretable
            (resolution rate of rated cmpids, outside-Worldscope share,
            expected misassignment); regions with fy_resolution below
            COVERAGE_MIN_RESOLUTION are marked as not interpretable in both panels.
            the rated side is matched through ws_cmpid_bridged (link resolved via the
            CUSIP bridge, see build_raw_esg_scores); the unbridged resolution is
            reported alongside. the panel itself does not use the bridge - figure
            captions must say so
output:     VIZ_ESG_COVERAGE - panel A: coverage per region with resolution rate,
            panel B: coverage per Tier-1 region over time

            run 2026-09-26, BEFORE the CUSIP bridge (numerator via worldscopecmpid):
            rated (cmpid, year) keys 126047, found in the universe 88283 (0.7004);
            universe firm-years rated 88283 of 927562 (0.0952)
            global coverage per year: 2009 0.0446 -> 2015 0.0657 -> 2020 0.1155
                                      -> 2024 0.1548 -> 2025 0.1509

            lvl3permid  firms  coverage          tier  rated_ent  fy_resol  exp_misas
                100277    586    0.2813        Tier 1        171    0.9989     0.0028
                100334   3827    0.2096        Tier 1       1010    0.9717     0.0047
                100223   6594    0.2009        Tier 1       1758    0.9638     0.0170
                100024   3588    0.1693        Tier 1        724    0.9973     0.0093
                103384   2362    0.1546        Tier 1        505    0.9658     0.0258
                100279   2477    0.1180        Tier 1        341    0.9971     0.0098
                100089  22425    0.0941        Tier 1       2887    0.9933     0.0026
                103401   3212    0.0913        Tier 1        542    0.9796     0.0062
                100276   6656    0.0912        Tier 1       1203    0.9933     0.0084
                100218    484    0.0848        Tier 1         93    1.0000     0.0000
                110000     24    0.0634        Tier 3          2    1.0000     0.0000
                100278   7469    0.0515        Tier 1        755    0.9956     0.0004
                100090   2977    0.0412        Tier 1        132    0.9965     0.0028
                100057      2    0.0400  not in panel          1    1.0000     0.0000
                100219  20043    0.0322        Tier 1       5079    0.1398     0.0426
                100060    123    0.0247        Tier 2          6    0.9211     0.0269
                  <NA>     93    0.0183      unmapped        884    0.9286        NaN
                100087    289    0.0154        Tier 2         12    1.0000     0.0084
                100332    264    0.0126        Tier 2         14    1.0000     0.0000

            findings: Tier-1 coverage spans 0.28 (100277) to 0.04 (100090);
            Tier-2 regions hold 123-289 Worldscope firms at 0.013-0.025 coverage;
            100219 is not interpretable (fy_resolution 0.14: permorgref links
            most rated NA firms via CUSIPs instead of Worldscope ids, see
            diagnose_cmpid_resolution; a CUSIP bridge would lift it to 0.93,
            see check_cusip_link); misassignment <= 0.043 everywhere;
            100057 (2 firms) and 110000 (24 firms) are single-firm noise
'''
##################################################

import pandas as pd
import numpy as np
import config as con
import matplotlib.pyplot as plt

LINK = 'ws_cmpid_bridged'   
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print(f"\n[°°°] determining ESG-rating coverage per region {con.YEAR_MIN}-{con.YEAR_MAX} [°°°]\n")

# load data
scores = pd.read_parquet(con.RAW_ESG_SCORES)
universe = pd.read_parquet(con.RAW_WS_UNIVERSE)
print(f"    [+++] loaded scores {scores.shape}, universe {universe.shape}")

assert not universe.duplicated(['worldscopecmpid', 'year']).any(), "duplicate (cmpid, year) in the universe"



print(f"    [+++] flagging rated universe firm-years...")


rated_keys = (scores.dropna(subset=[LINK])[[LINK, 'year']]
              .rename(columns={LINK: 'worldscopecmpid'})
              .drop_duplicates())
key_u = pd.MultiIndex.from_frame(universe[['worldscopecmpid', 'year']])
key_r = pd.MultiIndex.from_frame(rated_keys)


universe['rated'] = key_u.isin(key_r)
n_found = key_r.isin(key_u).sum()
assert universe['rated'].sum() == n_found, \
    f"rated firm-years {universe['rated'].sum()} != rated keys found in the universe {n_found}"

print(f"      [---] rated (cmpid, year) keys: {len(key_r)}, found in the universe: {n_found} "
      f"({n_found / len(key_r):.4f})")
print(f"      [---] universe firm-years rated: {universe['rated'].sum()} of {len(universe)} "
      f"({universe['rated'].mean():.4f})")



print(f"    [+++] computing coverage per region and per region-year...")
tier_map = {**{r: 'Tier 1' for r in con.TIER1_REGS},
            **{r: 'Tier 2' for r in con.TIER2_REGS},
            **{r: 'Tier 3' for r in con.TIER3_REGS}}


pooled = (universe.groupby('lvl3permid', dropna=False)
          .agg(firm_years=('rated', 'size'), rated=('rated', 'sum'), firms=('worldscopecmpid', 'nunique'))
          .reset_index())
pooled['coverage'] = pooled['rated'] / pooled['firm_years']
pooled['tier'] = pooled['lvl3permid'].map(tier_map).astype('string')
pooled.loc[pooled['lvl3permid'].isna(), 'tier'] = 'unmapped'
pooled['tier'] = pooled['tier'].fillna('not in panel')
pooled = pooled.sort_values('coverage', ascending=False).reset_index(drop=True)

yearly = (universe.groupby(['lvl3permid', 'year'], dropna=False)
          .agg(firm_years=('rated', 'size'), rated=('rated', 'sum'))
          .reset_index())
yearly['coverage'] = yearly['rated'] / yearly['firm_years']

for name, frame in (('pooled', pooled), ('yearly', yearly)):
    assert frame['coverage'].between(0, 1).all(), f"{name}: coverage outside [0, 1]"
    assert frame['firm_years'].sum() == len(universe), f"{name}: firm-year cells do not sum to the universe"
    assert frame['rated'].sum() == universe['rated'].sum(), f"{name}: rated cells do not sum to the universe"

print(f"      [---] coverage per region, pooled firm-years {con.YEAR_MIN}-{con.YEAR_MAX}:")
print(pooled.to_string(index=False, float_format='{:.4f}'.format))
print(f"      [---] global coverage per year: "
      f"{universe.groupby('year')['rated'].mean().round(4).to_dict()}")
print(f"      [---] coverage per region and year:")
print(yearly.dropna(subset=['lvl3permid'])
      .pivot(index='lvl3permid', columns='year', values='coverage')
      .to_string(float_format='{:.3f}'.format))



print(f"    [+++] diagnosing interpretability per region...")
xwalk = pd.read_parquet(con.REF_WS_NATION_XWALK)


ent = scores.drop_duplicates('orgpermid')[['orgpermid', 'worldscopecmpid', LINK, 'lvl3permid']]
ent = ent.assign(has_cmpid=ent['worldscopecmpid'].notna(),
                 resolved=ent[LINK].isin(set(universe['worldscopecmpid'])))
diag = ent.groupby('lvl3permid', dropna=False).agg(rated_entities=('orgpermid', 'size'),
                                                   with_cmpid=('has_cmpid', 'sum'),
                                                   resolved=('resolved', 'sum'))
diag['share_no_cmpid'] = 1 - diag['with_cmpid'] / diag['rated_entities']
diag['entity_resolution'] = diag['resolved'] / diag['with_cmpid']


fy = scores.loc[scores['worldscopecmpid'].notna(), ['worldscopecmpid', LINK, 'year', 'lvl3permid']]
fy = fy.assign(found=pd.MultiIndex.from_frame(fy[[LINK, 'year']]).isin(key_u),
               found_raw=pd.MultiIndex.from_frame(fy[['worldscopecmpid', 'year']]).isin(key_u))
diag['fy_resolution'] = fy.groupby('lvl3permid', dropna=False)['found'].mean()
diag['fy_resolution_unbridged'] = fy.groupby('lvl3permid', dropna=False)['found_raw'].mean()


firms = (universe.drop_duplicates('worldscopecmpid')
         .merge(xwalk[['ws_nation', 'purity']], on='ws_nation', how='left', validate='many_to_one'))
exp_misassigned = (1 - firms['purity']).groupby(firms['lvl3permid'], dropna=False).mean()


report = (pooled.set_index('lvl3permid')
          .join(diag[['rated_entities', 'share_no_cmpid', 'entity_resolution', 'fy_resolution',
                      'fy_resolution_unbridged']], how='outer')
          .join(exp_misassigned.rename('exp_misassigned'), how='left')
          .sort_values('coverage', ascending=False))
print(f"      [---] coverage (crosswalk region) with rated-side diagnostics (Refinitiv region):")
print(report.to_string(float_format='{:.4f}'.format))


### plot ---------------------------------------------------------
print(f"    [+++] plotting coverage asymmetry...")
plt.style.use('seaborn-v0_8-whitegrid')

tier_colors = {'Tier 1': "#1f4e79", 'Tier 2': "#6fa8dc", 'Tier 3': "#a6a6a6", 'not in panel': "#a6a6a6"}
color_resolution = "#d9534f"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.25]})

plot_a = report.loc[report.index.notna() & report['coverage'].notna()].sort_values('coverage')
y = np.arange(len(plot_a))
labels = [f"{int(r)}{'*' if t == 'Tier 2' else ''}  (n={n:,})".replace(',', '.')
          for r, t, n in zip(plot_a.index, plot_a['tier'], plot_a['firms'])]

unreliable = (plot_a['fy_resolution'] < con.COVERAGE_MIN_RESOLUTION).to_numpy()
bar_colors = plot_a['tier'].map(tier_colors).tolist()
print(f"      [---] regions below resolution {con.COVERAGE_MIN_RESOLUTION}: {[int(r) for r in plot_a.index[unreliable]]}")

ax_a.barh(y[~unreliable], plot_a['coverage'][~unreliable], height=0.7, zorder=2,
          color=[c for c, u in zip(bar_colors, unreliable) if not u])
ax_a.barh(y[unreliable], plot_a['coverage'][unreliable], height=0.7, zorder=2, color='white', hatch='///',
          edgecolor=[c for c, u in zip(bar_colors, unreliable) if u])
ax_a.scatter(plot_a['fy_resolution'], y, marker='D', s=28, color=color_resolution, zorder=3)
for yi, cov, res in zip(y[unreliable], plot_a['coverage'][unreliable], plot_a['fy_resolution'][unreliable]):
    ax_a.text(max(cov, res) + 0.025, yi, f"not interpretable - {res:.0%} of rated firm-years resolvable",
              va='center', fontsize=8, color='#555555', fontstyle='italic')

ax_a.set_yticks(y)
ax_a.set_yticklabels(labels, fontsize=9)
ax_a.set_xlim(0, 1.02)
ax_a.set_title(f"Rating Coverage per Region\nrated / Worldscope firm-years, {con.YEAR_MIN}-{con.YEAR_MAX}, CUSIP-bridged link",
               pad=15, fontweight='bold')
ax_a.set_xlabel("Share of firm-years", labelpad=10)
ax_a.set_ylabel("Level 3 PermID (n = Worldscope firms)")

present = set(plot_a['tier'])
handles = [Patch(facecolor=tier_colors[t], label=t) for t in ('Tier 1', 'Tier 2') if t in present]
if present & {'Tier 3', 'not in panel'}:
    handles.append(Patch(facecolor="#a6a6a6", label='Tier 3 / not in panel'))
if unreliable.any():
    handles.append(Patch(facecolor='white', edgecolor="#1f4e79", hatch='///',
                         label=f"Resolution < {con.COVERAGE_MIN_RESOLUTION:.0%}: not interpretable"))
handles.append(Line2D([], [], ls='none', marker='D', ms=6, color=color_resolution,
                      label='Rated firm-years resolvable in Worldscope (CUSIP-bridged)'))
ax_a.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
ax_a.grid(True, axis='x', linestyle="--", alpha=0.5)
ax_a.grid(False, axis='y')

t1 = yearly.loc[yearly['lvl3permid'].isin(con.TIER1_REGS)]
order = [r for r in plot_a.index[::-1] if r in con.TIER1_REGS]
cmap = plt.get_cmap('tab20')

unreliable_regs = set(plot_a.index[unreliable])

for i, reg in enumerate(order):
    s = t1.loc[t1['lvl3permid'] == reg].sort_values('year')
    if reg in unreliable_regs:
        ax_b.plot(s['year'], s['coverage'], color='#a6a6a6', lw=1.5, ls=':', marker='o', ms=3,
                  label=f"{reg} (not interpretable)")
    else:
        ax_b.plot(s['year'], s['coverage'], color=cmap(i % 20), lw=1.5, marker='o', ms=3, label=str(reg))
glob = universe.groupby('year')['rated'].mean()
ax_b.plot(glob.index, glob.values, color='#333333', lw=2.2, ls='--', label='All regions')

ax_b.set_ylim(0, max(t1['coverage'].max(), glob.max()) * 1.12)
ax_b.axvspan(con.YEAR_MAX - 0.5, con.YEAR_MAX + 0.5, color='#cccccc', alpha=0.4, zorder=0)
ax_b.text(con.YEAR_MAX, ax_b.get_ylim()[1], "possible\nfinalization lag", ha='center', va='top',
          fontsize=8, color='#555555', fontstyle='italic')

ax_b.set_xticks(range(con.YEAR_MIN, con.YEAR_MAX + 1, 2))
ax_b.set_title("Rating Coverage over Time\nTier-1 regions", pad=15, fontweight='bold')
ax_b.set_xlabel("Year", labelpad=10)
ax_b.set_ylabel("Share of firm-years")
ax_b.legend(title="Level 3 PermID", bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=8)
ax_b.grid(True, axis='y', linestyle="--", alpha=0.5)
ax_b.grid(False, axis='x')

for ax in (ax_a, ax_b):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

plt.tight_layout()
plt.savefig(con.VIZ_ESG_COVERAGE, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n[°°°] {con.VIZ_ESG_COVERAGE.name} saved [°°°]")