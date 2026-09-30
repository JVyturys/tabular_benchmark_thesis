##################################################
'''
src.exploration.WRDS_database.determine_lvl3_region_names

input:      ref_geo_table.parquet, raw SQL: tr_common.tmcregncntrymap
purpose:    resolve every lvl3permid of Refinitiv's regional taxonomy to its
            constituent countries (lvl5isocntry) as the evidence base for the
            hand-assigned labels in con.REGION_LABELS;
output:     printed rollup per lvl3permid: iso_taxonomy, iso_panel (entities per
            country), n_entities, in_panel, tier, region_label

            run 2026-09-30: taxonomy 240 (lvl3, iso) pairs,
 
            lvl3permid  tier      n  region_label                     top country (share)
            100089  Tier 1   2743  Eastern Asia                     CN 0.52
            100223  Tier 1   1396  Northern Europe                  GB 0.51
            100276  Tier 1   1055  South-eastern Asia               MY 0.52
            100334  Tier 1    925  Western Europe                   DE 0.32
            100278  Tier 1    693  Southern Asia                    IN 0.98
            100024  Tier 1    668  Australia and New Zealand        AU 0.90
            100219  Tier 1    609  Northern America                 CA 0.79
            103401  Tier 1    459  Western Asia                     TR 0.25
            103384  Tier 1    452  Latin America and the Caribbean  BR 0.36
            100279  Tier 1    325  Southern Europe                  IT 0.49
            100277  Tier 1    151  Southern Africa                  ZA 0.99
            100090  Tier 1    124  Eastern Europe                   RU 0.40
            100218  Tier 1     88  Northern Africa                  MA 0.52
            100087  Tier 2     11  Eastern Africa                   KE 0.36
            100332  Tier 2      9  Western Africa                   NG 0.67
            100060  Tier 2      5  Central Asia                     KZ 1.00
            110000  Tier 3      2  Melanesia                        PG 1.00

'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

print(f"\n[°°°] determining lvl3 region names from the taxonomy [°°°]\n")

geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid', 'lvl5isocntry'])
print(f"    [+++] loaded ref_geo_table {geo.shape}")

assert geo['orgpermid'].is_unique, \
    "ref_geo_table duplicates orgpermid - entity counts per region would double-count"
assert geo['lvl3permid'].dtype.kind == 'i', \
    f"ref_geo_table carries lvl3permid as {geo['lvl3permid'].dtype}, tier lists and REGION_LABELS keys are integer"
assert geo['lvl5isocntry'].notna().all(), \
    f"{geo['lvl5isocntry'].isna().sum()} entities without lvl5isocntry - the pair check (iv) cannot cover them"

db = wrds.Connection(wrds_username=lc.wrds_log)

print(f"    [+++] probing tr_common.tmcregncntrymap schema...")
schema = db.describe_table(library='tr_common', table='tmcregncntrymap')
REQUIRED_COLS = {'lvl3permid', 'lvl5isocntry'}
missing_cols = REQUIRED_COLS - set(schema['name'])
assert not missing_cols, \
    f"tmcregncntrymap lacks {sorted(missing_cols)} - columns present: {sorted(schema['name'])}"
for _, col in schema.loc[schema['name'].isin(REQUIRED_COLS)].iterrows():
    print(f"      [---] {col['name']}: {col['type']}, nullable={col['nullable']}")

print(f"    [+++] pulling lvl3permid -> lvl5isocntry pairs from tmcregncntrymap...")
tax = db.raw_sql("""
    SELECT DISTINCT lvl3permid, lvl5isocntry
    FROM tr_common.tmcregncntrymap
""")
print(f"      [---] distinct pairs: {len(tax)}")

incomplete = tax['lvl3permid'].isna() | tax['lvl5isocntry'].isna()
print(f"      [---] pairs dropped for a null lvl3permid or lvl5isocntry: {incomplete.sum()}")
if incomplete.any():
    print(tax.loc[incomplete].to_string(index=False))
tax = tax.loc[~incomplete].copy()


tax['lvl3permid'] = pd.to_numeric(tax['lvl3permid']).astype('Int64').astype('int64')

regions_per_iso = tax.groupby('lvl5isocntry')['lvl3permid'].nunique()
multi = regions_per_iso.index[regions_per_iso > 1]
assert multi.empty, \
    f"ISO codes under several lvl3permid:\n{tax.loc[tax['lvl5isocntry'].isin(multi)].sort_values('lvl5isocntry').to_string(index=False)}"
print(f"      [---] taxonomy: {tax['lvl3permid'].nunique()} lvl3 regions, {tax['lvl5isocntry'].nunique()} countries")


print(f"    [+++] cross-checking panel geography against the taxonomy...")
panel_regs = set(geo['lvl3permid'])
tax_regs = set(tax['lvl3permid'])

unknown = panel_regs - tax_regs
assert not unknown, f"panel regions absent from tmcregncntrymap: {sorted(unknown)}"

tier_regs = [*con.TIER1_REGS, *con.TIER2_REGS, *con.TIER3_REGS]
assert len(tier_regs) == len(set(tier_regs)), "tier lists overlap - a region would carry two tiers"
assert panel_regs == set(tier_regs), \
    f"untiered panel regions: {sorted(panel_regs - set(tier_regs))}, tiered regions absent from the panel: {sorted(set(tier_regs) - panel_regs)}"
print(f"      [---] panel regions: {len(panel_regs)} of {len(tax_regs)} taxonomy regions, all tiered")

pairs = geo[['lvl3permid', 'lvl5isocntry']].drop_duplicates()
key_tax = pd.MultiIndex.from_frame(tax[['lvl3permid', 'lvl5isocntry']])
orphans = pairs.loc[~pd.MultiIndex.from_frame(pairs).isin(key_tax)]
assert orphans.empty, f"panel pairs absent from the taxonomy:\n{orphans.to_string(index=False)}"
print(f"      [---] panel (lvl3permid, lvl5isocntry) pairs: {len(pairs)}, all in the taxonomy")

for k, (lvl3, iso) in con.DOMICILE_OVERRIDES.items():
    assert (lvl3, iso) in key_tax, f"override {k} -> ({lvl3}, {iso}) contradicts the taxonomy"
    print(f"      [---] override domcntrypermid {k} -> ({lvl3}, {iso}) confirmed by the taxonomy")


print(f"    [+++] building the lvl3 rollup...")
iso_taxonomy = (tax.sort_values('lvl5isocntry')
                .groupby('lvl3permid')['lvl5isocntry']
                .agg(', '.join)
                .rename('iso_taxonomy'))

ent_per_iso = (geo.groupby(['lvl3permid', 'lvl5isocntry'], dropna=False).size()
               .rename('n').reset_index()
               .sort_values(['lvl3permid', 'n', 'lvl5isocntry'], ascending=[True, False, True]))
assert ent_per_iso['n'].sum() == len(geo), \
    f"per-country cells sum to {ent_per_iso['n'].sum()}, ref_geo_table holds {len(geo)} entities"
iso_panel = (ent_per_iso.assign(cell=ent_per_iso['lvl5isocntry'] + ' ' + ent_per_iso['n'].astype(str))
             .groupby('lvl3permid')['cell']
             .agg(', '.join)
             .rename('iso_panel'))
n_entities = geo.groupby('lvl3permid', dropna=False).size().rename('n_entities')

tier_of = {**{r: 'Tier 1' for r in con.TIER1_REGS},
           **{r: 'Tier 2' for r in con.TIER2_REGS},
           **{r: 'Tier 3' for r in con.TIER3_REGS}}

rollup = iso_taxonomy.to_frame().join(iso_panel, how='left').join(n_entities, how='left')
rollup['n_entities'] = rollup['n_entities'].fillna(0).astype('int64')
rollup['in_panel'] = rollup.index.isin(panel_regs)
rollup['tier'] = rollup.index.map(tier_of).fillna('not in panel')
rollup['region_label'] = rollup.index.map(con.REGION_LABELS)
rollup = rollup.sort_values(['in_panel', 'n_entities'], ascending=False)

assert rollup.index.is_unique, "rollup duplicates lvl3permid"
assert rollup['n_entities'].sum() == len(geo), \
    f"rollup carries {rollup['n_entities'].sum()} entities, ref_geo_table holds {len(geo)} - a panel region fell out of the join"
assert (rollup['iso_panel'].notna() == rollup['in_panel']).all(), "iso_panel populated outside the panel regions or missing inside"
print(f"      [---] rollup: {len(rollup)} regions, {rollup['in_panel'].sum()} in the panel, "
      f"{rollup['n_entities'].sum()} entities")

print(f"      [---] lvl3 rollup (panel: entities per country):")
for reg, row in rollup.iterrows():
    label = row['region_label'] if pd.notna(row['region_label']) else '<unlabelled>'
    print(f"\n        {reg}  {row['tier']:<12}  n={row['n_entities']:<6}  label: {label}")
    print(f"          taxonomy: {row['iso_taxonomy']}")
    if row['in_panel']:
        print(f"          panel:    {row['iso_panel']}")



print(f"\n    [+++] checking con.REGION_LABELS...")
labels = pd.Series(con.REGION_LABELS)
assert labels.map(lambda v: isinstance(v, str) and bool(v.strip())).all(), \
    f"REGION_LABELS holds empty or non-string labels: {labels[~labels.map(lambda v: isinstance(v, str) and bool(v.strip()))].to_dict()}"
assert labels.is_unique, \
    f"REGION_LABELS reuses labels - plots would merge regions: {labels[labels.duplicated(keep=False)].to_dict()}"
stray = set(labels.index) - tax_regs
assert not stray, f"REGION_LABELS names regions outside the taxonomy: {sorted(stray)}"

unlabelled = rollup.loc[rollup['in_panel'] & rollup['region_label'].isna()]
assert unlabelled.empty, (
    f"{len(unlabelled)} panel regions without a label in con.REGION_LABELS - assign from the rollup above:\n"
    + unlabelled[['tier', 'n_entities', 'iso_panel']].to_string())

print(f"\n[°°°] all {rollup['in_panel'].sum()} panel regions labelled [°°°]")