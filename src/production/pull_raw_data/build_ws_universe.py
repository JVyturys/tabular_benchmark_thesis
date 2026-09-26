##################################################
'''
src.production.pull_raw_data.build_ws_universe

input:      raw SQL: tr_worldscope.wrds_ws_funda, tr_worldscope.wrds_ws_company,
            tr_common.permorgref, tr_common.tmcregncntrymap
purpose:    build the Worldscope annual company universe, the denominator of
            the rating-coverage analysis: base cmpids with a freq='A' record per
            year in YEAR_MIN-YEAR_MAX, their Worldscope nation (item6026), and a
            region from an empirical crosswalk item6026 -> lvl3permid derived
            from permorgref-linked cmpids (DOMICILE_OVERRIDES applied), so that
            linked and unlinked firms share one region definition;
            nations with a tied modal region get no crosswalk entry
output:     raw_ws_universe.parquet with columns worldscopecmpid, year, ws_nation, lvl3permid
            ref_ws_nation_xwalk.parquet with columns ws_nation, lvl3permid,
            n_votes (linked cmpid votes for that nation), purity (modal share)

'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

print(f"\n[°°°] building Worldscope annual universe {con.YEAR_MIN}-{con.YEAR_MAX} [°°°]\n")
db = wrds.Connection(wrds_username=lc.wrds_log)

print(f"    [+++] pulling annual (cmpid, year) records from wrds_ws_funda...")
universe = db.raw_sql(f"""
    SELECT DISTINCT item6105 AS worldscopecmpid, year_ AS year
    FROM tr_worldscope.wrds_ws_funda
    WHERE freq = 'A'
        AND item6105 IS NOT NULL
        AND year_ BETWEEN {con.YEAR_MIN} AND {con.YEAR_MAX}
""")
print(f"      [---] distinct (cmpid, year) rows: {len(universe)}")

print(f"    [+++] pulling Worldscope nation (item6026) from wrds_ws_company...")
nation = db.raw_sql("""
    SELECT item6105 AS worldscopecmpid, item6026 AS ws_nation
    FROM tr_worldscope.wrds_ws_company
    WHERE item6105 IS NOT NULL
""")
print(f"      [---] company rows: {len(nation)}")

print(f"    [+++] pulling permorgref links with Refinitiv geography...")
links = db.raw_sql("""
    SELECT DISTINCT p.worldscopecmpid, p.domcntrypermid, g.lvl3permid
    FROM tr_common.permorgref p
    LEFT JOIN tr_common.tmcregncntrymap g
        ON p.domcntrypermid = g.lvl5permid
    WHERE p.worldscopecmpid IS NOT NULL
""")
print(f"      [---] distinct (cmpid, domicile, lvl3) link rows: {len(links)}")


TOP_N = 15
BASE_CMPID = r'.{8}\d'         
SHARE_CLASS = r'.{8}[A-Z]'     

print(f"    [+++] restricting to base cmpids and casting dtypes...")
universe['worldscopecmpid'] = universe['worldscopecmpid'].astype('string')
nation['worldscopecmpid'] = nation['worldscopecmpid'].astype('string')
links['worldscopecmpid'] = links['worldscopecmpid'].astype('string')
ws_ids = set(universe['worldscopecmpid']) | set(nation['worldscopecmpid'])

is_base_u = universe['worldscopecmpid'].str.fullmatch(BASE_CMPID)
is_base_n = nation['worldscopecmpid'].str.fullmatch(BASE_CMPID)
is_base_l = links['worldscopecmpid'].str.fullmatch(BASE_CMPID)


for name, ids, keep in (('universe', universe['worldscopecmpid'], is_base_u),
                        ('nation', nation['worldscopecmpid'], is_base_n)):
    other = ids[~keep & ~ids.str.fullmatch(SHARE_CLASS)]
    assert other.empty, f"{name}: base filter would drop ids that are no share class: {sorted(other.unique())[:TOP_N]}"


unmatched = links.loc[~is_base_l, 'worldscopecmpid']
assert not unmatched.isin(ws_ids).any(), \
    f"non-base permorgref ids match Worldscope records: {sorted(unmatched[unmatched.isin(ws_ids)].unique())[:TOP_N]}"

print(f"      [---] universe: dropped {(~is_base_u).sum()} share-class rows "
      f"({universe.loc[~is_base_u, 'worldscopecmpid'].nunique()} cmpids)")
print(f"      [---] nation: dropped {(~is_base_n).sum()} share-class rows")
print(f"      [---] links: dropped {(~is_base_l).sum()} non-base ids without Worldscope record: "
      f"{sorted(unmatched.unique())[:TOP_N]}")
universe = universe.loc[is_base_u].reset_index(drop=True)
nation = nation.loc[is_base_n].reset_index(drop=True)
links = links.loc[is_base_l].reset_index(drop=True)

universe['year'] = pd.to_numeric(universe['year']).astype('int64')
nation['ws_nation'] = nation['ws_nation'].astype('string')
links[['domcntrypermid', 'lvl3permid']] = links[['domcntrypermid', 'lvl3permid']].apply(pd.to_numeric).astype('Int64')

assert not universe.duplicated(['worldscopecmpid', 'year']).any(), "duplicate (cmpid, year) in the universe"
dup_n = nation['worldscopecmpid'].duplicated(keep=False)
assert not dup_n.any(), \
    f"{dup_n.sum()} rows share a cmpid in the nation table:\n{nation.loc[dup_n].sort_values('worldscopecmpid').head(10)}"


print(f"    [+++] applying DOMICILE_OVERRIDES to permorgref links...")
override_keys = list(con.DOMICILE_OVERRIDES)
lvl3_map = {k: v[0] for k, v in con.DOMICILE_OVERRIDES.items()}

in_map = links['lvl3permid'].notna() & links['domcntrypermid'].isin(override_keys)
assert not in_map.any(), \
    f"override keys {sorted(int(k) for k in links.loc[in_map, 'domcntrypermid'].unique())} resolve in tmcregncntrymap - review DOMICILE_OVERRIDES"

fill = links['lvl3permid'].isna() & links['domcntrypermid'].isin(override_keys)
lvl3_before = links['lvl3permid'].copy()
links.loc[fill, 'lvl3permid'] = links.loc[fill, 'domcntrypermid'].map(lvl3_map).astype('Int64')

assert links.loc[fill, 'lvl3permid'].notna().all(), "override left a targeted row without lvl3permid"
assert links.loc[~fill, 'lvl3permid'].equals(lvl3_before[~fill]), "override touched a row outside its mask"

for k, (lvl3, iso) in con.DOMICILE_OVERRIDES.items():
    n = links.loc[fill & links['domcntrypermid'].isin([k]), 'worldscopecmpid'].nunique()
    print(f"      [---] domcntrypermid {k} -> ({lvl3}, {iso}): {n} cmpids")


print(f"    [+++] deriving crosswalk ws_nation -> lvl3permid from linked cmpids...")
votes = (links.dropna(subset=['lvl3permid'])
         .merge(nation.dropna(subset=['ws_nation']), on='worldscopecmpid', how='inner')
         [['worldscopecmpid', 'ws_nation', 'lvl3permid']]
         .drop_duplicates())
split = votes['worldscopecmpid'].duplicated(keep=False)
print(f"      [---] voting cmpids: {votes['worldscopecmpid'].nunique()}, "
      f"split across several lvl3permid: {votes.loc[split, 'worldscopecmpid'].nunique()}")

tally = votes.groupby(['ws_nation', 'lvl3permid']).size().rename('n').reset_index()
tally['n_votes'] = tally.groupby('ws_nation')['n'].transform('sum')
tally['purity'] = tally['n'] / tally['n_votes']
is_max = tally['n'] == tally.groupby('ws_nation')['n'].transform('max')
n_max = is_max.groupby(tally['ws_nation']).transform('sum')

ties = tally.loc[is_max & (n_max > 1), 'ws_nation'].unique()
if len(ties):
    print(f"      [---] nations with a tied modal region, left without crosswalk entry: {sorted(ties)}")

xwalk = (tally.loc[is_max & (n_max == 1), ['ws_nation', 'lvl3permid', 'n_votes', 'purity']]
         .sort_values('n_votes', ascending=False).reset_index(drop=True))
assert xwalk['ws_nation'].is_unique, "crosswalk maps a nation to more than one lvl3permid"

agreement = (xwalk['purity'] * xwalk['n_votes']).sum() / tally['n'].sum()
print(f"      [---] crosswalk entries: {len(xwalk)}, lvl3 agreement of linked votes: {agreement:.4f}")
print(f"      [---] least pure entries:")
print(xwalk.sort_values('purity').head(TOP_N).to_string(index=False))


print(f"    [+++] assigning nation and region to the universe...")
n_rows = len(universe)
universe = universe.merge(nation, on='worldscopecmpid', how='left', validate='many_to_one')
assert len(universe) == n_rows, f"nation join changed the row count: {n_rows} -> {len(universe)}"

# pandas matches NULL keys with NULL keys, so a NULL nation in the crosswalk would leak a region
assert xwalk['ws_nation'].notna().all(), "NULL ws_nation in the crosswalk"
universe = universe.merge(xwalk[['ws_nation', 'lvl3permid']], on='ws_nation', how='left', validate='many_to_one')
assert len(universe) == n_rows, f"crosswalk join changed the row count: {n_rows} -> {len(universe)}"

firms = universe.drop_duplicates('worldscopecmpid')       
no_company = ~firms['worldscopecmpid'].isin(nation['worldscopecmpid'])
no_nation = firms['ws_nation'].isna() & ~no_company
no_xwalk = firms['ws_nation'].notna() & firms['lvl3permid'].isna()

print(f"      [---] universe rows {n_rows}, cmpids {len(firms)}, "
      f"share linked to permorgref {firms['worldscopecmpid'].isin(links['worldscopecmpid']).mean():.4f}")
print(f"      [---] cmpids without region: {firms['lvl3permid'].isna().sum()} "
      f"(no company row {no_company.sum()}, NULL nation {no_nation.sum()}, "
      f"nation without crosswalk entry {no_xwalk.sum()})")
if no_xwalk.any():
    print(firms.loc[no_xwalk, 'ws_nation'].value_counts().head(TOP_N).to_string())
print(f"      [---] universe rows with region: {universe['lvl3permid'].notna().mean():.4f}")


universe = (universe[['worldscopecmpid', 'year', 'ws_nation', 'lvl3permid']]
            .sort_values(['worldscopecmpid', 'year']).reset_index(drop=True))
universe.to_parquet(con.RAW_WS_UNIVERSE, index=False)
xwalk.to_parquet(con.REF_WS_NATION_XWALK, index=False)
print(f"\n[°°°] {con.RAW_WS_UNIVERSE.name} saved - shape {universe.shape}; "
      f"{con.REF_WS_NATION_XWALK.name} saved - {len(xwalk)} entries [°°°]")