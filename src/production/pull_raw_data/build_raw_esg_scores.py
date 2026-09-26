##################################################
'''
src.production.pull_raw_data.build_raw_esg_scores

input:      raw SQL: tr_esg.wrds_ref_esg, tr_common.permorgref, tr_common.tmcregncntrymap,
            tr_worldscope.wrds_ws_company
purpose:    pull every available ESGCombinedScore (the panel target) for
            YEAR_MIN-YEAR_MAX without the typecode / worldscopecmpid universe
            filters, and attach the Refinitiv geography
            (domcntrypermid -> tmcregncntrymap, DOMICILE_OVERRIDES applied);
            unmapped entities are kept and reported, not dropped.
            ws_cmpid_bridged resolves the link to a Worldscope company id: kept as is
            where worldscopecmpid is a Worldscope id, mapped via CUSIP (item6004)
            where it is a CUSIP (check_cusip_link: most rated North American firms),
            NULL where neither route resolves. used for the coverage figure only;
            the panel pipeline does not read this artifact.
            numerator of the rating-coverage analysis and source of the
            outside-Worldscope share
output:     raw_esg_scores.parquet with columns orgpermid, year, esg_combined_score,
            typecode, worldscopecmpid, domcntrypermid, lvl3permid, lvl5isocntry,
            ws_cmpid_bridged
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

BASE_CMPID = r'.{8}\d'      

print(f"\n[°°°] building raw ESG score table {con.YEAR_MIN}-{con.YEAR_MAX} [°°°]\n")
db = wrds.Connection(wrds_username=lc.wrds_log)

esg_base = f"""
    SELECT DISTINCT orgpermid, year, valuescore
    FROM tr_esg.wrds_ref_esg
    WHERE fieldname = 'ESGCombinedScore'
        AND valuescore IS NOT NULL
        AND year BETWEEN {con.YEAR_MIN} AND {con.YEAR_MAX}
"""

print(f"    [+++] counting distinct ESGCombinedScore rows...")
n_base = db.raw_sql(f"SELECT COUNT(*) AS n FROM ({esg_base}) e")['n'].iloc[0]
print(f"      [---] distinct (orgpermid, year, score) rows: {n_base}")

print(f"    [+++] pulling scores with entity attributes and geography...")
scores = db.raw_sql(f"""
    SELECT
        e.orgpermid,
        e.year,
        e.valuescore AS esg_combined_score,
        p.typecode,
        p.worldscopecmpid,
        p.domcntrypermid,
        g.lvl3permid,
        g.lvl5isocntry
    FROM ({esg_base}) e
    LEFT JOIN tr_common.permorgref p
        ON p.orgpermid = e.orgpermid
    LEFT JOIN tr_common.tmcregncntrymap g
        ON p.domcntrypermid = g.lvl5permid
""")
print(f"      [---] pulled rows: {len(scores)}")


print(f"    [+++] checking value contracts and casting dtypes...")
assert len(scores) == n_base, \
    f"LEFT JOINs changed the row count: {n_base} -> {len(scores)} - permorgref or tmcregncntrymap fans out"

id_cols = ['orgpermid', 'year', 'esg_combined_score', 'domcntrypermid', 'lvl3permid']
scores[id_cols] = scores[id_cols].apply(pd.to_numeric)          # raises on non-numeric values

assert scores['orgpermid'].notna().all(), "NULL orgpermid in wrds_ref_esg"
assert scores['esg_combined_score'].notna().all(), "NULL score despite the valuescore filter"
assert scores['esg_combined_score'].between(0, 1).all(), \
    f"scores outside [0, 1]: min {scores['esg_combined_score'].min()}, max {scores['esg_combined_score'].max()}"
assert scores['year'].between(con.YEAR_MIN, con.YEAR_MAX).all(), \
    f"years outside the window: {sorted(scores['year'].unique())}"

scores = scores.astype({'orgpermid': 'int64', 'year': 'int64',
                        'domcntrypermid': 'Int64', 'lvl3permid': 'Int64',
                        'typecode': 'string', 'worldscopecmpid': 'string', 'lvl5isocntry': 'string'})

dup = scores.duplicated(subset=['orgpermid', 'year'], keep=False)
assert not dup.any(), \
    f"{dup.sum()} rows share an (orgpermid, year) with differing scores:\n" \
    f"{scores.loc[dup].sort_values(['orgpermid', 'year']).head(10)}"


print(f"    [+++] applying DOMICILE_OVERRIDES...")
override_keys = list(con.DOMICILE_OVERRIDES)
lvl3_map = {k: v[0] for k, v in con.DOMICILE_OVERRIDES.items()}
iso_map = {k: v[1] for k, v in con.DOMICILE_OVERRIDES.items()}

in_map = scores['lvl3permid'].notna() & scores['domcntrypermid'].isin(override_keys)
assert not in_map.any(), \
    f"override keys {sorted(int(k) for k in scores.loc[in_map, 'domcntrypermid'].unique())} resolve in tmcregncntrymap - review DOMICILE_OVERRIDES"

fill = scores['lvl3permid'].isna() & scores['domcntrypermid'].isin(override_keys)
lvl3_before = scores['lvl3permid'].copy()
scores.loc[fill, 'lvl3permid'] = scores.loc[fill, 'domcntrypermid'].map(lvl3_map).astype('Int64')
scores.loc[fill, 'lvl5isocntry'] = scores.loc[fill, 'domcntrypermid'].map(iso_map).astype('string')

assert scores.loc[fill, 'lvl3permid'].notna().all(), "override left a targeted row without lvl3permid"
assert scores.loc[~fill, 'lvl3permid'].equals(lvl3_before[~fill]), "override touched a row outside its mask"

for k, (lvl3, iso) in con.DOMICILE_OVERRIDES.items():
    n = scores.loc[fill & scores['domcntrypermid'].isin([k]), 'orgpermid'].nunique()
    print(f"      [---] domcntrypermid {k} -> ({lvl3}, {iso}): {n} entities")


print(f"    [+++] bridging CUSIP-type link ids to Worldscope company ids...")
company = db.raw_sql("""
    SELECT item6105, item6004 AS cusip
    FROM tr_worldscope.wrds_ws_company
    WHERE item6105 IS NOT NULL
""")
company['item6105'] = company['item6105'].astype('string')
company['cusip'] = company['cusip'].astype('string').str.strip().str.upper()

is_base = company['item6105'].str.fullmatch(BASE_CMPID)
company['base_id'] = company['item6105'].where(is_base, company['item6105'].str[:-1] + '0')

targets = company.dropna(subset=['cusip']).groupby('cusip')['base_id'].agg(['nunique', 'first'])
bridge = targets.loc[targets['nunique'] == 1, 'first']
ambiguous = set(targets.index[targets['nunique'] > 1])

link = scores['worldscopecmpid']
is_ws = link.isin(set(company['item6105']))
scores['ws_cmpid_bridged'] = link.where(is_ws, link.map(bridge)).astype('string')

via_cusip = ~is_ws & scores['ws_cmpid_bridged'].notna()
assert scores.loc[via_cusip, 'ws_cmpid_bridged'].str.fullmatch(BASE_CMPID).all(), "bridged id is not a base Worldscope id"
print(f"      [---] rows: Worldscope id {is_ws.sum()}, bridged via CUSIP {via_cusip.sum()}, "
      f"unresolved link {(link.notna() & scores['ws_cmpid_bridged'].isna()).sum()}, no link {link.isna().sum()}")
print(f"      [---] ambiguous CUSIPs hit by link ids (left unbridged): {link.isin(ambiguous).sum()} rows")
print(f"      [---] entities bridged via CUSIP per lvl3permid: "
      f"{({int(k): v for k, v in scores.loc[via_cusip].drop_duplicates('orgpermid')['lvl3permid'].value_counts().head(5).items()})}")


print(f"    [+++] reporting entity coverage...")
attrs = ['typecode', 'worldscopecmpid', 'domcntrypermid', 'lvl3permid', 'lvl5isocntry', 'ws_cmpid_bridged']
n_attr = scores.groupby('orgpermid')[attrs].nunique(dropna=False).max()
assert (n_attr <= 1).all(), f"entity attributes vary within orgpermid: {n_attr[n_attr > 1].to_dict()}"
entities = scores.drop_duplicates('orgpermid')[['orgpermid', *attrs]]

print(f"      [---] rows {len(scores)}, entities {len(entities)}")
print(f"      [---] entities per year: {scores.groupby('year')['orgpermid'].nunique().to_dict()}")
print(f"      [---] entities without typecode (no permorgref match or NULL typecode): "
      f"{entities['typecode'].isna().sum()}")

by_type = entities.assign(
    has_wscmpid=entities['worldscopecmpid'].notna(),
    has_domicile=entities['domcntrypermid'].notna(),
    has_lvl3=entities['lvl3permid'].notna(),
).groupby('typecode', dropna=False).agg(
    entities=('orgpermid', 'size'),
    with_wscmpid=('has_wscmpid', 'sum'),
    with_domicile=('has_domicile', 'sum'),
    with_lvl3=('has_lvl3', 'sum'),
).sort_values('entities', ascending=False)
print(f"      [---] Worldscope link and geography per typecode:")
print(by_type.to_string())

unmapped = entities.loc[entities['domcntrypermid'].notna() & entities['lvl3permid'].isna(), 'domcntrypermid']
print(f"      [---] entities with domcntrypermid but no lvl3permid: {len(unmapped)}")
if len(unmapped):
    print(unmapped.value_counts().head(10).to_string())


### save ---------------------------------------------------------
cols = ['orgpermid', 'year', 'esg_combined_score', *attrs]
scores = scores[cols].sort_values(['orgpermid', 'year']).reset_index(drop=True)
scores.to_parquet(con.RAW_ESG_SCORES, index=False)
print(f"\n[°°°] {con.RAW_ESG_SCORES.name} saved - shape {scores.shape} [°°°]")