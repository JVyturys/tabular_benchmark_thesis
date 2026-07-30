######################################################################
'''
src.production.from_raw.build_ref_geo_table
input: ref_geo_raw.parquet
purpose: override missing geo-IDs of CN and SD entities with
        in config.py hardcoded lvl3permids,
        build final geography reference table,
output: ref_geo_table.parquet
'''
######################################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

# intitialize database connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# load data (only orgpermid needed from the panel)
df = pd.read_parquet(con.RAW_PANEL, columns=['orgpermid'])
ref_geo_raw = pd.read_parquet(con.REF_GEO_RAW)

# cast ids to int for reliable set operations
df['orgpermid'] = df['orgpermid'].astype('int64')
ref_geo_raw['orgpermid'] = ref_geo_raw['orgpermid'].astype('int64')

# panel entities vs. entities already mapped to a geography
panel_orgids   = set(df['orgpermid'].unique())          # expected 9721
mapped_orgids  = set(ref_geo_raw['orgpermid'].unique()) # expected 8283
missing_orgids = panel_orgids - mapped_orgids           # expected 1438

print(f"panel entities:   {len(panel_orgids)}")
print(f"mapped entities:  {len(mapped_orgids)}")
print(f"missing entities: {len(missing_orgids)}")

# build ID-string for SQL query
orgid_string = ','.join(str(i) for i in missing_orgids)

# pull the domicile-country permid for the unmapped entities. this is the
# key override: CN / SD countries are absent from
# tr_common.tmcregncntrymap, which is why the original join dropped them.
query = f"""
        SELECT DISTINCT
            p.orgpermid,
            p.domcntrypermid
        FROM tr_common.permorgref p
        WHERE p.orgpermid IN ({orgid_string})
            AND p.typecode = 'COM'
"""
missing_geo = db.raw_sql(query)

# cast ids; domcntrypermid is nullable for entities with no geography at all
missing_geo['orgpermid'] = missing_geo['orgpermid'].astype('int64')
missing_geo['domcntrypermid'] = missing_geo['domcntrypermid'].astype('Int64')

# guard against multiple permorgref rows collapsing to one entity
missing_geo = missing_geo.drop_duplicates(subset='orgpermid')

# keep only entities whose domicile country is covered by the override map;
# the remaining entities (the 6 with no resolvable geography) are dropped
override = missing_geo[missing_geo['domcntrypermid'].isin(con.DOMICILE_OVERRIDES.keys())].copy()
override['domcntrypermid'] = override['domcntrypermid'].astype('int64')  # no NaN left

# map the hardcoded lvl3permid, lvl5isocntry pairs onto each entity
override['lvl3permid']   = override['domcntrypermid'].map({k: v[0] for k, v in con.DOMICILE_OVERRIDES.items()})
override['lvl5isocntry'] = override['domcntrypermid'].map({k: v[1] for k, v in con.DOMICILE_OVERRIDES.items()})

# report how many entities each override rule captured
for k, (lvl3, iso) in con.DOMICILE_OVERRIDES.items():
    n = int((override['domcntrypermid'] == k).sum())
    print(f"  domcntrypermid {k} -> ({lvl3}, {iso}): {n} entities")

override = override[['orgpermid', 'lvl3permid', 'lvl5isocntry']]

# report entities that stayed unresolved even after the override
dropped = missing_orgids - set(override['orgpermid'])
print(f"override rows added: {len(override)}")
print(f"entities dropped (no resolvable geography): {len(dropped)}")

# assemble the final geography reference table
ref_geo_table = pd.concat([ref_geo_raw, override], ignore_index=True)

# align dtypes across the mapped and overridden rows
ref_geo_table['orgpermid'] = ref_geo_table['orgpermid'].astype('int64')
if ref_geo_table['lvl3permid'].notna().all():
    ref_geo_table['lvl3permid'] = ref_geo_table['lvl3permid'].astype('int64')

# data-integrity checks: no entity should map to two geographies
assert ref_geo_table['orgpermid'].is_unique, "duplicate orgpermid in ref_geo_table!"

# save output
ref_geo_table.to_parquet(con.REF_GEOGRAPHY)
print(f'''"ref_geo_table.parquet" successfully saved. shape={ref_geo_table.shape}''')