##################################################
'''
src.production.from_raw.exclude_obs_wo_geo_ref

input: raw_panel.parquet, ref_geo_table.parquet
purpose: exclude observations from raw_panel that are not
        geographically referenceable
output: panel.parquet
'''
##################################################
import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.RAW_PANEL)
ref_geo_table = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid'])

# cast ids to int
panel['orgpermid'] = panel['orgpermid'].astype('int64')
ref_geo_table['orgpermid'] = ref_geo_table['orgpermid'].astype('int64')

# define set of geographically referenceable entities
geo_orgids = set(ref_geo_table['orgpermid'].unique())

# keep only observations whose entity is referenceable
mask = panel['orgpermid'].isin(geo_orgids)
panel_filtered = panel[mask].copy()

# reporting
n_entities_before = panel['orgpermid'].nunique()
n_entities_after  = panel_filtered['orgpermid'].nunique()
dropped_entities  = set(panel['orgpermid'].unique()) - geo_orgids

print(f"rows:     {len(panel)} -> {len(panel_filtered)} ({len(panel) - len(panel_filtered)} dropped)")
print(f"entities: {n_entities_before} -> {n_entities_after} ({len(dropped_entities)} dropped)")

# sanity check: no unreferenceable entity should remain
assert panel_filtered['orgpermid'].isin(geo_orgids).all(), \
    "unreferenceable entity still present in panel!"

# save output
panel_filtered.to_parquet(con.PANEL)
print(f'''"panel.parquet" successfully saved. shape={panel_filtered.shape}''')