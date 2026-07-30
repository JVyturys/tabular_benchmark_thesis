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

# the set of geographically referenceable entities
geo_orgids = set(ref_geo_table['orgpermid'].unique())

# keep only observations whose entity is referenceable
mask = panel['orgpermid'].isin(geo_orgids)
panel_filtered = panel[mask].copy()

# reporting
panel_orgids      = set(panel['orgpermid'].unique())
kept_orgids       = set(panel_filtered['orgpermid'].unique())
dropped_entities  = panel_orgids - geo_orgids
phantom_entities  = geo_orgids - panel_orgids  # in ref_geo_table but never in panel

print(f"rows:     {len(panel)} -> {len(panel_filtered)} ({len(panel) - len(panel_filtered)} dropped)")
print(f"entities: {len(panel_orgids)} -> {len(kept_orgids)} ({len(dropped_entities)} dropped)")

# sanity check: the kept entities must be exactly the referenceable set.
# equality fails if ref_geo_table carries a orgpermid that never
# appears in the panel
assert kept_orgids == geo_orgids, \
    f"kept entities != referenceable set; {len(phantom_entities)} phantom entity/ies " \
    f"in ref_geo_table not present in panel: {sorted(phantom_entities)[:10]}"

# save output
panel_filtered.to_parquet(con.PANEL)
print(f'''"panel.parquet" successfully saved. shape={panel_filtered.shape}''')