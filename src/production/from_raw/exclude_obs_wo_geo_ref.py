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
from collections import Counter


# load data
df = pd.read_parquet(con.RAW_PANEL)
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)

# filter only referenceable entities
included_entities = ref_geo['orgpermid'].to_list()

# create boolean index array  
indicies = []
for entity in df['orgpermid']:
    idx = entity in included_entities
    indicies.append(idx)

frequency = Counter(indicies)
df_clean = df[indicies]
assert df_clean['orgpermid'].nunique() == ref_geo['orgpermid'].nunique(), \
    "Entity mismatch between panel and geography reference"
print("Assertion passed — panel and geography table describe identical entity universe")

df_clean.to_parquet(con.PANEL, index=False)