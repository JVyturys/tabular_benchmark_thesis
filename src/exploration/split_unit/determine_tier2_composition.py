##################################################
'''

input: panel.partquet, ref_geography.parquet, ref_cluster_keys
purpose: explore the composition of each tier 2 region
output: 
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)
cluster= pd.read_parquet(con.REF_CLUSTER_KEYS)

# merge data
df_org =panel[['orgpermid']]
df_geo = geo[['orgpermid', 'lvl3permid']]
df_clust = cluster[['orgpermid', 'cluster_key']]

