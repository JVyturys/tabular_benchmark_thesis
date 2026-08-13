##################################################
'''
src.production.from_final_panel.build_split_partitions

input: panel.parquet, ref_geography.parquet, cluster_keys.parquet,
        regions-tier list
purpose: produce train, test, fit and validation partitions based on 
        the in the methodology part described stratification-, placement-, and
        ordering strategy.
output: idx-sets for each partition
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)
cluster = pd.read_parquet(con.REF_CLUSTER_KEYS)

# merge data
df_org =panel[['orgpermid']]
df_geo = geo[['orgpermid', 'lvl3permid']]
df_clust = cluster[['orgpermid', 'cluster_key']]
df_split = df_org.merge(df_org, on='orgpermid', how='left')
df_split = df_org.merge(df_geo, on='orgpermid', how='left')
df_split = df_org.merge(df_clust, on='orgpermid', how='left')
