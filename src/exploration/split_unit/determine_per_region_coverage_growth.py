##################################################
'''
src.exploration.split_unit.determine_per_region_coverage_growth

input: ref_cluster_keys.parquet
purpose: determine the number of clusters based on the cluster key
        assignment
output: clusters-based on ultpar: 8865
        clusters-based on immpar: 847
        singleton-clusters: 3
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)

# slice relevant vars
df_org = panel[['orgpermid', 'year']]
df_geo = ref_geo[['orgpermid', 'lvl3permid']]
