##################################################
'''
cluster_condensation.py 
Load ref_cluster_keys.parquet, assert 8123 nunique cluster_keys,
assert 3 singletons present,
reconcile the 9,695 entities to 8,123 clusters.

'''
##################################################

import pandas as pd
import config as con
import log_config as lc

# Initializing paths
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / 'data'
results_path = con.RESULTS_DIR/ 'joined_panel'

# load data
df = pd.read_parquet(data_path) # (83040, 245)
parent = pd.read_parquet(con.PROJECT_ROOT / "data" / "raw" / "ref_parent_table.parquet") # (9695, 3)

# cast ids to int
df = df[['orgpermid']].astype('int64')
parent['ultimateparentorgpermid'] = parent['ultimateparentorgpermid'].astype('Int64')
parent['immediateparentorgpermid'] = parent['immediateparentorgpermid'].astype('Int64')




