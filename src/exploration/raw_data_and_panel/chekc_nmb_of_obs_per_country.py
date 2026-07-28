##################################################
'''
Find out the number of observations per entity & 
the total observation per country in the final panel.
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc
from utils.reporting import AnalysisLogger as Al

# Initializing paths and data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / 'data'
results_path = con.RESULTS_DIR/ 'joined_panel'

# Load data
df = pd.read_parquet(data_path)
geo = pd.read_csv(con.PROJECT_ROOT / "data" / "raw" / "geo_ref.csv")

# create table: observations & country ID
df_orgs = df[['orgpermid']] # entitiy IDs 
geo_cntry = geo[['orgpermid','lvl5isocntry']] # iso-country codes

# merge data sets
df_geo_merge = df_orgs.merge(geo_cntry, on='orgpermid', how='left')

# count observations per entity
obs_per_ent = df_geo_merge.value_counts()

# count observation per entity
df_obs_cntry = df_geo_merge['lvl5isocntry'] # isolating countries
obs_per_cntry = df_obs_cntry.value_counts()



obs_per_ent.to_csv("obs_per_ent.csv", index=True)
obs_per_cntry.to_csv("obs_per_cntry.csv", index=True)

