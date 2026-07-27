##################################################
'''
Create the cluster assigment that is needed for the 
stratified split. 

Read parent_ent_type.parquet + ref_parents.parquet.
Produce ref_cluster_keys.parquet with orgpermid,
cluster_key, parent_typecode, key_source.

Cast all ID columns to a non-float type. 

The assignment logic is build from two masks.

Close with four assertions:
i) cluster_key has exactly 3 nulls before the singleton fill
ii) the ultimate-branch and immediate-branch key sets
    are disjoint 
iii) entity count in == entity count out
iv) every orgpermid in the panel has exactly one cluster_key
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

# load data
df = pd.read_parquet(data_path)
geo = pd.read_csv(con.PROJECT_ROOT / "data" / "raw" / "geo_ref.csv")
parent = pd.read_parquet(con.PROJECT_ROOT / "data" / "raw" / "parent_ref.parquet")

# cast ids to int
org_id = parent['orgpermid'].astype('int').to_list()
parent_id = parent['ultimateparentorgpermid'].filna(0).astype('int').to_list()

# turn into string for SQL query
org_string = ','.join(f"{id}" for id in org_id)
ultparent_string = ','.join(f"{par_id}" for par_id in parent_id)

# select entity meta-information of all parents in the panel 
query = f"""
     SELECT DISTINCT orgpermid,comname,typecode,ultimateparentorgpermid,immediateparentorgpermid,
                     domcntrypermid
     FROM tr_common.permorgref
     WHERE orgpermid IN ({ultparent_string})
""" 

df_parent_enttype = db.raw_sql(query)
df_parent_enttype.to_parque('parent_ent_type.parquet', index=False)
