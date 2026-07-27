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
import numpy as np
import config as con
import log_config as lc

# Initializing paths and data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / 'data'
results_path = con.RESULTS_DIR/ 'joined_panel'

# load data
df = pd.read_parquet(data_path)
geo = pd.read_csv(con.PROJECT_ROOT / "data" / "raw" / "geo_ref.csv")
parent = pd.read_parquet(con.PROJECT_ROOT / "data" / "raw" / "ref_parent.parquet")

# cast ids to int
org_id = parent['orgpermid'].astype('int').to_list()
parent_id = parent['ultimateparentorgpermid'].fillna(0).astype('int').to_list()

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
df_parent_enttype.to_parquet('parent_ent_type.parquet', index=False)

# create table with set of entity IDs in panel
df_cluster_keys = df[['orgpermid']].drop_duplicates()
ent_nmb = len(df_cluster_keys) # storing number of entities for post merge assertion 

# add parent inforamtion
df_cluster_keys = df_cluster_keys.merge(parent, how='left', on='orgpermid')
df_cluster_keys = df_cluster_keys.merge(df_parent_enttype,
                                        how='left',
                                        left_on='ultimateparentorgpermid',
                                        right_on='orgpermid')

# rename columns
df_cluster_keys = df_cluster_keys[['orgpermid_x', 'immediateparentorgpermid_x', 'ultimateparentorgpermid_x', 'typecode']]
df_cluster_keys = df_cluster_keys.rename(columns={'ultimateparentorgpermid_x':'ultimateparentorgpermid',
                                                  'orgpermid_x':'orgpermid',
                                                  'immediateparentorgpermid_x':'immediateparentorgpermid',
                                                  'typecode':'ultparent_ent_type'})

# create placeholder column
df_cluster_keys['cluster_key'] = np.nan

# create index list for conditional value assignment
mask_ult = df_cluster_keys['ultparent_ent_type'].isin(['COM', 'UNK', 'NGO', 'CLGUN'])
mask_imm = df_cluster_keys['ultparent_ent_type'].isin(['GVT', 'GVTDA', 'CINV'])

# assign cluster-keys
df_cluster_keys.loc[mask_ult, 'cluster_key'] = df_cluster_keys.loc[mask_ult, 'ultimateparentorgpermid']
df_cluster_keys.loc[mask_imm, 'cluster_key'] = df_cluster_keys.loc[mask_imm, 'immediateparentorgpermid']

# check: are any keys assiged that are an ultimateparerent for one and an immediate parent for another?
keys_assigned_from_ult = df_cluster_keys.loc[mask_ult, 'cluster_key']
keys_assigned_from_imm = df_cluster_keys.loc[mask_imm, 'cluster_key']

## build intersection of keys assigned as ultimate and immediate parent
overlapping_keys = set(keys_assigned_from_ult).intersection(set(keys_assigned_from_imm))

## filter dataframe for intersection 
df_overlaps = df_cluster_keys[df_cluster_keys['cluster_key'].isin(overlapping_keys)]

# test data quality
assert len(df_cluster_keys) == ent_nmb, "Possible merge error: entities pre-merge != entities post-merge"


print(df_cluster_keys.shape)



print(df_cluster_keys['ultparent_ent_type'].value_counts(dropna=False))
print(df_cluster_keys[['cluster_key']].isnull().sum())
print(df_cluster_keys.query('cluster_key.isnull()'))
print(df_overlaps)
print(f"Number of overlapping keys: {len(overlapping_keys)}")
print(f"Rows with intersecting keys: {len(df_overlaps)}")
