##################################################
'''
src.production.from_final_panel.build_cluster_keys

input: parent_ent_type.parquet, ref_parents.parquet
purpose: build the cluster assignment needed for the
        stratified split from two masks (ultimate- and
        immediate-branch), cast all ID columns to a
        non-float type,
output: ref_cluster_keys.parquet with columns orgpermid,
        cluster_key, parent_typecode, key_source
assertions:
        i)   cluster_key has exactly 3 nulls before the
             singleton fill and zero after singleton
             cluster assignment,
        ii)  the ultimate-branch and immediate-branch key
             sets are disjoint,
        iii) entity count pre merge equals entity count
             post merge,
        iv)  every orgpermid in the panel has exactly one
             cluster_key
'''
##################################################

import pandas as pd
import numpy as np
import config as con

# Initializing paths and data base connection
data_path = con.PANEL

# load data
df = pd.read_parquet(data_path)
parent = pd.read_parquet(con.REF_PARENT)
df_parent_enttype = pd.read_parquet(con.REF_PARENT_ENT_TYPE)

# cast ids to int
df['orgpermid'] = df['orgpermid'].astype('int64')
parent['ultimateparentorgpermid'] = parent['ultimateparentorgpermid'].astype('Int64')
parent['immediateparentorgpermid'] = parent['immediateparentorgpermid'].astype('Int64')

# create table with set of entity IDs in panel
df_cluster_keys = df[['orgpermid']].drop_duplicates()
ent_nmb = len(df_cluster_keys) # storing number of entities for post merge assertion 

# add parent inforamtion
df_cluster_keys = df_cluster_keys.merge(parent, how='left', on='orgpermid')
df_cluster_keys = df_cluster_keys.merge(df_parent_enttype,
                                        how='left',
                                        left_on='ultimateparentorgpermid',
                                        right_on='orgpermid')

assert len(df_cluster_keys) == ent_nmb, "Possible merge error: entities pre-merge != entities post-merge" # data-quality check

# rename columns
df_cluster_keys = df_cluster_keys[['orgpermid_x', 'immediateparentorgpermid_x', 'ultimateparentorgpermid_x', 'typecode']]
df_cluster_keys = df_cluster_keys.rename(columns={'ultimateparentorgpermid_x':'ultimateparentorgpermid',
                                                  'orgpermid_x':'orgpermid',
                                                  'immediateparentorgpermid_x':'immediateparentorgpermid',
                                                  'typecode':'parent_typecode'})

# create placeholder column
df_cluster_keys['cluster_key'] = np.nan

# create index list for conditional value assignment
mask_ult = df_cluster_keys['parent_typecode'].isin(con.ULTIMATE_KEY_TYPECODES)
mask_imm = df_cluster_keys['parent_typecode'].isin(con.IMMEDIATE_KEY_TYPECODES)

# assign cluster-keys
df_cluster_keys.loc[mask_ult, 'cluster_key'] = df_cluster_keys.loc[mask_ult, 'ultimateparentorgpermid']
df_cluster_keys.loc[mask_imm, 'cluster_key'] = df_cluster_keys.loc[mask_imm, 'immediateparentorgpermid']
df_cluster_keys['cluster_key'] = df_cluster_keys['cluster_key'].astype('Int64')

# data quality check
singleton_cluster_mask = df_cluster_keys['cluster_key'].isnull()
n_nan = singleton_cluster_mask.sum()
assert n_nan == 3, "Error: check data input, 3 entities should not have an ultimateparentorgpermid"

# assign singleton cluster IDs to entities without parents
singleton_ids = df_cluster_keys.loc[singleton_cluster_mask, 'orgpermid'].tolist()
df_cluster_keys.loc[singleton_cluster_mask, 'cluster_key'] = singleton_ids

# check if all NaN cluster keys have been resolved
n_nan = df_cluster_keys['cluster_key'].isnull().sum()
assert n_nan == 0, "Error: Unresolved NaNs in the cluster key"

# check: are any keys assiged that are an ultimateparerent for one and an immediate parent for another?
keys_assigned_from_ult = df_cluster_keys.loc[mask_ult, 'cluster_key']
keys_assigned_from_imm = df_cluster_keys.loc[mask_imm, 'cluster_key']
keys_assigned_from_sngltn = df_cluster_keys.loc[singleton_cluster_mask, 'cluster_key']

## build intersection of keys assigned as ultimate and immediate parent
overlapping_keys = set(keys_assigned_from_ult).intersection(set(keys_assigned_from_imm)).intersection(set(keys_assigned_from_sngltn))

## filter dataframe for intersection 
df_overlaps = df_cluster_keys[df_cluster_keys['cluster_key'].isin(overlapping_keys)]

## check if resulting intersection is empty 
assert len(df_overlaps) == 0,  '''Unresolved key intersection:
 a cluster key is an immediateparent ID and at the same time an 
 ultimateparent ID'''

print(df_cluster_keys.columns)

df_cluster_keys = df_cluster_keys[['orgpermid', 'cluster_key',
                                   'parent_typecode']]

df_cluster_keys['key_source'] = np.nan
df_cluster_keys.loc[df_cluster_keys['parent_typecode'].isin(['COM', 'UNK', 'NGO', 'CLGUN']), 'key_source'] = 1
df_cluster_keys.loc[df_cluster_keys['parent_typecode'].isin(['GVT', 'GVTDA', 'CINV']), 'key_source'] = 2
df_cluster_keys.loc[df_cluster_keys['parent_typecode'].isna(), 'key_source'] = 0

assert df_cluster_keys['key_source'].isnull().sum() == 0, '''Unresolved NaN in "key-source" column''' 

assert df_cluster_keys.groupby('orgpermid')['cluster_key'].nunique().max() == 1, '''Mismatch between orgpermid and cluster key allocation.'''
print(df_cluster_keys.groupby(['orgpermid', 'cluster_key']).size())

# ouput cluster reference file
df_cluster_keys.to_parquet(con.PROJECT_ROOT / "data" / "raw" / "ref_cluster_keys.parquet", index=False)


