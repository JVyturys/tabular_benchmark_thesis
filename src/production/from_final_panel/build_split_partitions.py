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

import numpy as np
import pandas as pd
import config as con
import math 

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)
cluster = pd.read_parquet(con.REF_CLUSTER_KEYS)

# load non-degenerate region list
regs =[*con.TIER1_REGS, *con.TIER2_REGS]

# merge data
df_org =panel[['orgpermid']]
df_geo = geo[['orgpermid', 'lvl3permid']]
df_clust = cluster[['orgpermid', 'cluster_key']]
df_split = df_org.merge(df_geo, on='orgpermid', how='left')
df_split = df_split.merge(df_clust, on='orgpermid', how='left')
df_split = df_split.query('lvl3permid.isin(@regs)')
pre_merge_len = len(df_split)

# append cluster-home-region columns
# # define region-cluster distribution h(cluster_key_i, lvl3permid_j)
biv_dist_clgeo = df_split[['lvl3permid', 'cluster_key']].groupby(['cluster_key', 'lvl3permid'], dropna=False).size()
df_biv_clgeo = pd.DataFrame({
    "n_observations":biv_dist_clgeo
})

# # determine cluster-regions with highest frequency of observations per cluster
idx_max = df_biv_clgeo.groupby('cluster_key', dropna=False)['n_observations'].idxmax()
max_obs = df_biv_clgeo.loc[idx_max]
df_max_obs = max_obs.reset_index() 
df_max_obs = df_max_obs.rename(columns={'lvl3permid':'cluster_home'})
print(df_max_obs)
df_max_obs = df_max_obs[['cluster_key', 'cluster_home']]

# append home-region information to split data frame
df_split = df_split.merge(df_max_obs, on="cluster_key")

# determine share of observations that have a from physical lvl3permid distinct cluster home
off_home_obs_share = len(df_split[df_split['lvl3permid']!=df_split['cluster_home']])/len(df_org)

# perform assertions
assert math.isclose(off_home_obs_share, 0.025, abs_tol = 0.0009), 'share of off-home observations does not allign'
assert df_split['cluster_home'].isna().sum() == 0, "rows without cluster home in df_split"
assert df_split['lvl3permid'].nunique() == len(regs), f"number of physical regions in df_split does not allign with number of regions in tier list ({len(regs)})"
assert len(df_split) == pre_merge_len, "lenght of df_split changed during merge with home regions"

# perform index partition allocation
# # initiate random number generator and set seed
rng = np.random.default_rng(seed=con.SEED)

# # create partition key
df_partition = df_split[['orgpermid', 'cluster_key', 'lvl3permid', 'cluster_home']]
df_partition['partition'] = np.nan

# # set up per-region deviation report
deviation_report = {} 

# # test-train idex split 
for current_region in regs:

    # select rows of one region
    df_current = df_split.query('cluster_home == @current_region')
    df_current['test_train'] = np.nan 

    # create ordered list of clusters based on cluster-sample-size in current region
    counts = df_current['cluster_key'].value_counts().rename_axis('cluster_key').reset_index(name='freq')
    counts['rand_weight'] = rng.random(len(counts))
    ordered_clusters = counts.sort_values(
                                        by=['freq', 'rand_weight'], 
                                        ascending=[False, True]
                                        )['cluster_key'].tolist()

    # determine partition goals for current regions; rps
    synth_reg_size = df_current.groupby('cluster_home').size().sum()
    train_size_goal = synth_reg_size*con.TRAIN_SHARE
    test_size_goal = synth_reg_size*con.TEST_SHARE
    test_size_current = len(df_current.query('test_train=="test"'))
    train_size_current = len(df_current.query('test_train=="train"'))

    # assign each cluster to a partition
    for parent in ordered_clusters:
        test_size_current = len(df_current.query('test_train=="test"'))
        train_size_current = len(df_current.query('test_train=="train"'))

        # determine distance to partition goal
        dist_train = 1 - train_size_current / train_size_goal
        dist_test = 1 - test_size_current / test_size_goal

        # direct cluster to partition 
        idx = df_current['cluster_key'] == parent

        if dist_train > dist_test:
            df_current.loc[idx, 'test_train'] = 'train'
            
        elif dist_train < dist_test: # Fixed typo: was dist_train < dist_train
            df_current.loc[idx, 'test_train'] = 'test'
            
        elif dist_train - dist_test == con.DIST_TOL: 
            df_current.loc[idx, 'test_train'] = 'test'

        # save allocation outside the loop 
        merged = df_partition.merge(df_current, on=['orgpermid', 'cluster_key', 'lvl3permid', 'cluster_home'], how='left')
        df_partition['partition'] = merged['test_train'].fillna(df_partition['partition'])

    # add deviation entry for current region to report
    current_deviation = len(df_current[['test_train']].query('test_train=="test"'))/synth_reg_size-con.TEST_SHARE
    deviation_report.update({current_region:current_deviation})
       
df_deviation_rep = pd.DataFrame.from_dict(deviation_report, orient="index", columns=["deviation"])
print(df_deviation_rep)

    
