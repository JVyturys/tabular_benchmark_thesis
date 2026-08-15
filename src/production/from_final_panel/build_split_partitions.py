##################################################
'''
src.production.from_final_panel.build_split_partitions

input: panel.parquet, ref_geography.parquet, cluster_keys.parquet,
        regions-tier list
purpose: produce train, test, fit and validation partitions based on 
        the in the methodology part described stratification-, placement-, and
        ordering strategy.
output: idx-sets for each partition, deviation-report, off-home-leakage

        outer dev report:
                deviation
        100277  -0.000057
        100090   0.000574
        100334  -0.000022
        103384   0.000089
        100024   0.000000
        100279   0.000165
        100089   0.000020
        100219  -0.000036
        100223   0.000033
        103401   0.000208
        100276   0.000086
        100218   0.003030
        100278  -0.000050
        100060   0.170588
        100087   0.026923
        100332   0.033333

               inner_deviation
        100277         0.000234
        100090         0.001758
        100334         0.000090
        103384         0.000274
        100024         0.000000
        100279         0.000269
        100089         0.000032
        100219         0.000148
        100223         0.000100
        103401         0.000000
        100276         0.000059
        100218         0.006211
        100278         0.000558
        100060         0.357143
        100087         0.171429
        100332         0.084416

        off-home leakage per partition and region
                partition cluster_home               
        fit     100024         35  0.009289
                100060          0  0.000000
                100087          0  0.000000
                100089        212  0.013936
                100090         54  0.064748
                100218          0  0.000000
                100219         78  0.023537
                100223        164  0.022368
                100276        125  0.029947
                100277         53  0.050573
                100278         35  0.014505
                100279         74  0.040659
                100332          0  0.000000
                100334        247  0.045421
                103384        105  0.039150
                103401         34  0.016815
        test    100024         49  0.026008
                100060          0  0.000000
                100087          0  0.000000
                100089        102  0.013409
                100090         18  0.042959
                100218          0  0.000000
                100219         16  0.009656
                100223        106  0.028906
                100276        110  0.052682
                100277         12  0.022901
                100278          4  0.003314
                100279         54  0.059276
                100332          0  0.000000
                100334        119  0.043766
                103384         94  0.070045
                103401          6  0.005929
        val     100024          0  0.000000
                100060          0  0.000000
                100087          0  0.000000
                100089         12  0.004732
                100090          0  0.000000
                100218          0  0.000000
                100219         17  0.030741
                100223         39  0.031889
                100276          0  0.000000
                100277         20  0.114286
                100278         39  0.096535
                100279         17  0.055921
                100332          0  0.000000
                100334         54  0.059537
                103384         33  0.073661
                103401          0  0.000000








'''
##################################################

import numpy as np
import pandas as pd
import config as con 
from alive_progress import alive_bar


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
nmb_ents = df_split['orgpermid'].nunique()

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
df_max_obs = df_max_obs[['cluster_key', 'cluster_home']]

# # append home-region information to split-data-frame
df_split = df_split.merge(df_max_obs, on="cluster_key")

# # determine share of observations that have a from physical lvl3permid distinct cluster home
off_home_obs_share = len(df_split[df_split['lvl3permid']!=df_split['cluster_home']])/pre_merge_len

# perform assertions
assert df_split['cluster_home'].isna().sum() == 0, "rows without cluster home in df_split"
assert df_split['lvl3permid'].nunique() == len(regs), f"number of physical regions in df_split does not allign with number of regions in tier list ({len(regs)})"
assert len(df_split) == pre_merge_len, "lenght of df_split changed during merge with home regions"

# initiate random number generator and set seed
rng = np.random.default_rng(seed=con.SEED)

# perform index partition allocation for train-test
# # set up per-region deviation report and result placeholders 
deviation_report = {}
collector = []
i_regions = []
i_assigned_clusters = []
i_decision = []
i_deviations = []
i_dist_trains = []
i_dist_tests = []
i_dist_trains_post = []
i_dist_tests_post = []
i_assigned_obs = []

# # test-train idex split 
with alive_bar(len(regs), title="outer split") as bar:
    for current_region in regs:

        # select rows of one region
        df_current = df_split.query('cluster_home == @current_region').copy()
        df_current['test_train'] = None 

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
            train_size_current = len(df_current.query('test_train=="train"'))
            test_size_current = len(df_current.query('test_train=="test"'))

            # determine distance to partition goal
            dist_train = 1 - train_size_current / train_size_goal
            dist_test = 1 - test_size_current / test_size_goal

            # direct cluster to partition 
            idx = df_current['cluster_key'] == parent

            if dist_train > dist_test:
                df_current.loc[idx, 'test_train'] = 'train'
                decision = "train"
                
            elif dist_train < dist_test: 
                df_current.loc[idx, 'test_train'] = 'test'
                decision = "test"
                
            elif dist_train - dist_test == con.DIST_TOL: 
                df_current.loc[idx, 'test_train'] = 'test'
                decision = "test"

            train_size_post = len(df_current.query('test_train == "train"'))
            test_size_post = len(df_current.query('test_train == "test"'))


            # determine distance to partition goal POST ASSIGNMENT
            dist_train_post = 1 - train_size_post / train_size_goal
            dist_test_post = 1 - test_size_post / test_size_goal

            # report iteration based deviation and distance
            current_deviation = len(df_current[['test_train']].query('test_train=="test"'))/synth_reg_size-con.TEST_SHARE 
            i_regions.append(current_region)
            i_assigned_clusters.append(parent)
            i_deviations.append(current_deviation)
            i_dist_trains.append(dist_train)
            i_dist_tests.append(dist_test)
            i_decision.append(decision)
            i_dist_trains_post.append(dist_train_post)
            i_dist_tests_post.append(dist_test_post)
            i_assigned_obs.append(len(idx[idx==True]))

        # collect test-train allocations
        assert max(df_current.groupby('orgpermid')['test_train'].nunique()) == 1, "entity leakage in train-test partitions"
        assert max(df_current.groupby('cluster_key')['test_train'].nunique()) == 1, "cluster leakage in train-test partitions"
        collector += [df_current.drop_duplicates(subset='orgpermid')]

        # add deviation entry for current region to report
        current_deviation = len(df_current[['test_train']].query('test_train=="test"'))/synth_reg_size-con.TEST_SHARE
        deviation_report.update({current_region:current_deviation})

        print(f"size of region {current_region}: {len(df_current)}")
        print(f'''post-split deviation: {current_deviation}\n''')

        bar()

df_partition = pd.concat(collector)

# # check if all entities are assigned
assert nmb_ents == df_partition['orgpermid'].nunique(), f"number of entities after partitioning != {nmb_ents}"

df_deviation_rep = pd.DataFrame.from_dict(deviation_report, orient="index", columns=["deviation"])
outer_iteration_log = pd.DataFrame ({
    "region": i_regions,
    "cluster":i_assigned_clusters,
    "dist_train":i_dist_trains,
    "dist_test":i_dist_tests,
    "decision": i_decision,
    "deviation":i_deviations,
    "post_alloc_train_dist": i_dist_trains_post,
    "post_alloc_test_dist": i_dist_tests_post,
    "assigned_obs":i_assigned_obs

})

print(f'''train-test split performed \n''')

# perform index partition allocation for fit-val
# # set up per-region deviation report and result placeholders 
inner_deviation_report = {}
collector = []
i_regions = []
i_assigned_clusters = []
i_decision = []
i_deviations = []
i_dist_fits = []
i_dist_vals = []
i_dist_fits_post = []
i_dist_vals_post = []
i_assigned_obs = []
print(df_partition.columns)
# enrich obs-grain data frame with test-train partition info
df_inner_split = df_split.merge(df_partition[['orgpermid', 'test_train']], on='orgpermid', validate='many_to_one')
nmb_train_ents = df_inner_split.loc[df_inner_split["test_train"]=="train", 'orgpermid'].nunique()
    
# # fit-val index split 
with alive_bar(len(regs), title="processing inner split") as bar:
    for current_region in regs:
        # select rows of one region
        df_current = df_inner_split.query('cluster_home == @current_region and test_train == "train" ').copy()
        df_current['fit_val'] = None 

        # create ordered list of clusters based on cluster-sample-size in current region, random order on ties
        counts = df_current['cluster_key'].value_counts().rename_axis('cluster_key').reset_index(name='freq')
        counts['rand_weight'] = rng.random(len(counts))
        ordered_clusters = counts.sort_values(
                                            by=['freq', 'rand_weight'], 
                                            ascending=[False, True]
                                            )['cluster_key'].tolist()

        # determine partition goals for current regions; rps
        synth_reg_size = df_current.groupby('cluster_home').size().sum()

        print(f'''synth-reg for {current_region}: {synth_reg_size} 
                vs. size of train partition: {len(df_inner_split.query('cluster_home==@current_region and test_train=="train" '))} ''')

        fit_size_goal = synth_reg_size*con.FIT_SHARE
        val_size_goal = synth_reg_size*con.VAL_SHARE
        fit_size_current = len(df_current.query('fit_val=="fit"'))
        val_size_current = len(df_current.query('fit_val=="val"'))
        
        # assign each cluster to a partition
        for parent in ordered_clusters:
            fit_size_current = len(df_current.query('fit_val=="fit"'))
            val_size_current = len(df_current.query('fit_val=="val"'))

            # determine distance to partition goal
            dist_fit = 1 - fit_size_current / fit_size_goal
            dist_val = 1 - val_size_current / val_size_goal

                    # direct cluster to partition 
            idx = df_current['cluster_key'] == parent

            if dist_fit > dist_val:
                df_current.loc[idx, 'fit_val'] = 'fit'
                decision = "fit"
                
            elif dist_fit < dist_val:
                df_current.loc[idx, 'fit_val'] = 'val'
                decision = "val"
                
            elif dist_fit - dist_val == con.DIST_TOL: 
                df_current.loc[idx, 'fit_val'] = 'val'
                decision = "val"

            fit_size_post = len(df_current.query('fit_val=="fit"'))
            val_size_post = len(df_current.query('fit_val=="val"'))

            # determine distance to partition goal POST ASSIGNMENT
            dist_fit_post = 1 - fit_size_post / fit_size_goal
            dist_val_post = 1 - val_size_post / val_size_goal

            # report iteration based deviation and distance
            current_deviation = len(df_current[['fit_val']].query('fit_val=="val"'))/synth_reg_size-con.VAL_SHARE 
            i_regions.append(current_region)
            i_assigned_clusters.append(parent)
            i_deviations.append(current_deviation)
            i_dist_fits.append(dist_fit)
            i_dist_vals.append(dist_val)
            i_decision.append(decision)
            i_dist_fits_post.append(dist_fit_post)
            i_dist_vals_post.append(dist_val_post)
            i_assigned_obs.append(len(idx[idx==True]))

            # collect test-train allocations
        assert max(df_current.groupby('orgpermid')['fit_val'].nunique()) == 1, "entity leakage in fit-val partitions"
        assert max(df_current.groupby('cluster_key')['fit_val'].nunique()) == 1, "cluster leakage in fit-val partitions"
        collector += [df_current.drop_duplicates(subset='orgpermid')]

        # add deviation entry for current region to report
        current_deviation = len(df_current[['fit_val']].query('fit_val=="val"'))/synth_reg_size-con.VAL_SHARE
        inner_deviation_report.update({current_region:current_deviation})

        print(f"size of region {current_region}: {len(df_current)}")
        print(f'''post split deviation: {current_deviation}\n''')

        bar()

df_val_partition = pd.concat(collector)

# # check if all entities are assigned
assert nmb_train_ents == df_val_partition['orgpermid'].nunique(), f"number of entities after partitioning != {nmb_train_ents} (= {df_val_partition['orgpermid'].nunique()})"

df_inn_deviation_rep = pd.DataFrame.from_dict(inner_deviation_report, orient="index", columns=["inner_deviation"])
inner_iteration_log = pd.DataFrame ({
    "region": i_regions,
    "cluster":i_assigned_clusters,
    "dist_fit":i_dist_fits,
    "dist_val":i_dist_vals,
    "decision": i_decision,
    "deviation":i_deviations,
    "post_alloc_fit_dist": i_dist_fits_post,
    "post_alloc_val_dist": i_dist_vals_post,
    "assigned_obs":i_assigned_obs
})

df_partition = df_partition.merge(df_val_partition, on=['orgpermid', 'lvl3permid', 'cluster_key',
                                                        'cluster_home', 'test_train'], how="left" )


print(f'''outer dev report:''')
print(df_deviation_rep)
print(f'''\n''')
print(f'''outer iteration log''')
print(outer_iteration_log[["cluster","decision"]])
print(f'''\n''')
print(f'''inner dev report''')
print(df_inn_deviation_rep)
print(f'''\n''')
print(f'''innter iteration log''')
print(inner_iteration_log[["cluster","decision"]])
print(f'''\n''')
print(f'''final partitions''')

print(df_partition[['test_train']].groupby(['test_train']).size())
print(df_partition[['test_train', 'fit_val']].groupby(['test_train', 'fit_val']).size())

# merge partition columns
df_partition['partition'] = np.where(df_partition['test_train']=="train", df_partition['fit_val'], df_partition['test_train'])

# save split artifact
split = df_partition[['orgpermid','partition']].copy()
assert split['partition'].isna().sum() == 0, "NaN in partion columns" 
assert split['partition'].isin(['test','fit','val']).all(), "incorrect level(s) in partition column"
split.to_parquet(con.SPLIT)

# off-home leakage
# # fan out partitions to observations
df_leakage = df_split.merge(df_partition[['orgpermid', 'test_train', 'fit_val', 'partition']], on='orgpermid', validate='many_to_one')

# # create reports
df_leakage['off_home'] = df_leakage['cluster_home'] != df_leakage['lvl3permid']
report = (df_leakage.groupby('partition')
                    .apply(lambda g: g.groupby('cluster_home')['off_home'].agg(['sum','mean'])))

print(report)



