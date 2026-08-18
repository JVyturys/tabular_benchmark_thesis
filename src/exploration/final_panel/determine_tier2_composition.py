##################################################
'''
src.exploration.split_unit.determine_tier2_composition

input: panel.partquet, ref_geography.parquet, ref_cluster_keys
purpose: explore the composition of each tier 2 region
output: 

        Total number of obsevations in tier-2 regions:
        134

        total off home observations in tier-2 regions:0
        which amounts to a total share of 0.0

        Total number of obsevations per region in tier 2:
        cluster_home
        100060    34
        100087    62
        100332    38
        dtype: int64 

        number of observations per cluster in region 100060
                    n  share_of_total_obs
        cluster_key                        
        5000435671   16            0.470588
        5000063281    9            0.264706
        5000462835    5            0.147059
        5052145845    4            0.117647
        total obs in 100060: 34
        sum of total shares in 100060: 1.0
        number of observations per cluster in region 100087
                    n  share_of_total_obs
        cluster_key                        
        5000025223   14            0.225806
        4296813461   11            0.177419
        4295889219    7            0.112903
        5001208962    7            0.112903
        5000758280    4            0.064516
        4298008740    4            0.064516
        4295894791    3            0.048387
        5000026125    3            0.048387
        4298007709    3            0.048387
        5042239845    3            0.048387
        5053943433    3            0.048387
        total obs in 100087: 62
        sum of total shares in 100087: 0.9999999999999999
        number of observations per cluster in region 100332
                    n  share_of_total_obs
        cluster_key                       
        5036776503   6            0.157895
        4295885261   5            0.131579
        4295888681   5            0.131579
        4295885276   5            0.131579
        4296534101   5            0.131579
        4297161004   5            0.131579
        4296901683   4            0.105263
        4298000881   3            0.078947
        total obs in 100332: 38
        sum of total shares in 100332: 0.9999999999999999
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)
cluster= pd.read_parquet(con.REF_CLUSTER_KEYS)

# load regional filter
regs =[*con.TIER2_REGS]

# merge data
df_org =panel[['orgpermid']]
df_geo = geo[['orgpermid', 'lvl3permid']]
df_clust = cluster[['orgpermid', 'cluster_key']]
df_tier2 = df_org.merge(df_geo, on='orgpermid', how='left')
df_tier2 = df_tier2.merge(df_clust, on='orgpermid', how='left')
df_tier2 = df_tier2.query('lvl3permid.isin(@regs)')
total_obs_tier2 = len(df_tier2)

print(f'''Total number of obsevations in tier-2 regions:''')
print(f'''{total_obs_tier2}\n''')

# append cluster-home-region columns
# # define region-cluster distribution h(cluster_key_i, lvl3permid_j)
biv_dist_clgeo = df_tier2[['lvl3permid', 'cluster_key']].groupby(['cluster_key', 'lvl3permid'], dropna=False).size()
df_biv_clgeo = pd.DataFrame({
    "n_observations":biv_dist_clgeo
})

# # determine cluster-regions with highest frequency of observations per cluster
idx_max = df_biv_clgeo.groupby('cluster_key', dropna=False)['n_observations'].idxmax()
max_obs = df_biv_clgeo.loc[idx_max]
df_max_obs = max_obs.reset_index() 
df_max_obs = df_max_obs.rename(columns={'lvl3permid':'cluster_home'})
df_max_obs = df_max_obs[['cluster_key', 'cluster_home']]

# append home-region information
df_tier2 = df_tier2.merge(df_max_obs, on="cluster_key")

# determine share of observations that have a from physical lvl3permid distinct cluster home
off_home_obs = len(df_tier2[df_tier2['lvl3permid']!=df_tier2['cluster_home']])
off_home_obs_share = off_home_obs/total_obs_tier2

print(f'''total off home observations in tier-2 regions:{off_home_obs}\n
            which amounts to a total share of {off_home_obs_share}''')

print(f'''Total number of obsevations per region in tier 2:''')
print(df_tier2.groupby('cluster_home').size(), "\n")



for reg in regs:
    print(f'''number of observations per cluster in region {reg}''')
    df_dummy = pd.DataFrame({"n":df_tier2.query('cluster_home==@reg').groupby(['cluster_key']).size().sort_values(ascending=False)})
    df_dummy["share_of_total_obs"] = df_tier2.query('cluster_home==@reg').groupby(['cluster_key']).size().sort_values(ascending=False)/df_tier2.query('cluster_home==@reg').groupby(['cluster_key']).size().sort_values(ascending=False).sum()
    print(df_dummy)
    print(f'''total obs in {reg}: {df_dummy["n"].sum()}''')
    print(f'''sum of total shares in {reg}: {df_dummy["share_of_total_obs"].sum()}''')
        


