##################################################
'''
src.exploration.final_panel.determine_cluster_region_impurity

input: panel.parquet, ref_geography,  ref_cluster_keys.parquet
purpose: determine by which degree parent clusters are regionally impure, 
        determine the off-dominant share of the impure clusters
output:
    number of total region-parent pairs: 8870
    number of parents: 8606 (must be == 8606)
    number of pure region-parent pairs = number of pure parents: 8388
    number of impure parent region-pairs: 482
    number of non-pure parents: 218
    number observations relating to impure parents: 6393
    share of regional inpure observations in panel: 7.61%

    Top 10 parents by off-regional-dominant observation frequency:
                off_dom_share  max_obs_in_single_region  total_obs_in_region  obs_beyond_home
    cluster_key                                                                               
    4295871566           0.496                        69                  137               68
    4295875798           0.362                        83                  130               47
    4298459635           0.344                        63                   96               33
    4295877325           0.148                        69                   81               12
    4295866806           0.440                        42                   75               33
    4295880522           0.216                        58                   74               16
    5000000933           0.472                        38                   72               34
    4295889547           0.609                        27                   69               42
    4295890620           0.565                        30                   69               39
    4295869130           0.125                        56                   64                8
    observations beyond "regional-home": 2142
    share of observations in panel that relate to "beyond-regional-home" observations: 0.025

'''
##################################################

import pandas as pd
import config as con


# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)
cluster_key = pd.read_parquet(con.REF_CLUSTER_KEYS)

# define org-geo-clusterkey dataframe
df_org = panel[['orgpermid']]
df_geo = geo[['orgpermid', 'lvl3permid']]
df_clustky = cluster_key[['orgpermid', 'cluster_key']]
df_merged = df_org.merge(df_geo, on='orgpermid', how='left')
df_merged = df_merged.merge(df_clustky, on='orgpermid', how='left')

# define region-cluster distribution h(cluster_key_i, lvl3permid_j)
biv_dist_clgeo = df_merged.groupby(['cluster_key', 'lvl3permid'], dropna=False).size()

# define marginal distribution of cluster-parents: h(cluster_key)
mrg_dist_par = df_merged.groupby('cluster_key', dropna=False).size()
print(len(mrg_dist_par))

# determine conditional distribution of region given parent: f(lvl3permid_i|cluster_key_j)
cdl_dist_regpar = biv_dist_clgeo.div(mrg_dist_par, level="cluster_key").reset_index(name='regional_purity')
print(cdl_dist_regpar)

# print impurity diagnostics
print(f'''number of total region-parent pairs: {len(cdl_dist_regpar)}''')
print(f'''number of parents: {cdl_dist_regpar['cluster_key'].nunique()} (must be == {len(mrg_dist_par)})''')
print(f'''number of pure region-parent pairs = number of pure parents: {len(cdl_dist_regpar.query('regional_purity == 1'))}''')
print(f'''number of impure parent region-pairs: {len(cdl_dist_regpar.query('regional_purity != 1'))}''')
print(f'''number of non-pure parents: {len(mrg_dist_par)-len(cdl_dist_regpar.query('regional_purity == 1'))}''')

impure_pars = cdl_dist_regpar.query('regional_purity != 1')['cluster_key'].unique().tolist() # get list of impure parent ids

print(f'''number observations relating to impure parents: {len(df_merged.query('cluster_key.isin(@impure_pars)'))}''')
print(f'''share of regional inpure observations in panel: {round(len(df_merged.query('cluster_key.isin(@impure_pars)'))/len(df_merged),4)*100}%''')

# determine off-dominant share of impure parents
biv_dist_clgeo_imp = biv_dist_clgeo[biv_dist_clgeo.index.get_level_values('cluster_key').isin(impure_pars)] # distribution for impure parents

biv_dist_clgeo_imp = pd.DataFrame({
    'n_observations': biv_dist_clgeo_imp
})

## get indicies for max obervations in region per parent
idx_obs_mass = biv_dist_clgeo_imp.groupby('cluster_key')['n_observations'].idxmax()

# get rows based on idx list
max_obs = biv_dist_clgeo_imp.loc[idx_obs_mass]
max_obs = max_obs.rename(columns={'n_observations' : 'max_obs_in_single_region'})
print(max_obs)

# calculate total observations across regions per impure parent
total_obs = biv_dist_clgeo_imp.groupby(['cluster_key']).sum()
total_obs = total_obs.rename(columns={'n_observations': 'total_obs_in_region'})
print(total_obs)

# calculate the share of observations, that lie in the region with most observations
off_dominant_share = 1 - max_obs['max_obs_in_single_region'].div(total_obs['total_obs_in_region'],
                                                                level='cluster_key')

df_off_dominant_share = pd.DataFrame({
    'off_dom_share': round(off_dominant_share, 3)
 })

df_off_dominant_share= df_off_dominant_share.merge(max_obs, on='cluster_key', how='left')
df_off_dominant_share = df_off_dominant_share.merge(total_obs, on='cluster_key', how='left')
df_off_dominant_share['obs_beyond_home'] = df_off_dominant_share['total_obs_in_region']- df_off_dominant_share['max_obs_in_single_region']
print(f'''Top 10 parents by off-regional-dominant observation frequency:''')
print(df_off_dominant_share.sort_values(by=['total_obs_in_region', 'off_dom_share', 'obs_beyond_home'], ascending=False)[:10])
print(f'''observations beyond "regional-home": {df_off_dominant_share['obs_beyond_home'].sum()}''')
print(f'''share of observations in panel that relate to "beyond-regional-home" observations: {round(df_off_dominant_share['obs_beyond_home'].sum()/len(df_merged),3)}''')
print(f'''highest off-dominant share: {df_off_dominant_share['off_dom_share'].max()}''')
print(f'''lowest off-dominant share: {df_off_dominant_share['off_dom_share'].min()}''')