##################################################
'''
src.exploration.final_panel.determine_geoID_dist

input: panel.parquet, ref_geo_table.parquet
purpose: determine the observations per region and the 
        number of entities per region,
        approximate split sizes
output:
        Observations per Region - Share
                entities  observations  obs / entity    70%   30%       val_60/10/30
        lvl3permid                                                                 
        100277           151          1734         11.48   1214   520           170
        100090           124          1412         11.39    988   424           138
        110000             2            22         11.00     15     7             2
        100334           925          8967          9.69   6277  2690           879
        103384           452          4364          9.65   3055  1309           428
        100024           668          6352          9.51   4446  1906           622
        100279           325          3036          9.34   2125   911           298
        100089          2743         25243          9.20  17670  7573          2474
        100219           609          5511          9.05   3858  1653           540
        100223          1396         12261          8.78   8583  3678          1202
        103401           459          3360          7.32   2352  1008           329
        100060             5            34          6.80     24    10             3
        100276          1055          6947          6.58   4863  2084           681
        100218            88           541          6.15    379   162            53
        100278           693          4165          6.01   2916  1250           408
        100087            11            62          5.64     43    19             6
        100332             9            38          4.22     27    11             4

        Numer of clusters for small-sized regions:
        lvl3permid  cluster_key
        100060      5000063281     1
                    5000435671     2
                    5000462835     1
                    5052145845     1
        100087      4295889219     1
                    4295894791     1
                    4296813461     1
                    4298007709     1
                    4298008740     1
                    5000025223     1
                    5000026125     1
                    5000758280     1
                    5001208962     1
                    5042239845     1
                    5053943433     1
        100332      4295885261     1
                    4295885276     1
                    4295888681     2
                    4296534101     1
                    4296901683     1
                    4297161004     1
                    4298000881     1
                    5036776503     1
        110000      4295857491     1
                    4297636264     1

'''
##################################################

import pandas as pd
import config as con

# load data
df = pd.read_parquet(con.PANEL) 
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)
clust_key = pd.read_parquet(con.REF_CLUSTER_KEYS)

# extract org IDs
df_ents = df[['orgpermid']].drop_duplicates()
print(f'''\nnmb. of unique entities in panel: {len(df_ents)}''')

# merge with geo IDs
df_ents = df_ents.merge(ref_geo, how='left', on='orgpermid')

# group by region to get entities per region
print(f'''\nentities per region:''')
print(f'''{df_ents.groupby('lvl3permid', dropna=False).size().sort_values(ascending=False)}''')   

# extract org IDs for observation count
df_obs = df[['orgpermid']]
print(f'''\nnmb. of observations in panel: {len(df_obs)}''')

# merge with geo IDs
df_obs = df_obs.merge(ref_geo, how='left', on='orgpermid')

# group by region to get entities per region
print(f'''\nobs per region:''')
print(f'''{df_obs.groupby('lvl3permid', dropna=False).size().sort_values(ascending=False)}''')   

# build grouped region tables side-by-sidegrp_ents_per_region
grp_ents_per_region = df_ents.groupby('lvl3permid', dropna=False).size()
grp_obs_per_region = df_obs.groupby('lvl3permid', dropna=False).size()
regional_obs_per_ent = grp_obs_per_region.div(grp_ents_per_region, level='lvl3permid')

split_30 = round(grp_obs_per_region*0.3).astype('int64')
split_70 = round(grp_obs_per_region*0.7).astype('int64')
split_val_14p = round(grp_obs_per_region*0.7*0.14).astype('int64')

df_reg_obs_per_ent = pd.DataFrame({
    "entities" : grp_ents_per_region,
    "observations" : grp_obs_per_region,
    "obs / entity" : round(regional_obs_per_ent, 2),
    "70%": split_70,
    "30%": split_30,
    "val_60/10/30":split_val_14p
})

print(f'''Observations per Region - Share''')
print(f'''{df_reg_obs_per_ent.sort_values('obs / entity', ascending=False)}''')

# merge wit cluster keys
df_clust = clust_key[['orgpermid', 'cluster_key']]

df_ents = df_ents.merge(df_clust, how='left', on='orgpermid')
print(f'''df_ents''')
print(df_ents)

small_regions = [110000,100060, 100087, 100332]
grp_ents_clust = df_ents.query('lvl3permid.isin(@small_regions)').groupby(['lvl3permid', 'cluster_key']).size()
print(grp_ents_clust)