##################################################
'''
src.exploration.final_panel.determine_geoID_dist

input: panel.parquet, ref_geo_table.parquet
purpose: determine the observations per region and the 
        number of entities per region
output:
        Observations per Region - Share
                    entities  observations  obs / entity
        lvl3permid                                      
        100277           151          1734         11.48
        100090           124          1412         11.39
        110000             2            22         11.00
        100334           925          8967          9.69
        103384           452          4364          9.65
        100024           668          6352          9.51
        100279           325          3036          9.34
        100089          2743         25243          9.20
        100219           609          5511          9.05
        100223          1396         12261          8.78
        103401           459          3360          7.32
        100060             5            34          6.80
        100276          1055          6947          6.58
        100218            88           541          6.15
        100278           693          4165          6.01
        100087            11            62          5.64
        100332             9            38          4.22

'''
##################################################

import pandas as pd
import config as con
import math

# load data
df = pd.read_parquet(con.PANEL) 
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)

# extract org IDs
df_ents_per_region = df[['orgpermid']].drop_duplicates()
print(f'''\nnmb. of unique entities in panel: {len(df_ents_per_region)}''')

# merge with geo IDs
df_ents_per_region = df_ents_per_region.merge(ref_geo, how='left', on='orgpermid')

# group by region to get entities per region
print(f'''\nentities per region:''')
print(f'''{df_ents_per_region.groupby('lvl3permid', dropna=False).size().sort_values(ascending=False)}''')   

# extract org IDs for observation count
df_obs_per_region = df[['orgpermid']]
print(f'''\nnmb. of observations in panel: {len(df_obs_per_region)}''')

# merge with geo IDs
df_obs_per_region = df_obs_per_region.merge(ref_geo, how='left', on='orgpermid')

# group by region to get entities per region
print(f'''\nobs per region:''')
print(f'''{df_obs_per_region.groupby('lvl3permid', dropna=False).size().sort_values(ascending=False)}''')   

# build grouped region tables side-by-side
grp_ents_per_region = df_ents_per_region.groupby('lvl3permid', dropna=False).size()
grp_obs_per_region = df_obs_per_region.groupby('lvl3permid', dropna=False).size()
regional_obs_per_ent = grp_obs_per_region.div(grp_ents_per_region, level='lvl3permid')

split_30 = round(grp_obs_per_region*0.3).astype('int64')
split_70 = round(grp_obs_per_region*0.7).astype('int64')

split_20 = round(grp_obs_per_region*0.2).astype('int64')
split_80 = round(grp_obs_per_region*0.8).astype('int64')

split_25 = round(grp_obs_per_region*0.25).astype('int64')
split_75 = round(grp_obs_per_region*0.75).astype('int64')

df_reg_obs_per_ent = pd.DataFrame({
    "entities" : grp_ents_per_region,
    "observations" : grp_obs_per_region,
    "obs / entity" : round(regional_obs_per_ent, 2),
    "70%": split_70,
    "30%": split_30,
    "80%": split_80,
    "20%": split_20,
    "75%": split_75,
    "25%": split_25
})

print(f'''Observations per Region - Share''')
print(f'''{df_reg_obs_per_ent.sort_values('obs / entity', ascending=False)}''')