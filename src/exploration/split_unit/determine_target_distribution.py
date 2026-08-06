##################################################
'''
src.exploration.split_unit.determine_target_distribution


input: panel.parquet, ref_geography.parquet
purpose: determine shape of the target variable globally
        and per region,
        determine regional shift against reference region
        using the Wasserstein-Distance as a distance metric
output:

'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)

# load tier-one region list
tier1_regs = con.TIER1_REGS
print(tier1_regs)

# slice data
df_target = panel[['orgpermid', 'esg_combined_score']]
df_geo = geo[['orgpermid', 'lvl3permid']]

# merge 
df_target = df_target.merge(df_geo, on="orgpermid", how="left")

# query R² breakout relevant regions
df_target_t1 = df_target.query('lvl3permid.isin(@tier1_regs)')

# group by region and get summary stats
grp_target = df_target_t1.groupby('lvl3permid', dropna = False).agg(
    count = ('orgpermid','count'),
    min = ('esg_combined_score', 'min'),
    q25 = ('esg_combined_score',lambda x: x.quantile(0.25)),
    mean = ('esg_combined_score','mean'),
    q50 = ('esg_combined_score',lambda x: x.quantile(0.5)),
    q75 = ('esg_combined_score',lambda x: x.quantile(0.75)),
    max = ('esg_combined_score', 'max'),
).sort_values('count', ascending=False)

print(grp_target)









