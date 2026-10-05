##################################################
'''
src.exploration.final_panel.determine_target_distribution


input: panel.parquet, ref_geography.parquet
purpose: determine shape of the target variable globally
        and per region,
        determine regional shift against reference region
        using the Wasserstein-Distance as a distance metric       
output: printed table below; the region figure (distribution + Wasserstein distance) is built by
        src.production.descriptives.build_desc_target_distribution


            count      mean       std       min       q25       q75       max  Wasserstein_dist
    lvl3permid                                                                                     
    100089      25243  0.441597  0.201334  0.000726  0.279514  0.599019  0.918634               0.0
    100223      12261  0.467295  0.185513  0.000609  0.333194  0.602181  0.942029          0.025707
    100334       8967  0.508874  0.200363  0.000594  0.374077  0.665676  0.947581          0.067331
    100276       6947  0.452589  0.188406  0.009745  0.305002  0.596404    0.9184          0.013574
    100024       6352  0.364155  0.187458  0.007767  0.212343  0.492615  0.919269           0.07747
    100219       5511  0.376293  0.189569  0.005823  0.226321   0.51579  0.912548          0.065313
    103384       4364    0.4603  0.215653  0.007312  0.291002  0.623864   0.93732           0.02399
    100278       4165   0.46415  0.165248  0.037376  0.338505  0.581338  0.924948          0.034579
    103401       3360  0.404509  0.229455  0.007136  0.210881  0.581978    0.9479          0.048811
    100279       3036  0.551952  0.196404  0.006267  0.427292  0.701441  0.939816            0.1104
    100277       1734  0.479048  0.176881  0.006502  0.359959  0.605467  0.901359           0.03867
    100090       1412  0.445063  0.181365  0.004173  0.307016  0.576605  0.870799          0.020047
    100218        541  0.337845  0.189577  0.003943  0.168683  0.477939   0.80247          0.103752



'''
##################################################

import pandas as pd
import config as con
from scipy.stats import wasserstein_distance as wd

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)

# load tier-one region list
tier1_regs = con.TIER1_REGS

# slice data
df_target = panel[['orgpermid', 'esg_combined_score']]
df_geo = geo[['orgpermid', 'lvl3permid']]

# merge 
df_target = df_target.merge(df_geo, on="orgpermid", how="left")

assert df_target.isna().sum().sum() == 0, "DataFrame contains missing values."

# query R^2 breakout relevant regions
df_target_t1 = df_target.query('lvl3permid.isin(@tier1_regs)')

# determine WassersteinDistance(WS)-reference-region based on observation frequency
ws_ref_regID = df_target_t1['lvl3permid'].value_counts(dropna=False).idxmax()

# extract reference distribution vector
vec_wsref_target = df_target_t1.query('lvl3permid == @ws_ref_regID')['esg_combined_score']

# calculate WS distance for each region
grp_target = df_target_t1.groupby('lvl3permid', dropna = False).agg(
    count = ('orgpermid','count'),
    mean = ('esg_combined_score','mean'),
    std = ('esg_combined_score','std'),
    min = ('esg_combined_score', 'min'),
    q25 = ('esg_combined_score',lambda x: x.quantile(0.25)),
    q75 = ('esg_combined_score',lambda x: x.quantile(0.75)),
    max = ('esg_combined_score', 'max'),
    Wasserstein_dist = ('esg_combined_score', lambda x: wd(vec_wsref_target, x))
).sort_values('count', ascending=False)

print(grp_target)