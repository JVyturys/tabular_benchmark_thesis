##################################################
'''
src.exploration.split_unit.determine_target_distribution


input: panel.parquet, ref_geography.parquet
purpose: determine shape of the target variable globally
        and per region,
        determine regional shift against reference region
        using the Wasserstein-Distance as a distance metric       
output: plots: target distribution per region, Distribution of wasserstein distance


                count       min       q25      mean       q50       q75       max  Wasserstein_dist
        lvl3permid                                                                                     
        100089      25243  0.000726  0.279514  0.441597   0.44891  0.599019  0.918634               0.0
        100223      12261  0.000609  0.333194  0.467295  0.473012  0.602181  0.942029          0.025707
        100334       8967  0.000594  0.374077  0.508874  0.522427  0.665676  0.947581          0.067331
        100276       6947  0.009745  0.305002  0.452589  0.447874  0.596404    0.9184          0.013574
        100024       6352  0.007767  0.212343  0.364155  0.341924  0.492615  0.919269           0.07747
        100219       5511  0.005823  0.226321  0.376293  0.353527   0.51579  0.912548          0.065313
        103384       4364  0.007312  0.291002    0.4603  0.476554  0.623864   0.93732           0.02399
        100278       4165  0.037376  0.338505   0.46415  0.457724  0.581338  0.924948          0.034579
        103401       3360  0.007136  0.210881  0.404509  0.384985  0.581978    0.9479          0.048811
        100279       3036  0.006267  0.427292  0.551952  0.583058  0.701441  0.939816            0.1104
        100277       1734  0.006502  0.359959  0.479048  0.486443  0.605467  0.901359           0.03867
        100090       1412  0.004173  0.307016  0.445063  0.451578  0.576605  0.870799          0.020047
        100218        541  0.003943  0.168683  0.337845  0.328974  0.477939   0.80247          0.103752

                      count      mean       std       min       25%       50%       75%       max
        lvl3permid                                                                               
        100089      25243.0  0.441597  0.201334  0.000726  0.279514   0.44891  0.599019  0.918634
        100223      12261.0  0.467295  0.185513  0.000609  0.333194  0.473012  0.602181  0.942029
        100334       8967.0  0.508874  0.200363  0.000594  0.374077  0.522427  0.665676  0.947581
        100276       6947.0  0.452589  0.188406  0.009745  0.305002  0.447874  0.596404    0.9184
        100024       6352.0  0.364155  0.187458  0.007767  0.212343  0.341924  0.492615  0.919269
        100219       5511.0  0.376293  0.189569  0.005823  0.226321  0.353527   0.51579  0.912548
        103384       4364.0    0.4603  0.215653  0.007312  0.291002  0.476554  0.623864   0.93732
        100278       4165.0   0.46415  0.165248  0.037376  0.338505  0.457724  0.581338  0.924948
        103401       3360.0  0.404509  0.229455  0.007136  0.210881  0.384985  0.581978    0.9479
        100279       3036.0  0.551952  0.196404  0.006267  0.427292  0.583058  0.701441  0.939816
        100277       1734.0  0.479048  0.176881  0.006502  0.359959  0.486443  0.605467  0.901359
        100090       1412.0  0.445063  0.181365  0.004173  0.307016  0.451578  0.576605  0.870799
        100218        541.0  0.337845  0.189577  0.003943  0.168683  0.328974  0.477939   0.80247



'''
##################################################

import pandas as pd
import config as con
from scipy.stats import wasserstein_distance as wd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

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

# query R² breakout relevant regions
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




plot
## plot WD barplot
wd_sorted = grp_target['Wasserstein_dist'].sort_values(ascending=False)
wd_sorted.plot(
    kind='barh',
    color='steelblue',
    figsize=(10, 6),
    title='Wasserstein Distance from Reference Region (100089) '
)
plt.xlabel("Distance")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig(con.VIZ_WD, dpi=600)
plt.show()


## plot target dist per region - sample size sorted boxplots with logscale color gradient 
sns.set_theme(style="whitegrid")

group_counts = df_target_t1["lvl3permid"].value_counts()
group_order = group_counts.index

norm = LogNorm(vmin=group_counts.min(), vmax=group_counts.max())
cmap = plt.get_cmap("Blues")
color_palette = {
    group: cmap(0.35 + 0.65 * norm(count)) for group, count in group_counts.items()
}

plt.figure(figsize=(14, 7))

ax = sns.boxplot(
    data=df_target_t1,
    x="lvl3permid",
    y="esg_combined_score",
    order=group_order,
    hue="lvl3permid",
    palette=color_palette,
    legend=False,
)

y_min = df_target_t1["esg_combined_score"].min()
y_max = df_target_t1["esg_combined_score"].max()
y_range = y_max - y_min
ax.set_ylim(y_min - y_range * 0.02, y_max + y_range * 0.18)

ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order, rotation=45, ha="right", fontsize=10)

for i, group in enumerate(group_order):
    count = group_counts[group]
    formatted_count = f"n={count:,}".replace(",", ".")

    ax.text(
        x=i,
        y=y_max + (y_range * 0.03),
        s=formatted_count,
        ha="left",  
        va="bottom",
        rotation=45,  
        fontsize=8,
        color="#444444",
        fontstyle="italic",
    )

plt.title(
    "ESG Combined Score Distribution \n Per Region",
    fontsize=14,
    pad=25,
)
plt.xlabel("Level 3 PermID", fontsize=11, labelpad=15)
plt.ylabel("ESG Combined Score", fontsize=11)

plt.tight_layout()
plt.savefig(con.VIZ_TDIST, dpi=600)
plt.show()
