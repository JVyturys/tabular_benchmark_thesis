##################################################
'''
src.exploration.split_unit.determine_NaN_geo_shares

input: panel.parquet, split.parquet, ref_geo_table.parquet, pre_processing_constants.parquet
purpose: determine relation between missingness ratios and respective regions

output: 4x4 plot of feature-nan composition per region

                            n  rate_retained  rate_dropped
            lvl3permid                                    
            100089      15117       0.009417      0.593668
            100223       7406       0.015506      0.607499
            100334       5344       0.017955      0.578494
            100276       4170       0.021408      0.660898
            100024       3830       0.021571      0.664927
            100219       3288       0.025287      0.657336
            103384       2648       0.020997      0.639189
            100278       2472       0.011246      0.575449
            103401       2009       0.029228      0.698349
            100279       1799       0.019335      0.583380
            100277       1067       0.015033      0.587713
            100090        844       0.032076      0.609153
            100218        322       0.029391      0.748059
            100087         34       0.016941      0.652836
            100332         22       0.016000      0.640828
            100060          9       0.045333      0.713294



'''
##################################################
import pandas as pd
import config as con
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# load data
panel = pd.read_parquet(con.PANEL)
panel = panel.drop(columns=['year', 'esg_combined_score'])
split = pd.read_parquet(con.SPLIT)
geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])

features = panel.drop(columns=['orgpermid']).columns
retained_features = pd.read_parquet(con.PRE_PROS_CONTS, columns=['variable'])['variable'].tolist()
dropped_features = con.CUTOFF_VARS
degvar_features = con.DEGVAR_VARS

# slice fit partition
fit = panel.merge(split, on='orgpermid', how='left')
fit = fit.query('partition=="fit"')
assert fit['partition'].unique() == ['fit'], "conatminated fit partition"
fit = fit.merge(geo, on='orgpermid', how='left')

regions = [*con.TIER1_REGS, *con.TIER2_REGS]

# calculate missingness share for each varible per region
region_sizes = fit.groupby('lvl3permid').size().sort_values(ascending=False)
nan_ana = fit.groupby('lvl3permid')[features].apply(lambda x: x.isna().mean()).T
nan_ana_dro = fit.groupby('lvl3permid')[dropped_features].apply(lambda x: x.isna().mean()).T
nan_ana_ret = fit.groupby('lvl3permid')[retained_features].apply(lambda x: x.isna().mean()).T

pop_drop_quant = pd.DataFrame(fit.groupby('lvl3permid')[features].apply(lambda x: (len(x)-x.isna().sum())/len(x)).T)

pop_dropped = pop_drop_quant.loc[dropped_features]   
usable_dropped = pd.DataFrame({
    "variables_dropped":(pop_dropped >= 0.75).sum(axis=0),
    "n_obs":region_sizes},
    index=region_sizes.index)
print(usable_dropped)
    

report_regions = []
avg_retrate = []
avg_drorate = []
report_regsize = []

for reg in region_sizes.index:
    avg_retrate.append(nan_ana_ret[reg].mean())
    avg_drorate.append(nan_ana_dro[reg].mean())
    report_regsize.append(region_sizes[reg])


df_nan_report = pd.DataFrame({
    "n":report_regsize,
    "nan_rate_retained_features":avg_retrate,
    "nan_rate_dropped_features":avg_drorate
}, index=region_sizes.index)

print(df_nan_report)

# plot
plt.style.use('seaborn-v0_8-whitegrid')

color_retained = "#1f4e79"  
color_dropped  = "#d9534f"  
color_degvar   = "#cccccc"  
color_default  = "#5cb85c"

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), constrained_layout=True)
axes = axes.flatten()

for i, region in enumerate(region_sizes.index):
    if i >= len(axes): 
        break 
        
    ax = axes[i]
    
    region_data = nan_ana[region].sort_values(ascending=True)
    
    bar_colors = [
        color_retained if feature in retained_features else 
        color_dropped if feature in dropped_features else 
        color_degvar if feature in degvar_features else
        color_default 
        for feature in region_data.index
    ]
            
    ax.barh(range(len(region_data)), region_data.values, color=bar_colors, height=1.0, edgecolor='none')
    
    display_name = f"{region}*" if region in con.TIER2_REGS else str(region)
    ax.set_title(f'{display_name}, n = {region_sizes[region]}', fontweight='bold', pad=8, fontsize=12)
    
    ax.set_xlim(0, 1) 
    
    if i >= 12: 
        ax.set_xlabel('NaN Share', labelpad=5, fontsize=11)
    else:
        ax.set_xlabel('')
        
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.grid(False, axis='y')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    ax.spines['bottom'].set_color('#cccccc')

for j in range(len(regions), len(axes)):
    axes[j].set_visible(False)

legend_elements = [
    Patch(facecolor=color_retained, label='Retained Features'),
    Patch(facecolor=color_dropped, label='Dropped Features'),
    Patch(facecolor=color_degvar, label='Features with Degenerate Variance')
]

fig.legend(
    handles=legend_elements, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 1.01), 
    ncol=3,  
    fontsize=13, 
    frameon=False
)

plt.savefig(con.VIZ_NAN_SHARE, dpi=600, bbox_inches='tight')
plt.show()