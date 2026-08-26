##################################################
'''
src.exploration.split_unit.determine_NaN_geo_shares

input: panel.parquet, split.parquet, ref_geo_table.parquet, pre_processing_constants.parquet
purpose: determine relation between missingness ratios and respective regions

output: 4x4 plot of feature-nan composition per region

                        variables_dropped  n_obs
            lvl3permid                          
            100089                     22  15117
            100223                     23   7406
            100334                     32   5344
            100276                     18   4170
            100024                     14   3830
            100219                     16   3288
            103384                     18   2648
            100278                     33   2472
            103401                      9   2009
            100279                     24   1799
            100277                     26   1067
            100090                     22    844
            100218                      8    322
            100087                     20     34
            100332                     22     22
            100060                     25      9



'''
##################################################
import pandas as pd
import config as con
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

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
    



# plot

## plot missingness rates
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


## plot dropped usable columns
plt.style.use('seaborn-v0_8-whitegrid')

color_dropped = "#d9534f"  
color_nobs = "#1f4e79"     

df_plot = usable_dropped.sort_values(by="n_obs", ascending=False)

x_labels = df_plot.index
x_pos = np.arange(len(x_labels))

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.bar(
    x_pos, 
    df_plot["variables_dropped"], 
    color=color_dropped, 
    alpha=0.85, 
    width=0.6, 
    label="Dropped Variables (<25% NaN-Share)"
)
ax1.set_ylabel("Number of Dropped Variables", color=color_dropped, fontweight="bold", labelpad=10)
ax1.tick_params(axis="y", labelcolor=color_dropped)
ax1.set_xticks(x_pos)

formatted_labels = [f"{reg}*" if reg in con.TIER2_REGS else str(reg) for reg in x_labels]
ax1.set_xticklabels(formatted_labels, rotation=45, ha="right")

ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(
    x_pos, 
    df_plot["n_obs"], 
    color=color_nobs, 
    marker="o", 
    linewidth=2.5, 
    markersize=6, 
    label="Sample Size (n_obs)"
)
ax2.set_ylabel("Sample Size (n_obs)", color=color_nobs, fontweight="bold", labelpad=10)
ax2.tick_params(axis="y", labelcolor=color_nobs)

ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.spines["left"].set_color(color_dropped)
ax2.spines["right"].set_color(color_nobs)

plt.title("Dropped Variables with NaN Share <25% vs. Sample Size per Region", fontweight="bold", pad=20, fontsize=14)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
fig.legend(
    lines_1 + lines_2, 
    labels_1 + labels_2, 
    loc="lower center", 
    bbox_to_anchor=(0.5, 0.98), 
    ncol=2, 
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.savefig(con.VIZ_USABLE_DROPPED, dpi=600, bbox_inches='tight')
plt.show()