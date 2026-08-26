##################################################
'''
src.exploration.split_unit.determine_NaN_geo_shares

input: panel.parquet, split.parquet, ref_geo_table.parquet, pre_processing_constants.parquet
purpose: determine relation between missingness ratios and respective regions

output:
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



fit_regions = fit.groupby('lvl3permid').size()
fit_ret_regions = fit[[*retained_features, 'lvl3permid']].dropna().groupby('lvl3permid').size()
fit_dro_regions = fit[[*dropped_features, 'lvl3permid']].dropna().groupby('lvl3permid').size()

df_nan_ana = pd.DataFrame({
    "n_obs":fit_regions,
    "rate_dropped": fit_dro_regions.div(fit_regions),
    "rate_retained": fit_ret_regions.div(fit_regions)
})

print(df_nan_ana.reindex(regions))

# calculate missingness share for each varible per region
nan_ana = fit.groupby('lvl3permid')[features].apply(lambda x: x.isna().mean()).T
nan_ana_dro = fit.groupby('lvl3permid')[dropped_features].apply(lambda x: x.isna().mean()).T
nan_ana_ret = fit.groupby('lvl3permid')[retained_features].apply(lambda x: x.isna().mean()).T





# # plot
# plt.style.use('seaborn-v0_8-whitegrid')

# color_retained = "#1f4e79"  
# color_dropped  = "#d9534f"  
# color_degvar   = "#cccccc"  
# color_default  = "#5cb85c"

# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), constrained_layout=True)
# axes = axes.flatten()

# for i, region in enumerate(regions):
#     if i >= len(axes): 
#         break 
        
#     ax = axes[i]
    
#     region_data = nan_ana[region].sort_values(ascending=True)
    
#     bar_colors = [
#         color_retained if feature in retained_features else 
#         color_dropped if feature in dropped_features else 
#         color_degvar if feature in degvar_features else
#         color_default 
#         for feature in region_data.index
#     ]
            
#     ax.barh(range(len(region_data)), region_data.values, color=bar_colors, height=1.0, edgecolor='none')
    
#     display_name = f"{region}*" if region in con.TIER2_REGS else str(region)
#     ax.set_title(f'{display_name}', fontweight='bold', pad=8, fontsize=12)
    
#     ax.set_xlim(0, 1) 
    
#     if i >= 12: 
#         ax.set_xlabel('NaN Share', labelpad=5, fontsize=11)
#     else:
#         ax.set_xlabel('')
        
#     ax.set_yticks([])
#     ax.set_yticklabels([])
    
#     ax.grid(True, axis='x', linestyle='--', alpha=0.4)
#     ax.grid(False, axis='y')
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False) 
#     ax.spines['bottom'].set_color('#cccccc')

# for j in range(len(regions), len(axes)):
#     axes[j].set_visible(False)


# legend_elements = [
#     Patch(facecolor=color_retained, label='Retained Features'),
#     Patch(facecolor=color_dropped, label='Dropped Features'),
#     Patch(facecolor=color_degvar, label='Degvar Features')
# ]

# fig.legend(
#     handles=legend_elements, 
#     loc='lower center', 
#     bbox_to_anchor=(0.5, 1.01), 
#     ncol=3,  
#     fontsize=13, 
#     frameon=False
# )

# plt.savefig(con.VIZ_NAN_SHARE, dpi=600, bbox_inches='tight')
# plt.show()