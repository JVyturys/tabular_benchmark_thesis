##################################################
'''
src.exploration.split_unit.determine_partition_target_distributions

input: panel.parquet, split.parquet
purpose: determine target distribution of fit and validation
        partition
output: 
'''
##################################################
import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL, columns=['orgpermid', 'esg_combined_score'])
geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
split = pd.read_parquet(con.SPLIT)

# merge data
panel = panel.merge(geo, on="orgpermid", how='left')
panel = panel.merge(split, on="orgpermid", how='left')

fit_target = panel.query('partition == "fit"').copy()
val_target = panel.query('partition == "val"').copy()

print(fit_target.shape)
print(val_target.shape)



