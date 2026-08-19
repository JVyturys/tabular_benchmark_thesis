##################################################
'''
src.exploration.split_unit.determine_partition_target_distributions

input: panel.parquet, split.parquet
purpose: determine target distribution of fit and validation
        partition
output: 
        fit vs. val target-distribution:
        wsd: 0.002406875068561614
        mean : 0.448734102804331
        mean : 0.44650085153302077
        std : 0.20013903078770717
        std : 0.19926944581152128

        fit vs. val target-distributions per region
        region  n_obs  wsd_target  fit_mean  val_mean   fit_std   val_std
        100277    155    0.054573  0.478099  0.514964  0.162593  0.218581
        100090    158    0.019227  0.435405  0.432210  0.180210  0.173613
        100334    914    0.010506  0.509997  0.500033  0.199947  0.194495
        103384    424    0.013363  0.458652  0.464529  0.214630  0.217220
        100024    645    0.007985  0.359509  0.364978  0.184339  0.182956
        100279    289    0.036578  0.557001  0.535366  0.199108  0.175368
        100089   2554    0.006793  0.440643  0.434876  0.201629  0.204514
        100219    542    0.017523  0.384701  0.399763  0.194425  0.184783
        100223   1226    0.020146  0.472114  0.453172  0.187044  0.180216
        103401    345    0.025129  0.399903  0.395418  0.223509  0.229274
        100276    705    0.023860  0.448075  0.471845  0.185411  0.188395
        100218     62    0.035066  0.331232  0.358546  0.182826  0.183524
        100278    381    0.016662  0.477885  0.488753  0.167943  0.182191
        100060      9    0.281535  0.503029  0.221494  0.072125  0.040022
        100087     11    0.051491  0.445781  0.469134  0.123702  0.059914
        100332      5    0.046295  0.410940  0.387229  0.091662  0.063648
'''
##################################################
import pandas as pd
import config as con
from scipy.stats import wasserstein_distance as wsd

# load data
panel = pd.read_parquet(con.PANEL, columns=['orgpermid', 'esg_combined_score'])
geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
split = pd.read_parquet(con.SPLIT)

# merge data
panel = panel.merge(geo, on="orgpermid", how='left')
panel = panel.merge(split, on="orgpermid", how='left')

# extract fit & val partitions
fit_part = panel.query('partition == "fit"').copy()
val_part = panel.query('partition == "val"').copy()

# determine Wasserstein Distance
wsd_target = wsd(fit_part['esg_combined_score'], val_part['esg_combined_score'])

print(f'''fit vs. val target-distribution:''')
print(f'''wsd: {wsd_target}''')
print(f'''mean : {fit_part['esg_combined_score'].mean()}''')
print(f'''mean : {val_part['esg_combined_score'].mean()}''')
print(f'''std : {fit_part['esg_combined_score'].std()}''')
print(f'''std : {val_part['esg_combined_score'].std()}''')


# determine Wasserstein Distance and distributional parameters per region
regs = [*con.TIER1_REGS, *con.TIER2_REGS]
distributions = []

for i, reg in enumerate(regs):
    curr_fit = fit_part.query('lvl3permid==@reg')
    curr_val = val_part.query('lvl3permid==@reg')
    target_dist = pd.DataFrame ({
    "region": reg,
    "n_obs" : curr_val.shape[0]    ,
    "wsd_target": wsd(curr_fit['esg_combined_score'], curr_val['esg_combined_score']),
    "fit_mean": curr_fit['esg_combined_score'].mean(),
    "val_mean": curr_val['esg_combined_score'].mean(),
    "fit_std": curr_fit['esg_combined_score'].std(),
    "val_std": curr_val['esg_combined_score'].std(),

    }, index=[i])
    distributions.append(target_dist)

        
distributions = pd.concat(distributions)

print(f'''fit vs. val target-distributions per region''')
print(distributions)







