##################################################
'''
src.production.from_final_panel.execute_preprocessing

input: panel_parquet,  split.parquet
purpose: define the preprocessing constants for feature selection 
        and scaling using the median-imputation induced variance loss.
        
output: pre_processing_constants.parquet
'''
##################################################
import pandas as pd
import config as con
from utils import variance_loss as vl
from utils import vl_cut_off as vlc
import numpy as np



# load data
panel = pd.read_parquet(con.PANEL)
panel = panel.drop(columns=['year', 'esg_combined_score'])
split = pd.read_parquet(con.SPLIT)

# slice fit partition
fit = panel.merge(split, on='orgpermid', how='left')
fit = fit.query('partition=="fit"')
assert fit['partition'].unique() == ['fit'], "conatminated fit partition"
fit_features = fit.drop(columns=['orgpermid', "partition"])
print(len(fit_features.columns))

# determine variance loss
cuoff_metrics = vl(fit_features)
cuoff_metrics = cuoff_metrics.loc[cuoff_metrics['variance_loss'].isna()==False].sort_values(by='variance_loss')

# determine kneedle
co_idx, co_vl = vlc(cuoff_metrics['variance_loss'])

prepros_constants = cuoff_metrics.query('variance_loss <= @co_vl').copy()
prepros_constants = prepros_constants[['variable', 'mean', 'median', 'variance']]
prepros_constants['variance'] = np.sqrt(prepros_constants['variance'])
prepros_constants = prepros_constants.rename(columns={'variance':"std"})
prepros_constants.to_parquet(con.PRE_PROS_CONTS)










    


