##################################################
'''
src.production.from_final_panel.execute_preprocessing

input: panel_parquet,  split.parquet
purpose: implement the preprocessing pipeline for feature selection 
        and scaling using the median-imputation induced variance loss.
output:
'''
##################################################
import pandas as pd
import config as con
from utils import variance_loss as vl
from utils import vl_cut_off as vlc
import seaborn as sns



# load data
panel = pd.read_parquet(con.PANEL)
panel = panel.drop(columns=['year', 'esg_combined_score'])
split = pd.read_parquet(con.SPLIT)

# slice fit partition
fit = panel.merge(split, on='orgpermid', how='left')
fit = fit.query('partition=="fit"')
assert fit['partition'].unique() == ['fit'], "conatminated fit partition"
fit_features = fit.drop(columns=['orgpermid', "partition"])

# determine variance loss
cuoff_metrics = vl(fit_features)
cuoff_metrics = cuoff_metrics.loc[cuoff_metrics['variance_loss'].isna()==False].sort_values(by='variance_loss')

# determine kneedle
co_idx, co_vl = vlc(cuoff_metrics['variance_loss'])

fit_post_co = cuoff_metrics.query('variance_loss > @co_vl').copy()
print(fit_post_co.shape)
print(fit_post_co.columns)







    


