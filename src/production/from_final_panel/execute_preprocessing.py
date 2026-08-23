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


# load data
panel = pd.read_parquet(con.PANEL)
panel = panel.drop(columns=['year', 'esg_combined_score'])
split = pd.read_parquet(con.SPLIT)

# slice fit partition
fit = panel.merge(split, on='orgpermid', how='left')
fit = fit.query('partition=="fit"')
assert fit['partition'].unique() == ['fit'], "conatminated fit partition"
fit_features = fit.drop(columns=['orgpermid', "partition"])

print(vl(fit_features))






    


