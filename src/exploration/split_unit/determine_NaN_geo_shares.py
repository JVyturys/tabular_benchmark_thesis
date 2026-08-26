##################################################
'''
src.exploration.split_unit.determine_NaN_geo_shares

input:
purpose: 
output:
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