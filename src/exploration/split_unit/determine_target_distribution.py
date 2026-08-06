##################################################
'''
src.exploration.split_unit.determine_target_distribution


input: panel.parquet
purpose: determine shape of the target variable globally
        and per region,
        determine regional shift against reference region
        using the Wasserstein-Distance as a distance metric
output:

'''
##################################################

import wrds
import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
geo = pd.read_parquet(con.REF_GEOGRAPHY)

