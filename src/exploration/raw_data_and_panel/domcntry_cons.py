##################################################
'''
This script looks at the 'domcntryid' of each entity 
and checks if it changes during the analysis time window.
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc
from utils.reporting import AnalysisLogger as Al

# Initializing paths and data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / 'data'
results_path = con.RESULTS_DIR/ 'joined_panel'

df = pd.read_parquet(data_path)
geo_ref = pd.read_csv(con.RAW_DATA_ROOT / "geo_ref.csv")
parent_red = pd.read_parquet(con.RAW_DATA_ROOT / "parent_ref.parquet")
