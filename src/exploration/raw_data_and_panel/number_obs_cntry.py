##################################################
'''
Find out the number of observations per country in the final panel.
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
geo = pd.read_csv(con.PROJECT_ROOT / "data" / "raw" / "geo_ref.csv")