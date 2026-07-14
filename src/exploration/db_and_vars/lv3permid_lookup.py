##################################################
'''
Pull the string descriptions for the 'lvl3permid' codes from the wrds database 
and generate lookup table.
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


