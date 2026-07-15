##################################################
'''
Pull the string descriptions for the 'lvl3permid' codes from the wrds database 
and generate lookup table. # WORK IN PROGRESS
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

# SQL
query_1 = f''' 
    SELECT * 
    FROM tr_common.tmcregncntrymap
'''

try:
    # pulling string names from wrds database
    print(db.raw_sql(query_1))
    pass
except Exception as error:
    print(f'An error occured during query pull. Error:{error}')


# lookup
name_lookup = {
    100024.0 : "",
    100060.0 : "",
    100087.0 : "",
    100089.0 : "",
    100090.0 : "",
    100218.0 : "",
    100219.0 : "",
    100223.0 : "",
    100276.0 : "",
    100277.0 : "",
    100278.0 : "",
    100279.0 : "",
    100332.0 : "",
    100334.0 : "",
    103384.0 : "",
    103401.0 : "",
    110000.0 : "",
}
