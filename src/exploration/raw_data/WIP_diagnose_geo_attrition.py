##################################################
'''
Investigate the attrition of 1438 entities after joining the geography
inforamtion.
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lg

# initialize database connection
db = wrds.Connection(wrds_username=lg.wrds_log)

#define paths
data_path = con.PANEL
results_path = con.RESULTS_DIR / "joined_panel" / "geo_id_merge_attrition"

# read data
panel = pd.read_parquet(data_path)

# define ID string for the SQL query 
ids = panel["orgpermid"].dropna().astype(int).unique()
id_list = ','.join(str(id) for id in ids)