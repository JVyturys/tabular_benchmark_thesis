##################################################
'''
VI.

Parent - subsidiary relations might cause data leakage in the split. To prevent this I am pulling 
a parent table to consider those relations later on during the data split.

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
output_path = con.PROJECT_ROOT / "data" / "raw"
results_path = con.RESULTS_DIR/ "exploration" / "joined_panel"

# loading data
panel = pd.read_parquet(data_path)
print(panel['orgpermid'].nunique()) # -> 9656 unique entities

# entity ID list for pulling entity parent table
ids = panel["orgpermid"].dropna().astype(int).unique()
id_list = ','.join(str(id) for id in ids)


# SQL
query_1 = f""" 
    SELECT DISTINCT 
        orgpermid,
        immediateparentorgpermid,
        ultimateparentorgpermid
    FROM tr_common.permorgref
    WHERE orgpermid IN ({id_list})
    """

try:
    # pull from database
    ref_parents = db.raw_sql(query_1)
    print(ref_parents.shape)
    print(ref_parents.isnull().sum())
    
    ref_parents.to_parquet(con.PROJECT_ROOT / "data" / "raw" / "parent_ref_table.parquet") # saving results

    print(f"Parent reference table successfully saved.")
except Exception as error:
    print(f"An error occured during execution of query 1.\n Error message:{error}")


