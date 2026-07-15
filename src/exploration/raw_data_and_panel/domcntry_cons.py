##################################################
'''
This script analyzes the way domcntryperm id is stored and 
whether it has a time axis within the wrds database.
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
parent_ref = pd.read_parquet(con.RAW_DATA_ROOT / "parent_ref.parquet")

# Find out whether the regional IDs are stored per entitiy or per entity-year
## Look at variables stored in 'permorgref' 
print(db.describe_table(library='tr_common', table='permorgref'))

## Define SQL query to inspect domcntrypermid
query_1 = f'''
    SELECT orgpermid, domcntrypermid
    FROM tr_common.permorgref
'''

try:
    org_and_cntry_id = db.raw_sql(query_1)
except Exception as error:
    print(f'An error occured during query pull: Error:{error}')

# domcntry ID is assigned once per orgpermid.
# historical id changes of the domcntryid are not stored.