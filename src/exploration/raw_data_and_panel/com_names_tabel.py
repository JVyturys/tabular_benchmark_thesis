##################################################
'''
Document the company names, that are included in the final panel.

'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lg
from utils.reporting import AnalysisLogger as Al

log_dir = con.RESULTS_DIR / "exploration" / "raw_data"
db = wrds.Connection(wrds_username=lg.wrds_log)

df = pd.read_parquet(con.FINAL_PANEL_PARQUET)
geo_ref = pd.read_csv(con.RAW_DATA_ROOT / "geo_ref.csv")
parent_ref = pd.read_parquet(con.RAW_DATA_ROOT / "parent_ref.parquet")

orgpermids = df['orgpermid'].astype('int').unique().tolist()
orgpermids_str = ','.join(f"{item}" for item in orgpermids)

try:
    query = f'''
    SELECT orgpermid, ultimateparentorgpermid, immediateparentorgpermid, comname
    FROM tr_common.permorgref
    WHERE orgpermid IN ({orgpermids_str})
'''
    df_com_names = db.raw_sql(query)
except Exception as error:
    print(f'An error occured during query pull. Error:{error}')

df_com_names.to_csv('panel_com_names.csv', index=False)