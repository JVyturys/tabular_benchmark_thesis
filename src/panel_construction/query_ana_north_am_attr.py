##################################################
'''
This script investigates the attrition of north american entities after main join.
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
output_path = con.PROJECT_ROOT / 'data' / "raw"
results_path = con.RESULTS_DIR/ 'exploration' / 'joined_panel'

# SQL

query_1 = f'''
    SELECT COUNT(DISTINCT item6105) as n_with_any_financial_record
    FROM tr_worldscope.wrds_ws_funda
    WHERE item6105 IN ( 
        SELECT worldscopecmpid
        FROM tr_common.permorgref
        WHERE orgpermid IN (
            SELECT orgpermid 
            FROM tr_esg.wrds_ref_esg
            WHERE fieldname = 'ESGCombinedScore'
            AND valuescore IS NOT NULL
            AND year BETWEEN 2009 AND 2025
        )
        AND lvl = 100219            
    )
    AND freq = 'A'
    '''