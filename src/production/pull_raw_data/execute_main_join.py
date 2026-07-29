#######################################################################
''' 
src.production.pull_raw_data.execute_main_join

input: ws_variables_final.csv, SQL
purpose: execute main join via SQL queries across WRDS libraries.
output: "raw_panel.parquet"                      
'''
#######################################################################

import wrds
import config as con
import log_config as ln
import pandas as pd
from utils.reporting import AnalysisLogger as Al

# initialize database connection
db = wrds.Connection(wrds_username = ln.wrds_log)


print(f'''DEBUG: LOADING DATA''')


# load variable set
ws_variables = pd.read_csv(con.RESULTS_DIR / "exploration" / "wrds_database" / "ws_variables_final.csv") 
ws_items = ws_variables.iloc[:,0].tolist()

# transform the item codes back to wrds_ws_funda column names
ws_items_col = [f"item{i:.0f}" for i in ws_items]

# define SQL SELECT string for join operation
feature_cols = ','.join(f"f.{item}" for item in ws_items_col)

print(f'''DEBUG: ENTERING SQL PULL''')
try:
    # execute join for panel data
    query = f"""
    WITH skeleton AS (
        SELECT DISTINCT 
            p.orgpermid, 
            p.worldscopecmpid, 
            e.year,
            e.valuescore as esg_combined_score
        FROM tr_common.permorgref p
        JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
        WHERE p.typecode = 'COM'
        AND p.worldscopecmpid IS NOT NULL
        AND e.fieldname = 'ESGCombinedScore'
        AND e.valuescore IS NOT NULL
        AND e.year BETWEEN 2009 AND 2025
    )
    SELECT s.orgpermid, s.year, s.esg_combined_score, {feature_cols}
    FROM skeleton s
    JOIN tr_worldscope.wrds_ws_funda f
        ON s.worldscopecmpid = f.item6105
        AND s.year = f.year_
    WHERE f.freq = 'A'
    """

    # save output to .parquet
    panel = db.raw_sql(query)
    panel.to_parquet(con.RAW_PANEL)
    print(f'''Dataset "raw_panel.parquet" successfully saved.''')
except:
    print(f"An error occured during the main data merge SQL pull!")

