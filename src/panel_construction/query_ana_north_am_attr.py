##################################################
'''
This script investigates the attrition of north american entities after the main join.
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
    SELECT lvl5isocntry, lvl5permid 
    FROM tr_common.tmcregncntrymap 
    WHERE lvl3permid = 100219
'''

try:
    test = db.raw_sql(query_1) # does 100219 lvl3permid yield north amereican countries? 
    print(test) # -> yes Bermuda, Canada, Greenland, Saint Pirre and Miquelon, USA
except Exception as error:
    print(f'An error occured during query pull X. Error:{error}')



query_2 = f''' 
    SELECT COUNT(DISTINCT item6105) as n_with_any_financial_record
    FROM tr_worldscope.wrds_ws_funda
    WHERE item6105 IN (
        SELECT p.worldscopecmpid
        FROM tr_common.permorgref p
        JOIN tr_common.tmcregncntrymap g
            ON p.domcntrypermid = g.lvl5permid
            WHERE p.orgpermid IN (
                SELECT orgpermid
                FROM tr_esg.wrds_ref_esg
                WHERE fieldname = 'ESGCombinedScore'
                    AND valuescore IS NOT NULL
                    AND year BETWEEN 2009 AND 2025
        )
        AND g.lvl3permid = 100219
        AND p.worldscopecmpid IS NOT NULL
        )
    AND freq = 'A'
'''

try:
    # Goal of queary: Count the amount of north american entites with esg score that have financial data in ws_funda.
    result = db.raw_sql(query_2)
except Exception as error:
    print(f'An error occured during query 2. \nError:{error}')

log = Al(results_path, 'North_Am_Attrition-Data_Coverage.txt')

log.section('Attrition of North American entities - Analysis')

message_string = f'''
Of the 4,579 North American entities confirmed in the Refinitiv ESG universe, only 607 (~13%) survived the Worldscope 
financial data join. A targeted diagnostic confirmed this attrition reflects a genuine Worldscope coverage gap — only 661 
North American entities have any annual financial record in wrds_ws_funda — consistent with Worldscope's known relative
coverage weakness in North American markets where Compustat is the dominant data provider. 
This structural gap means North America is underrepresented in the final panel relative to its actual ESG-rated universe,
which is a potential confound for the geographic generalization hypothesis and is acknowledged as a limitation.

'''

log.log(query_2)
log.log(result)
log.log(message_string)

