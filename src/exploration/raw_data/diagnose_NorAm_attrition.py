##################################################
'''
src.exploration.raw_data.diagnose_NorAm_attrition

input: raw SQL, 
purpose: diagnose_main_join_entity_attrition.py reveals that region 100219 shows the highest attrition 
        after the main join:4575 -> 609, delta: 3966 -> 86,6885% 
output: Of the 4,579 North American entities confirmed in the Refinitiv ESG universe, only 607 (~13%)
        survived the Worldscope financial data join. A targeted diagnostic confirmed this attrition reflects 
        a genuine Worldscope coverage gap; only 661 North American entities have any annual financial record
        in wrds_ws_funda — consistent with Worldscope's known relative coverage weakness in North American
        markets where Compustat is the dominant data provider. This structural gap means North America is
        underrepresented in the final panel relative to its actual ESG-rated universe, which is a potential
        confound for the geographic generalization hypothesis and is acknowledged as a limitation.
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc
from utils.reporting import AnalysisLogger as Al


# Initializing paths and data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)
data_path = con.RAW_PANEL


# define query to identify countries of origin for entities in region 100219
query_1 = f'''
    SELECT lvl5isocntry, lvl5permid 
    FROM tr_common.tmcregncntrymap 
    WHERE lvl3permid = 100219
'''

# execute SQL query
try:
    countries = db.raw_sql(query_1) 
    print(countries) # -> Bermuda, Canada, Greenland, Saint Pirre and Miquelon, USA -> North America
except Exception as error:
    print(f'An error occured during query pull X. Error:{error}')

# define SQL query to count entities with financial records in Worldscope and valid ESG score in relevant time frame 
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
    # execute SQL query
    result = db.raw_sql(query_2)
except Exception as error:
    print(f'An error occured during query 2. \nError:{error}')

