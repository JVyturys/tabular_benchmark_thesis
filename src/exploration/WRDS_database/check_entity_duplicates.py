#######################################################################
''' 
src.exploration.WRDS_database.check_entitiy_duplicates

input: raw SQL
purpose: load entity list based on parametrization and check 
        for duplicates.
output: Data frame with duplicate entires: Empty DataFrame                       
'''
#######################################################################

import wrds
import log_config as ln

# initialize database connection
db = wrds.Connection(wrds_username = ln.wrds_log)

try:
    # 1. Master entity-year list
    query_1 = f"""
        SELECT DISTINCT p.orgpermid,
                        p.worldscopecmpid,
                        e.year
        FROM tr_common.permorgref p 
        JOIN tr_esg.wrds_ref_esg e
            ON p.orgpermid = e.orgpermid
        WHERE p.typecode = 'COM'
            AND p.worldscopecmpid IS NOT NULL
            AND e.fieldname = 'ESGCombinedScore'
            AND e.valuescore IS NOT NULL
            AND e.value IS NOT NULL
            AND e.year BETWEEN 2009 AND 2025
        ORDER BY orgpermid
    """
    output = db.raw_sql(query_1)
except:
    print(f"An error occured during the entity-year master SQL pull!")

try:
    ## Make sure there are no duplicates within the company-year-financials table
    query_2 = """
    WITH skeleton AS (
        SELECT DISTINCT 
            p.orgpermid, 
            p.worldscopecmpid, 
            e.year
        FROM tr_common.permorgref p
        JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
        WHERE p.typecode = 'COM'
        AND p.worldscopecmpid IS NOT NULL
        AND e.fieldname = 'ESGCombinedScore'
        AND e.valuescore IS NOT NULL
        AND e.year BETWEEN 2009 AND 2025
    )
    SELECT s.orgpermid, s.year, COUNT(DISTINCT f.code) as n_codes
    FROM skeleton s
    JOIN tr_worldscope.wrds_ws_funda f
        ON s.worldscopecmpid = f.item6105
        AND s.year = f.year_
    WHERE f.freq = 'A'
    GROUP BY s.orgpermid, s.year
    HAVING COUNT(DISTINCT f.code) > 1
    """
    slice_ws = db.raw_sql(query_2)
    print(f"Data frame with duplicate entires:\n {slice_ws}")
except:
    print(f"An error occured during the duplicate-check SQL pull!")