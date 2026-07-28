######################################################################
'''
II.

Investigate the attrition of 1438 entities after joining the geography
inforamtion.
'''
######################################################################

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

try:
    # check number of entities with geo ID
    query_1 = f"""
        SELECT
            COUNT(*) AS total,
            COUNT(domcntrypermid) as with_country_id,
            COUNT(domcntrypermid) FILTER (
                WHERE domcntrypermid IN (
                    SELECT lvl5permid FROM tr_common.tmcregncntrymap
                )
            ) AS matched_in_taxonomy
        FROM tr_common.permorgref
        WHERE orgpermid IN ({id_list}) 
    """

    # execute SQL query
    print(db.raw_sql(query_1))

except Exception as error: 
    print(f"An error occured during the query_1 SQL pull:\n{error}")

try:
    # check number of entities that have an ID but are not referenced in tr database
    query_2 = f"""
        SELECT
            p.domcntrypermid,
            COUNT(*) as n_entities
        FROM tr_common.permorgref p
        WHERE p.orgpermid IN ({id_list}) 
            AND p.domcntrypermid IS NOT NULL
            AND p.domcntrypermid NOT IN (
                SELECT lvl5permid FROM tr_common.tmcregncntrymap
                )
        GROUP BY p.domcntrypermid
        ORDER BY n_entities DESC
    """

    # execute SQL query
    print(db.raw_sql(query_2))

except Exception as error:
    print(f"An error occured during the query_2 SQL pull:\n{error}")

try:
    # identify 105758 geo ID
    query_3 = f"""
        SELECT *
        FROM tr_common.tmcregncntrymap
        WHERE lvl5isocntry IN ('105758')
            OR lvl1permid = 105758
            OR lvl2permid = 105758
            OR lvl3permid = 105758
            OR lvl4permid = 105758
            OR lvl5permid = 105758
    """
    # execute SQL query
    print(db.raw_sql(query_3))
    
except Exception as error:
    print(f"An error occured during the query_3 SQL pull:\n{error}")

try:
    # emirically identify origin of entities with 105758 lvl3permid  
    query_4 = f"""
        SELECT 
            p.orgpermid,
            p.comname,
            p.typecode,
            p.status,
            p.domcntrypermid,
            p.inccntrypermid
        FROM tr_common.permorgref p
        WHERE p.orgpermid IN ({id_list})
            AND p.domcntrypermid = 105758
        LIMIT 30
"""
    # execute SQL query
    print(db.raw_sql(query_4))

except Exception as error:
    print(f"An error occured during the query_4 SQL pull:\n{error}")

try:
    query_5 = f"""
        SELECT 
            lvl5isocntry,
            lvl5permid,
            lvl3permid,
            lvl2permid
        FROM tr_common.tmcregncntrymap
        WHERE 
            lvl5isocntry = 'CN'      
"""
    # execute SQL query
    print(db.raw_sql(query_5))
    
except Exception as error:
    print(f"An error occured during the query_5 SQL pull:\n{error}")

try:
    # Select all chineese companies
    query_6 = f"""
        SELECT DISTINCT orgpermid
        FROM tr_common.permorgref
        WHERE orgpermid IN ({id_list})
        AND domcntrypermid = 105758    
"""
    chineese_entities = db.raw_sql(query_6)
    
    # appending lvl-3 and lvl-5-iso-country IDs 
    chineese_entities['lvl3permid'] = 100089.0
    chineese_entities['lvl5isocntry'] = 'CN'

    geo_ref = pd.read_csv(con.PROJECT_ROOT/"data"/"raw"/"geo_reference_table.csv") # loading geo reference table
    geo_ref_comp = pd.concat([geo_ref, chineese_entities], ignore_index=True) # concatenating wit geo reference table
    
except Exception as error:
    print(f"An error occured during the query_6 SQL pull:\n{error}")

# investigate last unknown entity
try:
    query_7 =  f"""
        SELECT 
            orgpermid,
            comname,
            domcntrypermid,
            inccntrypermid
        FROM tr_common.permorgref
        WHERE orgpermid IN (
                        SELECT orgpermid
                        FROM tr_common.permorgref
                        WHERE domcntrypermid = 110515)
    """

    # the last missing entitiy appears to be a sudaneese one

except Exception as error:
    print(f"An error occured during the query_7 SQL pull:\n{error}")


try:
    query_8 = f"""
        SELECT 
            lvl5permid,
            lvl5isocntry,
            lvl3permid,
            lvl2permid
        FROM tr_common.tmcregncntrymap
        WHERE 
            lvl5isocntry = 'SD'
    """
    # execute SQL query
    print(db.raw_sql(query_8))

except Exception as error:
    print(f"An error occured during the query_8 SQL pull:\n{error}")

try:
    # append sudaneese entitiy
    query_9 = f"""
                SELECT DISTINCT orgpermid
        FROM tr_common.permorgref
        WHERE orgpermid IN ({id_list})
        AND domcntrypermid = 110515   
    """

    sudan_ent = db.raw_sql(query_9)
    sudan_ent['lvl3permid'] = 100218
    sudan_ent['lvl5isocntry'] = 'SD'

    geo_ref_comp = pd.concat([geo_ref_comp, sudan_ent], ignore_index=True) # concatenating wit geo reference table
        
    # save clean reference table
    geo_ref_comp.to_csv(con.PROJECT_ROOT / "data" / "raw" / "geo_ref_table_clean", index=False)

    # print(geo_ref_comp.shape)
    # (9695, 3) -> panel has 9701 entities but reference table has 9695. I am excluding the entities that I cannot evaluate geographically.

except Exception as error:
    print(f"An error occured during the query_9 SQL pull:\n{error}")

