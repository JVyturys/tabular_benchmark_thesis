##################################################
'''
src.exploration.raw_data.diagnose_geo_attriton

input: raw_panel.parquet, ref_geo_raw.parquet, raw SQL
purpose: determine_geo_attrition.py reveals an attrition of 1438 entities,
        investigate country of origin and reason for attririon for those entities
output: out of 9721 entities 9715 have a valid regional ID,
        of those 1432 entities have no match within the tr_regcntrymap which
        is required for the lvl3permid mapping,
        1431 entities are listed within 105758 domcntrypermid (which is not a valid lvl3permid),
        a sample reveals that 105758 associates to chineese entities,
        lvl3permid for chineese companies: 100089
        1 entity is listed within 110515 domcntrypermid (which is not a valid lvl3permid),
        a sample reveals that 110515 associates to sudaneese entities,
        lvl3permid for sudaneese companies: 100218,
        
        ammend config with harcoded overrides for 110515 and 105758
        not geographically referenceable entities: 6
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lg

# initialize database connection
db = wrds.Connection(wrds_username=lg.wrds_log)

# read data
raw_panel = pd.read_parquet(con.RAW_PANEL)
ref_geo_raw = pd.read_parquet(con.REF_GEO_RAW)


##############################################
# define ID string for the SQL query 
org_id = raw_panel["orgpermid"].dropna().astype('Int64').unique()
orgid_string = ','.join(str(id) for id in org_id)

try:
    # determine number of entities with valid geography ID
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
        WHERE orgpermid IN ({orgid_string}) 
    """

    # execute SQL query
    print(f'''\nNumber of entities with valid geography ID:''')
    print(db.raw_sql(query_1))

except Exception as error: 
    print(f"An error occured during the query_1 SQL pull:\n{error}")

try:
    # investigate number of entities that have an ID but are not referenced in tr database
    query_2 = f"""
        SELECT
            p.domcntrypermid,
            COUNT(*) as n_entities
        FROM tr_common.permorgref p
        WHERE p.orgpermid IN ({orgid_string}) 
            AND p.domcntrypermid IS NOT NULL
            AND p.domcntrypermid NOT IN (
                SELECT lvl5permid FROM tr_common.tmcregncntrymap
                )
        GROUP BY p.domcntrypermid
        ORDER BY n_entities DESC
    """

    # execute SQL query
    print(f'''\nNumber of entities with geo ID but not referenced in tmcregncntrymap:''')
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
    print(f'''\nInvestigate ID 105758:''')
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
        WHERE p.orgpermid IN ({orgid_string})
            AND p.domcntrypermid = 105758
        LIMIT 30
"""
    # execute SQL query
    print(f'''\nSample of companies with lvl3permid == 105758\n''')
    entities_105758 = db.raw_sql(query_4) 
    print(entities_105758.columns)
    print(entities_105758['comname'])

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
    print(f'''\nInvestigate lvl3permid for chineese companies''')
    print(db.raw_sql(query_5))
    
except Exception as error:
    print(f"An error occured during the query_5 SQL pull:\n{error}")


# investigate origin of entity with domcntrypermid = 110515
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
    print(db.raw_sql(query_7))

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
    print(f'''tmcregncntrymap codes for sudaneese entities:''')    
    print(db.raw_sql(query_8))


except Exception as error:
    print(f"An error occured during the query_8 SQL pull:\n{error}")