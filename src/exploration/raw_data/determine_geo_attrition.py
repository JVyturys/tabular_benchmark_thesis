##################################################
'''
src.exploration.raw_data.determine_geo_attrition

input: raw_panel.parquet, ref_geo_raw.parquet
purpose: investigate number of entities with referenceable geography data,
        determine geography-reference induced entity attrition,
output: number of entities in raw panel: 9721
        number of entities in ref_geo_raw table: 8283
        attrition: 1438
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

# determine attrition due to geographic reference join
print(f'''number of entities in raw panel:''')
nmb_ents_raw = raw_panel['orgpermid'].nunique()
print(nmb_ents_raw)
print(f'''\nnumber of entities in ref_geo_raw table:''')
nmb_ents_geo = ref_geo_raw['orgpermid'].nunique()
print(nmb_ents_geo) 
print(f'''\nattrition: {nmb_ents_raw-nmb_ents_geo}''')


##############################################
# # define ID string for the SQL query 
# org_id = raw_panel["orgpermid"].dropna().astype('Int64').unique()
# orgid_string = ','.join(str(org_id) for id in org_id)

# try:
#     # check number of entities with geo ID
#     query_1 = f"""
#         SELECT
#             COUNT(*) AS total,
#             COUNT(domcntrypermid) as with_country_id,
#             COUNT(domcntrypermid) FILTER (
#                 WHERE domcntrypermid IN (
#                     SELECT lvl5permid FROM tr_common.tmcregncntrymap
#                 )
#             ) AS matched_in_taxonomy
#         FROM tr_common.permorgref
#         WHERE orgpermid IN ({org_id_string}) 
#     """

#     # execute SQL query
#     print(f'''\nNumber of entities with geo ID:\n''')
#     print(db.raw_sql(query_1))

# except Exception as error: 
#     print(f"An error occured during the query_1 SQL pull:\n{error}")

# try:
#     # check number of entities that have an ID but are not referenced in tr database
#     query_2 = f"""
#         SELECT
#             p.domcntrypermid,
#             COUNT(*) as n_entities
#         FROM tr_common.permorgref p
#         WHERE p.orgpermid IN ({org_id_string}) 
#             AND p.domcntrypermid IS NOT NULL
#             AND p.domcntrypermid NOT IN (
#                 SELECT lvl5permid FROM tr_common.tmcregncntrymap
#                 )
#         GROUP BY p.domcntrypermid
#         ORDER BY n_entities DESC
#     """

#     # execute SQL query
#     print(f'''\nNumber of entities with geo ID but not referenced in tmcregncntrymap:\n''')
#     print(db.raw_sql(query_2))

# except Exception as error:
#     print(f"An error occured during the query_2 SQL pull:\n{error}")

# try:
#     # identify 105758 geo ID
#     query_3 = f"""
#         SELECT *
#         FROM tr_common.tmcregncntrymap
#         WHERE lvl5isocntry IN ('105758')
#             OR lvl1permid = 105758
#             OR lvl2permid = 105758
#             OR lvl3permid = 105758
#             OR lvl4permid = 105758
#             OR lvl5permid = 105758
#     """
#     # execute SQL query
#     print(f'''\nCheck what type of ID 105758 is: :\n''')
#     print(db.raw_sql(query_3))
    
# except Exception as error:
#     print(f"An error occured during the query_3 SQL pull:\n{error}")

# try:
#     # emirically identify origin of entities with 105758 lvl3permid  
#     query_4 = f"""
#         SELECT 
#             p.orgpermid,
#             p.comname,
#             p.typecode,
#             p.status,
#             p.domcntrypermid,
#             p.inccntrypermid
#         FROM tr_common.permorgref p
#         WHERE p.orgpermid IN ({org_id_string})
#             AND p.domcntrypermid = 105758
#         LIMIT 30
# """
#     # execute SQL query
#     print(f'''\nSample of companies with lvl3permid == 105758\n''')
#     print(db.raw_sql(query_4))

# except Exception as error:
#     print(f"An error occured during the query_4 SQL pull:\n{error}")

# try:
#     query_5 = f"""
#         SELECT 
#             lvl5isocntry,
#             lvl5permid,
#             lvl3permid,
#             lvl2permid
#         FROM tr_common.tmcregncntrymap
#         WHERE 
#             lvl5isocntry = 'CN'      
# """
#     # execute SQL query
#     print(f'''\nDouble check lvl3permid for Chineese companies\n''')
#     print(db.raw_sql(query_5))
    
# except Exception as error:
#     print(f"An error occured during the query_5 SQL pull:\n{error}")


# # investigate origin of remaining entity
# try:
#     query_7 =  f"""
#         SELECT 
#             orgpermid,
#             comname,
#             domcntrypermid,
#             inccntrypermid
#         FROM tr_common.permorgref
#         WHERE orgpermid IN (
#                         SELECgeo_referenceT orgpermid
#                         FROM tr_common.permorgref
#                         WHERE domcntrypermid = 110515)
#     """
#     print(db.raw_sql(query_7))
#     # the last missing entitiy appears to be sudaneese

# except Exception as error:
#     print(f"An error occured during the query_7 SQL pull:\n{error}")

# try:
#     query_8 = f"""
#         SELECT 
#             lvl5permid,
#             lvl5isocntry,
#             lvl3permid,
#             lvl2permid
#         FROM tr_common.tmcregncntrymap
#         WHERE 
#             lvl5isocntry = 'SD'
#     """
#     # execute SQL query
#     print(db.raw_sql(query_8))

# except Exception as error:
#     print(f"An error occured during the query_8 SQL pull:\n{error}")