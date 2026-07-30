######################################################################
'''
src.production.from_raw.build_ref_geo_table

input: ref_geo_raw.parquet
purpose: override missing geo-IDs of CN and SD entities with
        in config.py hardcoded lvl3permids,
        build final geography reference table, 
output: ref_geo_table.parquet
'''
######################################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

# intitialize database connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# load data
df = pd.read_parquet(con.RAW_PANEL)

# cast ids to int
df['orgpermid'] = df['orgpermid'].astype('int64')

# extract org IDs
org_id = df['orgpermid'].drop_duplicates()
org_ids = org_id.to_list()

# build ID-string for SQL query 
orgid_string = ','.join(str(id) for id in org_ids)

# define SQL query to pull geography reference  
query = f"""
        SELECT DISTINCT
            p.orgpermid,
            g.lvl3permid,
            g.lvl5isocntry
        FROM tr_common.permorgref p
        JOIN tr_common.tmcregncntrymap g
            ON p.domcntrypermid = g.lvl5permid
        WHERE p.orgpermid IN ({orgid_string})
"""




