##################################################
'''
src.production.from_raw.build_ref_geo_raw

input: raw_panel.parquet
purpose: build parent reference table for raw panel    
output: ref_geo_raw.parquet'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

# initialize database
db = wrds.Connection(wrds_username = lc.wrds_log)

# define data path
data_path = con.RAW_PANEL

# load data
df = pd.read_parquet(data_path)

# cast ids to int
df['orgpermid'] = df['orgpermid'].astype('int64')

# extract org IDs
org_id = df['orgpermid'].drop_duplicates()
org_ids = org_id.to_list()

# build ID-string for SQL query 
orgid_string = ','.join(str(id) for id in org_ids)

# define SQL query to pull geo meta data from WRDS database
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
# execute SQL pull
ref_geo_raw = db.raw_sql(query)

# save output
ref_geo_raw.to_parquet(con.REF_GEO_RAW)
print(f'''"ref_geo_raw.parquet successfully saved."''')




