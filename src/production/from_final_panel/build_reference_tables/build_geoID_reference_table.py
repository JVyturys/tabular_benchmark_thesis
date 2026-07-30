#######################################################################
''' 
build_geoID_reference_table.py

This script creates the geography reference table for the lvl3permid 
geo codes. This tables allows me to check in which region and country 
an entity is based. Will be userd for geographic performance deivergence 
analysis.

# Output: ref_geo_table.parquet
'''
#######################################################################

import wrds
import config as con    
import log_config as lc
import pandas as pd

# initialize database connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# define paths
data_path = con.PANEL
output_path = con.PROJECT_ROOT / "data" / "raw"
results_path = con.RESULTS_DIR/ "exploration" / "joined_panel"

# define entitiy index for SQL query based on final panel
panel = pd.read_parquet(data_path)
org_ids = panel["orgpermid"].dropna().astype('Int64').unique()
orgid_string = ','.join(str(id) for id in org_ids)

try:
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

    # save output
    db.raw_sql(query).to_parquet(con.REF_GEOGRAPHY, index=False)

except:
    print(f"An error occured during the SQL-pull...")

