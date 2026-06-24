#######################################################################
# This script creates the geography reference table for the lvl3permid 
# geo codes.
#
# Output: geo_reference_table.csv
#######################################################################

import wrds
import config as con    
import log_config as lc
import pandas as pd

from utils.reporting import AnalysisLogger as Al 

db = wrds.Connection(wrds_username=lc.wrds_log)
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / "data" / "raw"
results_path = con.RESULTS_DIR/ "exploration" / "joined_panel"

# Defining entitiy index for SQL query based on final panel
panel = pd.read_parquet(data_path)
ids = panel["orgpermid"].dropna().astype(int).unique()
id_list = ','.join(str(id) for id in ids)

try:
    # Defining query
    query = f"""
        SELECT DISTINCT
            p.orgpermid,
            g.lvl3permid,
            g.lvl5isocntry
        FROM tr_common.permorgref p
        JOIN tr_common.tmcregncntrymap g
            ON p.domcntrypermid = g.lvl5permid
        WHERE p.orgpermid IN ({id_list})
    """

    # Saving as csv
    db.raw_sql(query).to_csv("geo_reference_table.csv", index=False)

    # Saving SQL pull as df for inspection
    ref_geography = db.raw_sql(query)

except:
    print(f"An error occured during the SQL-pull...")

geo_ref_shape = ref_geography.shape
geo_ref_nulls = ref_geography.isnull().sum()
geo_ref_preview = ref_geography[:5]

try:
    ref_log = Al(output_path=results_path, filename="geo_ref_table_stats.txt")

    ref_log.section("Summary Statistics of the Geography Reference Table")
    ref_log.log(query)
    ref_log.log("Shape:\n")
    ref_log.log(geo_ref_shape)
    ref_log.log("Null entires in table:")
    ref_log.log(geo_ref_nulls)
    ref_log.log("Data preview:")
    ref_log.log(geo_ref_preview)
except: 
    print(f"An error occured during logging...")
