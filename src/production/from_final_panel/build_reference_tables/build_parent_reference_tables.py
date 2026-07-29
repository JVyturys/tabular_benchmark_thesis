##################################################
'''
org_parent_table.py
VI.
Create two reference files:
- an org-id to parent-id refrence table
- and parent-id to parent entitiy type table  

'''
##################################################

import wrds
import pandas as pd 
import config as con
import log_config as lc

# Initializing paths and data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)

data_path = con.PANEL
output_path = con.PROJECT_ROOT / "data" / "raw"
results_path = con.RESULTS_DIR/ "exploration" / "joined_panel"

# loading data
df = pd.read_parquet(data_path)


# cast ids to int
df['orgpermid'] = df['orgpermid'].astype('int64')

# cast ids to lists
org_id = df['orgpermid'].unique().tolist()

# turn into string for SQL query
org_string = ','.join(f"{id}" for id in org_id)


# SQL
query_1 = f""" 
    SELECT DISTINCT 
        orgpermid,
        immediateparentorgpermid,
        ultimateparentorgpermid
    FROM tr_common.permorgref
    WHERE orgpermid IN ({org_string})
    """

try:
    # pull from database
    ref_parents = db.raw_sql(query_1)  
    ref_parents.to_parquet(con.PROJECT_ROOT / "data" / "raw" / "ref_parent_table.parquet", index=False) # saving results
    print(f"Parent reference table successfully saved.")

except Exception as error:
    print(f"An error occured during execution of query_1.\n Error: {error}")


# create parent entitiy type reference table
parent = ref_parents.copy()

# cast ids to int
parent['ultimateparentorgpermid'] = parent['ultimateparentorgpermid'].astype('Int64')
parent['immediateparentorgpermid'] = parent['immediateparentorgpermid'].astype('Int64')

# cast ids to lists
parent_id = parent['ultimateparentorgpermid'].dropna().to_list()

# turn into string for SQL query
ultparent_string = ','.join(f"{par_id}" for par_id in parent_id)

query_2 = f"""
     SELECT DISTINCT orgpermid,
                     comname,
                     typecode,
                     ultimateparentorgpermid,
                     immediateparentorgpermid,
                     domcntrypermid
     FROM tr_common.permorgref
     WHERE orgpermid IN ({ultparent_string})
""" 

try:
        
    df_parent_enttype = db.raw_sql(query_2)
    df_parent_enttype.to_parquet(con.PROJECT_ROOT / "data" / "raw" / "ref_parent_ent_type.parquet", index=False)
    print(f"Parent ent-type reference table successfully saved.")
except Exception as error:
    print(f"An error occured during execution of query_2.\n Error: {error}")




