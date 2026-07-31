##################################################
'''
src.production.from_final_panel.build_reference_tables.build_parent_reference_tables

input: panel.parquet, raw SQL
purpose: build a reference table for the organizational parents of
        the entities included in the panel,
        build a reference table that states the company type of the
        ultimate-parent organization of each entity in the panel,
output: ref_parent_table.parquet,
        ref_parent_ent_type.parquet
'''
##################################################
import wrds
import pandas as pd
import config as con
import log_config as lc

# initialize database connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# load data
df = pd.read_parquet(con.PANEL, columns=['orgpermid'])

# cast ids to int
df['orgpermid'] = df['orgpermid'].astype('int64')

# unique panel entities as SQL id-string
org_id = df['orgpermid'].unique().tolist()
org_string = ','.join(str(i) for i in org_id)

# pull immediate- and ultimate-parent ids for each panel entity
query_1 = f"""
    SELECT DISTINCT
        orgpermid,
        immediateparentorgpermid,
        ultimateparentorgpermid
    FROM tr_common.permorgref
    WHERE orgpermid IN ({org_string})
"""
ref_parents = db.raw_sql(query_1)
ref_parents.to_parquet(con.REF_PARENT, index=False)
print(f'''"ref_parent.parquet" successfully saved. shape={ref_parents.shape}''')

# build the parent entity-type reference table
parent = ref_parents.copy()

# cast ids to int
parent['immediateparentorgpermid'] = parent['immediateparentorgpermid'].astype('Int64')
parent['ultimateparentorgpermid']  = parent['ultimateparentorgpermid'].astype('Int64')

# define ultimate-parent ids as SQL id-string
parent_id = parent['ultimateparentorgpermid'].dropna().unique().tolist()
ultparent_string = ','.join(str(i) for i in parent_id)
print(f"unique ultimate parents to resolve: {len(parent_id)}")

# pull the company type of each ultimate-parent organization
query_2 = f"""
    SELECT DISTINCT
        orgpermid,
        comname,
        typecode,
        ultimateparentorgpermid,
        immediateparentorgpermid,
        domcntrypermid
    FROM tr_common.permorgref
    WHERE orgpermid IN ({ultparent_string})
"""
df_parent_enttype = db.raw_sql(query_2)
df_parent_enttype.to_parquet(con.REF_PARENT_ENT_TYPE, index=False)
print(f'''"ref_parent_ent_type.parquet" successfully saved. shape={df_parent_enttype.shape}''')