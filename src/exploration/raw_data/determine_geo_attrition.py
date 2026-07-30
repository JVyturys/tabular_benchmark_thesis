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