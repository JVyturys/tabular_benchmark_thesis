######################################################################
'''
src.production.from_raw.override_SD_CN_geo_ref

input: ref_geo_raw.parquet
purpose: override missing geo-IDs of CN and SD entities with
        in config.py hardcoded lvl3permids, restore 
output: ref_geography.parquet
'''
######################################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

# intitialize database connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# load data
ref_geo_raw= pd.read_parquet(con.REF_GEO_RAW)

print(ref_geo_raw.columns)
print(ref_geo_raw.shape)


