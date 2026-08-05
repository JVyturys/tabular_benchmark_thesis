##################################################
'''
src.exploration.split_unit.determine_per_region_coverage_growth

input: panel.parquet, ref_geo.parquet
purpose: determine the rating coverage development in total and 
        per region.
output: 
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)

# slice relevant vars
df_org = panel[['orgpermid', 'year']]
df_geo = ref_geo[['orgpermid', 'lvl3permid']]

# extract entry year per orgpermid
print(f'''Number of new entities per year (excluding 2009):''')
print(df_org.groupby('orgpermid').year.min().value_counts().sort_index())