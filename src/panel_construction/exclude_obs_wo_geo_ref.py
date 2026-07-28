##################################################
'''
V.

While constructing table for the geographic referencing of the entities , it turned out that 6 observations do not have 
geographic information. Since my main analysis perspective looks at model performance with respect to geographic subgroups of the data
I do not want to include observations, of which I can not evaluate the respective geo performance.

This script excludes observations from my final panel with no matching entry in the 'geo_ref_table_clean.csv'

'''
##################################################

import pandas as pd
import config as con
from collections import Counter

# define paths  
data_path = con.PANEL
results_path = con.RESULTS_DIR

# load data
df = pd.read_parquet(data_path)
geo_ref = pd.read_csv(con.REF_GEOGRAPHY)

# filter only referenceable entities
included_entities =  geo_ref['orgpermid'].to_list()
indicies = []
for entity in df['orgpermid']:
    idx = entity in included_entities
    indicies.append(idx)
#print(indicies)

frequency = Counter(indicies)
df_clean = df[indicies]
assert df_clean['orgpermid'].nunique() == geo_ref['orgpermid'].nunique(), \
    "Entity mismatch between panel and geography reference"
print("Assertion passed — panel and geography table describe identical entity universe")

df_clean.to_parquet("panel_clean", index=False)