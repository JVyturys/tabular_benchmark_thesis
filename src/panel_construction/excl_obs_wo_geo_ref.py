##################################################
'''
While constructing table for the geographic referencing of the entities , it turned out that 6 observations do not have 
geographic information. Since my main analysis perspective looks at model performance with respect to geographic subgroups of the data
I do not want to include observations, of which I can not evaluate the respective geo performance.

This script excludes observations from my final panel with no matching entry in the 'geo_ref_table_clean.csv'

'''
##################################################

import pandas as pd
import config as con
from collections import Counter
from utils.reporting import AnalysisLogger as Al

data_path = con.FINAL_PANEL_PARQUET
results_path = con.RESULTS_DIR

df = pd.read_parquet(data_path)
geo_ref = pd.read_csv(con.RAW_DATA_ROOT/"geo_ref_table_clean.csv")


# filtering only referenceable entities
included_entities =  geo_ref['orgpermid'].to_list()
indicies = []
for entity in df['orgpermid']:
    idx = entity in included_entities
    indicies.append(idx)
#print(indicies)

frequency = Counter(indicies)
df_clean = df[indicies]
print(f"Unique entities clean data: {df_clean['orgpermid'].nunique()}")

assert df_clean['orgpermid'].nunique() == geo_ref['orgpermid'].nunique(), \
    "Entity mismatch between panel and geography reference"
print("Assertion passed — panel and geography table describe identical entity universe")

df_clean.to_parquet("panel_clean", index=False)


#############################
# logging results -----------
#############################

logging = Al(con.RESULTS_DIR / "joined_panel", filename="cleaning_panel_geo_ref")
logging.section("Removing entities without geographic reference.")
logging.log("Number of entities WITH geographic reference:")
logging.log(len(geo_ref))

logging.log("\nNumber of unique entities in raw data panel:")
logging.log(df["orgpermid"].nunique())

logging.log("\nNumber of entities in panel after cleaning:")
logging.log(df_clean['orgpermid'].nunique())
