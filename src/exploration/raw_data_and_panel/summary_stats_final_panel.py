#########################################################
# Explorative statistics of the final data set.
#########################################################

import pandas as pd
import config as con
from utils.reporting import AnalysisLogger as Al

# Setting pandas setting to display two decimals
pd.set_option('display.float_format', lambda x: f'{x:.2f}')


# Defining paths and initializing AnalysisLogger instance for documentation
data_path = con.FINAL_PANEL_PARQUET
results_path = con.RESULTS_DIR / "joined_panel"
# panel_doc = Al(results_path, "summary_stats_panel.txt")

# Loading and formating 'year'&'orgpermid' to string to exclude them from .describe() output
panel = pd.read_parquet(data_path)
panel["orgpermid"] = panel["orgpermid"].dropna().astype(str)

min_year = panel["year"].min().astype(int) # Extracting min and max values from time range
max_year = panel["year"].max().astype(int)

panel["year"] = panel["year"].astype(str)
try:
    # Summary stats
    panel_shape = panel.shape
    panel_columns = panel.columns
    panel_summary = panel.describe()

    ## Finding out the number of unique entities
    panel_orgs = panel["orgpermid"].unique()
    panel_unq_orgs = pd.DataFrame(pd.unique(panel_orgs)) #'pd.unique()' method returns a numpy array -> value_counts() only works on pd-dataframes
    count_unique_ents = panel_unq_orgs.value_counts().sum()
    
except Exception as error:
    print(f"An error occured while calculating summary stats!\n Error: {error}")

# try:
#     panel_doc.section("Summary Statistics of the Final Data Set")
#     panel_doc.log("Shape of the panel:\n")
#     panel_doc.log(panel_shape)
#     panel_doc.log("\n Number of unique entities:")
#     panel_doc.log(count_unique_ents)
#     panel_doc.log("\nRange of years:") 
#     panel_doc.log(min_year)
#     panel_doc.log("-")
#     panel_doc.log(max_year)
#     panel_doc.log("\nPanel columns:\n")
#     panel_doc.log(panel_columns)
#     panel_doc.log("\nSummary table:\n")
#     panel_doc.log(panel_summary)
    

# except:
#     print(f"An error occured during the logging process!")


