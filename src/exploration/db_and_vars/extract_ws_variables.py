
# Commented-out code left for documentation
# For final set of variables run query1 & query4  

import wrds
import config as con
import log_config as lg
from utils.reporting import AnalysisLogger as Al

log_dir = con.RESULTS_DIR / "exploration" / "raw_data"
# log_var = Al(output_path=log_dir, filename= "ws_variables_i.txt")
#log_ws_industry = Al(output_path=log_dir, filename="ws_industry_codes.txt")

db = wrds.Connection(wrds_username=lg.wrds_log)

# Extracting variable names from tr_worldscope

## 1. Extracting all coded columns
query_1 = """
    SELECT *
    FROM tr_worldscope.wrds_ws_funda
    LIMIT 2
"""
df_ws = db.raw_sql(query_1)
cols = df_ws.columns
fin_cols = cols[4:]

## 2. Remove "item[...]" string from item-coding
item_numbers = [int(code.replace('item', '')) for code in fin_cols if code.startswith('item')]
item_list = ",".join(str(x) for x in item_numbers)
# print(item_list)

# ## Querying tr_worldscope.wscode with item-code filter
# query_2 =  f"""
#     SELECT number, name, datatype, frequency, industry
#     FROM tr_worldscope.wsitem
#     WHERE number IN ({item_list}) AND datatype = 'N'
#     ORDER BY number
# """
# output = db.raw_sql(query_2).to_csv('ws_variables_ii.txt', index=False)


## 3. Demystifying industry codes
### 3.1 Frequency of each industry code within set
# query_3 = f"""
#     SELECT industry, COUNT(*) as n
#     FROM tr_worldscope.wsitem
#     WHERE number IN ({item_list}) AND datatype = 'N'
#     GROUP BY industry
#     ORDER BY n DESC
# """
# output = db.raw_sql(query_3)


### 3.2 Filtering the variable list for 'all-industry' code excluding rolling averages and growth rates.
query_4 = f"""
    SELECT number, name, industry
    FROM tr_worldscope.wsitem
    WHERE number IN ({item_list})
        AND datatype = 'N'
        AND industry IN ('111111')
        AND name NOT LIKE '%%AVG%%'
        AND name NOT LIKE '%% YR %%'
        AND name NOT LIKE '%%YEAR%%'
        AND name NOT LIKE '%%QUART%%'
        AND name NOT LIKE '%%LENGHT OF FISC%%'
        AND name NOT LIKE '%%EXCHANGE RATE USED FOR TRANS%%'
    ORDER BY industry
"""
output = db.raw_sql(query_4).to_csv('ws_variables_final.csv', index=False)


