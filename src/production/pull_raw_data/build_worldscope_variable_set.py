##################################################
'''
src.production.pull_raw_data.build_worldscope_variable_set

input: raw SQL
purpose: build set of variables to be included in raw panel
output: ref_ws_variables.parquet
'''
##################################################

import wrds
import config as con
import log_config as lg
from utils.reporting import AnalysisLogger as Al

# define path
log_dir = con.RESULTS_DIR / "exploration" / "raw_data"

# initialize database connection
db = wrds.Connection(wrds_username=lg.wrds_log)

# Extract variable names from tr_worldscope
## Extract all coded columns
query_1 = """
    SELECT *
    FROM tr_worldscope.wrds_ws_funda
    LIMIT 2
"""

df_ws = db.raw_sql(query_1)

# exclude columns with non-financial meta information 
exclude_columns = ["code", "year_", "freq", "seq"]
fin_cols = [col for col in df_ws.columns if col not in exclude_columns]

## Remove "item[...]" string from item-coding
item_numbers = [int(code.replace('item', '')) for code in fin_cols if code.startswith('item')]
item_list = ",".join(str(x) for x in item_numbers)

## Filter the variable list for 'all-industry' code excluding rolling averages and growth rates.
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
db.raw_sql(query_4).to_parquet(con.REF_WSVAR, index=False)


