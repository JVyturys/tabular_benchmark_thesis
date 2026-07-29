##################################################
'''
src.exploration.WRDS_database.check_domcntryID_consistency

input: raw SQL
purpose: Check the way domcntryperm id is stored and 
        whether it has a time axis within the wrds database.
output: Domcntry ID is assigned once per orgpermid. 
        Historical id changes of the domcntryid are not stored.
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

# initialize data base connection
db = wrds.Connection(wrds_username = lc.wrds_log)

# find out whether the regional IDs are stored per entitiy or per entity-year
## look at variables stored in 'permorgref' 
print(db.describe_table(library='tr_common', table='permorgref'))

## define SQL query to inspect domcntrypermid
query_1 = f'''
    SELECT orgpermid, domcntrypermid
    FROM tr_common.permorgref
'''

try:
    org_and_cntry_id = db.raw_sql(query_1)
    print(org_and_cntry_id)
except Exception as error:
    print(f'An error occured during query pull: Error:{error}')
