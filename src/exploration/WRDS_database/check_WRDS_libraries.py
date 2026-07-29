##################################################
'''
src.exploration.WRDS_database.check_WRDS_libraries

input: raw SQL
purpose: explore WRDS library structure
output: overview WRDS libraries
'''
##################################################

import wrds 
import log_config as lg

# initialize database connection
db = wrds.Connection(wrds_username=lg.wrds_log)

try:
    wrds_libraries = db.list_libraries()
    # print library structure
    print(wrds_libraries)
except Exception as e:
    print(f"Fehler bei Compustat: {e}")