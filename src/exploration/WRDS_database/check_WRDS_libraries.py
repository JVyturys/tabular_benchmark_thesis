##################################################
'''
src.exploration.WRDS_database.check_WRDS_libraries

input: raw SQL
purpose: explore WRDS library structure
output: insight over WRDS library structure
'''
##################################################

import wrds 
import log_config as lg
from utils.reporting import write_log

# define path and initialize database connection
db = wrds.Connection(wrds_username=lg.wrds_log)

try:
    entries = {}
    wrds_libraries = db.list_libraries()
    for i in range(len(wrds_libraries)):
        dummy_dict={i: wrds_libraries[i]}
        entries.update(dummy_dict)

    # print library structure
    print(entries)
except Exception as e:
    print(f"Fehler bei Compustat: {e}")