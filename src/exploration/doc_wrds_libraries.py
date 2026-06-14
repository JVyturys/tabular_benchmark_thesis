import wrds 
import config as con 
import log_config as lg
from utils.reporting import write_log
 

import pandas as pd

log_dir = con.RESULTS_DIR / "exploration" / "raw_data"

db = wrds.Connection(wrds_username=lg.wrds_log)

entries = {} # for logging the results

try:
    print(f'[*] Starting library identification...\n')
    print(f'[**] Listing libraries...\n')
    wrds_libraries = db.list_libraries()
    #print(wrds_libraries)
    for i in range(len(wrds_libraries)):
        dummy_dict={i: wrds_libraries[i]}
        entries.update(dummy_dict)
    print(entries)
    #print(keylink_library.columns)

    write_log("WRDS Libraries", entries, log_dir, "wrds_libraries.txt", overwrite=True)        

    print(f"[*] Keylink Analysis finished ")
except Exception as e:
    print(f"Fehler bei Compustat: {e}")