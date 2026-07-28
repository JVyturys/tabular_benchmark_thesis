import wrds 
import config as con 
import log_config as lg
from utils.reporting import write_log
import pandas as pd

# define path and initialize database connection
log_dir = con.RESULTS_DIR / "exploration" / "raw_data"
db = wrds.Connection(wrds_username=lg.wrds_log)

try:
    entries = {}
    wrds_libraries = db.list_libraries()
    for i in range(len(wrds_libraries)):
        dummy_dict={i: wrds_libraries[i]}
        entries.update(dummy_dict)

    # document library structure
    write_log("WRDS Libraries", entries, log_dir, "wrds_libraries.txt", overwrite=True)
except Exception as e:
    print(f"Fehler bei Compustat: {e}")