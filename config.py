import os
from pathlib import Path

# DATA PATHS 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FINAL_PANEL_CSV = PROJECT_ROOT / "data" / "raw" / "panel_final.csv"
FINAL_PANEL_PARQUET = PROJECT_ROOT / "data" / "raw" / "panel_clean.parquet"  
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"


# OUTPUT NAMING CONVENTION
LOG     = "log.txt"
ARTIFACT_METRICS = "metrics.csv"

