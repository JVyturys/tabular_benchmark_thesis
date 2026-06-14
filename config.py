import os
from pathlib import Path

# DATA PATHS 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_COMP = PROJECT_ROOT / "data" / "00_raw" / "raw_merged_full.csv"
DATA_RAW_ESG = PROJECT_ROOT / "data" / "00_raw" / "thesis_Refinitiv_scores.csv"

# RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"


# OUTPUT NAMING CONVENTION
LOG     = "log.txt"
ARTIFACT_METRICS = "metrics.csv"

