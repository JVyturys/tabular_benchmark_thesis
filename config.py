import os
from pathlib import Path

# DATA PATHS 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RAW_PANEL = DATA_ROOT / "raw" / "panel_raw.parquet"


FINAL_DIR = PROJECT_ROOT / "data" / "final"
PANEL = FINAL_DIR / "panel.parquet"
REF_CLUSTER_KEYS = FINAL_DIR / "ref_cluster_keys.parquet"
REF_GEOGRAPHY = FINAL_DIR / "ref_geo_table.parquet"
REF_PARENT =  FINAL_DIR / "ref_parent_table.parquet"
REF_PARENT_ENT_TYPE = FINAL_DIR / "ref_parent_ent_type.parquet"


# RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"

# BUILD_CLUSTER_KEY CONSTANTS - build_cluster_keys.py
ULTIMATE_KEY_TYPECODES = ['COM', 'UNK', 'NGO', 'CLGUN']
IMMEDIATE_KEY_TYPECODES = ['GVT', 'GVTDA', 'CINV']
KEY_SOURCE = {'singleton': 0, 'ultimate': 1, 'immediate': 2}

# GEOGRAPHY OVERRIDES FOR SUDANESE AND CHINEESE ENTITIES
DOMICILE_OVERRIDES = {105758: (100089, 'CN'), 110515: (100218, 'SD')}