import os
from pathlib import Path

# PATHS 
PROJECT_ROOT = Path(__file__).resolve().parent

## DATA
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"

RAW_PANEL = DATA_RAW / "raw_panel.parquet"
REF_WSVAR = DATA_RAW / "ref_ws_variables.parquet"
REF_GEO_RAW = DATA_RAW / "ref_geo_raw.parquet"

# FINAL PANEL
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PANEL = FINAL_DIR / "panel.parquet"

## FINAL PANEL ARTIFACTS
REF_CLUSTER_KEYS = FINAL_DIR / "ref_cluster_keys.parquet"
REF_GEOGRAPHY = FINAL_DIR / "ref_geo_table.parquet"
REF_PARENT =  FINAL_DIR / "ref_parent_table.parquet"
REF_PARENT_ENT_TYPE = FINAL_DIR / "ref_parent_ent_type.parquet"

# RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ = RESULTS_DIR / "visualization"
VIZ_WD = VIZ / "wassersteindist.png"
VIZ_TDIST = VIZ / "target_distr_preg.png" 

# BUILD_CLUSTER_KEY CONSTANTS - build_cluster_keys.py
ULTIMATE_KEY_TYPECODES = ['COM', 'UNK', 'NGO', 'CLGUN']
IMMEDIATE_KEY_TYPECODES = ['GVT', 'GVTDA', 'CINV']
KEY_SOURCE = {'singleton': 0, 'ultimate': 1, 'immediate': 2}

# GEOGRAPHY OVERRIDES FOR SUDANESE AND CHINEESE ENTITIES
DOMICILE_OVERRIDES = {105758: (100089, 'CN'), 110515: (100218, 'SD')}

# TIER ONE SPLIT REGIONS
TIER1_REGS = [100277, 100090, 100334, 103384,
              100024, 100279, 100089, 100219,
              100223, 103401, 100276, 100218, 100278]
TIER2_REGS = [100060, 100087, 100332]
TIER3_REGS = [110000]

# SPLIT PARAMETERS
TRAIN_SHARE = 0.7
TEST_SHARE = 0.3
FIT_SHARE = (6/7)
VAL_SHARE = (1/7)
SEED = 17
