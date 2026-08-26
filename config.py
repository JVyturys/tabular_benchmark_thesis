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

## FINAL PANEL
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PANEL = FINAL_DIR / "panel.parquet"

## FINAL PANEL ARTIFACTS
REF_CLUSTER_KEYS = FINAL_DIR / "ref_cluster_keys.parquet"
REF_GEOGRAPHY = FINAL_DIR / "ref_geo_table.parquet"
REF_PARENT =  FINAL_DIR / "ref_parent_table.parquet"
REF_PARENT_ENT_TYPE = FINAL_DIR / "ref_parent_ent_type.parquet"

## RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ = RESULTS_DIR / "visualization"
VIZ_WD = VIZ / "wassersteindist.png"
VIZ_TDIST = VIZ / "target_distr_preg.png" 
VIZ_CUTOFF = VIZ / "cutoff_kneedle.png"
VIZ_NAN_SHARE = VIZ / "total_nan_shares_per_region.png"
VIZ_USABLE_DROPPED = VIZ / "dropped_usable_features.png"

## SPLIT PARTITIONS
SPLIT = FINAL_DIR / "split.parquet"

## PREPROCESSING
PRE_PROS_CONTS = FINAL_DIR / "pre_processing_constants.parquet"

# --------------------------------------------------------------------------

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
DIST_TOL = 0

# VARIABLES DROPPED DURING PREPROCESSING
CUTOFF_VARS = ['item4057', 'item3499', 'item18188', 'item4150', 'item1084', 'item1254',
          'item1269', 'item7011', 'item18183', 'item8256', 'item18184', 'item8351', 'item8406', 'item1155',
            'item1801', 'item4148', 'item1267', 'item1266', 'item8346', 'item8906', 'item1149', 'item4053',
              'item4840', 'item1306', 'item3260', 'item1253', 'item18280', 'item2513', 'item1352', 'item1503',
                'item4149', 'item4651', 'item3261', 'item18324', 'item2514', 'item2515', 'item18140',
                  'item2654', 'item2655', 'item18274', 'item18299', 'item4056', 'item2516', 'item2517',
                    'item4821', 'item18352', 'item4058', 'item3493', 'item2502', 'item2503', 'item1157',
                      'item4052', 'item1301', 'item2510', 'item18571', 'item2511', 'item2507', 'item2512',
                        'item18852', 'item2509', 'item1268', 'item2508', 'item1204', 'item18286', 'item18574',
                          'item4055', 'item3491', 'item1154', 'item18293', 'item18408', 'item18187', 'item1152',
                            'item18275', 'item18224', 'item18854', 'item1302', 'item18226', 'item2504', 'item2505',
                              'item2653', 'item18851', 'item18225', 'item2506', 'item18189', 'item18185', 'item18215',
                                'item18353', 'item1153', 'item18175', 'item18572', 'item18165', 'item4892',
                                  'item4891', 'item18575', 'item1156', 'item3494', 'item18159', 'item18173',
                                    'item18172', 'item18170', 'item18171', 'item18166', 'item3450', 'item4054',
                                      'item18168', 'item18167', 'item18853', 'item1351', 'item18065', 'item3490',
                                        'item1265', 'item3449']

DEGVAR_VARS = ['item4450', 'item3448', 'item4452' , 'item4799', 'item3257']



