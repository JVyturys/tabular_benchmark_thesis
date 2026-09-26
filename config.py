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

## DEPLETION DRAWS
DEPLETION_DRAWS = FINAL_DIR / "depletion_draws.parquet"
DEPLETION_COUNTS = FINAL_DIR / "depletion_counts.parquet"
CONTEXT_ROWS = 58806
TRIPWIRE_MEAN_SE_FACTOR = 4
TRIPWIRE_STD_RATIO = (0.5, 2.0)

## DEPLETION GRID RUN
DEPLETION_BUDGET_H = 40
GRID_MAX_CONSECUTIVE_FAILURES = 2

## ICL ENVIRONMENT EQUIVALENCE CHECK
ICL_EQUIV_REPLICATES = 2
ICL_EQUIV_NOISE_FACTOR = 3.0
ICL_EQUIV_ABS_TOL = 1e-4
ICL_ENV_CHECK_DIR = PROJECT_ROOT / "archive" / "icl_env_check"

## RESULTS
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions" 
PRED_DIR_MAN = PROJECT_ROOT / "model_scores" / "rf_xgb_scores"
FTT_VAL_TRIALS = PROJECT_ROOT / "model_scores" /"ftt_scores"
CTX_CAP_CHECK = RESULTS_DIR / "exploration" / "hardware"
ICL_SCORES = PROJECT_ROOT / "model_scores" /"icl_scores"
VIZ = RESULTS_DIR / "visualization"
VIZ_WD = VIZ / "wassersteindist.png"
VIZ_TDIST = VIZ / "target_distr_preg.png" 
VIZ_CUTOFF = VIZ / "cutoff_kneedle.png"
VIZ_NAN_SHARE = VIZ / "total_nan_shares_per_region.png"
VIZ_USABLE_DROPPED = VIZ / "dropped_usable_features.png"
VIZ_RMSE = VIZ / "ft_epoch_rmse_curve.png"

## ASSEMBLED RESULTS - assemble_results.py
RESULTS_TABLES = RESULTS_DIR / "tables"
TAB_AGGREGATE = RESULTS_TABLES / "aggregate_undepl.csv"
TAB_PER_REGION = RESULTS_TABLES / "per_region_undepl.csv"
TAB_DID = RESULTS_TABLES / "did_r_sq.csv"
TAB_GAP_CURVE = RESULTS_TABLES / "gap_curve.csv"
TAB_HEADLINE = RESULTS_TABLES / "headline_changes.csv"
VIZ_GAP_CURVE = VIZ / "gap_curve_r_sq.png"
## DESCRIPTIVE STATISTICS - src/production/descriptives/build_desc_*.py
DESC_DIR = RESULTS_DIR / "descriptives"
VIZ_DESC_DIR = VIZ / "descriptives"
### raw
TAB_DESC_ATTRITION = DESC_DIR / "desc_attrition.csv"
### panel
TAB_DESC_REGION_BALANCE = DESC_DIR / "desc_region_balance.csv"
TAB_DESC_YEAR_COVERAGE = DESC_DIR / "desc_year_coverage.csv"
TAB_DESC_OBS_PER_ENTITY = DESC_DIR / "desc_obs_per_entity.csv"
TAB_DESC_TARGET_REGION = DESC_DIR / "desc_target_by_region.csv"
TAB_DESC_TARGET_YEAR = DESC_DIR / "desc_target_by_year.csv"
TAB_DESC_FEATURE_NAN = DESC_DIR / "desc_feature_missingness.csv"
VIZ_DESC_COVERAGE = VIZ_DESC_DIR / "coverage_region_year.png"
VIZ_DESC_OBS_PER_ENTITY = VIZ_DESC_DIR / "obs_per_entity.png"
### clusters & split
TAB_DESC_CLUSTER_SIZES = DESC_DIR / "desc_cluster_sizes.csv"
TAB_DESC_CLUSTER_SOURCE = DESC_DIR / "desc_cluster_key_source.csv"
TAB_DESC_PARTITIONS = DESC_DIR / "desc_partition_by_region.csv"
TAB_DESC_TARGET_PARTITION = DESC_DIR / "desc_target_by_partition.csv"
VIZ_DESC_PARTITIONS = VIZ_DESC_DIR / "partition_shares.png"
### features
TAB_DESC_FEATURE_FUNNEL = DESC_DIR / "desc_feature_funnel.csv"
TAB_DESC_FEATURE_CONSTANTS = DESC_DIR / "desc_feature_constants.csv"
TAB_DESC_IMPUTATION = DESC_DIR / "desc_imputation_by_region.csv"
VIZ_DESC_IMPUTATION = VIZ_DESC_DIR / "imputation_by_region.png"
### depletion
TAB_DESC_DEPL_CONDITIONS = DESC_DIR / "desc_depletion_conditions.csv"
TAB_DESC_DEPL_COMPOSITION = DESC_DIR / "desc_depletion_composition.csv"
TAB_DESC_DEPL_LEVELS = DESC_DIR / "desc_depletion_levels.csv"
VIZ_DESC_POOL_SHARE = VIZ_DESC_DIR / "depletion_pool_share.png"
VIZ_DESC_DEPL_TARGET = VIZ_DESC_DIR / "depletion_target.png"
### predictions
TAB_DESC_PRED_REGION = DESC_DIR / "desc_prediction_by_region.csv"
TAB_DESC_PRED_LEVELS = DESC_DIR / "desc_prediction_levels.csv"
VIZ_DESC_RESID_BIAS = VIZ_DESC_DIR / "residual_bias_by_region.png"
VIZ_DESC_DISPERSION = VIZ_DESC_DIR / "prediction_dispersion.png"
### performance metrics (manifests)
TAB_DESC_METRIC_LEVELS = DESC_DIR / "desc_metric_levels.csv"
TAB_DESC_METRIC_REGION_LEVELS = DESC_DIR / "desc_metric_region_levels.csv"
TAB_DESC_METRIC_RANKS = DESC_DIR / "desc_metric_ranks.csv"
TAB_DESC_METRIC_REGION_SIZE = DESC_DIR / "desc_metric_region_size.csv"
VIZ_DESC_METRIC_CURVES = VIZ_DESC_DIR / "metric_curves.png"
VIZ_DESC_REGION_DELTA = VIZ_DESC_DIR / "region_delta_r_sq_deepest.png"
VIZ_DESC_REGION_SIZE = VIZ_DESC_DIR / "region_r_sq_vs_size.png"
### assembled results (assemble_results tables)
VIZ_DESC_POOLED_MACRO = VIZ_DESC_DIR / "pooled_vs_macro_r_sq.png"
VIZ_DESC_REGION_HEATMAP = VIZ_DESC_DIR / "region_r_sq_heatmap.png"
VIZ_DESC_GAP_SHARE = VIZ_DESC_DIR / "gap_share_removed.png"
VIZ_DESC_DID = VIZ_DESC_DIR / "did_decomposition.png"
VIZ_DESC_HEADLINE = VIZ_DESC_DIR / "headline_changes.png"
### depletion design schematic
TAB_DESC_DEPL_DESIGN = DESC_DIR / "desc_depletion_design_facts.csv"
VIZ_DESC_DEPL_DESIGN = VIZ_DESC_DIR / "depletion_design_schematic.png"
### depletion strata
TAB_DESC_STRATA = DESC_DIR / "desc_depletion_strata.csv"
VIZ_DESC_STRATA = VIZ_DESC_DIR / "depletion_strata.png"
VIZ_DESC_DRAW_DISTR = VIZ_DESC_DIR / "depletion_draw_distributions.png"

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
ANCHOR_REGION = 100089      # largest Tier-1 region by rows; depletion anchor and Wasserstein reference
EQUAL_N_REGION = 100218     # smallest Tier-1 region; deepest depletion level approximates its train+val N

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

# RESULTS ASSEMBLY 
INVARIANT_RTOL = 1e-9