##################################################
'''
src.exploration.final_panel.determine_region_exclusion_sensitivity

input:      results_*.yaml in PRED_DIR_MAN, FTT_VAL_TRIALS, ICL_SCORES (via assemble_results)
purpose:    recompute the reported aggregates without con.SENSITIVITY_EXCLUDED_REGION
            (100219: Canada-dominated through the CUSIP link artefact), from the
            per-region values already stored in the run manifests - no model is rerun.
            the global test variance is recovered per run as (ssr_r / n_r) / (1 - r_sq)
            and held fixed, so the excluded region leaves the Tier-1 set but not
            the denominator of the global scale (the region's own test mean is not
            stored, so a variance without it cannot be derived).
            reported: undepleted aggregates, DiD on r_sq (11 instead of 12
            comparators), headline changes deepest level vs. undepleted
output:     console report
            run 2026-09-26 on the manifests at GitHub HEAD da42004 (96 runs; all 7
            metrics and the production DiD reproduced):
            - undepleted, excluding 100219: regional bias gap +0.0026 to +0.0069 R2
              (e.g. xgb 0.0696 -> 0.0761); macro R2 -0.0066 to -0.0119 for trees and
              ICL (100219 is an above-average region for them, r_sq 0.35-0.43),
              +0.0007 for ftt (100219 r_sq 0.13); pooled R2 -0.0009 to -0.0057
            - rankings: gap order unchanged; RF baseline and TabPFN-3 swap in pooled
              and macro R2 (they differ by < 0.001 pooled with 100219 included)
            - DiD on r_sq: |change| <= 0.0084 (ftt, level 8000); <= 0.0017 for
              rf, xgb, tabicl; tabpfn3 -0.003 to -0.008; every DiD stays positive
            - gap removed at the deepest level (350): 72-81 % -> 71-79 % for rf, xgb,
              tabicl, tabpfn3; ftt reverses the gap in both cases (-127 % / -126 %)
            -> no qualitative conclusion depends on region 100219
'''
##################################################

import numpy as np
import pandas as pd
import config as con
import utils as ut
from src.production.from_final_panel import assemble_results as ar

EXCL = con.SENSITIVITY_EXCLUDED_REGION
TIER1_FULL = list(con.TIER1_REGS)
TIER1_EXCL = [r for r in TIER1_FULL if r != EXCL]
METRIC_NAMES = ['pooled_RMSE', 'pooled R2', 'average_RMSE', 'average R2 (global denominator)',
                'average_rmse_sq', 'regional bias gap', 'regional bias gap rmse']
REGION_COLS = ['rmse_r', 'ssr_r', 'n_r', 'denominator_r', 'r_sq', 'r_sq_reg']

pd.set_option('display.width', 200)
print(f"\n[°°°] sensitivity of the reported aggregates to excluding region {EXCL} [°°°]\n")

assert EXCL in TIER1_FULL and EXCL not in (con.ANCHOR_REGION, con.EQUAL_N_REGION), \
    f"{EXCL} must be a Tier-1 region other than the anchor and the equal-N region"

runs, regions = ar.load_manifests()
print(f"    [+++] manifests loaded - {len(runs)} runs, {len(regions)} region rows")


### recover the global test variance per run ---------------------------------------------------------
regions['sigma2'] = (regions['ssr_r'] / regions['n_r']) / (1 - regions['r_sq'])
spread = regions.groupby(ar.KEY)['sigma2'].agg(lambda s: s.max() - s.min())
assert np.allclose(spread, 0, atol=1e-15), f"global test variance differs within runs: max spread {spread.max()}"
sigma2 = regions.groupby(ar.KEY)['sigma2'].first()


### recompute run metrics through utils ---------------------------------------------------------
def recompute(tier1: list) -> pd.DataFrame:
    """Run-level metrics over the given Tier-1 set, via ut.macro_average_metrics / ut.report_metrics."""
    rows = []
    for key, frame in regions.groupby(ar.KEY):
        per_region = (frame.set_index('region')[REGION_COLS], None)   # report_metrics reads [0] only
        macro = ut.macro_average_metrics(per_region, tier1)
        _, *values = ut.report_metrics(per_region, (None, sigma2.loc[key]), macro, tier1)
        pooled_r2, pooled_rmse, avg_rmse, avg_r2, avg_rmse_sq, gap, gap_rmse = values
        rows.append({**dict(zip(ar.KEY, key)),
                     'pooled_RMSE': pooled_rmse, 'pooled R2': pooled_r2, 'average_RMSE': avg_rmse,
                     'average R2 (global denominator)': avg_r2, 'average_rmse_sq': avg_rmse_sq,
                     'regional bias gap': gap, 'regional bias gap rmse': gap_rmse})
    return pd.DataFrame(rows)

print(f"    [+++] recomputing run metrics with all Tier-1 regions (reproduction check)...")
full = recompute(TIER1_FULL).merge(runs[ar.KEY + METRIC_NAMES], on=ar.KEY, suffixes=('', '_manifest'),
                                   validate='one_to_one')
assert len(full) == len(runs), "recomputed runs do not match the manifests one-to-one"
for m in METRIC_NAMES:
    assert np.allclose(full[m], full[f"{m}_manifest"], rtol=con.INVARIANT_RTOL, atol=0), \
        f"recomputation does not reproduce '{m}'"
full = full[ar.KEY + METRIC_NAMES]
print(f"      [---] all {len(METRIC_NAMES)} metrics reproduced for {len(full)} runs")

print(f"    [+++] recomputing run metrics without {EXCL}...")
excl = recompute(TIER1_EXCL)

# depletion level and draw from the condition tag (format of run_ICL_depletion_grid._condition_tag)
for frame in (full, excl):
    tags = frame['condition'].str.extract(r'^depl_L(\d+)_d(\d+)$')
    frame['level'] = pd.to_numeric(tags[0]).astype('Int64')
    frame['draw'] = pd.to_numeric(tags[1]).astype('Int64')
    assert (frame['condition'].eq('undepl') | frame['level'].notna()).all(), "unparsed condition tag"


### 1. undepleted aggregates ---------------------------------------------------------
print(f"\n    [+++] undepleted aggregates, all Tier-1 vs. without {EXCL}:")
show = ['pooled R2', 'average R2 (global denominator)', 'regional bias gap', 'regional bias gap rmse']
u_full = full.loc[full['condition'].eq('undepl')].set_index(ar.CURVE_KEYS)[show]
u_excl = excl.loc[excl['condition'].eq('undepl')].set_index(ar.CURVE_KEYS)[show]
agg = pd.concat({'all': u_full, f'excl {EXCL}': u_excl, 'delta': u_excl - u_full}, axis=1)
print(agg.to_string(float_format='{:+.4f}'.format))

r_na = regions.loc[regions['condition'].eq('undepl') & regions['region'].eq(EXCL)].set_index(ar.CURVE_KEYS)
print(f"\n      [---] region {EXCL} itself, undepleted (r_sq on the global scale, test rows):")
print(r_na[['r_sq', 'rmse_r', 'n_r']].to_string(float_format='{:.4f}'.format))


### 2. DiD on r_sq: 12 vs. 11 comparators ---------------------------------------------------------
print(f"\n    [+++] DiD on r_sq, level means per model...")
comp_full = [r for r in TIER1_FULL if r != con.ANCHOR_REGION]
comp_excl = [r for r in comp_full if r != EXCL]

base = regions.loc[regions['condition'].eq('undepl'), ['model', 'configuration', 'region', 'r_sq']]
depl = regions.loc[regions['condition'].ne('undepl'), ar.KEY + ['region', 'r_sq']]
delta = depl.merge(base, on=['model', 'configuration', 'region'], suffixes=('', '_base'), validate='many_to_one')
delta['d_r_sq'] = delta['r_sq_base'] - delta['r_sq']
wide = delta.pivot(index=ar.KEY, columns='region', values='d_r_sq')

did = pd.DataFrame({'did_all': wide[con.ANCHOR_REGION] - wide[comp_full].mean(axis=1),
                    f'did_excl_{EXCL}': wide[con.ANCHOR_REGION] - wide[comp_excl].mean(axis=1)}).reset_index()
prod = ar.did_table(regions)[ar.KEY + ['did']]
check = did.merge(prod, on=ar.KEY, validate='one_to_one')
assert len(check) == len(did) and np.allclose(check['did_all'], check['did'], rtol=con.INVARIANT_RTOL, atol=0), \
    "DiD with all comparators does not reproduce assemble_results.did_table"

did['level'] = pd.to_numeric(did['condition'].str.extract(r'^depl_L(\d+)_d')[0])
did_sum = did.groupby(ar.CURVE_KEYS + ['level'])[['did_all', f'did_excl_{EXCL}']].mean()
did_sum['delta'] = did_sum[f'did_excl_{EXCL}'] - did_sum['did_all']
print(did_sum.to_string(float_format='{:+.4f}'.format))


### 3. headline changes, deepest level vs. undepleted ---------------------------------------------------------
print(f"\n    [+++] headline changes (relative), deepest level vs. undepleted:")
h_full = ar.headline_changes(full).set_index(ar.CURVE_KEYS)
h_excl = ar.headline_changes(excl).set_index(ar.CURVE_KEYS)
cols = ['pooled R2', 'average R2 (global denominator)', 'regional bias gap']
head = pd.concat({'all': h_full[cols], f'excl {EXCL}': h_excl[cols]}, axis=1)
print(f"      [---] deepest level {int(h_full['level'].iloc[0])}, draws {int(h_full['n_draws'].iloc[0])}")
print(head.to_string(float_format='{:+.4f}'.format))

print(f"\n[°°°] sensitivity complete [°°°]")