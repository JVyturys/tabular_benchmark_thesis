##################################################
'''
src.exploration.raw_data.diagnose_cmpid_resolution

input:      raw_esg_scores.parquet, raw_ws_universe.parquet, raw SQL:
            tr_worldscope.wrds_ws_company, tr_worldscope.wrds_ws_funda,
            tr_common.permorgref
purpose:    determine_esg_coverage_asymmetry flags regions whose rated firm-years
            rarely resolve into the Worldscope annual universe (fy_resolution below
            COVERAGE_MIN_RESOLUTION; run 2026-09-26: 100219 at 0.14).
            diagnose_NorAm_attrition read the matching panel attrition as a
            Worldscope coverage gap, yet the universe holds 20043 firms in 100219.
            Separate the mechanisms per unresolved cmpid of a flagged region:
            N)  id not in Worldscope base format
            A)  id absent from wrds_ws_company and wrds_ws_funda (link / id mismatch)
            B)  company record, but no funda record at all
            C1) annual funda records, but none inside YEAR_MIN-YEAR_MAX
            C2) non-annual funda records only
            and contrast how often each region's Worldscope firms are linked
            to any permorgref entity at all
output:     console report
            run 2026-09-26 (flagged: 100219, fy_resolution 0.1398):
            - rated entities with cmpid 4851, unresolved cmpids 4192 (0.8642)
            - mechanism: A (absent from wrds_ws_company and wrds_ws_funda) 4192 / 4192;
              no N, B, C1, C2
            - unresolved ids are 9 characters in CUSIP-like format (e.g. 000360206,
              00081T108, 00090Q103); none share an id prefix with the region's
              Worldscope firms -> permorgref.worldscopecmpid holds a different
              identifier for most rated North American firms
            - Worldscope-side link rate: 100219 0.2737 vs. 0.67-0.92 in every
              other region (next lowest 100278 0.6733)
            -> the North American attrition is an identifier mismatch in the
               permorgref link, not a Worldscope coverage gap: 20043 NA firms exist
               in the Worldscope annual universe 2009-2025
'''
##################################################

import wrds
import numpy as np
import pandas as pd
import config as con
import log_config as lc

TOP_N = 10
BASE_CMPID = r'.{8}\d'      # same rule as build_ws_universe

print(f"\n[°°°] diagnosing cmpid resolution of rated firms [°°°]\n")

# load data
scores = pd.read_parquet(con.RAW_ESG_SCORES)
universe = pd.read_parquet(con.RAW_WS_UNIVERSE)
universe_ids = set(universe['worldscopecmpid'])


### flag regions (same definition as determine_esg_coverage_asymmetry) ---------------------------------------------------------
print(f"    [+++] flagging regions with fy_resolution < {con.COVERAGE_MIN_RESOLUTION}...")
key_u = pd.MultiIndex.from_frame(universe[['worldscopecmpid', 'year']])
fy = scores.loc[scores['worldscopecmpid'].notna(), ['worldscopecmpid', 'year', 'lvl3permid']]
fy = fy.assign(found=pd.MultiIndex.from_frame(fy[['worldscopecmpid', 'year']]).isin(key_u))
fy_resolution = fy.groupby('lvl3permid')['found'].mean()
flagged = fy_resolution[fy_resolution < con.COVERAGE_MIN_RESOLUTION]
print(f"      [---] flagged: {({int(k): round(v, 4) for k, v in flagged.items()}) if len(flagged) else 'none - nothing to diagnose'}")

db = wrds.Connection(wrds_username=lc.wrds_log)


### Worldscope-side link rate per region ---------------------------------------------------------
print(f"\n    [+++] share of Worldscope firms linked to any permorgref entity, per crosswalk region...")
linked_ids = set(db.raw_sql("""
    SELECT DISTINCT worldscopecmpid FROM tr_common.permorgref
    WHERE worldscopecmpid IS NOT NULL
""")['worldscopecmpid'].astype(str))
firms = universe.drop_duplicates('worldscopecmpid')
link_rate = (firms.assign(linked=firms['worldscopecmpid'].isin(linked_ids))
             .groupby('lvl3permid', dropna=False)['linked']
             .agg(firms='size', linked_share='mean')
             .sort_values('linked_share'))
print(link_rate.to_string(float_format='{:.4f}'.format))


### unresolved cmpids per flagged region ---------------------------------------------------------
for reg, res in flagged.items():
    print(f"\n    [+++] region {reg}: fy_resolution {res:.4f}")
    ent = scores.loc[scores['lvl3permid'].isin([reg]) & scores['worldscopecmpid'].notna()].drop_duplicates('orgpermid')
    cmpids = pd.Series(ent['worldscopecmpid'].unique()).astype(str)
    unresolved = cmpids[~cmpids.isin(universe_ids)].reset_index(drop=True)
    print(f"      [---] rated entities with cmpid {len(ent)}, distinct cmpids {len(cmpids)}, "
          f"unresolved {len(unresolved)} ({len(unresolved) / len(cmpids):.4f})")
    if unresolved.empty:
        continue

    assert unresolved.str.fullmatch(r'[A-Za-z0-9]+').all(), "non-alphanumeric cmpid - the IN list would need escaping"
    id_list = ','.join(f"'{c}'" for c in unresolved)

    in_window = db.raw_sql(f"""
        SELECT DISTINCT item6105 FROM tr_worldscope.wrds_ws_funda
        WHERE item6105 IN ({id_list})
            AND freq = 'A'
            AND year_ BETWEEN {con.YEAR_MIN} AND {con.YEAR_MAX}
    """)
    assert in_window.empty, \
        f"{len(in_window)} unresolved cmpids have annual records in the window - universe construction inconsistent: " \
        f"{in_window['item6105'].head(TOP_N).tolist()}"

    company_ids = set(db.raw_sql(f"""
        SELECT DISTINCT item6105 FROM tr_worldscope.wrds_ws_company
        WHERE item6105 IN ({id_list})
    """)['item6105'].astype(str))
    funda = db.raw_sql(f"""
        SELECT item6105, freq, MIN(year_) AS first_year, MAX(year_) AS last_year, COUNT(*) AS n_rows
        FROM tr_worldscope.wrds_ws_funda
        WHERE item6105 IN ({id_list})
        GROUP BY item6105, freq
    """)
    funda['item6105'] = funda['item6105'].astype(str)

    diag = pd.DataFrame({'cmpid': unresolved})
    diag['base'] = diag['cmpid'].str.fullmatch(BASE_CMPID)
    diag['in_company'] = diag['cmpid'].isin(company_ids)
    diag['in_funda'] = diag['cmpid'].isin(set(funda['item6105']))
    diag['annual_any'] = diag['cmpid'].isin(set(funda.loc[funda['freq'] == 'A', 'item6105']))

    diag['category'] = np.select(
        [~diag['base'],
         ~diag['in_company'] & ~diag['in_funda'],
         diag['in_company'] & ~diag['in_funda'],
         diag['annual_any'],
         diag['in_funda']],
        ['N: non-base id format',
         'A: absent from company and funda',
         'B: company record, no funda record',
         'C1: annual records outside the window only',
         'C2: non-annual funda records only'],
        default='unclassified')
    assert (diag['category'] != 'unclassified').all(), "category conditions are not exhaustive"
    assert diag['category'].value_counts().sum() == len(unresolved), "categories do not partition the unresolved cmpids"

    print(f"      [---] mechanism per unresolved cmpid:")
    print(diag['category'].value_counts().to_frame('cmpids')
          .assign(share=lambda d: d['cmpids'] / len(diag)).to_string(float_format='{:.4f}'.format))

    c1 = funda.loc[(funda['freq'] == 'A') & funda['item6105'].isin(diag.loc[diag['annual_any'], 'cmpid'])]
    if not c1.empty:
        print(f"      [---] C1: last annual year -> cmpids: {c1['last_year'].value_counts().sort_index().to_dict()}")
    if diag['in_funda'].any():
        print(f"      [---] funda freq among unresolved cmpids -> cmpids: "
              f"{funda.groupby('freq')['item6105'].nunique().to_dict()}")

    # Worldscope ids embed an issuer prefix; a different prefix mix hints at a different id scheme
    region_ids = firms.loc[firms['lvl3permid'].isin([reg]), 'worldscopecmpid'].astype(str)
    prefixes = pd.concat({'unresolved rated': unresolved.str[:4].value_counts(normalize=True),
                          'Worldscope firms': region_ids.str[:4].value_counts(normalize=True)}, axis=1)
    print(f"      [---] id prefix mix (first 4 characters, share):")
    print(prefixes.fillna(0).sort_values('unresolved rated', ascending=False)
          .head(TOP_N).to_string(float_format='{:.4f}'.format))
    print(f"      [---] sample of unresolved cmpids: {sorted(unresolved)[:TOP_N]}")

print(f"\n[°°°] diagnosis complete [°°°]")