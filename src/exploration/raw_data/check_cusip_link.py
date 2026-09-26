##################################################
'''
src.exploration.raw_data.check_cusip_link

input:      raw_esg_scores.parquet, raw_ws_universe.parquet, raw SQL:
            tr_worldscope.wsitem, wrds_ws_company, wrds_ws_ids
purpose:    diagnose_cmpid_resolution found that the unresolved cmpids of flagged
            regions (100219) exist nowhere in Worldscope and look like CUSIPs.
            test whether they are, and whether they recover the firms:
            i)   CUSIP check-digit validity, with resolved Worldscope ids as control
                 (a random 9-character string passes with probability 0.1)
            ii)  locate Worldscope's CUSIP item(s) in wrds_ws_company / wrds_ws_ids
            iii) match unresolved ids on CUSIP-9, else CUSIP-8, map hits to base
                 Worldscope ids, and count the firms and rated firm-years that
                 would enter the annual universe YEAR_MIN-YEAR_MAX
output:     console report - record findings here after the run
'''
##################################################

import wrds
import pandas as pd
import config as con
import log_config as lc

TOP_N = 10
BASE_CMPID = r'.{8}\d'      # same rule as build_ws_universe
WS_TABLES = ['wrds_ws_company', 'wrds_ws_ids']
CUSIP_VALUES = {**{str(d): d for d in range(10)},
                **{chr(ord('A') + i): 10 + i for i in range(26)},
                '*': 36, '@': 37, '#': 38}


def cusip_check_ok(code: str) -> bool:
    """Standard CUSIP mod-10 check: double every second of the first eight values, add the digits."""
    if len(code) != 9 or any(c not in CUSIP_VALUES for c in code[:8]) or not code[8].isdigit():
        return False
    total = 0
    for i, c in enumerate(code[:8]):
        v = CUSIP_VALUES[c] * (2 if i % 2 else 1)
        total += v // 10 + v % 10
    return (10 - total % 10) % 10 == int(code[8])


print(f"\n[°°°] checking the CUSIP hypothesis for unresolved cmpids [°°°]\n")

# load data
scores = pd.read_parquet(con.RAW_ESG_SCORES)
universe = pd.read_parquet(con.RAW_WS_UNIVERSE)
universe_ids = set(universe['worldscopecmpid'])
key_u = pd.MultiIndex.from_frame(universe[['worldscopecmpid', 'year']])


### flagged regions and unresolved ids (same definition as determine_esg_coverage_asymmetry) ----------------
print(f"    [+++] collecting unresolved cmpids of regions with fy_resolution < {con.COVERAGE_MIN_RESOLUTION}...")
fy = scores.loc[scores['worldscopecmpid'].notna(), ['orgpermid', 'worldscopecmpid', 'year', 'lvl3permid']]
fy = fy.assign(found=pd.MultiIndex.from_frame(fy[['worldscopecmpid', 'year']]).isin(key_u))
fy_resolution = fy.groupby('lvl3permid')['found'].mean()
flagged = fy_resolution[fy_resolution < con.COVERAGE_MIN_RESOLUTION]

fy_flag = fy.loc[fy['lvl3permid'].isin(flagged.index)]
unres_rows = fy_flag.loc[~fy_flag['worldscopecmpid'].isin(universe_ids)]
unresolved = pd.Series(unres_rows['worldscopecmpid'].unique()).astype(str).str.strip().str.upper()
control = pd.Series(fy.loc[fy['worldscopecmpid'].isin(universe_ids), 'worldscopecmpid'].unique()).astype(str)

print(f"      [---] flagged: {({int(k): round(v, 4) for k, v in flagged.items()})}")
print(f"      [---] rated firm-years in flagged regions {len(fy_flag)}, unresolved {len(unres_rows)}, "
      f"unresolved distinct ids {len(unresolved)}")


### i) check-digit validity ---------------------------------------------------------
print(f"\n    [+++] CUSIP check-digit validity...")
valid = unresolved.map(cusip_check_ok)
print(f"      [---] unresolved ids: {valid.sum()} of {len(valid)} pass ({valid.mean():.4f})")
print(f"      [---] control, resolved Worldscope ids: {control.map(cusip_check_ok).mean():.4f} pass "
      f"(expected ~0.1 or 0 for non-CUSIPs)")
if (~valid).any():
    print(f"      [---] sample failing ids: {sorted(unresolved[~valid])[:TOP_N]}")


### ii) locate CUSIP items ---------------------------------------------------------
print(f"\n    [+++] locating CUSIP items in Worldscope...")
db = wrds.Connection(wrds_username=lc.wrds_log)
cand = db.raw_sql("""
    SELECT DISTINCT number, name FROM tr_worldscope.wsitem
    WHERE UPPER(name) LIKE '%%CUSIP%%'
    ORDER BY number
""")
print(cand.to_string(index=False))
cand_cols = [f"item{n:.0f}" for n in cand['number']]
columns = {t: set(db.describe_table(library='tr_worldscope', table=t)['name']) for t in WS_TABLES}
print(pd.DataFrame({t: [c in columns[t] for c in cand_cols] for t in WS_TABLES}, index=cand_cols).to_string())
print(f"      [---] item6105 present: {({t: 'item6105' in columns[t] for t in WS_TABLES})}")

located = [(t, c) for t in WS_TABLES for c in cand_cols if c in columns[t] and 'item6105' in columns[t]]
assert located, "no CUSIP item found in a table that carries item6105 - inspect the listing above"


### iii) match and recoverability ---------------------------------------------------------
ids = pd.DataFrame({'cmpid': unresolved, 'cusip8': unresolved.str[:8]})

for t, c in located:
    print(f"\n    [+++] matching against {t}.{c}...")
    pairs = db.raw_sql(f"""
        SELECT item6105, {c} AS cusip FROM tr_worldscope.{t}
        WHERE item6105 IS NOT NULL AND {c} IS NOT NULL
    """).astype(str)
    pairs['cusip'] = pairs['cusip'].str.strip().str.upper()
    # share-class rows carry their own CUSIP; their base id is the stem + '0' (archive.check_ws_country_field)
    is_base = pairs['item6105'].str.fullmatch(BASE_CMPID)
    pairs['base_id'] = pairs['item6105'].where(is_base, pairs['item6105'].str[:-1] + '0')
    print(f"      [---] rows {len(pairs)}, share-class rows {(~is_base).sum()}, "
          f"value length -> rows: {pairs['cusip'].str.len().value_counts().sort_index().to_dict()}")

    p9 = pairs[['cusip', 'base_id']].drop_duplicates()
    p8 = pairs.assign(cusip8=pairs['cusip'].str[:8])[['cusip8', 'base_id']].drop_duplicates()
    hit9 = ids.merge(p9, left_on='cmpid', right_on='cusip', how='inner')[['cmpid', 'base_id']]
    hit8 = (ids.loc[~ids['cmpid'].isin(hit9['cmpid'])]
            .merge(p8, on='cusip8', how='inner')[['cmpid', 'base_id']])
    hits = pd.concat([hit9, hit8], ignore_index=True).drop_duplicates()

    n_targets = hits.groupby('cmpid')['base_id'].nunique()
    unique = hits.loc[hits['cmpid'].isin(n_targets.index[n_targets == 1])]
    print(f"      [---] ids matched: {n_targets.size} of {len(ids)} ({n_targets.size / len(ids):.4f}) - "
          f"CUSIP-9 {hit9['cmpid'].nunique()}, CUSIP-8 only {hit8['cmpid'].nunique()}")
    print(f"      [---] unique base id {len(unique)}, ambiguous (several base ids) {(n_targets > 1).sum()}")

    # recoverable rated firm-years: unresolved rows whose mapped base id has an annual record that year
    mapped = unres_rows.merge(unique, left_on='worldscopecmpid', right_on='cmpid', how='inner', validate='many_to_one')
    recovered = pd.MultiIndex.from_frame(mapped[['base_id', 'year']].astype({'base_id': 'string'})).isin(key_u)
    assert recovered.sum() <= len(unres_rows), "recovered firm-years exceed the unresolved firm-years"

    in_universe = unique['base_id'].isin(universe_ids)
    already_linked = unique['base_id'].isin(set(fy.loc[fy['found'], 'worldscopecmpid']))
    projected = (fy_flag['found'].sum() + recovered.sum()) / len(fy_flag)
    print(f"      [---] mapped base ids in the annual universe: {in_universe.sum()} of {len(unique)}")
    print(f"      [---] recoverable rated firm-years: {recovered.sum()} of {len(unres_rows)} unresolved "
          f"-> projected fy_resolution {projected:.4f} (now {fy_flag['found'].mean():.4f})")
    print(f"      [---] mapped base ids already linked to another rated entity (possible duplicates): "
          f"{already_linked.sum()}")
    unmatched = ids.loc[~ids['cmpid'].isin(n_targets.index), 'cmpid']
    if not unmatched.empty:
        print(f"      [---] sample unmatched ids: {sorted(unmatched)[:TOP_N]}")

print(f"\n[°°°] check complete [°°°]")