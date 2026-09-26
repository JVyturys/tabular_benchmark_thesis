##################################################
'''
src.production.descriptives.build_desc_raw

input:      raw_panel.parquet, ref_geo_raw.parquet, ref_geo_table.parquet, panel.parquet
purpose:    on-disk attrition from the raw pull to the modelling universe
            (tier-1 + tier-2), in rows and entities
output:     desc_attrition.csv - step, role, rows, entities, row_share_raw, entity_share_raw

'''
##################################################
import pandas as pd

import config as con


def _int_ids(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    ids = frame['orgpermid']
    assert ids.notna().all(), f"{name}: {int(ids.isna().sum())} null orgpermid"
    cast = ids.astype('int64')
    assert (cast.astype(ids.dtype) == ids).all(), f"{name}: orgpermid ({ids.dtype}) does not cast to int64 losslessly"
    return frame.assign(orgpermid=cast)


def _record(step: str, role: str, ids: pd.Series) -> dict:
    return {'step': step, 'role': role, 'rows': int(len(ids)), 'entities': int(ids.nunique())}


def build_attrition() -> pd.DataFrame:
    raw = _int_ids(pd.read_parquet(con.RAW_PANEL, columns=['orgpermid']), 'raw_panel')
    geo_raw = _int_ids(pd.read_parquet(con.REF_GEO_RAW, columns=['orgpermid', 'lvl3permid', 'lvl5isocntry']), 'ref_geo_raw')
    geo = _int_ids(pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid', 'lvl5isocntry']), 'ref_geo_table')
    panel = _int_ids(pd.read_parquet(con.PANEL, columns=['orgpermid']), 'panel')
    print(f"    [+++] loaded - raw {len(raw)}, geo_raw {len(geo_raw)}, geo {len(geo)}, panel {len(panel)} rows")

    assert geo_raw['orgpermid'].is_unique, "ref_geo_raw duplicates orgpermid"
    assert geo['orgpermid'].is_unique, "ref_geo_table duplicates orgpermid"
    assert geo['lvl3permid'].notna().all(), "ref_geo_table carries a null lvl3permid"
    assert set(geo_raw['orgpermid']) <= set(geo['orgpermid']), "ref_geo_raw entity missing from ref_geo_table"
    assert set(geo['orgpermid']) <= set(raw['orgpermid']), "ref_geo_table entity absent from the raw panel"

    direct = geo_raw.merge(geo, on='orgpermid', how='left', suffixes=('_raw', ''), validate='one_to_one')
    assert (direct['lvl3permid_raw'].astype('float64') == direct['lvl3permid'].astype('float64')).all(), \
        "override step changed a direct lvl3permid mapping"
    assert (direct['lvl5isocntry_raw'] == direct['lvl5isocntry']).all(), "override step changed a direct iso mapping"

    override = geo.loc[~geo['orgpermid'].isin(geo_raw['orgpermid'])]
    pairs = set(zip(override['lvl3permid'].astype('int64'), override['lvl5isocntry']))
    assert pairs == set(con.DOMICILE_OVERRIDES.values()), \
        f"override pairs {sorted(pairs)} differ from con.DOMICILE_OVERRIDES {sorted(con.DOMICILE_OVERRIDES.values())}"

    ids = raw['orgpermid']
    is_direct = ids.isin(geo_raw['orgpermid'])
    is_override = ids.isin(override['orgpermid'])
    is_unresolved = ~ids.isin(geo['orgpermid'])
    assert (is_direct.astype(int) + is_override.astype(int) + is_unresolved.astype(int)).eq(1).all(), \
        "raw rows not partitioned by direct / override / unresolvable"

    assert len(panel) == int((~is_unresolved).sum()), \
        f"panel holds {len(panel)} rows, raw restricted to referenceable entities holds {int((~is_unresolved).sum())}"
    assert set(panel['orgpermid']) == set(geo['orgpermid']), "panel and ref_geo_table disagree on the entity set"

    records = [_record('raw_panel', 'stage', ids), _record('geo_direct', 'retained', ids[is_direct])]
    iso_of = override.set_index('orgpermid')['lvl5isocntry']
    for iso in sorted(iso_of.unique()):
        records.append(_record(f'geo_override_{iso}', 'retained', ids[ids.isin(iso_of.index[iso_of == iso])]))
    records += [_record('geo_unresolvable', 'dropped', ids[is_unresolved]), _record('panel', 'stage', panel['orgpermid'])]

    tiered = panel.merge(geo[['orgpermid', 'lvl3permid']], on='orgpermid', how='left', validate='many_to_one')
    assert len(tiered) == len(panel), "panel grew or shrank on the geo merge"
    tiers = {'tier1': con.TIER1_REGS, 'tier2': con.TIER2_REGS, 'tier3': con.TIER3_REGS}
    untiered = set(tiered['lvl3permid']) - {r for regs in tiers.values() for r in regs}
    assert not untiered, f"panel regions without a tier: {sorted(untiered)}"
    for tier, regs in tiers.items():
        records.append(_record(tier, 'dropped' if tier == 'tier3' else 'retained',
                               tiered.loc[tiered['lvl3permid'].isin(regs), 'orgpermid']))
    in_tier = tiered.loc[tiered['lvl3permid'].isin([*con.TIER1_REGS, *con.TIER2_REGS]), 'orgpermid']
    records.append(_record('model_universe', 'stage', in_tier))

    out = pd.DataFrame(records)
    stage_idx = out.index[out['role'].eq('stage')].tolist()
    for parent, child in zip(stage_idx[:-1], stage_idx[1:]):
        parts = out.loc[parent + 1:child - 1]
        for col in ('rows', 'entities'):
            assert parts[col].sum() == out.loc[parent, col], \
                f"components after {out.loc[parent, 'step']} sum to {parts[col].sum()} {col}, stage holds {out.loc[parent, col]}"
            retained = parts.loc[parts['role'].eq('retained'), col].sum()
            assert retained == out.loc[child, col], \
                f"retained {col} after {out.loc[parent, 'step']} ({retained}) differ from {out.loc[child, 'step']} ({out.loc[child, col]})"

    out['row_share_raw'] = out['rows'] / out.loc[0, 'rows']
    out['entity_share_raw'] = out['entities'] / out.loc[0, 'entities']
    return out


if __name__ == '__main__':
    print(f"\n[°°°] building raw-to-model-universe attrition [°°°]\n")
    attrition = build_attrition()
    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    attrition.to_csv(con.TAB_DESC_ATTRITION, index=False)
    print(attrition.to_string(index=False))
    print(f"\n      [---] written {con.TAB_DESC_ATTRITION.name} - {attrition.shape[0]} rows x {attrition.shape[1]} cols")
    print(f"\n[°°°] attrition complete [°°°]")