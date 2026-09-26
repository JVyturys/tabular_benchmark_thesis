##################################################
'''
src.production.descriptives.build_desc_split

input:      panel.parquet, ref_geo_table.parquet, ref_cluster_keys.parquet, split.parquet
purpose:    describe the split unit (parent clusters) and the realised partitions
            per physical region against the configured shares
output:     desc_cluster_sizes.csv, desc_cluster_key_source.csv,
            desc_partition_by_region.csv, desc_target_by_partition.csv,
            partition_shares.png

'''
##################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import config as con

PARTITIONS = ['fit', 'val', 'test']
PART_COLORS = {'fit': "#1f4e79", 'val': "#6fa8dc", 'test': "#a6a6a6"}
TIERS = {'tier1': con.TIER1_REGS, 'tier2': con.TIER2_REGS, 'tier3': con.TIER3_REGS}


def _tier(region: int) -> str:
    return next(t for t, regs in TIERS.items() if region in regs)


def _label(region: int) -> str:
    return f"{region}*" if region in con.TIER2_REGS else str(region)


def _style(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')


def _save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"      [---] written {path.name}")


def _int_col(frame: pd.DataFrame, col: str, name: str) -> None:
    assert frame[col].notna().all(), f"{name}: null {col}"
    cast = frame[col].astype('int64')
    assert (cast.astype(frame[col].dtype) == frame[col]).all(), f"{name}: {col} ({frame[col].dtype}) does not cast to int64 losslessly"
    frame[col] = cast


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(con.PANEL, columns=['orgpermid', 'esg_combined_score'])
    geo = pd.read_parquet(con.REF_GEOGRAPHY, columns=['orgpermid', 'lvl3permid'])
    clusters = pd.read_parquet(con.REF_CLUSTER_KEYS, columns=['orgpermid', 'cluster_key', 'parent_typecode', 'key_source'])
    split = pd.read_parquet(con.SPLIT, columns=['orgpermid', 'partition'])
    for frame, name, cols in ((panel, 'panel', ['orgpermid']), (geo, 'ref_geo_table', ['orgpermid', 'lvl3permid']),
                              (clusters, 'ref_cluster_keys', ['orgpermid', 'cluster_key', 'key_source']),
                              (split, 'split', ['orgpermid'])):
        for col in cols:
            _int_col(frame, col, name)
    for frame, name in ((geo, 'ref_geo_table'), (clusters, 'ref_cluster_keys'), (split, 'split')):
        assert frame['orgpermid'].is_unique, f"{name} duplicates orgpermid"

    assert set(clusters['orgpermid']) == set(panel['orgpermid']), "cluster keys and panel disagree on the entity set"
    assert set(clusters['key_source']) <= set(con.KEY_SOURCE.values()), \
        f"key_source values {sorted(set(clusters['key_source']))} outside con.KEY_SOURCE"
    assert split['partition'].isin(PARTITIONS).all(), f"partition values {sorted(set(split['partition']))}"

    n = len(panel)
    frame = (panel.merge(geo, on='orgpermid', how='left', validate='many_to_one')
                  .merge(clusters, on='orgpermid', how='left', validate='many_to_one')
                  .merge(split, on='orgpermid', how='left', validate='many_to_one'))
    assert len(frame) == n, "panel grew or shrank on merge"
    assert frame[['lvl3permid', 'cluster_key']].notna().all().all(), "row without region or cluster key"
    frame['tier'] = frame['lvl3permid'].map(_tier)

    in_tier = frame['tier'].isin(['tier1', 'tier2'])
    assert set(split['orgpermid']) == set(frame.loc[in_tier, 'orgpermid']), \
        "split entities differ from the tier-1 + tier-2 entity set"
    assert frame.loc[in_tier, 'partition'].notna().all(), "in-tier row without a partition"

    parts_per_cluster = frame.loc[in_tier].groupby('cluster_key')['partition'].nunique()
    assert parts_per_cluster.eq(1).all(), f"{int((parts_per_cluster > 1).sum())} clusters span more than one partition"

    tiered = frame.loc[in_tier].copy()
    counts = tiered['partition'].value_counts()
    assert int(counts.sum()) == int(in_tier.sum())
    assert int(counts.get('fit', 0) + counts.get('val', 0)) == con.CONTEXT_ROWS, \
        f"fit + val holds {int(counts.get('fit', 0) + counts.get('val', 0))} rows, con.CONTEXT_ROWS is {con.CONTEXT_ROWS}"
    print(f"    [+++] frame assembled - panel rows {n}, in-tier rows {len(tiered)}, "
          f"clusters {frame['cluster_key'].nunique()}, partitions {counts.to_dict()}")
    return frame, tiered


def cluster_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cluster = frame.groupby('cluster_key').agg(entities=('orgpermid', 'nunique'), rows=('orgpermid', 'size'),
                                                   regions=('lvl3permid', 'nunique'))
    sizes = per_cluster.groupby('entities').agg(clusters=('rows', 'size'), rows=('rows', 'sum'),
                                               region_impure=('regions', lambda r: int((r > 1).sum())))
    sizes = sizes.rename_axis('cluster_size').reset_index()
    sizes['entities'] = sizes['cluster_size'] * sizes['clusters']
    sizes['cluster_share'] = sizes['clusters'] / sizes['clusters'].sum()
    sizes['entity_share'] = sizes['entities'] / sizes['entities'].sum()
    assert sizes['entities'].sum() == frame['orgpermid'].nunique()
    assert sizes['rows'].sum() == len(frame)

    source_name = {v: k for k, v in con.KEY_SOURCE.items()}
    source = (frame.assign(parent_typecode=frame['parent_typecode'].fillna('none'))
                   .groupby(['key_source', 'parent_typecode'])
                   .agg(entities=('orgpermid', 'nunique'), clusters=('cluster_key', 'nunique'), rows=('orgpermid', 'size'))
                   .reset_index())
    source.insert(1, 'key_source_name', source['key_source'].map(source_name))
    assert source['entities'].sum() == frame['orgpermid'].nunique() and source['rows'].sum() == len(frame)
    return sizes[['cluster_size', 'clusters', 'entities', 'rows', 'region_impure', 'cluster_share', 'entity_share']], source


def partition_table(tiered: pd.DataFrame) -> pd.DataFrame:
    def _row(g: pd.DataFrame) -> dict:
        rec = {}
        for p in PARTITIONS:
            s = g.loc[g['partition'].eq(p)]
            rec[f'rows_{p}'], rec[f'entities_{p}'] = len(s), s['orgpermid'].nunique()
        train = rec['rows_fit'] + rec['rows_val']
        rec['rows_total'] = len(g)
        rec['train_share'] = train / len(g)
        rec['train_share_dev'] = rec['train_share'] - con.TRAIN_SHARE
        rec['fit_share_in_train'] = rec['rows_fit'] / train if train else np.nan
        rec['fit_share_dev'] = rec['fit_share_in_train'] - con.FIT_SHARE
        return rec

    records = [{'scope': 'all', 'lvl3permid': pd.NA, 'tier': pd.NA, **_row(tiered)}]
    records += [{'scope': 'region', 'lvl3permid': r, 'tier': _tier(r), **_row(g)} for r, g in tiered.groupby('lvl3permid')]
    out = pd.DataFrame(records)
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    reg = out.loc[out['scope'].eq('region')]
    for p in PARTITIONS:
        assert reg[f'rows_{p}'].sum() == out.loc[0, f'rows_{p}'], f"region {p} rows do not sum to the total"
    return out.sort_values(['scope', 'rows_total'], ascending=[True, False]).reset_index(drop=True)


def target_partition_table(tiered: pd.DataFrame) -> pd.DataFrame:
    def _row(g: pd.DataFrame) -> dict:
        y = {p: g.loc[g['partition'].eq(p), 'esg_combined_score'] for p in PARTITIONS}
        y['train'] = g.loc[g['partition'].isin(['fit', 'val']), 'esg_combined_score']
        rec = {}
        for p, s in y.items():
            rec.update({f'rows_{p}': len(s), f'mean_{p}': s.mean(), f'std_{p}': s.std(), f'median_{p}': s.median()})
        rec['w1_fit_val'] = wasserstein_distance(y['fit'], y['val']) if len(y['fit']) and len(y['val']) else np.nan
        rec['w1_train_test'] = wasserstein_distance(y['train'], y['test']) if len(y['train']) and len(y['test']) else np.nan
        return rec

    records = [{'scope': 'all', 'lvl3permid': pd.NA, 'tier': pd.NA, **_row(tiered)}]
    records += [{'scope': 'region', 'lvl3permid': r, 'tier': _tier(r), **_row(g)} for r, g in tiered.groupby('lvl3permid')]
    out = pd.DataFrame(records)
    out['lvl3permid'] = out['lvl3permid'].astype('Int64')
    assert out.loc[out['scope'].eq('region'), 'rows_train'].sum() == out.loc[0, 'rows_train']
    return out.sort_values(['scope', 'rows_train'], ascending=[True, False]).reset_index(drop=True)


def plot_partitions(parts: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    reg = parts.loc[parts['scope'].eq('region')].sort_values('rows_total', ascending=True)
    y = np.arange(len(reg))
    fig, ax = plt.subplots(figsize=(10, 7))
    left = np.zeros(len(reg))
    for p in PARTITIONS:
        share = (reg[f'rows_{p}'] / reg['rows_total']).to_numpy()
        ax.barh(y, share, left=left, color=PART_COLORS[p], height=0.75, label=p, edgecolor='white', linewidth=0.5)
        left += share
    for x, lab in ((con.TRAIN_SHARE * con.FIT_SHARE, 'fit target'), (con.TRAIN_SHARE, 'train target')):
        ax.axvline(x, color="#d9534f", linestyle=":", linewidth=1.5, zorder=4)
        ax.text(x, len(reg) - 0.3, lab, color="#d9534f", fontsize=8.5, ha='center', va='bottom')
    for yi, n in zip(y, reg['rows_total']):
        ax.text(1.01, yi, f"n={n:,}".replace(",", "."), va='center', fontsize=8.5, color="#555555", fontstyle='italic')
    ax.set_yticks(y)
    ax.set_yticklabels([_label(r) for r in reg['lvl3permid']])
    ax.set_xlim(0, 1)
    ax.set_title("Realised Partition Shares per Region\nRows, Tier-1 and Tier-2 (*) Regions", pad=20, fontweight='bold')
    ax.set_xlabel("Share of region rows", labelpad=10)
    ax.set_ylabel("Level 3 PermID")
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.grid(False)
    _style(ax)
    _save(fig, con.VIZ_DESC_PARTITIONS)


if __name__ == '__main__':
    print(f"\n[°°°] building cluster and split descriptives [°°°]\n")
    frame, tiered = load_frame()

    sizes, source = cluster_tables(frame)
    parts = partition_table(tiered)
    target = target_partition_table(tiered)
    print(f"    [+++] tables built - singleton clusters {int(sizes.loc[sizes['cluster_size'].eq(1), 'clusters'].sum())} "
          f"of {int(sizes['clusters'].sum())}")

    con.DESC_DIR.mkdir(parents=True, exist_ok=True)
    for tab, path in ((sizes, con.TAB_DESC_CLUSTER_SIZES), (source, con.TAB_DESC_CLUSTER_SOURCE),
                      (parts, con.TAB_DESC_PARTITIONS), (target, con.TAB_DESC_TARGET_PARTITION)):
        tab.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {tab.shape[0]} rows x {tab.shape[1]} cols")

    plot_partitions(parts)

    print(f"\n    [+++] partitions\n{parts[['scope', 'lvl3permid', 'rows_fit', 'rows_val', 'rows_test', 'train_share', 'fit_share_in_train']].to_string(index=False)}")
    print(f"\n[°°°] cluster and split descriptives complete [°°°]")