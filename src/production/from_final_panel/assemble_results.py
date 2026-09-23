##################################################
'''
src.production.from_final_panel.assemble_results

input:      results_*.yaml in PRED_DIR_MAN, FTT_VAL_TRIALS, ICL_SCORES (non-recursive),
            depletion_counts.parquet
purpose:    assemble every run manifest into frames and derive the reported
            results.
output:     TAB_AGGREGATE, TAB_PER_REGION, TAB_DID, TAB_GAP_CURVE, TAB_HEADLINE (csv)
            VIZ_GAP_CURVE (png)

'''
##################################################
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config as con

SCORE_DIRS = (con.PRED_DIR_MAN, con.FTT_VAL_TRIALS, con.ICL_SCORES)
KEY = ["model", "configuration", "condition"]
CURVE_KEYS = ["model", "configuration"]
GAP_COLS = {"regional bias gap": "gap_r_sq", "regional bias gap rmse": "gap_rmse"}
N_COLS = ["level", "draw", "anchor_rows_retained", "anchor_entities_retained"]
REGION_METRICS = ["rmse_r", "r_sq", "r_sq_reg"]

MODEL_LABELS = {"rf": "RF", "xgb": "XGBoost", "ftt": "FT-Transformer", "tabicl": "TabICL", "tabpfn3": "TabPFN-3"}
MODEL_COLORS = {"rf": "#1f4e79", "xgb": "#6fa8dc", "ftt": "#c0392b", "tabicl": "#2e8b57", "tabpfn3": "#e69f00"}


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> runs (one row per manifest), regions (one row per manifest x Tier-1 region)."""
    runs, regions = [], []
    for d in SCORE_DIRS:
        for path in sorted(d.glob("results_*.yaml")):  # non-recursive: the env-check report stays out
            m = yaml.safe_load(path.read_text(encoding="utf-8"))
            key = {k: m["meta"][k] for k in KEY}
            runs.append({**key, **m["metrics"], "file": path.name})
            for reg, vals in m["region metrics"]["results"].items():
                regions.append({**key, "region": int(reg), **vals})
    runs, regions = pd.DataFrame(runs), pd.DataFrame(regions)
    assert len(runs) == 96 and not runs.duplicated(KEY).any()
    assert regions.groupby(KEY).size().eq(13).all()

    # (1 - r_sq)/(1 - r_sq_reg) = sigma2_r / sigma2_global: predictions cancel, only the frozen
    # test targets remain, so any drift means a run was scored on a different test set or denominator
    ratio = (1 - regions["r_sq"]) / (1 - regions["r_sq_reg"])
    ref = ratio.groupby(regions["region"]).transform("first")
    assert np.allclose(ratio, ref, rtol=con.INVARIANT_RTOL, atol=0), "test-set variance ratio differs between runs"
    return runs, regions


def attach_realised_n(frame: pd.DataFrame) -> pd.DataFrame:
    """Realised anchor train+val rows and entities per condition, from the frozen counts artifact.
    """
    counts = pd.read_parquet(con.DEPLETION_COUNTS, columns=N_COLS + ["anchor_rows_full", "anchor_entities_full"])
    full_rows = counts["anchor_rows_full"].unique()
    full_ents = counts["anchor_entities_full"].unique()
    assert len(full_rows) == 1 and len(full_ents) == 1, "full anchor size differs between conditions"
    assert not counts.duplicated(["level", "draw"]).any()

    lookup = counts[N_COLS].copy()
    lookup["condition"] = [f"depl_L{int(l)}_d{int(d)}" for l, d in zip(lookup["level"], lookup["draw"])]
    undepl = pd.DataFrame({"condition": ["undepl"], "level": [pd.NA], "draw": [pd.NA],
                           "anchor_rows_retained": full_rows, "anchor_entities_retained": full_ents})
    lookup = pd.concat([lookup, undepl], ignore_index=True).astype({"level": "Int64", "draw": "Int64"})
    assert lookup["condition"].is_unique
    assert (lookup.loc[lookup["condition"].ne("undepl"), "anchor_rows_retained"] < full_rows[0]).all()

    # tag format is duplicated from run_ICL_depletion_grid._condition_tag (import would pull in run_icl);
    # set equality below catches any drift between the two
    depl_tags = set(frame.loc[frame["condition"].ne("undepl"), "condition"])
    grid_tags = set(lookup["condition"]) - {"undepl"}
    assert depl_tags == grid_tags, f"runs-only {sorted(depl_tags - grid_tags)}, counts-only {sorted(grid_tags - depl_tags)}"

    assert not set(N_COLS) & set(frame.columns), "frame already carries realised-N columns"
    out = frame.merge(lookup, on="condition", how="left", validate="many_to_one")
    assert len(out) == len(frame), "merge changed row count"
    assert out[["anchor_rows_retained", "anchor_entities_retained"]].notna().all().all(), "run without a realised N"
    return out


def gap_curve(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tier-1 bias gap, both forms, per model x level, with the share of the undepleted gap removed.
    """
    gaps = list(GAP_COLS.values())
    shares = [f"share_removed_{g}" for g in gaps]
    assert set(N_COLS) <= set(runs.columns), "run attach_realised_n first"

    # only configurations with a depletion curve; drops rf/baseline
    has_curve = runs.loc[runs["condition"].ne("undepl"), CURVE_KEYS].drop_duplicates()
    points = runs.merge(has_curve, on=CURVE_KEYS, how="inner")
    points = points[KEY + N_COLS + list(GAP_COLS)].rename(columns=GAP_COLS)
    assert points[gaps].notna().all().all()

    base = points.loc[points["condition"].eq("undepl"), CURVE_KEYS + gaps]
    assert len(base) == len(has_curve) and not base.duplicated(CURVE_KEYS).any(), "each curve needs exactly one undepl base"
    assert (base[gaps] != 0).all().all(), "undepl gap exactly 0 - share undefined"

    n_before = len(points)
    points = points.merge(base, on=CURVE_KEYS, how="left", suffixes=("", "_undepl"), validate="many_to_one")
    assert len(points) == n_before
    for g, s in zip(gaps, shares):
        points[s] = (points[f"{g}_undepl"] - points[g]) / points[f"{g}_undepl"]
    assert np.allclose(points.loc[points["condition"].eq("undepl"), shares], 0.0)

    # mean and min-max only: 2-5 draws per level do not support a dispersion estimate
    summary = points.groupby(CURVE_KEYS + ["level"], dropna=False).agg(
        n_draws=("draw", "size"),
        anchor_rows_mean=("anchor_rows_retained", "mean"),
        **{f"{c}_{stat}": (c, stat) for c in gaps + shares for stat in ("mean", "min", "max")},
    ).reset_index()
    assert summary["n_draws"].sum() == len(points)

    # denominator check: undepl gap against the draw noise of the same model
    print(f"    [+++] undepl gap vs. draw noise (largest within-level min-max range)")
    for g, s in zip(gaps, shares):
        noise = summary.assign(r=summary[f"{g}_max"] - summary[f"{g}_min"]).groupby(CURVE_KEYS)["r"].max()
        check = base.set_index(CURVE_KEYS)[g].to_frame("undepl").join(noise.rename("max_range"))
        for (model, cfg), row in check.iterrows():
            flag = "UNSTABLE " if abs(row["undepl"]) <= row["max_range"] else ""
            print(f"      [---] {flag}{g} {model}/{cfg}: undepl {row['undepl']:+.4f}, max draw range {row['max_range']:.4f}")
        reversed_runs = points.loc[points[s] > 1, "model"]
        if len(reversed_runs):
            print(f"      [---] {s} > 1 (gap reversed) in {len(reversed_runs)} runs: {sorted(set(reversed_runs))}")
    return points, summary


def did_table(regions: pd.DataFrame) -> pd.DataFrame:
    """Per model x condition: anchor change, comparator change, their difference, on r_sq.

    change = base - depleted (positive = r_sq fell), base = same model AND
    configuration at undepl. Comparator = unweighted mean change over the 12
    non-anchor Tier-1 regions, so each region counts once regardless of its
    test size, matching the equal region weights of the macro average.
    con.EQUAL_N_REGION, the smallest Tier-1 region, is also reported on its own:
    the deepest level depletes the anchor to about its train+val size.
    Depleted conditions only: undepl rows are 0 by definition.
    """
    base_key = ["model", "configuration", "region"]
    comparators = [r for r in con.TIER1_REGS if r != con.ANCHOR_REGION]
    assert len(comparators) == 12 and con.EQUAL_N_REGION in comparators

    # 1. change per region against the own undepleted base
    is_base = regions["condition"].eq("undepl")
    base = regions.loc[is_base, base_key + ["r_sq"]].rename(columns={"r_sq": "r_sq_base"})
    depl = regions.loc[~is_base, KEY + ["region", "r_sq"]]
    assert not base.duplicated(base_key).any()

    delta = depl.merge(base, on=base_key, how="left", validate="many_to_one")
    assert len(delta) == len(depl), "merge changed row count"
    assert delta["r_sq_base"].notna().all(), "depleted run without an undepl base of the same configuration"
    delta["d_r_sq"] = delta["r_sq_base"] - delta["r_sq"]

    # 2. split into anchor, comparators, equal-N region
    wide = delta.pivot(index=KEY, columns="region", values="d_r_sq")  # raises on duplicate (KEY, region)
    assert set(wide.columns) == set(con.TIER1_REGS) and wide.notna().all().all()

    # 3.-4. reduce comparators, form the differences
    out = pd.DataFrame({
        "d_anchor": wide[con.ANCHOR_REGION],
        "d_comp": wide[comparators].mean(axis=1),
        "d_equal_n": wide[con.EQUAL_N_REGION],
    })
    out["did"] = out["d_anchor"] - out["d_comp"]
    out["did_equal_n"] = out["d_anchor"] - out["d_equal_n"]
    out = out.reset_index()

    assert len(out) == depl[KEY].drop_duplicates().shape[0]
    assert not out["configuration"].eq("baseline").any()
    return out


def headline_changes(runs: pd.DataFrame) -> pd.DataFrame:
    """Relative change per metric, deepest level vs undepleted, per model; one column per metric.

    change = (level mean at the deepest level - undepl) / |undepl|, same model AND
    configuration. The sign follows the metric: RMSE up is worse, R2 up is better;
    for the two gap columns it equals minus the share of the gap removed.
    """
    metrics = [c for c in runs.columns if c not in KEY + N_COLS + ["file"]]
    assert all(pd.api.types.is_float_dtype(runs[c]) for c in metrics), "non-numeric metric column"

    deepest = runs["level"].min()
    deep = runs.loc[runs["level"].eq(deepest).fillna(False)]
    n_draws = deep.groupby(CURVE_KEYS).size()
    assert n_draws.nunique() == 1, "deepest level holds different draw counts per curve"

    base = runs.loc[runs["condition"].eq("undepl")].set_index(CURVE_KEYS)[metrics]
    deep_mean = deep.groupby(CURVE_KEYS)[metrics].mean()
    assert deep_mean.index.isin(base.index).all()
    base = base.loc[deep_mean.index]  # drops rf/baseline
    assert (base != 0).all().all(), "undepl metric exactly 0 - relative change undefined"

    out = (deep_mean - base) / base.abs()
    out.insert(0, "level", int(deepest))
    out.insert(1, "n_draws", int(n_draws.iloc[0]))
    return out.reset_index()


def plot_gap_curve(points: pd.DataFrame, summary: pd.DataFrame) -> None:
    """R2 gap over realised anchor rows (log): level means as lines, draws as points, undepl marked."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for (model, cfg), s in summary.groupby(CURVE_KEYS):
        color = MODEL_COLORS[model]
        s = s.sort_values("anchor_rows_mean")
        p = points.loc[(points["model"] == model) & (points["configuration"] == cfg)]
        is_undepl = p["condition"].eq("undepl")

        ax.plot(s["anchor_rows_mean"], s["gap_r_sq_mean"], color=color, lw=1.6, marker="o", ms=4,
                label=MODEL_LABELS[model], zorder=3)
        ax.scatter(p.loc[~is_undepl, "anchor_rows_retained"], p.loc[~is_undepl, "gap_r_sq"],
                   color=color, s=16, alpha=0.45, edgecolor="none", zorder=2)
        ax.scatter(p.loc[is_undepl, "anchor_rows_retained"], p.loc[is_undepl, "gap_r_sq"],
                   marker="D", s=42, facecolor="white", edgecolor=color, lw=1.4, zorder=4)

    ax.axhline(0, color="#333333", lw=0.9, zorder=1)

    ax.set_xscale("log")
    ticks = sorted(summary["level"].dropna().astype(int).unique()) + [int(points["anchor_rows_retained"].max())]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, color="#888888", alpha=0.45),
                plt.Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec="#888888")]
    labels += ["individual draw", "undepleted (full anchor)"]
    ax.legend(handles, labels, frameon=False, loc="best")

    ax.set_title(f"Tier-1 Regional Bias Gap under Anchor Depletion ({con.ANCHOR_REGION})", pad=15, fontweight='bold')
    ax.set_xlabel("Realised anchor rows, train+val (log scale)", labelpad=10)
    ax.set_ylabel("Regional bias gap (R², global scale)")

    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    con.VIZ_GAP_CURVE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(con.VIZ_GAP_CURVE, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print(f"\n[°°°] assembling results - anchor {con.ANCHOR_REGION}, equal-N region {con.EQUAL_N_REGION} [°°°]\n")

    runs, regions = load_manifests()
    print(f"    [+++] manifests loaded - {len(runs)} runs, {len(regions)} region rows")

    runs = attach_realised_n(runs)
    did = attach_realised_n(did_table(regions))
    print(f"    [+++] realised N attached - runs {len(runs)}, DiD rows {len(did)}")

    points, summary = gap_curve(runs)
    headline = headline_changes(runs)

    is_undepl = runs["condition"].eq("undepl")
    aggregate = runs.loc[is_undepl].drop(columns=["condition", "level", "draw"])
    per_region = regions.loc[regions["condition"].eq("undepl"), CURVE_KEYS + ["region"] + REGION_METRICS]
    assert len(per_region) == is_undepl.sum() * len(con.TIER1_REGS)

    con.RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    for frame, path in ((aggregate, con.TAB_AGGREGATE), (per_region, con.TAB_PER_REGION),
                        (did, con.TAB_DID), (summary, con.TAB_GAP_CURVE), (headline, con.TAB_HEADLINE)):
        frame.to_csv(path, index=False)
        print(f"      [---] written {path.name} - {frame.shape[0]} rows x {frame.shape[1]} cols")

    plot_gap_curve(points, summary)
    print(f"      [---] written {con.VIZ_GAP_CURVE.name}")

    print(f"\n[°°°] results assembled [°°°]")