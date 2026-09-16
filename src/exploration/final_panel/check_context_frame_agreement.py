##################################################
'''
src.exploration.final_panel.check_context_frame_agreement

input:      panel.parquet, ref_geo_table.parquet, split.parquet,
            pre_processing_constants.parquet
purpose:    prove that build_depletion_draws._load_context_frame and
            Gatekeeper.stage_three_data() are two derivations of one object.
            Run before build_depletion_draws.
output:     none 
'''
##################################################
import config as con, utils as ut, pandas as pd, numpy as np
from stageguard import Gatekeeper
from src.production.from_final_panel.build_depletion_draws import _load_context_frame, ANCHOR_REGION

print(f"\n[°°°] checking context frame agreement [°°°]\n")

context = _load_context_frame()

gk = Gatekeeper(model='ICL')
X_ctx, y_ctx = gk.stage_three_data()

print(f"\n    [+++] comparing derivations - draw frame {len(context)} rows, gatekeeper {len(X_ctx)} rows")

assert len(context) == len(X_ctx), \
    f"row counts disagree - draw frame {len(context)}, gatekeeper stage 3 {len(X_ctx)}. " \
    f"The draws would be cut against a population the models never see"

draw_positions = pd.Index(np.sort(context.index.to_numpy()))
gate_positions = pd.Index(np.sort(X_ctx.index.to_numpy()))
assert draw_positions.equals(gate_positions), \
    f"panel positions disagree - {len(draw_positions.difference(gate_positions))} rows only in the draw frame, " \
    f"{len(gate_positions.difference(draw_positions))} only in the gatekeeper slice"
print(f"    [+++] panel positions identical - {len(draw_positions)} rows")

keys = ut.resolve_keys(gk, X_ctx.index)
gate_counts = keys.groupby('orgpermid').size().sort_index()
draw_counts = context.groupby('orgpermid').size().sort_index()

only_gate = gate_counts.index.difference(draw_counts.index)
only_draw = draw_counts.index.difference(gate_counts.index)
assert len(only_gate) == 0 and len(only_draw) == 0, \
    f"entity sets disagree - {len(only_gate)} entities only in the gatekeeper slice (e.g. {list(only_gate[:5])}), " \
    f"{len(only_draw)} only in the draw frame (e.g. {list(only_draw[:5])})"

disagreeing = draw_counts.index[draw_counts.to_numpy() != gate_counts.to_numpy()]
assert len(disagreeing) == 0, \
    f"{len(disagreeing)} entities carry a different row count in each derivation (e.g. {list(disagreeing[:5])}) - " \
    f"the aggregate row count can still match while the composition differs"
print(f"    [+++] entity sets and per-entity row counts identical - {len(draw_counts)} entities")

gate_regions = keys.merge(gk.geo_id, on='orgpermid', how='left', validate='many_to_one')
assert gate_regions['lvl3permid'].notna().all(), "a gatekeeper context row resolves to no region"
gate_region_counts = gate_regions.groupby('lvl3permid').size().sort_index()
draw_region_counts = context.groupby('lvl3permid').size().sort_index()

assert gate_region_counts.index.equals(draw_region_counts.index), \
    f"region coverage disagrees - gatekeeper {list(gate_region_counts.index)}, draw frame {list(draw_region_counts.index)}"
region_gap = (gate_region_counts - draw_region_counts)
assert (region_gap == 0).all(), f"per-region row counts disagree:\n{region_gap[region_gap != 0]}"
print(f"    [+++] per-region row counts identical across {len(draw_region_counts)} regions")

shared_target = context.loc[X_ctx.index, 'esg_combined_score']
assert np.allclose(shared_target.to_numpy(), y_ctx.to_numpy(), rtol=0, atol=0), \
    f"the target disagrees on {int((shared_target.to_numpy() != y_ctx.to_numpy()).sum())} shared positions - " \
    f"same rows, different values, so one derivation is reading a stale panel"
print(f"    [+++] target agrees row by row")

anchor_rows = int((context['lvl3permid'] == ANCHOR_REGION).sum())
print(f"\n    [+++] agreement established")
print(f"      [---] context rows        {len(context)}   <- con.CONTEXT_ROWS must equal this, currently {con.CONTEXT_ROWS}")
print(f"      [---] context entities    {len(draw_counts)}")
print(f"      [---] anchor {ANCHOR_REGION} rows  {anchor_rows}")
print(f"      [---] tier-1 regions      {len([r for r in draw_region_counts.index if r in con.TIER1_REGS])}")
print(f"      [---] tier-2 regions      {len([r for r in draw_region_counts.index if r in con.TIER2_REGS])}")
print(f"\n[°°°] context frame agreement check complete [°°°]")