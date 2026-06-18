"""
Detailed ER-style diagram of the ESG / Worldscope panel join chain.
Produces both a PNG and a PDF 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import config as con

# ---------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------
ROW_H = 0.28
HEADER_H = 0.42
KEY_COLOR = "#a83232"
NOTE_COLOR = "#666666"
TEXT_COLOR = "#262626"
ROW_BG_EVEN = "#f5f5f2"
ROW_BG_ODD = "#ffffff"

HEADER_COLORS = {
    "source": "#2c6f9b",     # ESG + Worldscope financials
    "bridge": "#4a4a48",     # permorgref hub
    "lookup": "#b5651d",     # geography taxonomy
}


def draw_table(ax, x, y_top, width, title, rows, header_kind):
    """
    rows: list of (column_name, note, is_key)
    Returns: dict {column_name: (left_x, center_y, right_x)}, bottom_y
    """
    n = len(rows)
    height = HEADER_H + n * ROW_H

    header_color = HEADER_COLORS[header_kind]
    ax.add_patch(Rectangle((x, y_top - HEADER_H), width, HEADER_H,
                            facecolor=header_color, edgecolor="black",
                            linewidth=1.0, zorder=2))
    ax.text(x + width / 2, y_top - HEADER_H / 2, title,
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=3)

    positions = {}
    for i, (col, note, is_key) in enumerate(rows):
        row_top = y_top - HEADER_H - i * ROW_H
        row_center = row_top - ROW_H / 2
        bg = ROW_BG_EVEN if i % 2 == 0 else ROW_BG_ODD
        ax.add_patch(Rectangle((x, row_top - ROW_H), width, ROW_H,
                                facecolor=bg, edgecolor="#cccccc",
                                linewidth=0.6, zorder=2))
        color = KEY_COLOR if is_key else TEXT_COLOR
        weight = "bold" if is_key else "normal"
        ax.text(x + 0.15, row_center, col, ha="left", va="center",
                fontsize=9, color=color, fontweight=weight, zorder=3)
        if note:
            ax.text(x + width - 0.15, row_center, note, ha="right",
                    va="center", fontsize=7.5, color=NOTE_COLOR,
                    style="italic", zorder=3)
        positions[col] = (x, row_center, x + width)

    bottom_y = y_top - height
    ax.add_patch(Rectangle((x, bottom_y), width, height, facecolor="none",
                            edgecolor="black", linewidth=1.3, zorder=4))
    return positions, bottom_y


def connector(ax, p_from, p_to, rad, color="#333333"):
    cp = ConnectionPatch(p_from, p_to, coordsA="data", coordsB="data",
                          axesA=ax, axesB=ax, arrowstyle="-|>",
                          connectionstyle=f"arc3,rad={rad}",
                          color=color, lw=1.4, mutation_scale=14, zorder=5)
    ax.add_patch(cp)


# ---------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------
esg_rows = [
    ("orgpermid", "JOIN \u2192 permorgref", True),
    ("year", "filter 2009\u20132025", False),
    ("fieldname", "= 'ESGCombinedScore'", True),
    ("valuescore", "TARGET (0\u20131 scale)", True),
    ("pillar", "", False),
    ("hierarchy", "", False),
    ("comname", "", False),
    ("cusip / ticker", "", False),
    ("siccode / naicscode", "sparse \u2014 unused", False),
    ("currency", "", False),
]

permorgref_rows = [
    ("orgpermid", "PK \u2014 bridges ESG \u2194 Worldscope", True),
    ("worldscopecmpid", "JOIN \u2192 wrds_ws_funda.item6105", True),
    ("domcntrypermid", "JOIN \u2192 tmcregncntrymap.lvl5permid", True),
    ("typecode", "filter = 'COM'", True),
    ("status", "Active / Inactive flag", False),
    ("immediateparentorgpermid", "parent-entity tracking", False),
    ("ultimateparentorgpermid", "split contamination control", False),
    ("comname", "", False),
    ("lei / cik", "", False),
]

funda_rows = [
    ("item6105", "JOIN \u2192 permorgref.worldscopecmpid", True),
    ("code", "internal Worldscope security ID", False),
    ("year_", "JOIN key (= esg.year)", True),
    ("freq", "filter = 'A'", True),
    ("seq", "always 1 \u2014 no filter needed", False),
    ("item1004 \u2026 item19112", "242 selected financial features", True),
]

geo_rows = [
    ("lvl5permid", "JOIN \u2192 permorgref.domcntrypermid", True),
    ("lvl5isocntry", "ISO country code", False),
    ("lvl4permid", "sub-region", False),
    ("lvl3permid", "EVALUATION LABEL (region)", True),
    ("lvl2permid", "continent", False),
    ("lvl1permid", "global", False),
]

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 13.5))
ax.set_xlim(-1.2, 10.2)
ax.axis("off")

ax.text(4.6, 10.9, "ESG \u2013 Financial panel: join chain & key columns",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color="#1a1a1a")
ax.text(4.6, 10.55, "Master's thesis data pipeline \u2014 generated from WRDS schema exploration",
        ha="center", va="center", fontsize=9.5, color="#666666", style="italic")

main_x, main_w = 0.5, 4.4
geo_x, geo_w = 5.6, 3.6

esg_pos, esg_bottom = draw_table(ax, main_x, 10.0, main_w,
                                  "tr_esg.wrds_ref_esg", esg_rows, "source")

perm_top = esg_bottom - 0.6
perm_pos, perm_bottom = draw_table(ax, main_x, perm_top, main_w,
                                    "tr_common.permorgref", permorgref_rows, "bridge")

funda_top = perm_bottom - 0.6
funda_pos, funda_bottom = draw_table(ax, main_x, funda_top, main_w,
                                      "tr_worldscope.wrds_ws_funda", funda_rows, "source")

geo_pos, geo_bottom = draw_table(ax, geo_x, perm_top, geo_w,
                                  "tr_common.tmcregncntrymap", geo_rows, "lookup")

# ---------------------------------------------------------------
# Connectors (bulge left into empty margin to avoid crossing boxes)
# ---------------------------------------------------------------
l1, y1, _ = esg_pos["orgpermid"]
l2, y2, _ = perm_pos["orgpermid"]
connector(ax, (l1 - 0.05, y1), (l2 - 0.05, y2), rad=0.15, color=KEY_COLOR)

l1, y1, _ = perm_pos["worldscopecmpid"]
l2, y2, _ = funda_pos["item6105"]
connector(ax, (l1 - 0.05, y1), (l2 - 0.05, y2), rad=0.15, color=KEY_COLOR)

_, y1, r1 = perm_pos["domcntrypermid"]
l2, y2, _ = geo_pos["lvl5permid"]
connector(ax, (r1, y1), (l2, y2), rad=0.15, color=KEY_COLOR)

# ---------------------------------------------------------------
# Legend
# ---------------------------------------------------------------
legend_y = min(funda_bottom, geo_bottom) - 0.55
legend_items = [
    ("source table", HEADER_COLORS["source"]),
    ("bridge / reference table", HEADER_COLORS["bridge"]),
    ("lookup table", HEADER_COLORS["lookup"]),
]
lx = 0.5
for label, color in legend_items:
    ax.add_patch(Rectangle((lx, legend_y), 0.22, 0.22, facecolor=color, edgecolor="black", linewidth=0.6))
    ax.text(lx + 0.32, legend_y + 0.11, label, ha="left", va="center", fontsize=9, color=TEXT_COLOR)
    lx += 0.32 + len(label) * 0.072 + 0.5

ax.add_patch(Rectangle((lx, legend_y), 0.22, 0.22, facecolor="white", edgecolor=KEY_COLOR, linewidth=1.6))
ax.text(lx + 0.32, legend_y + 0.11, "join key / active filter (bold red text)",
        ha="left", va="center", fontsize=9, color=TEXT_COLOR)

ax.set_ylim(legend_y - 0.4, 11.2)

plt.tight_layout()
fig.savefig(con.RESULTS_DIR/"join_chain_diagram.png", dpi=600, bbox_inches="tight")
fig.savefig(con.RESULTS_DIR/"join_chain_diagram.pdf", bbox_inches="tight")
print("Saved PNG and PDF.")