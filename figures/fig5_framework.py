#!/usr/bin/env python3
"""
fig5_framework.py — Fig.5 Dynamic grounding framework (Task 3.5)

Three-layer architecture:
  Layer 1: Static AI prediction (5 foundation / benchmark models)
  Layer 2: Dynamic grounding constraint (turnover + localization priors)
  Layer 3: Biology-aware calibrated output

Walkthrough example uses NEAT1 (nuclear, stable) as a concrete flow.
Pure matplotlib. Paired with fig1_concept.py in visual style.

Output:
    fig5_framework.pdf
    fig5_framework.png (300 dpi)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = Path(__file__).parent

# Palette (matched to fig1_concept.py)
C_L1 = "#E8E8E8"            # static layer band
C_L1_EDGE = "#999999"
C_L2 = "#C9E4CA"            # dynamic grounding band
C_L2_EDGE = "#4A8B5C"
C_L3 = "#F5D0A5"            # biology-aware output band
C_L3_EDGE = "#C2803A"

C_MODEL = "#FFFFFF"
C_MODEL_EDGE = "#777777"
C_TURNOVER = "#5B8BB5"
C_LOCAL = "#8E6BB5"
C_ARROW = "#555555"
C_ARROW_DN = "#4A8B5C"
C_ARROW_OUT = "#C2803A"
C_EXAMPLE = "#1F4E79"
C_TEXT = "#222222"
ARROW = "\u2192"
APPROX = "\u2248"
HALF = "\u00bd"
GTE = "\u2265"


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, text=None,
                     fontsize=9, fontweight="normal", textcolor=None,
                     rounding=0.06):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(box)
    if text is not None:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor or C_TEXT, zorder=5,
        )


def draw_arrow(ax, xy_from, xy_to, color=C_ARROW, lw=1.6, style="-|>",
               mutation_scale=16, connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=mutation_scale,
        linewidth=lw, color=color,
        connectionstyle=connectionstyle, zorder=4,
    )
    ax.add_patch(arr)


def layer_band(ax, x, y, w, h, color, edge, label, sublabel=None):
    draw_rounded_box(ax, x, y, w, h, color, edge, lw=1.4, rounding=0.1)
    # Header placed ABOVE the band to avoid overlap with inner worked-example columns
    ax.text(x + 0.15, y + h + 0.18, label,
            ha="left", va="center",
            fontsize=11, fontweight="bold", color=edge)
    if sublabel:
        ax.text(x + w - 0.15, y + h + 0.18, sublabel,
                ha="right", va="center",
                fontsize=8, fontstyle="italic", color=edge)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        7.5, 10.05,
        "Dynamic grounding framework: static prediction "
        f"{ARROW} dynamic constraint "
        f"{ARROW} biology-aware output",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_TEXT,
    )
    ax.text(
        7.5, 9.65,
        "Worked example: NEAT1 (nuclear, long half-life)",
        ha="center", va="center",
        fontsize=9.5, fontstyle="italic", color="#555555",
    )

    # ------------------------------------------------------------------
    # Left column: worked example flow (narrower)
    # Right: three stacked layer bands
    # ------------------------------------------------------------------

    # Input column (left)
    # RNA sequence
    draw_rounded_box(ax, 0.25, 8.55, 2.1, 0.5, "#FFFFFF", "#BBBBBB", lw=0.9,
                     text="NEAT1 (4-kb isoform)", fontsize=8.5, fontweight="bold")
    ax.text(1.3, 8.20, "RNA sequence input", ha="center",
            fontsize=7.5, fontstyle="italic", color="#777777")

    # Vertical flow line connecting layers on the left
    # Arrow: seq → L1
    draw_arrow(ax, (1.3, 8.50), (1.3, 7.65), lw=1.8)

    # ------------------------------------------------------------------
    # Layer 1: Static AI prediction (top band)
    # ------------------------------------------------------------------
    L1_X, L1_Y, L1_W, L1_H = 0.25, 5.95, 14.5, 1.65
    layer_band(ax, L1_X, L1_Y, L1_W, L1_H, C_L1, C_L1_EDGE,
               "Layer 1  |  Static AI prediction",
               "sequence-only foundation & benchmark models")

    # Five model boxes in a row
    models = [
        ("RNA-FM", "640-dim"),
        ("RiNALMo*", "4-mer 256"),
        ("Evo*", "ERNIE-RNA 768"),
        ("RhoFold+*", "2D 9-dim"),
        ("DeepLncLoc", "3-mer 64"),
    ]
    m_w, m_h = 1.55, 0.75
    start_x = 3.7
    gap = 0.30
    model_centers = []
    for i, (name, sub) in enumerate(models):
        xx = start_x + i * (m_w + gap)
        yy = L1_Y + 0.30
        draw_rounded_box(ax, xx, yy, m_w, m_h, C_MODEL, C_MODEL_EDGE, lw=1.0,
                         rounding=0.05)
        ax.text(xx + m_w / 2, yy + m_h * 0.62, name,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_TEXT)
        ax.text(xx + m_w / 2, yy + m_h * 0.22, sub,
                ha="center", va="center",
                fontsize=7, fontstyle="italic", color="#666666")
        model_centers.append((xx + m_w / 2, yy + m_h / 2))

    # Small caption about point estimate output
    ax.text(L1_X + L1_W - 0.3, L1_Y + 0.18,
            "output: point estimate  (P(stable), logit, or embedding)",
            ha="right", va="center",
            fontsize=7.5, fontstyle="italic", color="#555555")

    # Worked example block on the left of Layer 1
    draw_rounded_box(ax, 0.45, L1_Y + 0.45, 2.7, 0.75, "#FFFFFF", "#555555", lw=0.8,
                     rounding=0.05)
    ax.text(1.8, L1_Y + 1.00, "NEAT1 raw prediction",
            ha="center", fontsize=7.5, fontweight="bold", color=C_TEXT)
    ax.text(1.8, L1_Y + 0.72, f"P(stable) {APPROX} 0.52",
            ha="center", fontsize=8.5, color="#B03030", fontweight="bold")
    ax.text(1.8, L1_Y + 0.52, "(uncertain)",
            ha="center", fontsize=7, fontstyle="italic", color="#777777")

    # ------------------------------------------------------------------
    # Vertical bridge: Layer 1 → Layer 2
    # ------------------------------------------------------------------
    # Center bridge arrow
    draw_arrow(ax, (7.5, 5.90), (7.5, 5.30), color=C_ARROW_DN, lw=2.2, mutation_scale=20)
    ax.text(8.1, 5.60, "condition on measurable biology",
            ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=C_ARROW_DN, fontstyle="italic")

    # ------------------------------------------------------------------
    # Layer 2: Dynamic grounding constraint (middle band)
    # ------------------------------------------------------------------
    L2_X, L2_Y, L2_W, L2_H = 0.25, 3.10, 14.5, 2.15
    layer_band(ax, L2_X, L2_Y, L2_W, L2_H, C_L2, C_L2_EDGE,
               "Layer 2  |  Dynamic grounding constraint",
               "wet-lab priors condition static output")

    # Turnover axis sub-panel
    T_X, T_Y, T_W, T_H = 3.7, L2_Y + 0.35, 4.9, 1.35
    draw_rounded_box(ax, T_X, T_Y, T_W, T_H, "#FFFFFF", C_TURNOVER, lw=1.1)
    ax.text(T_X + T_W / 2, T_Y + T_H - 0.14,
            f"turnover prior  p(t{HALF} | sequence, cell line)",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_TURNOVER)

    # Turnover distribution inset — placed in data coords to stay inside the host box
    t_inset_bounds = [T_X + 0.20, T_Y + 0.28, T_W - 0.40, T_H - 0.58]
    ax_inset_t = ax.inset_axes(t_inset_bounds, transform=ax.transData)
    x_t = np.linspace(-1, 3, 200)
    # unstable: peak around 0.3, stable: peak around 1.8 (log2 hours scale)
    y_u = np.exp(-((x_t - 0.3) ** 2) / (2 * 0.45 ** 2))
    y_s = np.exp(-((x_t - 1.9) ** 2) / (2 * 0.5 ** 2))
    ax_inset_t.fill_between(x_t, 0, y_u, color="#E34A33", alpha=0.35)
    ax_inset_t.fill_between(x_t, 0, y_s, color="#2C7FB8", alpha=0.35)
    ax_inset_t.plot(x_t, y_u, color="#E34A33", lw=1.2)
    ax_inset_t.plot(x_t, y_s, color="#2C7FB8", lw=1.2)
    # Inline labels on the curves (replaces legend to avoid curve overlap)
    ax_inset_t.text(0.30, 1.12, "unstable", ha="center", fontsize=6.5,
                    color="#B0361F", fontweight="bold")
    ax_inset_t.text(1.55, 1.12, "stable", ha="center", fontsize=6.5,
                    color="#1F5E8E", fontweight="bold")
    # Mark NEAT1 position
    neat1_pos = 2.05
    ax_inset_t.axvline(neat1_pos, color="#1F4E79", lw=1.3, linestyle="--")
    ax_inset_t.text(neat1_pos + 0.08, 0.70, "NEAT1",
                    fontsize=7, color="#1F4E79", fontweight="bold",
                    ha="left", va="center")
    ax_inset_t.set_xlim(-1, 3)
    ax_inset_t.set_ylim(0, 1.35)
    tick_positions = [-0.5, 0.5, 1.5, 2.5]
    tick_labels = ["0.7h", "1.4h", "2.8h", "5.7h"]
    ax_inset_t.set_xticks(tick_positions)
    ax_inset_t.set_xticklabels(tick_labels, fontsize=6.2, color=C_TURNOVER)
    ax_inset_t.set_yticks([])
    ax_inset_t.set_xlabel("half-life (log scale)", fontsize=6.2, color=C_TURNOVER, labelpad=1)
    for spine in ax_inset_t.spines.values():
        spine.set_edgecolor(C_TURNOVER)
        spine.set_linewidth(0.7)
    ax_inset_t.tick_params(axis="x", colors=C_TURNOVER, length=2, pad=1)

    # Turnover data source label — placed just below header, above the inset
    ax.text(T_X + T_W / 2, T_Y + T_H - 0.30,
            "data: BRIC-seq, SLAM-seq, TimeLapse-seq",
            ha="center", va="center", fontsize=6.8, fontstyle="italic", color=C_TURNOVER)

    # Localization axis sub-panel
    L_X, L_Y, L_W, L_H = 8.85, L2_Y + 0.35, 3.7, 1.35
    draw_rounded_box(ax, L_X, L_Y, L_W, L_H, "#FFFFFF", C_LOCAL, lw=1.1)
    ax.text(L_X + L_W / 2, L_Y + L_H - 0.14,
            "localization prior  p(nuc | seq)",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_LOCAL)

    # Localization bar inset — data-coord positioning keeps it inside the panel
    l_inset_bounds = [L_X + 0.20, L_Y + 0.28, L_W - 0.40, L_H - 0.58]
    ax_inset_l = ax.inset_axes(l_inset_bounds, transform=ax.transData)
    bin_names = ["cytoplasm", "nuclear", "chromatin"]
    vals = [0.10, 0.55, 0.35]
    colors_bar = ["#BFB2D8", "#8E6BB5", "#5A3C86"]
    xpos = np.arange(len(bin_names))
    ax_inset_l.bar(xpos, vals, color=colors_bar, edgecolor=C_LOCAL, lw=0.8)
    ax_inset_l.set_ylim(0, 0.78)
    ax_inset_l.set_yticks([])
    ax_inset_l.set_xticks(xpos)
    ax_inset_l.set_xticklabels(bin_names, fontsize=6.2, color=C_LOCAL)
    ax_inset_l.tick_params(axis="x", colors=C_LOCAL, length=0, pad=1)
    for spine_name, spine in ax_inset_l.spines.items():
        if spine_name == "bottom":
            spine.set_edgecolor(C_LOCAL)
            spine.set_linewidth(0.7)
        else:
            spine.set_visible(False)
    # Highlight NEAT1's call above the nuclear bar (bar height 0.55 → label at 0.66)
    ax_inset_l.text(1, 0.68, "NEAT1", ha="center", fontsize=7,
                    color=C_LOCAL, fontweight="bold")

    # Localization data source label — placed just below header
    ax.text(L_X + L_W / 2, L_Y + L_H - 0.30,
            "data: CeFra-seq, APEX-seq, Lnc-GPS",
            ha="center", va="center", fontsize=6.8, fontstyle="italic", color=C_LOCAL)

    # Worked example block on the left of Layer 2
    draw_rounded_box(ax, 0.45, L2_Y + 0.35, 2.7, 1.35, "#FFFFFF", "#555555", lw=0.8,
                     rounding=0.05)
    ax.text(1.8, L2_Y + 1.50, "NEAT1 measured priors",
            ha="center", fontsize=7.5, fontweight="bold", color=C_TEXT)
    ax.text(1.8, L2_Y + 1.22, f"t{HALF}  {GTE}  5 h  (BRIC-seq)",
            ha="center", fontsize=8, color=C_TURNOVER, fontweight="bold")
    ax.text(1.8, L2_Y + 0.92, "nuclear, paraspeckle",
            ha="center", fontsize=8, color=C_LOCAL, fontweight="bold")
    ax.text(1.8, L2_Y + 0.62, "(fractionation + FISH)",
            ha="center", fontsize=7, fontstyle="italic", color="#777777")

    # Right-most legend for the 4-quadrant grid in Layer 3
    draw_rounded_box(ax, 12.75, L2_Y + 0.35, 1.85, 1.35, "#FFFFFF", "#777777", lw=0.8,
                     rounding=0.05)
    ax.text(13.67, L2_Y + 1.50, "joint constraint", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(13.67, L2_Y + 1.20, "bin = (stability,", ha="center", fontsize=7.5)
    ax.text(13.67, L2_Y + 0.95, "localization)", ha="center", fontsize=7.5)
    ax.text(13.67, L2_Y + 0.65, f"4 {ARROW} 6 biology bins",
            ha="center", fontsize=7.5, fontstyle="italic", color="#555555")

    # (Arrows from turnover/localization into biology grid removed —
    # the central L2→L3 bridge arrow already communicates the flow,
    # and the panels are visually connected through the grounded band.)

    # ------------------------------------------------------------------
    # Layer 3: Biology-aware output (bottom band)
    # ------------------------------------------------------------------
    L3_X, L3_Y, L3_W, L3_H = 0.25, 0.25, 14.5, 2.15
    layer_band(ax, L3_X, L3_Y, L3_W, L3_H, C_L3, C_L3_EDGE,
               "Layer 3  |  Biology-aware output",
               "calibrated prediction mapped to interpretable bins")

    # 2x2 biology grid (stability × localization)
    grid_x0 = 4.5
    grid_y0 = L3_Y + 0.35
    cell_w = 1.15
    cell_h = 0.55
    rows = ["nuclear", "cytoplasmic"]
    cols = ["unstable", "intermediate", "stable"]

    # header row (stability)
    for ci, col in enumerate(cols):
        ax.text(grid_x0 + cell_w * (ci + 0.5), grid_y0 + cell_h * 2 + 0.20, col,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#555555")
    # side labels (localization)
    for ri, row in enumerate(rows):
        ax.text(grid_x0 - 0.12, grid_y0 + cell_h * (len(rows) - ri - 0.5), row,
                ha="right", va="center",
                fontsize=7.5, fontweight="bold", color="#555555")

    # Cells
    # heatmap-ish probabilities for NEAT1 (strong on stable-nuclear)
    cell_probs = [
        [0.04, 0.10, 0.58],   # nuclear row (top)
        [0.02, 0.05, 0.21],   # cytoplasmic row (bottom)
    ]
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            px = grid_x0 + ci * cell_w
            py = grid_y0 + (len(rows) - ri - 1) * cell_h
            p = cell_probs[ri][ci]
            cell_color = (1.0 - p * 0.6, 0.78 - p * 0.35, 0.45 + p * 0.10)  # peach gradient
            draw_rounded_box(ax, px, py, cell_w * 0.95, cell_h * 0.92, cell_color,
                             C_L3_EDGE, lw=0.8, rounding=0.04)
            label = f"{p:.2f}"
            fw = "bold" if p > 0.25 else "normal"
            color = "#3B2010" if p > 0.25 else "#7A4A1C"
            ax.text(px + cell_w * 0.95 / 2, py + cell_h * 0.92 / 2,
                    label, ha="center", va="center",
                    fontsize=8.5, fontweight=fw, color=color)

    # Outline best cell (nuclear × stable)
    best_x = grid_x0 + 2 * cell_w
    best_y = grid_y0 + 1 * cell_h
    draw_rounded_box(ax, best_x - 0.03, best_y - 0.03, cell_w * 0.95 + 0.06,
                     cell_h * 0.92 + 0.06, "none", "#B03030", lw=1.8,
                     rounding=0.04)

    # Output interpretation
    draw_rounded_box(ax, 9.55, L3_Y + 0.35, 5.05, 1.35, "#FFFFFF", C_L3_EDGE, lw=1.0,
                     rounding=0.05)
    ax.text(12.08, L3_Y + 1.50, "biology-aware prediction",
            ha="center", fontsize=8.5, fontweight="bold", color="#6B4226")
    ax.text(9.75, L3_Y + 1.20,
            f"NEAT1 {ARROW} nuclear, stable  (P = 0.58)",
            ha="left", fontsize=8.5, color="#B03030", fontweight="bold")
    ax.text(9.75, L3_Y + 0.93,
            "role: paraspeckle scaffold (consistent)",
            ha="left", fontsize=7.8, color="#333333")
    ax.text(9.75, L3_Y + 0.68,
            f"calibrated to cell-line-specific t{HALF} distribution",
            ha="left", fontsize=7.3, fontstyle="italic", color="#555555")
    ax.text(9.75, L3_Y + 0.45,
            "uncertainty reported per bin (not just top-1)",
            ha="left", fontsize=7.3, fontstyle="italic", color="#555555")

    # Worked example block on the left of Layer 3
    draw_rounded_box(ax, 0.45, L3_Y + 0.35, 2.7, 1.35, "#FFFFFF", "#555555", lw=0.8,
                     rounding=0.05)
    ax.text(1.8, L3_Y + 1.50, "NEAT1 final call", ha="center",
            fontsize=7.8, fontweight="bold", color=C_TEXT)
    ax.text(1.8, L3_Y + 1.22, "nuclear x stable",
            ha="center", fontsize=8.5, color="#B03030", fontweight="bold")
    ax.text(1.8, L3_Y + 0.98, "P = 0.58",
            ha="center", fontsize=8.5, color="#B03030", fontweight="bold")
    ax.text(1.8, L3_Y + 0.70, f"{APPROX} matches published",
            ha="center", fontsize=7, fontstyle="italic", color="#666666")
    ax.text(1.8, L3_Y + 0.50, "paraspeckle biology",
            ha="center", fontsize=7, fontstyle="italic", color="#666666")

    # Vertical flow arrows on the left connecting all three example blocks
    draw_arrow(ax, (1.8, 5.90), (1.8, 5.30), color=C_ARROW, lw=1.6, mutation_scale=14)
    draw_arrow(ax, (1.8, 3.05), (1.8, 2.48), color=C_ARROW, lw=1.6, mutation_scale=14)
    # Layer 2 → Layer 3 central bridge
    draw_arrow(ax, (7.5, 3.05), (7.5, 2.48), color=C_ARROW_OUT, lw=2.2, mutation_scale=20)
    ax.text(8.1, 2.77, "project onto biology bins",
            ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=C_ARROW_OUT, fontstyle="italic")

    # Asterisk note for CPU substitutes
    ax.text(7.5, 0.05,
            "* Phase 2 CPU-feasible substitutes (RiNALMo 650M -> 4-mer; Evo 7B -> ERNIE-RNA 86M; "
            "RhoFold+ -> ViennaRNA 2D). "
            "Same framework applies to full-scale models in Phase 3.",
            ha="center", va="center",
            fontsize=7.5, fontstyle="italic", color="#666666")

    fig.savefig(OUT_DIR / "fig5_framework.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig5_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_DIR / 'fig5_framework.pdf'}")
    print(f"Wrote {OUT_DIR / 'fig5_framework.png'}")


if __name__ == "__main__":
    main()
