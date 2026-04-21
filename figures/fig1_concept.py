#!/usr/bin/env python3
"""
fig1_concept.py — Fig.1 Conceptual schematic (Task 3.1)

From "black box" sequence models → BRIC-seq/SLAM-seq dynamic ground truth
→ biology-aware predictor. Pure matplotlib, no BioRender dependency.

Output:
    fig1_concept.pdf
    fig1_concept.png (300 dpi)
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

# Muted scientific palette
C_BLACK_BOX = "#2B2B2B"
C_BLACK_BOX_EDGE = "#1A1A1A"
C_MODEL = "#E8E8E8"
C_MODEL_EDGE = "#999999"
C_GROUND = "#D9E7F5"
C_GROUND_EDGE = "#5B8BB5"
C_DYNAMIC = "#C9E4CA"
C_DYNAMIC_EDGE = "#4A8B5C"
C_OUT = "#F5D0A5"
C_OUT_EDGE = "#C2803A"
C_ARROW = "#555555"
C_ARROW_HL = "#D64545"
C_ARROW_GR = "#4A8B5C"
C_TEXT = "#222222"
ARROW = "\u2192"  # DejaVu supports this


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, text=None,
                     fontsize=9, fontweight="normal", textcolor=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
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


def main():
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        7.5, 7.65,
        "From black boxes to biology: grounding AI predictions in RNA turnover",
        ha="center", va="center",
        fontsize=13.5, fontweight="bold", color=C_TEXT,
    )

    # ------------------------------------------------------------------
    # Panel A: Current static AI (left)
    # ------------------------------------------------------------------
    ax.text(2.4, 6.9, "A. Current static AI", ha="center", fontsize=11.5, fontweight="bold", color=C_TEXT)

    # Sequence input
    ax.text(1.35, 6.3, "RNA sequence", ha="center", fontsize=8.5, color="#555555", style="italic")
    draw_rounded_box(ax, 0.3, 5.65, 2.1, 0.42, "#FFFFFF", "#BBBBBB", lw=0.9,
                     text="AUCGGCUAACGUAAUGCA...", fontsize=7.5)

    # Black box container
    bb_x, bb_y, bb_w, bb_h = 0.3, 1.7, 4.2, 3.5
    draw_rounded_box(ax, bb_x, bb_y, bb_w, bb_h, C_BLACK_BOX, C_BLACK_BOX_EDGE, lw=1.5)
    ax.text(bb_x + bb_w / 2, bb_y + bb_h - 0.3, "black box", ha="center", va="center",
            fontsize=10.5, fontstyle="italic", color="#EFEFEF", fontweight="bold")

    # 5 model boxes in clean 2-column grid inside black box (no size labels for clarity)
    model_rows = [
        ["RNA-FM", "RiNALMo"],
        ["Evo", "RhoFold+"],
        ["DeepLncLoc"],
    ]
    model_w, model_h = 1.75, 0.48
    row_gap = 0.62
    col_x = [bb_x + 0.25, bb_x + 0.25 + model_w + 0.2]
    top_row_y = bb_y + 2.45
    for r, row in enumerate(model_rows):
        yy = top_row_y - r * row_gap
        for c, name in enumerate(row):
            xx = col_x[c]
            draw_rounded_box(ax, xx, yy, model_w, model_h, C_MODEL, C_MODEL_EDGE, lw=0.8,
                             text=name, fontsize=8.8, fontweight="bold")

    # ? at bottom, indicating unknown output
    ax.text(bb_x + bb_w / 2, bb_y + 0.45, "?", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#E74C3C")
    ax.text(bb_x + bb_w / 2, bb_y + 0.15, "turnover-blind", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#D0D0D0")

    # Arrow: sequence → black box
    draw_arrow(ax, (1.35, 5.6), (1.35, bb_y + bb_h + 0.05), color=C_ARROW, lw=1.8)

    # Arrow: black box → output
    ax.text(2.4, 1.35, "function prediction\nof unknown biological validity",
            ha="center", va="top", fontsize=8, color="#888888", fontstyle="italic")

    # ------------------------------------------------------------------
    # Panel B: Dynamic ground truth (middle)
    # ------------------------------------------------------------------
    ax.text(7.5, 6.9, "B. Dynamic ground truth", ha="center", fontsize=11.5, fontweight="bold", color=C_TEXT)

    gt_x, gt_y, gt_w, gt_h = 5.3, 1.7, 4.4, 4.5
    draw_rounded_box(ax, gt_x, gt_y, gt_w, gt_h, C_GROUND, C_GROUND_EDGE, lw=1.4)
    ax.text(gt_x + gt_w / 2, gt_y + gt_h - 0.35,
            "RNA turnover & localization\nmeasurements",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1F4E79")

    # BRIC-seq decay inset
    ax_inset1 = fig.add_axes([0.42, 0.40, 0.1, 0.14])
    t = np.linspace(0, 1, 40)
    ax_inset1.plot(t, 2 ** (-t / 0.35), color="#2C7FB8", lw=1.8, label="stable")
    ax_inset1.plot(t, 2 ** (-t / 0.12), color="#E34A33", lw=1.8, label="unstable")
    ax_inset1.set_xticks([])
    ax_inset1.set_yticks([])
    ax_inset1.set_title("BRIC-seq decay", fontsize=7.5, fontweight="bold", color="#1F4E79", pad=2)
    ax_inset1.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax_inset1.text(0.5, -0.18, "time after 5-BrU", ha="center", transform=ax_inset1.transAxes, fontsize=6, color="#1F4E79")
    ax_inset1.text(-0.08, 0.5, "fraction", ha="right", va="center", rotation=90,
                   transform=ax_inset1.transAxes, fontsize=6, color="#1F4E79")
    for spine in ax_inset1.spines.values():
        spine.set_edgecolor("#5B8BB5")
        spine.set_linewidth(0.7)

    # SLAM-seq schematic
    ax_inset2 = fig.add_axes([0.545, 0.40, 0.1, 0.14])
    positions = np.linspace(0.08, 0.92, 12)
    labels = list("AUGCAUGCAUGC")
    convert_idx = [1, 5, 9]
    for i, (p, lab) in enumerate(zip(positions, labels)):
        if lab == "U":
            color = "#E34A33" if i in convert_idx else "#888888"
            ax_inset2.text(p, 0.68, lab, fontsize=7, ha="center", va="center",
                           color=color, fontweight="bold" if i in convert_idx else "normal")
            if i in convert_idx:
                ax_inset2.annotate("", xy=(p, 0.28), xytext=(p, 0.58),
                                   arrowprops=dict(arrowstyle="->", color="#E34A33", lw=0.8))
                ax_inset2.text(p, 0.16, "C", fontsize=7, ha="center", va="center",
                               color="#E34A33", fontweight="bold")
        else:
            ax_inset2.text(p, 0.68, lab, fontsize=7, ha="center", va="center", color="#888888")
    ax_inset2.set_xlim(0, 1)
    ax_inset2.set_ylim(0, 1)
    ax_inset2.set_title(f"SLAM-seq T{ARROW}C", fontsize=7.5, fontweight="bold", color="#1F4E79", pad=2)
    ax_inset2.axis("off")
    rect = mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                              edgecolor="#5B8BB5", lw=0.7, transform=ax_inset2.transAxes)
    ax_inset2.add_patch(rect)

    # Half-life axis at bottom of Panel B
    ax.text(gt_x + gt_w / 2, gt_y + 0.9, "half-life axis",
            ha="center", fontsize=8, fontweight="bold", color="#1F4E79")
    ax.plot([gt_x + 0.3, gt_x + gt_w - 0.3], [gt_y + 0.55, gt_y + 0.55], color="#1F4E79", lw=2)
    for frac, lab in [(0.0, "0.5h"), (0.25, "2h"), (0.5, "8h"), (0.75, "24h"), (1.0, ">72h")]:
        xp = gt_x + 0.3 + frac * (gt_w - 0.6)
        ax.plot([xp, xp], [gt_y + 0.5, gt_y + 0.6], color="#1F4E79", lw=1.2)
        ax.text(xp, gt_y + 0.28, lab, ha="center", fontsize=7, color="#1F4E79")
    ax.text(gt_x + 0.35, gt_y + 0.68, "unstable", fontsize=7, color="#E34A33", fontweight="bold")
    ax.text(gt_x + gt_w - 0.8, gt_y + 0.68, "stable", fontsize=7, color="#2C7FB8", fontweight="bold")

    # ------------------------------------------------------------------
    # Panel C: Biology-aware predictor (right)
    # ------------------------------------------------------------------
    ax.text(12.6, 6.9, "C. Biology-aware predictor", ha="center", fontsize=11.5, fontweight="bold", color=C_TEXT)

    bap_x, bap_y, bap_w, bap_h = 10.5, 1.7, 4.2, 4.5
    draw_rounded_box(ax, bap_x, bap_y, bap_w, bap_h, C_DYNAMIC, C_DYNAMIC_EDGE, lw=1.5)
    ax.text(bap_x + bap_w / 2, bap_y + bap_h - 0.35, "dynamic grounding",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#2B5A36")

    draw_rounded_box(ax, bap_x + 0.3, bap_y + 2.9, bap_w - 0.6, 0.55, "#FFFFFF", "#4A8B5C", lw=1.0,
                     text=f"turnover axis (t\u00bd)",
                     fontsize=8.5, fontweight="bold", textcolor="#2B5A36")
    draw_rounded_box(ax, bap_x + 0.3, bap_y + 2.15, bap_w - 0.6, 0.55, "#FFFFFF", "#4A8B5C", lw=1.0,
                     text="localization axis (nuc/cyto)",
                     fontsize=8.5, fontweight="bold", textcolor="#2B5A36")
    draw_rounded_box(ax, bap_x + 0.3, bap_y + 1.1, bap_w - 0.6, 0.8, C_OUT, C_OUT_EDGE, lw=1.2,
                     text="function prediction\n(tiered: static x dynamic)",
                     fontsize=8, fontweight="bold")

    ax.text(bap_x + bap_w / 2, bap_y + 0.55,
            f"NEAT1 {ARROW} nuclear scaffold, stable",
            ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#6B4226")
    ax.text(bap_x + bap_w / 2, bap_y + 0.27,
            f"FIRRE {ARROW} nuclear, unstable",
            ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#6B4226")

    # Inter-panel arrows — placed above the insets to avoid overlap
    draw_arrow(ax, (4.55, 5.7), (5.2, 5.7), color=C_ARROW_HL, lw=2.2, mutation_scale=20)
    ax.text(4.88, 5.95, "ground", ha="center", fontsize=8, color=C_ARROW_HL, fontweight="bold")

    draw_arrow(ax, (9.75, 5.7), (10.45, 5.7), color=C_ARROW_GR, lw=2.2, mutation_scale=20)
    ax.text(10.1, 5.95, "condition", ha="center", fontsize=8, color=C_ARROW_GR, fontweight="bold")

    # Bottom caption
    ax.text(7.5, 0.55,
            "Five publicly released RNA AI models (Panel A) predict lncRNA function from sequence alone, "
            "missing the dynamic axis (turnover, localization) captured by BRIC-seq / SLAM-seq (Panel B). "
            "Dynamic grounding (Panel C) conditions static predictions on measurable biological state.",
            ha="center", va="center", fontsize=8.5, color="#555555", wrap=True)

    fig.savefig(OUT_DIR / "fig1_concept.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "fig1_concept.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_DIR / 'fig1_concept.pdf'}")
    print(f"Wrote {OUT_DIR / 'fig1_concept.png'}")


if __name__ == "__main__":
    main()
