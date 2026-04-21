#!/usr/bin/env python3
"""
supp_fig1_length_tertile.py — Supplementary Fig. 1

Length-stratified classification performance across the five representation
classes, binned by transcript-length tertile (short < 2,387 nt; mid
2,387–4,159 nt; long > 4,159 nt). Data source: ablation_results.csv
(length stratum only).

Output:
    supp_fig1_length_tertile.pdf
    supp_fig1_length_tertile.png  (300 dpi)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["rna_fm", "rinalmo", "evo", "rhofold_plus", "deeplncloc"]
MODEL_LABELS = {
    "rna_fm": "RNA-FM",
    "rinalmo": "RiNALMo (shallow-CNN proxy)",
    "evo": "Evo (ERNIE-RNA proxy)",
    "rhofold_plus": "RhoFold+ (ViennaRNA)",
    "deeplncloc": "DeepLncLoc",
}
STRATA = ["short", "mid", "long"]
STRATA_LABELS = {
    "short": "short\n(<2,387 nt)",
    "mid": "mid\n(2,387–4,159 nt)",
    "long": "long\n(>4,159 nt)",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def main() -> None:
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "benchmark" / "results" / "ablation_results.csv")
    df = df[df["stratum"] == "length"].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)

    x = np.arange(len(STRATA))
    width = 0.16
    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("value")
        vals = [sub.loc[s, "AUROC"] if s in sub.index else np.nan for s in STRATA]
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset, vals, width, label=MODEL_LABELS[model],
            color=COLORS[i], edgecolor="black", linewidth=0.5,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATA_LABELS[s] for s in STRATA])
    ax.set_ylabel("AUROC (5-fold stratified CV, MLP)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Length-stratified classification performance")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, fontsize=8,
    )
    fig.tight_layout()

    out_pdf = root / "figures" / "supp_fig1_length_tertile.pdf"
    out_png = root / "figures" / "supp_fig1_length_tertile.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
