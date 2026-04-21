#!/usr/bin/env python3
"""
fig2_auroc_heatmap.py — Fig.2 AUROC ヒートマップ（Task 3.2）

目的:
    5モデル × 3データセット（HeLa/mESC/K562）の AUROC を
    ヒートマップで可視化する。5-fold stratified CV と
    leave-one-cell-out CV を別パネルで示す。

入力:
    ../benchmark/results/metrics_summary.csv   (eval.py の出力)

出力:
    fig2_auroc_heatmap.pdf
    fig2_auroc_heatmap.png  (300 dpi)

配色（plan.md Step 3.2.2）:
    - AUROC 0.5 (gray/baseline) → 0.9 (viridis bright)
    - 0.05 刻みで annotation

使い方:
    python fig2_auroc_heatmap.py
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ORDER = ["rna_fm", "rinalmo", "evo", "rhofold_plus", "deeplncloc"]
MODEL_LABELS = {
    "rna_fm": "RNA-FM",
    "rinalmo": "RiNALMo (4-mer)*",
    "evo": "Evo (ERNIE-RNA)*",
    "rhofold_plus": "RhoFold+ (ViennaRNA)*",
    "deeplncloc": "DeepLncLoc (3-mer)",
}


def build_heatmap_matrix(
    df: pd.DataFrame, cv_scheme: str, classifier: str = "logreg"
) -> pd.DataFrame:
    """Pivot to (model × held_out_cell_line) of mean AUROC from metrics_table.csv rows."""
    sub = df[
        (df["task"] == "classification")
        & (df["metric"] == "AUROC")
        & (df["cv_scheme"] == cv_scheme)
        & (df["classifier"] == classifier)
    ]
    if cv_scheme == "5fold_stratified":
        pivot = sub.groupby("model")["value"].mean().to_frame("AUROC").reindex(MODEL_ORDER)
        return pivot
    return (
        sub.pivot_table(values="value", index="model", columns="held_out", aggfunc="mean")
        .reindex(MODEL_ORDER)
    )


def plot_heatmap(data: pd.DataFrame, ax, title: str) -> None:
    labels = [MODEL_LABELS.get(m, m) for m in data.index]
    sns.heatmap(
        data.values,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.5,
        vmax=0.9,
        cbar_kws={"label": "AUROC"},
        xticklabels=data.columns if len(data.columns) > 1 else ["5-fold CV"],
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Held-out cell line" if len(data.columns) > 1 else "")
    ax.set_ylabel("Model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark" / "results" / "metrics_table.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(__file__).parent / "fig2_auroc_heatmap.pdf",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(__file__).parent / "fig2_auroc_heatmap.png",
    )
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "mlp"])
    args = parser.parse_args()

    if not args.metrics_csv.exists():
        log.error(f"Missing {args.metrics_csv}. Run benchmark/eval.py first.")
        sys.exit(1)

    df = pd.read_csv(args.metrics_csv)
    log.info(f"Loaded {len(df)} summary rows")

    fold_data = build_heatmap_matrix(df, "5fold_stratified", args.classifier)
    loco_data = build_heatmap_matrix(df, "leave_one_cell_out", args.classifier)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]})
    plot_heatmap(fold_data, axes[0], "5-fold stratified CV")
    plot_heatmap(loco_data, axes[1], "Leave-one-cell-out CV")
    fig.suptitle(
        f"Binary classification AUROC — {args.classifier.upper()} head",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    log.info(f"Wrote {args.output_pdf}")
    log.info(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
