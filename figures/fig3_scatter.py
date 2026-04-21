#!/usr/bin/env python3
"""
fig3_scatter.py — Fig.3 予測 vs 実測 散布図（Task 3.3）

目的:
    5モデル横並びで、連続値回帰の予測 log2(t½) vs 実測 log2(t½) を散布図化する。
    既知 stable/unstable lncRNA を色分けハイライト。

入力:
    ../benchmark/results/predictions/*.csv  (regression pred per model, if produced)
    OR
    ../benchmark/results/embeddings/*.npz + ../data/processed/test_set_final.csv
      → スクリプト内で RidgeRegressor を 5-fold OOF で再走行して予測を得る

出力:
    fig3_scatter.pdf
    fig3_scatter.png (300 dpi)

強調lncRNA:
    stable  : NEAT1, MALAT1, KCNQ1OT1
    unstable: FIRRE, LINC-PINT
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
HIGHLIGHT_STABLE = {"NEAT1", "MALAT1", "KCNQ1OT1"}
HIGHLIGHT_UNSTABLE = {"FIRRE", "LINC-PINT"}


def load_embeddings(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return data["gene_ids"], data["embeddings"]


def oof_regression_predictions(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> np.ndarray:
    from sklearn.model_selection import KFold

    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
    from classifiers import RidgeRegressor

    pred = np.full(len(y), np.nan)
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X):
        model = RidgeRegressor().fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark" / "results" / "embeddings",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "test_set_final.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(__file__).parent / "fig3_scatter.pdf",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(__file__).parent / "fig3_scatter.png",
    )
    args = parser.parse_args()

    if not args.test_csv.exists():
        log.error(f"Missing {args.test_csv}")
        sys.exit(1)

    test_df = pd.read_csv(args.test_csv)
    test_df["composite_key"] = (
        test_df["gencode_gene_id"].astype(str)
        + "|" + test_df["gencode_gene_symbol"].astype(str)
        + "|" + test_df["cell_line"].astype(str)
    )
    y_true_map = dict(zip(test_df["composite_key"], test_df["half_life_log2"]))
    symbol_map = dict(zip(test_df["composite_key"], test_df["gencode_gene_symbol"].str.upper()))

    # collect (gids, X, y) per model
    per_model = {}
    for m in MODEL_ORDER:
        npz = args.embeddings_dir / f"{m}.npz"
        if not npz.exists():
            log.warning(f"Skipping {m}: {npz} missing")
            continue
        gids, emb = load_embeddings(npz)
        gids = np.array([str(g) for g in gids])
        mask = np.array([g in y_true_map for g in gids])
        gids, emb = gids[mask], emb[mask]
        y = np.array([y_true_map[g] for g in gids])
        per_model[m] = (gids, emb, y)

    if not per_model:
        log.error("No model embeddings found. Run Phase 2 first.")
        sys.exit(1)

    ncols = len(per_model)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.4), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (m, (gids, X, y)) in zip(axes, per_model.items()):
        pred = oof_regression_predictions(X, y)
        rho = spearmanr(y, pred)[0] if len(y) >= 3 else np.nan

        syms = np.array([symbol_map.get(g, "") for g in gids])
        base_mask = np.array(
            [s not in HIGHLIGHT_STABLE and s not in HIGHLIGHT_UNSTABLE for s in syms]
        )
        stable_mask = np.array([s in HIGHLIGHT_STABLE for s in syms])
        unstable_mask = np.array([s in HIGHLIGHT_UNSTABLE for s in syms])

        ax.scatter(y[base_mask], pred[base_mask], s=15, alpha=0.45, color="grey", edgecolor="none")
        if stable_mask.any():
            ax.scatter(
                y[stable_mask], pred[stable_mask], s=48, color="#2c7fb8", label="stable reference", edgecolor="white"
            )
            for gi in np.where(stable_mask)[0]:
                ax.annotate(syms[gi], (y[gi], pred[gi]), fontsize=7, alpha=0.8)
        if unstable_mask.any():
            ax.scatter(
                y[unstable_mask], pred[unstable_mask], s=48, color="#e34a33", label="unstable reference", edgecolor="white"
            )
            for gi in np.where(unstable_mask)[0]:
                ax.annotate(syms[gi], (y[gi], pred[gi]), fontsize=7, alpha=0.8)

        lo, hi = min(y.min(), pred.min()) - 0.5, max(y.max(), pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=0.7)
        ax.set_title(f"{MODEL_LABELS.get(m, m)}\nρ = {rho:.2f}")
        ax.set_xlabel("Observed log2(t½)")
        if ax is axes[0]:
            ax.set_ylabel("Predicted log2(t½)")

    axes[-1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    log.info(f"Wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
