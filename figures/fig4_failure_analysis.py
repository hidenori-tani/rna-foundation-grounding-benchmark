#!/usr/bin/env python3
"""
fig4_failure_analysis.py — Fig.4 失敗事例分析（Task 3.4）

目的:
    5モデル全てで予測を外した lncRNA（consensus failure）の特徴量を比較し、
    何がAIに解釈できていないかを可視化する。

出力パネル:
    A) Consensus failure の配列長・GC含量・局在スコアの violin/box
       （成功群との比較）
    B) モデル A / モデル B の disagreement 4象限マトリクス
       （例：RNA-FM正解 × Evo正解 の 2×2 分割）

入力:
    ../benchmark/results/metrics_table.csv (foldごとの生データ)
    ../benchmark/results/embeddings/*.npz
    ../data/processed/test_set_final.csv
    ../data/processed/test_set_sequences.fa

出力:
    fig4_failure_analysis.pdf
    fig4_failure_analysis.png (300 dpi)
    fig4_failure_table.csv (consensus failure リスト)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ORDER = ["rna_fm", "rinalmo", "evo", "rhofold_plus", "deeplncloc"]
MODEL_LABELS = {
    "rna_fm": "RNA-FM",
    "rinalmo": "RiNALMo (shallow CNN)*",
    "evo": "Evo (ERNIE-RNA)*",
    "rhofold_plus": "RhoFold+ (ViennaRNA)*",
    "deeplncloc": "DeepLncLoc (3-mer)",
}
SEED = 42


def parse_fasta(path: Path) -> dict[str, str]:
    seqs, cur_id, cur_seq = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].strip().split("|")[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


def oof_classification(X: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """Return out-of-fold predicted labels."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
    from classifiers import LogisticRegressionClassifier

    pred = np.full(len(y), -1)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for tr, te in splitter.split(X, y):
        clf = LogisticRegressionClassifier().fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
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
        "--sequences-fa",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "test_set_sequences.fa",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(__file__).parent / "fig4_failure_analysis.pdf",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path(__file__).parent / "fig4_failure_table.csv",
    )
    args = parser.parse_args()

    if not args.test_csv.exists():
        log.error(f"Missing {args.test_csv}")
        sys.exit(1)

    test_df = pd.read_csv(args.test_csv)
    classifiable = (
        test_df[test_df["label_binary"].isin(["stable", "unstable"])]
        .copy()
        .reset_index(drop=True)
    )
    classifiable["label_int"] = (classifiable["label_binary"] == "stable").astype(int)
    classifiable["composite_key"] = (
        classifiable["gencode_gene_id"].astype(str)
        + "|" + classifiable["gencode_gene_symbol"].astype(str)
        + "|" + classifiable["cell_line"].astype(str)
    )

    seqs = parse_fasta(args.sequences_fa)
    classifiable["seq"] = classifiable["gencode_gene_id"].map(seqs)
    classifiable = classifiable.dropna(subset=["seq"])
    classifiable["length"] = classifiable["seq"].str.len()
    classifiable["gc"] = classifiable["seq"].str.upper().apply(
        lambda s: (s.count("G") + s.count("C")) / max(len(s), 1)
    )

    # Per-model OOF predictions
    pred_cols = {}
    for m in MODEL_ORDER:
        npz = args.embeddings_dir / f"{m}.npz"
        if not npz.exists():
            log.warning(f"Skipping {m}: {npz} missing")
            continue
        data = np.load(npz, allow_pickle=True)
        gid_to_idx = {str(g): i for i, g in enumerate(data["gene_ids"])}
        mask = classifiable["composite_key"].isin(gid_to_idx)
        sub = classifiable[mask].reset_index(drop=True)
        if sub.empty:
            continue
        X = data["embeddings"][[gid_to_idx[g] for g in sub["composite_key"]]]
        y = sub["label_int"].values
        pred = oof_classification(X, y)
        col = f"pred_{m}"
        pred_cols[col] = dict(zip(sub["composite_key"], pred))

    for col, d in pred_cols.items():
        classifiable[col] = classifiable["composite_key"].map(d)

    pred_col_names = list(pred_cols.keys())
    if not pred_col_names:
        log.error("No pred columns. Phase 2 incomplete.")
        sys.exit(1)

    classifiable["n_correct"] = sum(
        (classifiable[c] == classifiable["label_int"]).astype(int) for c in pred_col_names
    )
    classifiable["consensus_failure"] = classifiable["n_correct"] == 0

    # Save failure table
    fail = classifiable[classifiable["consensus_failure"]][
        ["gencode_gene_id", "gencode_gene_symbol", "cell_line", "half_life_h", "label_binary", "length", "gc"]
    ]
    fail.to_csv(args.output_table, index=False)
    log.info(f"Consensus failures: {len(fail)}")
    log.info(f"Wrote {args.output_table}")

    # Panel A: feature comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, ylabel in zip(axes, ["length", "gc", "half_life_h"], ["Length (nt)", "GC fraction", "t½ (h)"]):
        groups = [
            classifiable[classifiable["consensus_failure"]][col],
            classifiable[~classifiable["consensus_failure"]][col],
        ]
        ax.boxplot(groups, labels=["consensus\nfailure", "other"])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if col == "length":
            ax.set_yscale("log")

    fig.suptitle("A. Feature comparison: consensus-failure vs others", y=1.03)
    fig.tight_layout()
    fig.savefig(args.output_pdf, bbox_inches="tight")
    png_path = args.output_pdf.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    log.info(f"Wrote {png_path}")

    # Panel B: disagreement matrix (first two models)
    if len(pred_col_names) >= 2:
        m1, m2 = pred_col_names[:2]
        correct1 = classifiable[m1] == classifiable["label_int"]
        correct2 = classifiable[m2] == classifiable["label_int"]
        matrix = np.array(
            [
                [(correct1 & correct2).sum(), (correct1 & ~correct2).sum()],
                [(~correct1 & correct2).sum(), (~correct1 & ~correct2).sum()],
            ]
        )
        panel_b = args.output_pdf.with_name(args.output_pdf.stem + "_panelB.pdf")
        fig2, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{MODEL_LABELS.get(m2.replace('pred_', ''), m2)}\ncorrect",
                            f"{MODEL_LABELS.get(m2.replace('pred_', ''), m2)}\nwrong"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"{MODEL_LABELS.get(m1.replace('pred_', ''), m1)}\ncorrect",
                            f"{MODEL_LABELS.get(m1.replace('pred_', ''), m1)}\nwrong"])
        ax.set_title(f"B. Disagreement: {MODEL_LABELS.get(m1.replace('pred_', ''), m1)} × {MODEL_LABELS.get(m2.replace('pred_', ''), m2)}")
        fig2.tight_layout()
        fig2.savefig(panel_b, bbox_inches="tight")
        panel_b_png = panel_b.with_suffix(".png")
        fig2.savefig(panel_b_png, bbox_inches="tight", dpi=300)
        log.info(f"Wrote {panel_b}")
        log.info(f"Wrote {panel_b_png}")


if __name__ == "__main__":
    main()
