#!/usr/bin/env python3
"""
eval.py — Cross-validation + 評価指標 + 5モデル × 2タスク × 2CV 走行（Task 2.7 + 2.8）

入力:
    benchmark/results/embeddings/*.npz (各モデルの gene_ids + embeddings)
    data/processed/test_set_final.csv (gene_id, label_binary, half_life_log2, cell_line)

出力:
    benchmark/results/metrics_table.csv
        columns: model, task, cv_scheme, metric, value, fold, cell_line_held_out

評価プロトコル（spec §4.2）:
    - 5-fold stratified CV（label_binary ベース、excluded は除外）
    - Leave-one-cell-out CV（HeLa / mESC / K562 など）
    - 分類指標: AUROC, F1, MCC
    - 回帰指標: Spearman, Pearson, RMSE
    - 軽量分類器 2 種（LR, MLP）× embedding 5モデル

Seed固定: 42（plan.md Step 2.7.3）

使い方:
    python eval.py --embeddings-dir results/embeddings --test-csv ../data/processed/test_set_final.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
MODEL_FILES = ["rna_fm", "rinalmo", "evo", "rhofold_plus", "deeplncloc"]


def load_embeddings(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (gene_ids, embeddings)."""
    data = np.load(npz_path, allow_pickle=True)
    return data["gene_ids"], data["embeddings"]


def align_data(
    gene_ids: np.ndarray, emb: np.ndarray, test_df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align embedding rows to test_df rows.

    npz の gene_ids は '{gid}|{symbol}|{cell_line}' の複合キー
    （per-cell-line サンプルのため同一 gene_id が複数出現する）。
    test_df 側にも同じ複合キーを作ってマッチする。
    """
    gid_to_idx = {str(g): i for i, g in enumerate(gene_ids)}
    match_key = (
        test_df["gencode_gene_id"].astype(str)
        + "|" + test_df["gencode_gene_symbol"].astype(str)
        + "|" + test_df["cell_line"].astype(str)
    )
    keep_mask = match_key.isin(gid_to_idx)
    sub = test_df[keep_mask].reset_index(drop=True)
    sub_keys = (
        sub["gencode_gene_id"].astype(str)
        + "|" + sub["gencode_gene_symbol"].astype(str)
        + "|" + sub["cell_line"].astype(str)
    )
    rows = [gid_to_idx[k] for k in sub_keys]
    return emb[rows], sub


def metrics_classification(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import (
        roc_auc_score, f1_score, matthews_corrcoef
    )

    y_pred = (y_score >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else np.nan,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error

    return {
        "Pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    classifier_factory,
    k: int = 5,
) -> list[dict]:
    """Stratified K-Fold on classification; regular K-Fold for regression."""
    from sklearn.model_selection import KFold, StratifiedKFold

    results = []
    if task == "classification":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=SEED)
        splits = splitter.split(X)

    for fold, (tr, te) in enumerate(splits):
        model = classifier_factory()
        model.fit(X[tr], y[tr])
        if task == "classification":
            y_score = model.predict_proba(X[te])[:, 1]
            m = metrics_classification(y[te], y_score)
        else:
            y_pred = model.predict(X[te])
            m = metrics_regression(y[te], y_pred)
        for metric, value in m.items():
            results.append({"fold": fold, "metric": metric, "value": value, "held_out": None})
    return results


def run_leave_one_cell_out(
    X: np.ndarray,
    y: np.ndarray,
    cell_lines: np.ndarray,
    task: str,
    classifier_factory,
) -> list[dict]:
    results = []
    for cl in np.unique(cell_lines):
        tr = cell_lines != cl
        te = cell_lines == cl
        if te.sum() < 5 or len(set(y[tr])) < 2:
            log.warning(f"Skipping LOCO held-out {cl}: insufficient data")
            continue
        model = classifier_factory()
        model.fit(X[tr], y[tr])
        if task == "classification":
            y_score = model.predict_proba(X[te])[:, 1]
            m = metrics_classification(y[te], y_score)
        else:
            y_pred = model.predict(X[te])
            m = metrics_regression(y[te], y_pred)
        for metric, value in m.items():
            results.append({"fold": -1, "metric": metric, "value": value, "held_out": cl})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "embeddings",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "test_set_final.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).parent / "results" / "metrics_table.csv",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "mlp", "both"],
        default="both",
    )
    args = parser.parse_args()

    if not args.test_csv.exists():
        log.error(f"Missing test_set_final.csv: {args.test_csv}")
        sys.exit(1)

    from classifiers import (
        LogisticRegressionClassifier,
        MLPClassifier,
        RidgeRegressor,
        MLPRegressor,
    )

    test_df = pd.read_csv(args.test_csv)
    classifiable = test_df[test_df["label_binary"].isin(["stable", "unstable"])].copy()
    classifiable["label_int"] = (classifiable["label_binary"] == "stable").astype(int)

    rows: list[dict] = []

    for model_name in MODEL_FILES:
        npz = args.embeddings_dir / f"{model_name}.npz"
        if not npz.exists():
            log.warning(f"Skipping {model_name}: embedding not found at {npz}")
            continue
        gids, emb = load_embeddings(npz)
        log.info(f"{model_name}: loaded {emb.shape}")

        # Classification (binary stable vs unstable)
        X_cls, cls_df = align_data(gids, emb, classifiable)
        y_cls = cls_df["label_int"].values
        cl_cls = cls_df["cell_line"].values

        # Regression (log2 half-life, all classifiable + excluded)
        X_reg, reg_df = align_data(gids, emb, test_df)
        y_reg = reg_df["half_life_log2"].values
        cl_reg = reg_df["cell_line"].values

        clf_factories = {}
        if args.classifier in ("logreg", "both"):
            clf_factories["logreg"] = lambda: LogisticRegressionClassifier()
        if args.classifier in ("mlp", "both"):
            clf_factories["mlp"] = lambda: MLPClassifier(in_dim=X_cls.shape[1], n_classes=2, epochs=100)

        reg_factories = {}
        if args.classifier in ("logreg", "both"):
            reg_factories["ridge"] = lambda: RidgeRegressor()
        if args.classifier in ("mlp", "both"):
            reg_factories["mlp"] = lambda: MLPRegressor(in_dim=X_reg.shape[1], epochs=100)

        # 5-fold CV (classification)
        for clf_name, factory in clf_factories.items():
            out = run_stratified_kfold(X_cls, y_cls, "classification", factory, k=5)
            for r in out:
                rows.append(
                    dict(
                        model=model_name,
                        task="classification",
                        cv_scheme="5fold_stratified",
                        classifier=clf_name,
                        **r,
                    )
                )
            out = run_leave_one_cell_out(X_cls, y_cls, cl_cls, "classification", factory)
            for r in out:
                rows.append(
                    dict(
                        model=model_name,
                        task="classification",
                        cv_scheme="leave_one_cell_out",
                        classifier=clf_name,
                        **r,
                    )
                )

        # 5-fold CV (regression)
        for reg_name, factory in reg_factories.items():
            out = run_stratified_kfold(X_reg, y_reg, "regression", factory, k=5)
            for r in out:
                rows.append(
                    dict(
                        model=model_name,
                        task="regression",
                        cv_scheme="5fold",
                        classifier=reg_name,
                        **r,
                    )
                )
            out = run_leave_one_cell_out(X_reg, y_reg, cl_reg, "regression", factory)
            for r in out:
                rows.append(
                    dict(
                        model=model_name,
                        task="regression",
                        cv_scheme="leave_one_cell_out",
                        classifier=reg_name,
                        **r,
                    )
                )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info(f"Wrote {args.output_csv} ({len(df)} rows)")

    # Summary: per (model, task, cv_scheme, metric) mean±std
    if len(df):
        summary = (
            df.groupby(["model", "task", "cv_scheme", "classifier", "metric"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary_path = args.output_csv.with_name("metrics_summary.csv")
        summary.to_csv(summary_path, index=False)
        log.info(f"Wrote {summary_path}")

        # R3 risk check: if all models' AUROC spread < 0.05, trigger tertile mode
        cls_auc = summary[
            (summary["task"] == "classification")
            & (summary["metric"] == "AUROC")
            & (summary["cv_scheme"] == "5fold_stratified")
        ]
        if len(cls_auc) >= 2:
            spread = cls_auc["mean"].max() - cls_auc["mean"].min()
            log.info(f"AUROC spread across models: {spread:.3f}")
            if spread < 0.05:
                log.warning(
                    "**R3 TRIGGERED**: all models cluster within 0.05 AUROC. "
                    "Re-run with tertile classification per Step 2.11.3."
                )


if __name__ == "__main__":
    main()
