#!/usr/bin/env python3
"""
interpretability.py — 特徴重要度分析（Task 2.10）

目的:
    spec §4.2 に基づき、embedding 次元ごとの予測寄与を可視化する：
        - MLP 分類器には Integrated Gradients（captum）を適用
        - LogisticRegression には SHAP 値を計算
        - 代表 5 lncRNA（正解/不正解の両方から選定）の寄与マップを JSON 保存

入力:
    benchmark/results/embeddings/*.npz
    data/processed/test_set_final.csv

出力:
    benchmark/results/feature_importance/
        {model}_{gene_id}_{method}.json

NOTE:
    埋め込み空間での寄与度は塩基レベルの解釈にそのまま落ちないため、
    本総説では「モデルごとに異なる次元を見ているか」の定性比較に使う（§5 failure modes）。
    塩基レベル attribution が必要な場合は Task 2.4.3 以降の sequence-level re-encode が必要。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def select_representatives(df: pd.DataFrame, preds: np.ndarray, n: int = 5) -> list[int]:
    """
    Pick representative lncRNAs: mix of correct/incorrect predictions with
    wide half-life spread.
    """
    df = df.copy().reset_index(drop=True)
    df["pred"] = preds
    df["correct"] = (df["pred"] == df["label_int"]).astype(int)

    # Prefer: 2 correct confident, 2 incorrect, 1 borderline
    out: list[int] = []
    correct = df[df["correct"] == 1]
    incorrect = df[df["correct"] == 0]
    if len(correct) >= 2:
        out.extend(correct.index[:2].tolist())
    if len(incorrect) >= 2:
        out.extend(incorrect.index[:2].tolist())
    remaining = [i for i in df.index if i not in out]
    out.extend(remaining[: max(0, n - len(out))])
    return out[:n]


def ig_for_mlp(model, X: np.ndarray, y: np.ndarray, indices: list[int]) -> dict:
    """Integrated Gradients using captum (MLP classifier from classifiers.py)."""
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ImportError:
        log.error("Install: pip install captum")
        return {}

    ig = IntegratedGradients(model.net)
    results: dict[str, list[float]] = {}
    for i in indices:
        x = torch.tensor(X[i : i + 1], dtype=torch.float32, device=model.device)
        target = int(y[i])
        attributions = ig.attribute(x, target=target)
        results[str(i)] = attributions.squeeze(0).cpu().numpy().tolist()
    return results


def shap_for_logreg(model, X: np.ndarray, indices: list[int]) -> dict:
    """SHAP values for logistic regression via shap.LinearExplainer."""
    try:
        import shap
    except ImportError:
        log.error("Install: pip install shap")
        return {}

    Xs = model.scaler.transform(X)
    expl = shap.LinearExplainer(model.model, Xs)
    shap_vals = expl.shap_values(Xs[indices])
    if isinstance(shap_vals, list):
        # multi-class: take positive-class
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    return {str(i): shap_vals[j].tolist() for j, i in enumerate(indices)}


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
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "feature_importance",
    )
    parser.add_argument("--n-representatives", type=int, default=5)
    args = parser.parse_args()

    if not args.test_csv.exists():
        log.error(f"Missing {args.test_csv}")
        sys.exit(1)

    from classifiers import LogisticRegressionClassifier, MLPClassifier

    test_df = pd.read_csv(args.test_csv)
    classifiable = test_df[test_df["label_binary"].isin(["stable", "unstable"])].copy().reset_index(drop=True)
    classifiable["label_int"] = (classifiable["label_binary"] == "stable").astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for npz in args.embeddings_dir.glob("*.npz"):
        data = np.load(npz, allow_pickle=True)
        gids = data["gene_ids"]
        emb = data["embeddings"]
        gid_to_idx = {g: i for i, g in enumerate(gids)}
        mask = classifiable["gencode_gene_id"].isin(gid_to_idx)
        sub = classifiable[mask].reset_index(drop=True)
        X = emb[[gid_to_idx[g] for g in sub["gencode_gene_id"]]]
        y = sub["label_int"].values
        model_name = npz.stem

        if len(sub) < 10:
            log.warning(f"Skipping {model_name}: only {len(sub)} examples")
            continue

        # Fit LR and MLP on full data for interpretability
        lr = LogisticRegressionClassifier().fit(X, y)
        mlp = MLPClassifier(in_dim=X.shape[1], n_classes=2, epochs=100).fit(X, y)

        # Pick representatives from MLP predictions
        preds = mlp.predict(X)
        rep_idx = select_representatives(sub, preds, n=args.n_representatives)

        # IG for MLP
        ig_result = ig_for_mlp(mlp, X, y, rep_idx)
        # SHAP for LR
        shap_result = shap_for_logreg(lr, X, rep_idx)

        for i in rep_idx:
            gid = sub.loc[i, "gencode_gene_id"]
            symbol = sub.loc[i, "gencode_gene_symbol"]
            out = {
                "model": model_name,
                "gene_id": gid,
                "gene_symbol": symbol,
                "true_label": int(y[i]),
                "mlp_pred": int(preds[i]),
                "ig_mlp": ig_result.get(str(i), []),
                "shap_lr": shap_result.get(str(i), []),
            }
            safe_gid = gid.replace("/", "_")
            out_path = args.output_dir / f"{model_name}_{safe_gid}.json"
            out_path.write_text(json.dumps(out, indent=2))
            log.info(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
