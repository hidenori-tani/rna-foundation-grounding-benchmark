#!/usr/bin/env python3
"""ablation.py — Task 2.9 (length only, v1): length tertile 層別 AUROC

test_set_final.csv の length カラムで3分割し、
out-of-fold predictions を層別集計。

出力: benchmark/results/ablation_results.csv
columns: model, stratum, value, n, AUROC

Note: GC / motif は FASTA 再取得後に v2 で追加予定。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from classifiers import LogisticRegressionClassifier  # noqa: E402
from eval import align_data, load_embeddings  # noqa: E402

SEED = 42
MODELS = ["rna_fm", "rinalmo", "evo", "rhofold_plus", "deeplncloc"]
BASE = Path(__file__).parent
EMB_DIR = BASE / "results" / "embeddings"
TEST_CSV = BASE.parent / "data" / "processed" / "test_set_final.csv"


def main() -> None:
    test_df = pd.read_csv(TEST_CSV)
    cls = test_df[test_df["label_binary"].isin(["stable", "unstable"])].copy()
    cls["label_int"] = (cls["label_binary"] == "stable").astype(int)

    cls["len_bin"] = pd.qcut(cls["length"], 3, labels=["short", "mid", "long"])
    print("Length tertile bounds:", cls["length"].quantile([0.33, 0.67]).tolist())
    print("len_bin × label counts:")
    print(cls.groupby(["len_bin", "label_binary"]).size().unstack(fill_value=0))

    rows: list[dict] = []
    for model_name in MODELS:
        npz_path = EMB_DIR / f"{model_name}.npz"
        if not npz_path.exists():
            print(f"skip {model_name}: npz not found")
            continue
        gids, emb = load_embeddings(npz_path)
        X, sub = align_data(gids, emb, cls)
        y = sub["label_int"].values
        if len(set(y)) < 2:
            print(f"skip {model_name}: only one class present")
            continue

        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        probs = np.zeros(len(y), dtype=float)
        for tr, te in skf.split(X, y):
            clf = LogisticRegressionClassifier()
            clf.fit(X[tr], y[tr])
            probs[te] = clf.predict_proba(X[te])[:, 1]
        print(f"{model_name}: OOF AUROC overall = {roc_auc_score(y, probs):.3f}")

        for bin_val in sub["len_bin"].cat.categories:
            mask = (sub["len_bin"] == bin_val).values
            if mask.sum() < 5 or len(set(y[mask])) < 2:
                continue
            auc = roc_auc_score(y[mask], probs[mask])
            rows.append(dict(
                model=model_name, stratum="length",
                value=str(bin_val), n=int(mask.sum()), AUROC=float(auc)
            ))

    out = pd.DataFrame(rows)
    out_path = BASE / "results" / "ablation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
