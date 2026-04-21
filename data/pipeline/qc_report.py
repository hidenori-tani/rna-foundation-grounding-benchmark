#!/usr/bin/env python3
"""
qc_report.py — Phase 1 最終品質レビュー（Task 1.8）

目的:
    test_set_final.csv の内容を品質チェックし、QC_report.md を自動生成する。

入力:
    ../processed/test_set_final.csv
    ../processed/halflife_merged.csv

出力:
    ../QC_report.md

チェック項目（Spec §8, plan.md Task 1.8 参照）:
    1. テストセットの N（total / stable / unstable / excluded）
    2. 長さ分布・GC含量分布
    3. Sanity check: 既知の安定/不安定 lncRNA との整合
         - NEAT1, MALAT1（stable）
         - FIRRE, XIST（多くはやや不安定寄り）
         - 既存文献との order-of-magnitude 整合性
    4. R1発動判定: N < 30（classifiable）→ 公開SLAM-seqの追加を検討
    5. データセット別の bias（source 列で層別）

NOTE:
    本スクリプトは scaffold。Phase 1 実行時に実データで確定する。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

KNOWN_STABLE = ["NEAT1", "MALAT1", "KCNQ1OT1"]
KNOWN_UNSTABLE_CANDIDATES = ["FIRRE", "LINC-PINT"]  # 相対的に短命と報告あり


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def generate_report(test_csv: Path, merged_csv: Path, out_md: Path) -> None:
    df = pd.read_csv(test_csv)
    merged = pd.read_csv(merged_csv) if merged_csv.exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Phase 1 QC Report")
    lines.append("")
    lines.append(f"Generated from: `{test_csv.name}`")
    lines.append("")

    lines.append("## 1. Test set size")
    lines.append("")
    lines.append(f"- Total rows: **{len(df)}**")
    lines.append(f"- Unique genes: **{df['gencode_gene_id'].nunique()}**")
    lines.append("")
    if "label_binary" in df.columns:
        for lbl, n in df["label_binary"].value_counts().items():
            lines.append(f"- `label_binary = {lbl}`: {n}")
    lines.append("")

    lines.append("## 2. Cell line coverage")
    lines.append("")
    if "cell_line" in df.columns:
        for cl, n in df["cell_line"].value_counts().items():
            lines.append(f"- {cl}: {n}")
    lines.append("")

    lines.append("## 3. Length & GC content")
    lines.append("")
    if "length" in df.columns and len(df):
        ln = df["length"].describe()
        lines.append(f"- length median={ln['50%']:.0f}, min={ln['min']:.0f}, max={ln['max']:.0f}")
    elif not len(df):
        lines.append("- (test set empty — no length stats available)")
    if "sequence" in df.columns and len(df):
        gc = df["sequence"].apply(gc_content)
        lines.append(f"- GC median={gc.median():.3f}, IQR=[{gc.quantile(0.25):.3f}, {gc.quantile(0.75):.3f}]")
    lines.append("")

    lines.append("## 4. Sanity check vs known lncRNAs")
    lines.append("")
    for sym in KNOWN_STABLE + KNOWN_UNSTABLE_CANDIDATES:
        hit = df[df["gencode_gene_symbol"].str.upper() == sym]
        if len(hit):
            t = hit["half_life_h"].mean()
            lbl = hit["label_binary"].iloc[0] if "label_binary" in hit.columns else "?"
            expected = "stable" if sym in KNOWN_STABLE else "unstable-ish"
            ok = "✅" if (
                (expected == "stable" and lbl == "stable") or
                (expected == "unstable-ish" and lbl in ("unstable", "excluded"))
            ) else "⚠️"
            lines.append(f"- {sym}: t½={t:.2f}h, label={lbl}, expected={expected} {ok}")
        else:
            lines.append(f"- {sym}: not in test set")
    lines.append("")

    lines.append("## 5. Risk flags")
    lines.append("")

    # R1-a: classifiable N
    n_class = (df["label_binary"] != "excluded").sum() if "label_binary" in df.columns else 0
    # R1-b: minority class の N
    n_stable = (df.get("label_binary") == "stable").sum() if "label_binary" in df.columns else 0
    n_unstable = (df.get("label_binary") == "unstable").sum() if "label_binary" in df.columns else 0
    n_minority = min(n_stable, n_unstable) if (n_stable and n_unstable) else 0
    # R1-c: GENCODE unmap rate among lncRNA-eligible rows (mapped entries in merged).
    # The old metric (len(df)/len(merged)) conflated mRNA-filter-out with ID-mapping failure,
    # producing misleading ~99% even when cross_mapping worked correctly on lncRNA candidates.
    mapped_in_merged = (
        merged[merged["gencode_gene_id"].notna() & (merged["gencode_gene_id"].astype(str) != "")]
        if "gencode_gene_id" in merged.columns else pd.DataFrame()
    )
    n_mapped_eligible = len(mapped_in_merged)
    unmap_rate = (
        1 - (len(df) / n_mapped_eligible) if n_mapped_eligible else 1.0
    )
    # R1-d: cell-line 偏り
    cl_counts = df["cell_line"].value_counts() if "cell_line" in df.columns else pd.Series(dtype=int)
    max_cl_share = (cl_counts.max() / cl_counts.sum()) if len(cl_counts) else 0.0
    # R1-e: known-label consistency
    known_hits_stable = [
        sym for sym in KNOWN_STABLE
        if len(df[df["gencode_gene_symbol"].str.upper() == sym]) > 0
    ]
    known_hits_unstable = [
        sym for sym in KNOWN_UNSTABLE_CANDIDATES
        if len(df[df["gencode_gene_symbol"].str.upper() == sym]) > 0
    ]

    triggers = []
    if n_class < 30:
        triggers.append(f"R1-a: classifiable N={n_class} < 30 (threshold for ML)")
    if n_minority < 10 and (n_stable or n_unstable):
        triggers.append(f"R1-b: minority class N={n_minority} < 10 (class imbalance extreme)")
    if unmap_rate > 0.30:
        triggers.append(
            f"R1-c: test_set drop rate from mapped-eligible={unmap_rate:.1%} > 30% "
            f"(of {n_mapped_eligible} lncRNA-mapped rows, {len(df)} survived length/sequence filters)"
        )
    if max_cl_share > 0.80:
        dominant = cl_counts.idxmax()
        triggers.append(f"R1-d: cell-line '{dominant}' = {max_cl_share:.1%} > 80% (LOCO CV unreliable)")
    if not known_hits_stable:
        triggers.append("R1-e: none of NEAT1/MALAT1/KCNQ1OT1 in test set (anchor lncRNAs missing)")

    if triggers:
        lines.append("**R1 RISK TRIGGERED**:")
        for t in triggers:
            lines.append(f"- ⚠️ {t}")
        lines.append("")
        lines.append("Mitigation options (plan.md Task 1.8.3):")
        lines.append("- R1-a/b: Relax binary thresholds (e.g., stable > 3h, unstable < 3h) OR switch to tertile labels")
        lines.append("- R1-c: Inspect `cross_mapping.py` output; check GENCODE version + Ensembl ID prefix consistency")
        lines.append("- R1-d: Down-sample the dominant cell line, or use 5-fold stratified by cell_line")
        lines.append("- R1-e: Manually inject anchor lncRNAs (spec §3.2 minimum set)")
    else:
        lines.append(f"- R1 OK: classifiable N={n_class}, minority={n_minority}")
        lines.append(f"- R1 OK: unmap rate={unmap_rate:.1%} ≤ 30%")
        lines.append(f"- R1 OK: max cell-line share={max_cl_share:.1%} ≤ 80%")
        lines.append(f"- R1 OK: anchors in set = {known_hits_stable + known_hits_unstable}")
    lines.append("")

    lines.append("## 6. Dataset contributions")
    lines.append("")
    if "sources" in df.columns:
        for src, n in df["sources"].value_counts().head(10).items():
            lines.append(f"- `{src}`: {n}")
    lines.append("")

    lines.append("## 7. Decision gate for Phase 2")
    lines.append("")
    lines.append("Phase 2 (GPU spend) は **R1 flag ゼロ** を確認してから実行する。")
    lines.append("triggers がある場合は対応策を講じた上で本スクリプトを再実行し、clear を確認すること。")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    log.info(f"Wrote {out_md}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base data/ dir where QC_report.md is placed",
    )
    args = parser.parse_args()

    test_csv = args.processed_dir / "test_set_final.csv"
    merged_csv = args.processed_dir / "halflife_merged.csv"
    out_md = args.data_dir / "QC_report.md"

    if not test_csv.exists():
        log.error(f"Run build_test_set.py first: missing {test_csv}")
        sys.exit(1)

    generate_report(test_csv, merged_csv, out_md)


if __name__ == "__main__":
    main()
