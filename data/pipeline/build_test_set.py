#!/usr/bin/env python3
"""
build_test_set.py — 最終テストセット（50〜100 lncRNAs）の構築

目的:
    halflife_merged.csv から本ベンチマークのテストセットを確定する。
    Spec §3.2 / §3.3 に従い、二値分類・連続値回帰・tertile（fallback）用のラベルを付与する。

入力:
    ../processed/halflife_merged.csv                  (Task 1.6 の出力)
    ../processed/gencode_v44_lncrna_sequences.fa      (Task 1.4)
    ../processed/gencode_vM33_lncrna_sequences.fa     (Task 1.4)
    ../processed/tpm_per_cell_line.csv                (別途公開RNA-seqから算出。Task 1.7.1)

出力:
    ../processed/test_set_final.csv
        columns: gencode_gene_id, gencode_gene_symbol, cell_line, half_life_h,
                 half_life_log2, normalized_half_life, label_binary, label_tertile, sources
    ../processed/test_set_sequences.fa

フィルタ順:
    1. TPM >= 3（各 cell_line）
    2. 3データセットで intersection（測定済みの lncRNA）
    3. gene_type == "lncRNA" (GENCODE)
    4. 長さ >= 200 nt, <= 20000 nt（極端な長さを除外）

ラベル（§3.3）:
    - label_binary ∈ {stable, unstable, excluded}
        stable:    half_life_h > 4
        unstable:  half_life_h < 2
        excluded:  2 <= half_life_h <= 4
    - label_tertile ∈ {low, mid, high}（全lncRNAでの tertile、R3 fallback用）

出力レポート:
    - 合計N、stable/unstable/excluded内訳、tertile内訳
    - 目標：50-100 lncRNAs（intersection後）
    - N<30 なら R1 発動：Task 1.8 でSLAM-seq比率拡大を検討
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_fasta(path: Path) -> dict[str, str]:
    """
    Parse GENCODE FASTA. Keys: stripped transcript_id AND stripped gene_id
    (both map to the same sequence — multiple transcripts per gene keeps the
    longest sequence as the representative).
    GENCODE header: >ENST00000...|ENSG...|-|...|gene_symbol|length|
    """
    if not path.exists():
        log.warning(f"FASTA missing, skipping: {path}")
        return {}
    seqs: dict[str, str] = {}
    current_tid: str | None = None
    current_gid: str | None = None
    current_seq: list[str] = []

    def _commit():
        if current_tid is None:
            return
        seq = "".join(current_seq)
        seqs[current_tid] = seq
        if current_gid is not None:
            # keep longest seq per gene_id
            prev = seqs.get(current_gid, "")
            if len(seq) > len(prev):
                seqs[current_gid] = seq

    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                _commit()
                header = line[1:].strip()
                parts = header.split("|")
                current_tid = parts[0].split(".")[0] if parts else None
                current_gid = parts[1].split(".")[0] if len(parts) > 1 else None
                current_seq = []
            else:
                current_seq.append(line.strip())
        _commit()
    log.info(f"Loaded {len(seqs)} sequences from {path.name} (tid+gid keys)")
    return seqs


def assign_labels(df: pd.DataFrame, stable_thr: float, unstable_thr: float) -> pd.DataFrame:
    """Assign binary and tertile labels per spec §3.3."""
    def _binary(h):
        if pd.isna(h):
            return "excluded"
        if h > stable_thr:
            return "stable"
        if h < unstable_thr:
            return "unstable"
        return "excluded"

    df["label_binary"] = df["half_life_h"].apply(_binary)

    # Tertile on full continuous values (not excluded)
    valid = df["half_life_h"].dropna()
    if len(valid) >= 3:
        t1, t2 = valid.quantile([1 / 3, 2 / 3])

        def _tertile(h):
            if pd.isna(h):
                return "na"
            if h <= t1:
                return "low"
            if h <= t2:
                return "mid"
            return "high"

        df["label_tertile"] = df["half_life_h"].apply(_tertile)
    else:
        df["label_tertile"] = "na"
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
    )
    parser.add_argument("--merged-csv", type=Path, default=None)
    parser.add_argument("--tpm-csv", type=Path, default=None)
    parser.add_argument("--tpm-threshold", type=float, default=3.0)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=20000)
    parser.add_argument("--stable-threshold", type=float, default=4.0, help="t½ > this is 'stable'")
    parser.add_argument("--unstable-threshold", type=float, default=2.0, help="t½ < this is 'unstable'")
    parser.add_argument(
        "--require-intersection",
        action="store_true",
        default=False,
        help="Require lncRNA measured in all datasets. Default: False (cross-species union is the intended behavior for D+C benchmark).",
    )
    parser.add_argument(
        "--min-sources",
        type=int,
        default=1,
        help="Minimum number of source datasets that must have measured a gene. Default 1 (any).",
    )
    args = parser.parse_args()

    pdir = args.processed_dir
    merged_path = args.merged_csv or (pdir / "halflife_merged.csv")
    if not merged_path.exists():
        log.error(f"halflife_merged.csv not found at {merged_path}. Run normalize_halflife.py first.")
        sys.exit(1)

    df = pd.read_csv(merged_path)
    log.info(f"Loaded {len(df)} merged rows")

    # Drop rows without a mapped gencode_gene_id (unmapped entries cannot be benchmarked)
    before = len(df)
    df = df[df["gencode_gene_id"].notna() & (df["gencode_gene_id"].astype(str) != "")]
    log.info(f"Drop unmapped: {before} -> {len(df)}")

    # Source coverage gating
    if args.require_intersection:
        sources_per_gene = df.groupby("gencode_gene_id")["source"].nunique()
        n_sources_required = df["source"].nunique()
        keep_ids = sources_per_gene[sources_per_gene >= n_sources_required].index
        df = df[df["gencode_gene_id"].isin(keep_ids)]
        log.info(f"After intersection ({n_sources_required} sources): {df['gencode_gene_id'].nunique()} genes, {len(df)} rows")
    elif args.min_sources > 1:
        sources_per_gene = df.groupby("gencode_gene_id")["source"].nunique()
        keep_ids = sources_per_gene[sources_per_gene >= args.min_sources].index
        df = df[df["gencode_gene_id"].isin(keep_ids)]
        log.info(f"After min_sources={args.min_sources}: {df['gencode_gene_id'].nunique()} genes, {len(df)} rows")

    # Collapse to one row per (gene_id, cell_line) — take mean half-life if duplicated
    df = df.groupby(
        ["gencode_gene_id", "gencode_gene_symbol", "cell_line"], as_index=False
    ).agg(
        half_life_h=("half_life_h", "mean"),
        half_life_log2=("half_life_log2", "mean"),
        normalized_half_life=("normalized_half_life", "mean"),
        sources=("source", lambda s: ",".join(sorted(set(s)))),
    )

    # TPM filter
    if args.tpm_csv or (pdir / "tpm_per_cell_line.csv").exists():
        tpm_path = args.tpm_csv or (pdir / "tpm_per_cell_line.csv")
        tpm = pd.read_csv(tpm_path)
        df = df.merge(tpm, on=["gencode_gene_id", "cell_line"], how="left")
        before = len(df)
        df = df[df["tpm"].fillna(0) >= args.tpm_threshold]
        log.info(f"TPM>={args.tpm_threshold} filter: {before} -> {len(df)}")
    else:
        log.warning("TPM table missing. Skipping expression filter (Task 1.7.1).")

    # Load lncRNA sequences (human v44 + mouse vM33)
    hsa = parse_fasta(pdir / "gencode_v44_lncrna_sequences.fa")
    mmu = parse_fasta(pdir / "gencode_vM33_lncrna_sequences.fa")
    all_seqs = {**hsa, **mmu}

    def seq_lookup(gene_id: str) -> str | None:
        # gene_id of form ENSG... (strip version)
        gid = gene_id.split(".")[0]
        # pick any transcript whose record header contained this gene (not implemented here)
        # placeholder — real implementation needs gene_id -> transcript_id lookup
        return all_seqs.get(gid)

    df["sequence"] = df["gencode_gene_id"].apply(seq_lookup)
    before = len(df)
    df = df[df["sequence"].notna()]
    log.info(f"With sequence available: {before} -> {len(df)}")

    # Length filter
    df["length"] = df["sequence"].str.len()
    df = df[(df["length"] >= args.min_length) & (df["length"] <= args.max_length)]
    log.info(f"Length [{args.min_length}, {args.max_length}]: {len(df)}")

    # Labels
    df = assign_labels(df, args.stable_threshold, args.unstable_threshold)

    # Save
    out_csv = pdir / "test_set_final.csv"
    out_fa = pdir / "test_set_sequences.fa"
    keep_cols = [
        "gencode_gene_id",
        "gencode_gene_symbol",
        "cell_line",
        "half_life_h",
        "half_life_log2",
        "normalized_half_life",
        "label_binary",
        "label_tertile",
        "sources",
        "length",
    ]
    present = [c for c in keep_cols if c in df.columns]
    df[present].to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv} ({len(df)} rows)")

    with open(out_fa, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['gencode_gene_id']}|{row['gencode_gene_symbol']}|{row['cell_line']}\n")
            seq = row["sequence"]
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")
    log.info(f"Wrote {out_fa}")

    # Summary report
    log.info("=" * 60)
    log.info("TEST SET SUMMARY")
    log.info(f"  Total: {len(df)}")
    for lbl, n in df["label_binary"].value_counts().items():
        log.info(f"  label_binary={lbl}: {n}")
    for lbl, n in df["label_tertile"].value_counts().items():
        log.info(f"  label_tertile={lbl}: {n}")
    for cl, n in df["cell_line"].value_counts().items():
        log.info(f"  cell_line={cl}: {n}")

    n_classifiable = (df["label_binary"] != "excluded").sum()
    log.info(f"  Classifiable (stable+unstable): {n_classifiable}")
    if n_classifiable < 30:
        log.warning("N<30: R1 risk triggered. See Task 1.8 for mitigation.")


if __name__ == "__main__":
    main()
