#!/usr/bin/env python3
"""
cross_mapping.py — 旧プローブ/転写物ID → GENCODE v44 への統一マッピング

目的:
    異なる時代・プラットフォームのデータセットを同一 ID 空間（GENCODE v44 human /
    vM33 mouse の gene_id）にマッピングする。

入力ID空間:
    - Tani 2012 BRIC-seq: Agilent microarray probe ID → Entrez Gene ID → Ensembl → GENCODE
    - Herzog 2017 SLAM-seq (mESC): Ensembl mouse gene_id (旧バージョン) → GENCODE vM33
    - Schofield 2018 TimeLapse-seq: RefSeq / Ensembl 混在 → GENCODE v44 (K562) / vM33 (MEF)

出力:
    ../processed/id_mapping.csv
        columns: source_dataset, source_id, source_id_type, gencode_gene_id,
                 gencode_transcript_id, gencode_gene_symbol, gencode_version, mapping_method

方針:
    1. 一次資料: 公式 cross-reference table (HGNC, MGI, Ensembl BioMart) を優先
    2. 補助: gene_symbol 一致（大文字小文字・別名テーブル考慮）
    3. 残余: liftOver（genomic coordinate ベース）は今回スコープ外。必要時に拡張
    4. unmappable は件数ログし、30%超なら手動キュレーション発動（R1リスク対処）

NOTE:
    本スクリプトは scaffold。各データセットの実ID空間を確認後に実装詳細を埋める。
    Phase 1 Task 1.5 で実装完了予定。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_gencode_lookup(gtf_path: Path) -> pd.DataFrame:
    """
    Parse GENCODE GTF to build gene_id ↔ gene_symbol lookup.

    出力列: gencode_gene_id (ENSG... or ENSMUSG...), gencode_transcript_id,
           gencode_gene_symbol, gene_type
    """
    if not gtf_path.exists():
        raise FileNotFoundError(f"GENCODE GTF not found: {gtf_path}")

    rows = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "transcript":
                continue
            attrs = dict(
                (kv.strip().split(" ", 1)[0], kv.strip().split(" ", 1)[1].strip('"'))
                for kv in parts[8].rstrip(";").split(";")
                if kv.strip() and " " in kv.strip()
            )
            rows.append({
                "gencode_gene_id": attrs.get("gene_id", ""),
                "gencode_transcript_id": attrs.get("transcript_id", ""),
                "gencode_gene_symbol": attrs.get("gene_name", ""),
                "gene_type": attrs.get("gene_type", ""),
            })
    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} transcripts from {gtf_path.name}")
    return df


def load_hgnc_refseq_lookup(hgnc_tsv: Path) -> pd.DataFrame:
    """
    Build NR_XXXXX → (symbol, ensembl_gene_id) lookup from HGNC complete set.
    Keeps only entries classified as lncRNA or generic RNA to reduce false matches.
    """
    if not hgnc_tsv.exists():
        log.warning(f"HGNC file missing, skipping NR_ mapping: {hgnc_tsv}")
        return pd.DataFrame(columns=["refseq_nr", "hgnc_symbol", "ensembl_gene_id"])
    hgnc = pd.read_csv(hgnc_tsv, sep="\t", low_memory=False)
    # Filter to lncRNA. Other ncRNA types (miRNA, snoRNA, rRNA) aren't in GENCODE lncRNA GTF.
    is_lnc = hgnc["locus_type"].fillna("").str.contains("long non-coding", case=False)
    hgnc = hgnc[is_lnc].copy()
    # refseq_accession can be "NR_015380" or "NR_015380|NR_131012". Explode on "|".
    hgnc["refseq_accession"] = hgnc["refseq_accession"].fillna("").astype(str)
    hgnc["refseq_list"] = hgnc["refseq_accession"].str.split("|")
    hgnc = hgnc.explode("refseq_list")
    hgnc["refseq_nr"] = hgnc["refseq_list"].str.strip()
    hgnc = hgnc[hgnc["refseq_nr"].str.startswith("NR_", na=False)]
    out = hgnc[["refseq_nr", "symbol", "ensembl_gene_id"]].rename(
        columns={"symbol": "hgnc_symbol"}
    )
    log.info(f"Loaded {len(out)} NR_→lncRNA mappings from HGNC")
    return out


def map_by_symbol_or_refseq(
    source_df: pd.DataFrame,
    source_symbol_col: str,
    gencode_df: pd.DataFrame,
    hgnc_nr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Two-stage mapping:
      1. Symbol exact (case-insensitive) → gencode_gene_id
      2. Fallback for unmapped: if gene_symbol looks like NR_XXXXX, lookup via HGNC →
         ensembl_gene_id → GENCODE.
    """
    sym_lookup = (
        gencode_df.assign(sym_lower=gencode_df["gencode_gene_symbol"].str.lower())
        .drop_duplicates("sym_lower")
        .set_index("sym_lower")[["gencode_gene_id", "gencode_transcript_id", "gencode_gene_symbol"]]
    )
    merged = source_df.assign(sym_lower=source_df[source_symbol_col].str.lower()).merge(
        sym_lookup, left_on="sym_lower", right_index=True, how="left"
    )
    merged["mapping_method"] = merged["gencode_gene_id"].apply(
        lambda x: "symbol_exact" if pd.notna(x) else "unmapped"
    )
    merged = merged.drop(columns=["sym_lower"])

    if hgnc_nr is None or hgnc_nr.empty:
        return merged

    # Build gencode lookup by stripped ensembl gene_id (without version)
    gc_by_gid = gencode_df.copy()
    gc_by_gid["gid_stripped"] = gc_by_gid["gencode_gene_id"].str.split(".").str[0]
    gc_by_gid = gc_by_gid.drop_duplicates("gid_stripped").set_index("gid_stripped")[
        ["gencode_gene_id", "gencode_transcript_id", "gencode_gene_symbol"]
    ]

    # NR_ lookup from HGNC (drop dups first — multiple HGNC entries sometimes share an NR_)
    nr_map = (
        hgnc_nr.drop_duplicates("refseq_nr")
        .set_index("refseq_nr")[["hgnc_symbol", "ensembl_gene_id"]]
        .to_dict("index")
    )

    def resolve_nr(row):
        if pd.notna(row["gencode_gene_id"]):
            return row
        raw = str(row[source_symbol_col])
        # Tani source_ids can be comma-separated; try each NR_ token
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok.startswith("NR_"):
                continue
            hgnc_hit = nr_map.get(tok)
            if not hgnc_hit:
                continue
            ensg = hgnc_hit.get("ensembl_gene_id")
            if not isinstance(ensg, str) or not ensg:
                continue
            gc_hit = gc_by_gid.loc[ensg] if ensg in gc_by_gid.index else None
            if gc_hit is None:
                continue
            row["gencode_gene_id"] = gc_hit["gencode_gene_id"]
            row["gencode_transcript_id"] = gc_hit["gencode_transcript_id"]
            row["gencode_gene_symbol"] = gc_hit["gencode_gene_symbol"]
            row["mapping_method"] = "refseq_nr_via_hgnc"
            return row
        return row

    merged = merged.apply(resolve_nr, axis=1)
    return merged


# Backwards-compat alias (callers that used the original name still work)
def map_by_symbol(source_df, source_symbol_col, gencode_df):
    return map_by_symbol_or_refseq(source_df, source_symbol_col, gencode_df, hgnc_nr=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
    )
    parser.add_argument(
        "--species",
        choices=["human", "mouse"],
        default="human",
        help="Which GENCODE reference to use for mapping (human=v44, mouse=vM33)",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help="Source half-life CSV to map (e.g., bricseq_halflife.csv)",
    )
    parser.add_argument(
        "--symbol-col",
        default="gene_symbol",
        help="Column in source CSV containing gene symbols",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output mapping CSV path (default: <source>_mapped.csv)",
    )
    parser.add_argument(
        "--hgnc-tsv",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "hgnc" / "hgnc_complete_set.txt",
        help="HGNC complete set TSV for NR_ RefSeq → lncRNA mapping (optional).",
    )
    args = parser.parse_args()

    gtf_name = "gencode_v44_lncrna.gtf" if args.species == "human" else "gencode_vM33_lncrna.gtf"
    gtf_path = args.processed_dir / gtf_name
    gencode_df = load_gencode_lookup(gtf_path)

    # HGNC only applies to human (Tani ncRNA_NR). For mouse, skip.
    hgnc_nr = load_hgnc_refseq_lookup(args.hgnc_tsv) if args.species == "human" else None

    src_df = pd.read_csv(args.source_csv)
    log.info(f"Loaded {len(src_df)} source rows from {args.source_csv.name}")

    mapped = map_by_symbol_or_refseq(src_df, args.symbol_col, gencode_df, hgnc_nr=hgnc_nr)

    n_mapped = (mapped["mapping_method"] != "unmapped").sum()
    n_total = len(mapped)
    unmap_rate = 1.0 - n_mapped / n_total if n_total else 1.0
    log.info(f"Mapped: {n_mapped} / {n_total} ({n_mapped / n_total:.1%})")
    log.info(f"Unmapped rate: {unmap_rate:.1%}")
    if unmap_rate > 0.30:
        log.warning("Unmapped rate > 30%. R1 risk: manual curation required.")

    out_path = args.output_csv or (
        args.processed_dir / f"{args.source_csv.stem}_mapped.csv"
    )
    mapped.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
