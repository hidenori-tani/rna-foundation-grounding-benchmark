#!/usr/bin/env python3
"""
fetch_bricseq.py — Tani et al. 2012 Genome Research BRIC-seq データ取得

目的:
    HeLa Tet-off 細胞の 11,052 mRNA + 1,418 ncRNA の半減期データを取得し、
    統一フォーマット（gene_id, half_life_h, cell_line, source）のCSVに変換する。

データソース:
    - 論文: Tani et al. 2012, Genome Research 22(5):947-956, PMC3337439
    - DOI: 10.1101/gr.130559.111
    - 生データ: DDBJ DRA accessions
        * DRA000357-DRA000361: 2つの独立実験（+/-ActD、3時点）
        * DRA000345-DRA000350: 半減期測定（5時点）
    - 半減期値: 論文 Supplementary Tables（genome.cshlp.org 付属ファイル）

使い方:
    python fetch_bricseq.py --output-dir ../raw/bricseq_tani2012 --mode supp
    python fetch_bricseq.py --output-dir ../raw/bricseq_tani2012 --mode dra    # 生データも欲しい場合

出力:
    ../processed/bricseq_halflife.csv
        columns: gene_id, gene_symbol, half_life_h, half_life_category, cell_line, source

NOTE:
    初回実行前に、Supplementary Table の具体的な URL/ファイル名を確定させること。
    先生ご自身の論文のため、manuscript側から直接取得する方が確実。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Tani 2012 Genome Research supplementary material URLs
# 2026-04-19 公式 Supplemental Material ページ（genome.cshlp.org/content/22/5/947/suppl/DC1）
# から転記。S1–S7 は単一 Excel ファイルに統合されている。
TANI2012_SUPP_URLS = {
    "tables": "https://genome.cshlp.org/content/suppl/2012/02/14/gr.130559.111.DC1/Tani_Supp_Tables_revised2.xls",
    "figures": "https://genome.cshlp.org/content/suppl/2012/02/14/gr.130559.111.DC1/Tani_Supp_Figures_revised2.pdf",
}

# DDBJ DRA accession list（raw dataが必要な場合）
DRA_ACCESSIONS = [
    "DRA000345", "DRA000346", "DRA000347", "DRA000348", "DRA000349", "DRA000350",
    "DRA000357", "DRA000358", "DRA000359", "DRA000360", "DRA000361",
]


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file from URL to dest, skipping if exists."""
    if dest.exists():
        log.info(f"Already exists, skipping: {dest.name}")
        return
    log.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def fetch_supplementary_tables(output_dir: Path) -> None:
    """Download Tani 2012 supplementary tables from Genome Research."""
    if not TANI2012_SUPP_URLS:
        log.error(
            "TANI2012_SUPP_URLS is empty. "
            "先生に確認：genome.cshlp.org の付属ファイルURLを埋めてください。"
        )
        sys.exit(2)
    for table_id, url in TANI2012_SUPP_URLS.items():
        fname = url.split("/")[-1]
        download_file(url, output_dir / fname)


def fetch_dra_raw(output_dir: Path) -> None:
    """Download DDBJ DRA raw sequencing data (large, ~数十GB)."""
    log.warning("DRA raw data download — this is very large (~数十GB).")
    log.warning("Consider using --mode supp unless re-quantification is required.")
    # DDBJ ftp mirror: https://ddbj.nig.ac.jp/resource/sra-submission/{accession}
    base = "https://ddbj.nig.ac.jp/public/ddbj_database/dra/sralite/ByExp/litesra"
    for acc in DRA_ACCESSIONS:
        log.info(f"Accession {acc}: fetch manually via DRA search interface")
        log.info(f"  URL: https://ddbj.nig.ac.jp/resource/sra-submission/{acc}")
    log.warning("本スクリプトではDRA自動取得は未実装。先生が必要と判断したら個別DL。")


def parse_halflife_table(supp_file: Path) -> pd.DataFrame:
    """
    Parse Tani 2012 supplementary table into unified format.

    入力: Genome Research supplementary Excel（実ファイル名とsheet構造は要確認）
    出力: DataFrame with columns [gene_id, gene_symbol, half_life_h, cell_line, source]

    NOTE:
        このパーサーは Supplementary Table のcolumn構造が確定した時点で実装する。
        先生の論文のため、ファイル構造は先生に直接確認するのが速い。
    """
    if not supp_file.exists():
        raise FileNotFoundError(f"Supp file not found: {supp_file}")

    # Tani_Supp_Tables_revised2.xls 実ファイル仕様（2026-04-20 確定）:
    #   - 先頭 4 行はタイトル/空行、実ヘッダは row 3（0-indexed）
    #   - 半減期列は "t1/2 (h)"（S1–S7）または "t1/2"（S9, S10, S14）
    #   - 遺伝子識別子は "RepName"（RefSeq NM_/NR_ 等）または "RepName or Genomic region"
    #   - Sheet 内訳:
    #       S1 = NM_ mRNA (n=18,001)
    #       S2 = NR_ ncRNA (n=2,829)
    #       S3 = FLJ cDNA (n=1,738)
    #       S4 = MGC/DKFZ/KIAA (n=36)
    #       S5 = lincRNA (Khalil 2009, n=261, 座標ベース・RepName列なし)
    #       S6 = long ncRNA (Jia 2010, n=5,445)
    #       S7 = lncRNAdb (n=831)
    # 本 review の対象は lncRNA。S2 + S5–S7 を取り、mRNA (S1) は source_category で区別。
    xl = pd.ExcelFile(supp_file)
    log.info(f"Sheets in {supp_file.name}: {xl.sheet_names}")

    # 取得対象 sheet: S1（mRNA baseline）と S2, S5, S6, S7（lncRNA）
    target_sheets = {
        "Table S1": "mRNA",
        "Table S2": "ncRNA_NR",
        "Table S5": "lincRNA_Khalil",
        "Table S6": "lncRNA_Jia",
        "Table S7": "lncRNAdb",
    }

    frames = []
    for sheet, category in target_sheets.items():
        if sheet not in xl.sheet_names:
            log.warning(f"  Sheet '{sheet}' not found, skipping")
            continue
        tmp = pd.read_excel(xl, sheet_name=sheet, header=3)
        tmp.columns = [str(c).strip() for c in tmp.columns]

        half_life_col = next(
            (c for c in tmp.columns if "t1/2" in c.lower() or "half" in c.lower()), None
        )
        if half_life_col is None:
            log.info(f"  Skip sheet '{sheet}' (no t1/2 column)")
            continue

        # 遺伝子 ID: RepName が優先。S5 は座標ベース → chromosome:start-end を ID 化
        if "RepName" in tmp.columns:
            gene_id_series = tmp["RepName"].astype(str).str.strip().str.rstrip(",")
            symbol_series = gene_id_series
        elif "RepName or Genomic region" in tmp.columns:
            gene_id_series = tmp["RepName or Genomic region"].astype(str).str.strip().str.rstrip(",")
            symbol_series = gene_id_series
        elif {"chromosome", "start", "end"}.issubset(tmp.columns):
            gene_id_series = (
                tmp["chromosome"].astype(str)
                + ":"
                + tmp["start"].astype(str)
                + "-"
                + tmp["end"].astype(str)
            )
            symbol_series = gene_id_series
        else:
            log.info(f"  Skip sheet '{sheet}' (no RepName or coordinate columns)")
            continue

        out = pd.DataFrame({
            "gene_id": gene_id_series,
            "gene_symbol": symbol_series,
            "half_life_h": pd.to_numeric(tmp[half_life_col], errors="coerce"),
        }).dropna(subset=["half_life_h"])
        out["source_sheet"] = sheet
        out["source_category"] = category
        log.info(f"  {sheet} ({category}): {len(out)} entries with t1/2")
        frames.append(out)

    if not frames:
        raise RuntimeError(
            f"No half-life column found in any target sheet of {supp_file.name}."
        )

    df = pd.concat(frames, ignore_index=True)
    df["cell_line"] = "HeLa_TetOff"
    df["source"] = "Tani2012_BRICseq"

    return df[["gene_id", "gene_symbol", "half_life_h", "cell_line", "source", "source_category"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "bricseq_tani2012",
        help="Directory to save raw supplementary files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
        help="Directory for unified CSV output",
    )
    parser.add_argument(
        "--mode",
        choices=["supp", "dra", "parse_only"],
        default="supp",
        help="supp=download supp tables, dra=download raw seq data, parse_only=skip download",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "supp":
        fetch_supplementary_tables(args.output_dir)
    elif args.mode == "dra":
        fetch_dra_raw(args.output_dir)

    # Parse: find the first .xls/.xlsx in output_dir as main half-life table
    supp_files = list(args.output_dir.glob("*.xls*"))
    if not supp_files:
        log.error("No supplementary files found. Run with --mode supp first.")
        sys.exit(1)

    # 実装時: どのSupplementary TableがHalf-life値を含むかを特定
    main_table = supp_files[0]
    log.info(f"Parsing {main_table.name} ...")
    df = parse_halflife_table(main_table)

    output_csv = args.processed_dir / "bricseq_halflife.csv"
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote {len(df)} rows to {output_csv}")

    # Pilot check: lncRNAの数をカウント（GENCODEクロスマッピング後が本番）
    log.info(f"Dataset summary:")
    log.info(f"  Total transcripts: {len(df)}")
    log.info(f"  Cell line: {df['cell_line'].iloc[0]}")
    log.info(f"  Half-life range: {df['half_life_h'].min():.2f} - {df['half_life_h'].max():.2f} h")


if __name__ == "__main__":
    main()
