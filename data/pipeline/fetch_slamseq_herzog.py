#!/usr/bin/env python3
"""
fetch_slamseq_herzog.py — Herzog et al. 2017 Nature Methods SLAM-seq データ取得

目的:
    mESC（mouse embryonic stem cell）で測定された 8,405 transcripts の
    SLAM-seq ベース半減期データを取得し、統一フォーマットCSVに変換する。

データソース:
    - 論文: Herzog VA et al. 2017, Nature Methods 14(12):1198-1204
    - DOI: 10.1038/nmeth.4435
    - 手法: 4-thiouridine (4sU) metabolic labeling + T>C conversion (iodoacetamide alkylation)
    - 時点: 0, 0.5, 1, 3, 6, 12, 24 h の pulse/chase
    - 生データ: GEO GSE99978
    - 再解析済み half-life: Zenodo (GRAND-SLAM pipeline による halflives.tsv)
        https://zenodo.org/records/7612564

使い方:
    python fetch_slamseq_herzog.py --output-dir ../raw/slamseq_herzog2017 --mode zenodo
    python fetch_slamseq_herzog.py --output-dir ../raw/slamseq_herzog2017 --mode geo     # raw fastq も欲しい場合

出力:
    ../processed/slamseq_herzog_halflife.csv
        columns: gene_id, gene_symbol, half_life_h, cell_line, source

NOTE:
    - Zenodo の halflives.tsv は GRAND-SLAM 再解析版（Jürges et al. 2018 Bioinformatics）。
      原論文 Supplementary との差分は解析パイプラインの違いによる。
    - 本総説では Zenodo 版を "re-analyzed canonical half-lives" として採用予定。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Zenodo record (GRAND-SLAM 再解析版) — 2026-04-19 公式 Zenodo ページから転記
# NOTE: ファイル名は "halflifes.tsv"（スペル "halflifes"、"halflives" ではない）
ZENODO_URLS = {
    "halflifes.tsv": "https://zenodo.org/records/7612564/files/halflifes.tsv?download=1",
    "mESC-end.gtf": "https://zenodo.org/records/7612564/files/mESC-end.gtf?download=1",
    "actD.tsv": "https://zenodo.org/records/7612564/files/actD.tsv?download=1",
}

# GEO accession（raw fastq が必要な場合）
GEO_ACCESSION = "GSE99978"
GEO_FTP_BASE = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/{GEO_ACCESSION}/"


def download_file(url: str, dest: Path) -> None:
    """Stream-download with skip-if-exists."""
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


def fetch_zenodo(output_dir: Path) -> None:
    """Download GRAND-SLAM re-analyzed half-lives from Zenodo."""
    if not ZENODO_URLS:
        log.error(
            "ZENODO_URLS is empty. "
            "Visit https://zenodo.org/records/7612564 and fill in halflives.tsv URL."
        )
        sys.exit(2)
    for fname, url in ZENODO_URLS.items():
        download_file(url, output_dir / fname)


def fetch_geo_supp(output_dir: Path) -> None:
    """Download GEO GSE99978 supplementary files (processed)."""
    log.info(f"Fetching GEO {GEO_ACCESSION} supplementary files")
    log.info(f"  Base URL: {GEO_FTP_BASE}")
    log.warning(
        "GEO supplementary file names vary. "
        "Manually inspect {GEO_FTP_BASE}suppl/ via browser and add to SUPP_FILES below."
    )
    # Placeholder — fill in after inspecting GEO supplementary listing
    supp_files: list[str] = []
    for fname in supp_files:
        download_file(GEO_FTP_BASE + "suppl/" + fname, output_dir / fname)


def parse_halflife_table(supp_file: Path) -> pd.DataFrame:
    """
    Parse Herzog 2017 half-life table into unified format.

    入力: halflives.tsv (Zenodo) または GEO supplementary
        典型列: transcript_id, gene_id, gene_name, half_life_h, confidence, etc.
    出力: DataFrame [gene_id, gene_symbol, half_life_h, cell_line, source]
    """
    if not supp_file.exists():
        raise FileNotFoundError(f"Supp file not found: {supp_file}")

    # halflifes.tsv 実カラム（2026-04-20 確定）:
    #   Chromosome, Start, End, Name, Length, Strand, Half-life (h),
    #   k (cpm/h), stderror Half-life, stderror k, Rsquare
    df = pd.read_csv(supp_file, sep="\t")
    log.info(f"Loaded {len(df)} rows from {supp_file.name}")
    log.info(f"Columns: {list(df.columns)[:15]}")

    df.columns = [str(c).strip() for c in df.columns]

    # 半減期列（"Half-life (h)" 完全一致優先、次に "halflife*" 系）
    hl_cols = [c for c in df.columns if c.lower() in ("half-life (h)", "halflife", "half_life")]
    if not hl_cols:
        hl_cols = [
            c for c in df.columns
            if (c.lower().startswith("halflife") or c.lower().startswith("half-life"))
            and "stderror" not in c.lower()
        ]
    if not hl_cols:
        raise RuntimeError(f"No halflife column found. Columns: {list(df.columns)[:20]}")

    # 遺伝子識別子: Name > Gene > gene_id の順で採用
    gene_col = next(
        (c for c in df.columns if c.lower() in ("name", "gene", "gene_id", "gene_name", "symbol")),
        None,
    )
    if gene_col is None:
        raise RuntimeError(f"Gene ID column not found. Columns: {list(df.columns)[:20]}")

    # 複数時点版は median を採用（GRAND-SLAM 標準）。単列なら素通し。
    if len(hl_cols) > 1:
        df["_half_life_h"] = df[hl_cols].median(axis=1, skipna=True)
    else:
        df["_half_life_h"] = pd.to_numeric(df[hl_cols[0]], errors="coerce")

    out = pd.DataFrame({
        "gene_id": df[gene_col].astype(str),
        "gene_symbol": df[gene_col].astype(str),
        "half_life_h": df["_half_life_h"],
    }).dropna(subset=["half_life_h"])
    out["cell_line"] = "mESC"
    out["source"] = "Herzog2017_SLAMseq"

    return out[["gene_id", "gene_symbol", "half_life_h", "cell_line", "source"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "slamseq_herzog2017",
        help="Directory to save raw supplementary / Zenodo files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
        help="Directory for unified CSV output",
    )
    parser.add_argument(
        "--mode",
        choices=["zenodo", "geo", "parse_only"],
        default="zenodo",
        help="zenodo=re-analyzed halflives.tsv, geo=GEO supp files, parse_only=skip download",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "zenodo":
        fetch_zenodo(args.output_dir)
    elif args.mode == "geo":
        fetch_geo_supp(args.output_dir)

    # Parse: find tsv/csv in output_dir
    supp_files = list(args.output_dir.glob("*.tsv")) + list(args.output_dir.glob("*.csv"))
    if not supp_files:
        log.error("No half-life tables found. Run with --mode zenodo first.")
        sys.exit(1)

    # halflifes.tsv を優先（actD.tsv は ActD 化学抑制の別解析で本 review の canonical ではない）
    priority = [p for p in supp_files if "halflif" in p.name.lower()]
    main_table = priority[0] if priority else supp_files[0]
    log.info(f"Parsing {main_table.name} ...")
    df = parse_halflife_table(main_table)

    output_csv = args.processed_dir / "slamseq_herzog_halflife.csv"
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote {len(df)} rows to {output_csv}")

    log.info("Dataset summary:")
    log.info(f"  Total transcripts: {len(df)}")
    log.info(f"  Cell line: mESC (mouse embryonic stem cell)")
    if "half_life_h" in df.columns:
        log.info(
            f"  Half-life range: {df['half_life_h'].min():.2f} - {df['half_life_h'].max():.2f} h"
        )


if __name__ == "__main__":
    main()
