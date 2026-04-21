#!/usr/bin/env python3
"""
fetch_timelapseseq_schofield.py — Schofield et al. 2018 Nature Methods TimeLapse-seq データ取得

目的:
    MEF（mouse embryonic fibroblast, 1h s4U）と K562（human CML, 4h s4U）で測定された
    TimeLapse-seq ベース半減期データを取得し、統一フォーマットCSVに変換する。

データソース:
    - 論文: Schofield JA, Duffy EE, Kiefer L, Sullivan MC, Simon MD 2018,
      Nature Methods 15(3):221-225
    - DOI: 10.1038/nmeth.4582
    - 手法: 4-thiouridine (4sU) metabolic labeling + oxidative-nucleophilic-aromatic
      substitution (OsO4化学) で 4sU → C 変換（U→C mutation として検出）
    - 注: SLAM-seq（Herzog 2017, iodoacetamide化学）とは変換機序が異なるが、
      同じく 4sU 標識新規転写物を T>C / U>C mutation で検出する metabolic labeling 系
    - 生データ: GEO GSE95854
    - 半減期: Supplementary Tables (MEF replicates + K562)

使い方:
    python fetch_timelapseseq_schofield.py --output-dir ../raw/timelapseseq_schofield2018 --mode supp

出力:
    ../processed/timelapseseq_schofield_halflife.csv
        columns: gene_id, gene_symbol, half_life_h, cell_line (MEF or K562), source

NOTE:
    - MEF と K562 は cell_line 列で区別して 1 ファイルに統合する。
    - Cross-species validation: MEF（mouse）+ K562（human）で cell-line + species generality 担保。
    - 本総説では Herzog 2017 mESC（SLAM-seq）と合わせ、「4sU-based dynamics」の3点測定として扱う。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEO_ACCESSION = "GSE95854"
GEO_FTP_BASE = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE95nnn/{GEO_ACCESSION}/"

# Nature Methods supplementary tables — 2026-04-19 Springer static-content から確認済
# MOESM5 が半減期 xlsx（710 KB、MEF + K562 を含む）。MOESM1 は Supplementary Figures PDF。
SUPP_URLS = {
    "nmeth4582_MOESM5_halflives.xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.4582/MediaObjects/41592_2018_BFnmeth4582_MOESM5_ESM.xlsx",
    "nmeth4582_MOESM1_figures.pdf": "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.4582/MediaObjects/41592_2018_BFnmeth4582_MOESM1_ESM.pdf",
}


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


def fetch_supplementary(output_dir: Path) -> None:
    """Download Nature Methods supplementary tables."""
    if not SUPP_URLS:
        log.error(
            "SUPP_URLS is empty. "
            "Visit https://www.nature.com/articles/nmeth.4582 and fill in Supplementary Table URLs."
        )
        sys.exit(2)
    for fname, url in SUPP_URLS.items():
        download_file(url, output_dir / fname)


def fetch_geo_supp(output_dir: Path) -> None:
    """Fallback: fetch GEO-level supplementary files."""
    log.info(f"Fetching GEO {GEO_ACCESSION} supplementary files")
    log.info(f"  Base URL: {GEO_FTP_BASE}")
    log.warning(
        "GEO supplementary file names vary. "
        "Inspect {GEO_FTP_BASE}suppl/ via browser first."
    )


def parse_halflife_table(supp_file: Path) -> pd.DataFrame:
    """
    Parse Schofield 2018 half-life supplementary into unified format.

    想定: xlsx で MEF と K562 が別 sheet、あるいは1 sheet に cell_line 列あり。
    出力: DataFrame [gene_id, gene_symbol, half_life_h, cell_line, source]
    """
    if not supp_file.exists():
        raise FileNotFoundError(f"Supp file not found: {supp_file}")

    if supp_file.suffix in (".xlsx", ".xls"):
        xl = pd.ExcelFile(supp_file)
        log.info(f"Sheets in {supp_file.name}: {xl.sheet_names}")

        # MEF / K562 を sheet 名または列値で区別する。どちらの形式にも対応。
        frames = []
        for sheet in xl.sheet_names:
            tmp = pd.read_excel(xl, sheet_name=sheet)
            tmp.columns = [str(c).strip() for c in tmp.columns]
            low = {c: c.lower() for c in tmp.columns}
            tmp = tmp.rename(columns=low)

            # 半減期列: mean_half_life を最優先（replicate平均値が canonical）
            hl_col = next(
                (c for c in tmp.columns if c == "mean_half_life"), None
            )
            if hl_col is None:
                hl_col = next(
                    (c for c in tmp.columns if "half" in c and "life" in c), None
                )
            if hl_col is None:
                hl_col = next((c for c in tmp.columns if c in ("kdeg", "lambda", "t_half")), None)
            symbol_col = next(
                (c for c in tmp.columns if c in ("gene", "symbol", "gene_symbol", "gene_name", "transcript")),
                None,
            )
            gene_id_col = next(
                (c for c in tmp.columns if c in ("gene_id", "ensembl_id", "transcript_id", "transcript")),
                None,
            ) or symbol_col

            if hl_col is None or symbol_col is None:
                log.info(f"  Skip sheet '{sheet}' (no half-life/symbol column)")
                continue

            sub = pd.DataFrame({
                "gene_id": tmp[gene_id_col].astype(str),
                "gene_symbol": tmp[symbol_col].astype(str),
                "half_life_h": pd.to_numeric(tmp[hl_col], errors="coerce"),
            }).dropna(subset=["half_life_h"])

            # cell_line 決定：sheet 名に含まれるキーワードで判定
            sheet_lower = sheet.lower()
            if "k562" in sheet_lower:
                sub["cell_line"] = "K562"
            elif "mef" in sheet_lower:
                sub["cell_line"] = "MEF"
            elif "cell_line" in tmp.columns:
                sub["cell_line"] = tmp["cell_line"].astype(str)
            else:
                sub["cell_line"] = sheet  # 後段で手動修正できるよう sheet 名を残す

            sub["_source_sheet"] = sheet
            frames.append(sub)

        if not frames:
            raise RuntimeError(
                f"No valid half-life sheets in {supp_file.name}. "
                f"Inspect sheets manually: {xl.sheet_names}"
            )
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(supp_file, sep=None, engine="python")

    log.info(f"Loaded {len(df)} rows from {supp_file.name}")
    df["source"] = "Schofield2018_TimeLapseseq"

    return df[["gene_id", "gene_symbol", "half_life_h", "cell_line", "source"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "timelapseseq_schofield2018",
        help="Directory to save raw supplementary / GEO files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
        help="Directory for unified CSV output",
    )
    parser.add_argument(
        "--mode",
        choices=["supp", "geo", "parse_only"],
        default="supp",
        help="supp=Nature Methods supp tables, geo=GEO supp, parse_only=skip download",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "supp":
        fetch_supplementary(args.output_dir)
    elif args.mode == "geo":
        fetch_geo_supp(args.output_dir)

    # Parse
    supp_files = (
        list(args.output_dir.glob("*.xls*"))
        + list(args.output_dir.glob("*.tsv"))
        + list(args.output_dir.glob("*.csv"))
    )
    if not supp_files:
        log.error("No half-life tables found. Run with --mode supp first.")
        sys.exit(1)

    main_table = supp_files[0]
    log.info(f"Parsing {main_table.name} ...")
    df = parse_halflife_table(main_table)

    output_csv = args.processed_dir / "timelapseseq_schofield_halflife.csv"
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote {len(df)} rows to {output_csv}")

    log.info("Dataset summary:")
    log.info(f"  Total transcripts: {len(df)}")
    if "cell_line" in df.columns:
        for cl in df["cell_line"].unique():
            n = (df["cell_line"] == cl).sum()
            log.info(f"  {cl}: {n} transcripts")


if __name__ == "__main__":
    main()
