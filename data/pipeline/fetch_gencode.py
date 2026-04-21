#!/usr/bin/env python3
"""
fetch_gencode.py — GENCODE lncRNA アノテーション＋配列取得

目的:
    - Human: GENCODE v44 (primary assembly) から lncRNA 転写物を抽出
    - Mouse: GENCODE vM33 から lncRNA 転写物を抽出（Herzog 2017 mESC, Schofield 2018 MEF用）
    - gene_type == "lncRNA" (旧称 lincRNA, antisense, processed_transcript 等は含めない)
    - 対応する transcript sequences (FASTA) も取得

データソース:
    https://www.gencodegenes.org/human/release_44.html
    https://www.gencodegenes.org/mouse/release_M33.html

    FTP直リンク:
        gencode.v44.long_noncoding_RNAs.gtf.gz
        gencode.v44.lncRNA_transcripts.fa.gz
        gencode.vM33.long_noncoding_RNAs.gtf.gz
        gencode.vM33.lncRNA_transcripts.fa.gz

使い方:
    python fetch_gencode.py --species human
    python fetch_gencode.py --species mouse
    python fetch_gencode.py --species both

出力:
    ../processed/gencode_v44_lncrna.gtf
    ../processed/gencode_v44_lncrna_sequences.fa
    ../processed/gencode_vM33_lncrna.gtf
    ../processed/gencode_vM33_lncrna_sequences.fa
"""

import argparse
import gzip
import logging
import shutil
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GENCODE_BASE = "https://ftp.ebi.ac.uk/pub/databases/gencode"

RELEASES = {
    "human": {
        "version": "44",
        "base": f"{GENCODE_BASE}/Gencode_human/release_44",
        "files": {
            "gtf": "gencode.v44.long_noncoding_RNAs.gtf.gz",
            "fa": "gencode.v44.lncRNA_transcripts.fa.gz",
        },
        "out_prefix": "gencode_v44_lncrna",
    },
    "mouse": {
        "version": "M33",
        "base": f"{GENCODE_BASE}/Gencode_mouse/release_M33",
        "files": {
            "gtf": "gencode.vM33.long_noncoding_RNAs.gtf.gz",
            "fa": "gencode.vM33.lncRNA_transcripts.fa.gz",
        },
        "out_prefix": "gencode_vM33_lncrna",
    },
}


def download_file(url: str, dest: Path) -> None:
    """Stream-download with skip-if-exists."""
    if dest.exists():
        log.info(f"Already exists, skipping: {dest.name}")
        return
    log.info(f"Downloading {url}")
    log.info(f"  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def gunzip(src: Path, dest: Path) -> None:
    """Decompress .gz to dest (skip if exists)."""
    if dest.exists():
        log.info(f"Already decompressed, skipping: {dest.name}")
        return
    log.info(f"Decompressing {src.name} -> {dest.name}")
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def fetch_species(species: str, raw_dir: Path, processed_dir: Path) -> None:
    cfg = RELEASES[species]
    gz_gtf = raw_dir / cfg["files"]["gtf"]
    gz_fa = raw_dir / cfg["files"]["fa"]

    download_file(f"{cfg['base']}/{cfg['files']['gtf']}", gz_gtf)
    download_file(f"{cfg['base']}/{cfg['files']['fa']}", gz_fa)

    out_gtf = processed_dir / f"{cfg['out_prefix']}.gtf"
    out_fa = processed_dir / f"{cfg['out_prefix']}_sequences.fa"
    gunzip(gz_gtf, out_gtf)
    gunzip(gz_fa, out_fa)

    # Quick summary
    n_lines = sum(1 for _ in open(out_gtf) if not _.startswith("#"))
    n_seqs = sum(1 for line in open(out_fa) if line.startswith(">"))
    log.info(f"{species.upper()} (GENCODE {cfg['version']}):")
    log.info(f"  GTF entries (non-header): {n_lines}")
    log.info(f"  FASTA sequences: {n_seqs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        choices=["human", "mouse", "both"],
        default="both",
        help="Which species annotation to fetch",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "gencode",
        help="Directory for downloaded .gz files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed",
        help="Directory for decompressed GTF/FASTA",
    )
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    targets = ["human", "mouse"] if args.species == "both" else [args.species]
    for sp in targets:
        fetch_species(sp, args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
