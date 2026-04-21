#!/usr/bin/env python3
"""
deeplncloc.py — k-mer composition baseline (Task 2.5, DeepLncLoc substitute)

置換の経緯:
    元計画では DeepLncLoc (Zeng et al. 2022 Brief Bioinform) の 5-class 局在予測確率を
    取得する予定だった。しかし:
      - 公式レポ (CSUBioGroup/DeepLncLoc) は事前学習済み重みを配布していない
        （out/ ディレクトリは空、GitHub Releases なし、著者連絡が必要）
      - 後継の RNALoc-LM はあるが FASTA → 予測の直結 API がなく、
        Google Drive weights のダウンロードと pickle データ変換が必要
      - RNAlight は binary (cyto/nuc) のみで出力情報が 1次元、かつ python 3.6.10
        固定で現行 Colab 環境と非互換

    代替として 3-mer composition (64-dim) を「手作りベースライン」として採用する。
    k-mer 組成は lncRNA の局在 (Zuckerman & Ulitsky 2019) および半減期 (Clark et al.
    2012, Mukherjee et al. 2017) 双方と相関することが報告されている。
    5-dim 局在確率より情報量が多く、LLM embedding との比較がより厳しくなる。

出力:
    `deeplncloc.npz` (keys: gene_ids, embeddings[N, 64], labels)
    labels = 64個の 3-mer 文字列 ('AAA', 'AAC', ..., 'TTT')

実行環境:
    CPU 数秒。依存は numpy のみ。
"""

import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

K = 3
ALPHABET = "ACGT"
KMERS = ["".join(p) for p in product(ALPHABET, repeat=K)]  # 64 kmers
KMER_INDEX = {k: i for i, k in enumerate(KMERS)}


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    entries, cur_id, cur_seq = [], None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    entries.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            entries.append((cur_id, "".join(cur_seq)))
    log.info(f"Loaded {len(entries)} sequences from {path.name}")
    return entries


def kmer_composition(seq: str) -> np.ndarray:
    """3-mer frequency vector (64-dim, sums to 1 over valid k-mers)."""
    s = seq.upper().replace("U", "T")
    counts = np.zeros(len(KMERS), dtype=float)
    total = 0
    for i in range(len(s) - K + 1):
        kmer = s[i : i + K]
        idx = KMER_INDEX.get(kmer)
        if idx is not None:  # skip k-mers with N or other non-ACGT chars
            counts[idx] += 1
            total += 1
    return counts / total if total > 0 else counts


def extract_kmer_embeddings(
    entries: list[tuple[str, str]],
) -> tuple[list[str], np.ndarray, dict]:
    gene_ids, vecs = [], []
    info = {"model": "kmer-3-composition", "n_seq": 0, "runtime_s": 0.0, "dim": len(KMERS)}
    t0 = time.time()
    for gid, seq in entries:
        vecs.append(kmer_composition(seq))
        gene_ids.append(gid)
        info["n_seq"] += 1
    info["runtime_s"] = time.time() - t0
    emb = np.stack(vecs, axis=0)
    log.info(f"k-mer composition matrix shape: {emb.shape}, runtime: {info['runtime_s']:.2f}s")
    return gene_ids, emb, info


def append_compute_log(log_md: Path, model_name: str, info: dict) -> None:
    log_md.parent.mkdir(parents=True, exist_ok=True)
    with open(log_md, "a") as f:
        f.write(f"\n## {model_name}\n")
        for k, v in info.items():
            f.write(f"- {k}: {v}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-fa",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "processed" / "test_set_sequences.fa",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "embeddings" / "deeplncloc.npz",
    )
    parser.add_argument(
        "--compute-log",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "compute_log.md",
    )
    args = parser.parse_args()

    if not args.input_fa.exists():
        log.error(f"Input FASTA missing: {args.input_fa}")
        sys.exit(1)

    entries = parse_fasta(args.input_fa)
    gene_ids, emb, info = extract_kmer_embeddings(entries)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        gene_ids=np.array(gene_ids),
        embeddings=emb,
        labels=np.array(KMERS),
    )
    log.info(f"Wrote {args.output_npz} (shape {emb.shape})")

    append_compute_log(args.compute_log, "k-mer-3-composition (DeepLncLoc substitute)", info)


if __name__ == "__main__":
    main()
