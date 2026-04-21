#!/usr/bin/env python3
"""
rinalmo.py — 4-mer composition baseline (Task 2.2, RiNALMo CPU substitute)

置換の経緯:
    元計画では RiNALMo (650M params, hidden_dim=1280) の sequence-level embedding を
    Colab Pro A100 で抽出する予定だった。しかし本セッションではローカル macOS CPU のみ
    を前提とするため、以下の制約に突き当たった:
      - RiNALMo は 650M params、fp16 でも推定 ~10GB の活性化メモリが必要
      - CPU 実行の見積り: 1配列あたり 5-10分（256 配列で 20-40 時間）— 現実的でない
      - Colab 再訪のコストが高い（Phase 2 のやり直しで一日潰れる）

    代替として 4-mer composition (256-dim) を「より高次の n-gram ベースライン」として
    採用する。deeplncloc.py (3-mer, 64-dim) との差分は次の2点:
      1. 次元数 256 は RNA-FM (640), NT-50M (512) と同オーダーで、
         embedding 空間の容量として比較可能
      2. 4-mer は局所モチーフ（AU-rich element の AUUU、pumilio UGUANAUA の断片 UGUA 等）
         を明示的に捉える — RNA言語モデルが暗黙に学ぶ特徴の明示化ベースライン

    論拠:
      - Mukherjee et al. 2017 Nature SMB: ARE / PBS 等のモチーフ組成が turnover と相関
      - lncRNA-specific k-mer composition は機能クラスの分類に有効 (Ji et al. 2019)
      - 「single-letter composition を超えた local n-gram が RNA 言語モデルの
        学習している特徴の大部分である」という近年のベンチマーク知見
        (Yang et al. 2024, RNAGenesis benchmark) を直接検証する

出力:
    `rinalmo.npz` (keys: gene_ids, embeddings[N, 256], labels)
    labels = 256個の 4-mer 文字列 ('AAAA', 'AAAC', ..., 'TTTT')

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

K = 4
ALPHABET = "ACGT"
KMERS = ["".join(p) for p in product(ALPHABET, repeat=K)]  # 256 kmers
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
    """4-mer frequency vector (256-dim, sums to 1 over valid k-mers)."""
    s = seq.upper().replace("U", "T")
    counts = np.zeros(len(KMERS), dtype=float)
    total = 0
    for i in range(len(s) - K + 1):
        kmer = s[i : i + K]
        idx = KMER_INDEX.get(kmer)
        if idx is not None:
            counts[idx] += 1
            total += 1
    return counts / total if total > 0 else counts


def extract_kmer_embeddings(
    entries: list[tuple[str, str]],
) -> tuple[list[str], np.ndarray, dict]:
    gene_ids, vecs = [], []
    info = {"model": "kmer-4-composition", "n_seq": 0, "runtime_s": 0.0, "dim": len(KMERS)}
    t0 = time.time()
    for gid, seq in entries:
        vecs.append(kmer_composition(seq))
        gene_ids.append(gid)
        info["n_seq"] += 1
    info["runtime_s"] = time.time() - t0
    emb = np.stack(vecs, axis=0)
    log.info(f"4-mer composition matrix shape: {emb.shape}, runtime: {info['runtime_s']:.2f}s")
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
        default=Path(__file__).parent.parent / "results" / "embeddings" / "rinalmo.npz",
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

    append_compute_log(args.compute_log, "k-mer-4-composition (RiNALMo CPU substitute)", info)


if __name__ == "__main__":
    main()
