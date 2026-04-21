#!/usr/bin/env python3
"""
rna_fm.py — RNA-FM (multimolecule/rnafm) embedding 抽出（Task 2.1）

目的:
    test_set_sequences.fa の全lncRNA配列から RNA-FM の sequence-level embedding を抽出し、
    `rna_fm.npz` (keys: gene_ids, embeddings[N, D]) として保存する。

モデル:
    - HuggingFace repo: multimolecule/rnafm
    - 元論文: Chen et al. 2022 (CVPR), RNA-FM: a RNA foundation model pre-trained
      on 23M ncRNA sequences
    - hidden_dim=640, 12 layers, trained on RNAcentral
    - 入力長制限: 通常 ≤ 1024 nt（超過配列は 1024 nt チャンクで切り出して平均化）

集約:
    mean-pooling（全 token embeddings の平均） → sequence-level embedding

使い方:
    python rna_fm.py \\
        --input-fa ../../data/processed/test_set_sequences.fa \\
        --output-npz ../results/embeddings/rna_fm.npz

実行環境:
    - CPU 可（遅い）
    - GPU 推奨（T4 で N=100, 10分程度）

NOTE:
    RNA-FM は U/T を区別せず受け取る。GENCODE FASTA は T なので、そのまま入力可。
    ※ multimolecule ライブラリは U/T 自動変換対応。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if current_id is not None:
                    entries.append((current_id, "".join(current_seq)))
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id is not None:
            entries.append((current_id, "".join(current_seq)))
    log.info(f"Loaded {len(entries)} sequences from {path.name}")
    return entries


def chunk_sequence(seq: str, max_len: int = 1022) -> list[str]:
    """Split a sequence into non-overlapping chunks of size max_len.

    Default 1022 leaves room for the tokenizer's [CLS] and [SEP] tokens so the
    final sequence stays within RNA-FM's 1024-position embedding limit.
    """
    if len(seq) <= max_len:
        return [seq]
    return [seq[i : i + max_len] for i in range(0, len(seq), max_len)]


def extract_embeddings(entries: list[tuple[str, str]], device: str = "cuda") -> tuple[list[str], np.ndarray]:
    """
    Return (gene_ids, embeddings[N, D]).

    Uses multimolecule's RNA-FM via HuggingFace transformers.
    """
    try:
        import torch
        from multimolecule import RnaTokenizer, RnaFmModel
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Install: pip install torch multimolecule")
        sys.exit(2)

    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    log.info(f"Using device: {dev}")

    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm").to(dev).eval()

    gene_ids: list[str] = []
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for gid, seq in entries:
            chunks = chunk_sequence(seq, max_len=1022)
            chunk_embs: list[np.ndarray] = []
            for chunk in chunks:
                toks = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(dev)
                out = model(**toks)
                # last_hidden_state: [1, L, D]
                hidden = out.last_hidden_state.squeeze(0)  # [L, D]
                chunk_embs.append(hidden.mean(dim=0).cpu().numpy())
            # Average over chunks
            emb = np.mean(np.stack(chunk_embs, axis=0), axis=0)
            gene_ids.append(gid)
            embeddings.append(emb)

    emb_matrix = np.stack(embeddings, axis=0)
    log.info(f"Extracted embeddings shape: {emb_matrix.shape}")
    nonzero = float((emb_matrix != 0).mean())
    log.info(f"Non-zero rate: {nonzero:.4f}")
    return gene_ids, emb_matrix


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
        default=Path(__file__).parent.parent / "results" / "embeddings" / "rna_fm.npz",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if not args.input_fa.exists():
        log.error(f"Input FASTA missing: {args.input_fa}. Run Phase 1 first.")
        sys.exit(1)

    entries = parse_fasta(args.input_fa)
    gene_ids, emb = extract_embeddings(entries, device=args.device)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, gene_ids=np.array(gene_ids), embeddings=emb)
    log.info(f"Wrote {args.output_npz}")


if __name__ == "__main__":
    main()
