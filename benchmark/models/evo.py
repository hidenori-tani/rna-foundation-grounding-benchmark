#!/usr/bin/env python3
"""
evo.py — ERNIE-RNA (multimolecule/ernierna) embedding 抽出
         (Task 2.3, Evo 7B CPU substitute)

置換の経緯:
    元計画では Evo 7B (togethercomputer/evo-1-8k-base) を Colab Pro+ A100 80GB で実行し、
    StripedHyena アーキテクチャによる DNA 文脈 embedding を取得する予定だった。しかし
    本セッションではローカル macOS CPU のみを前提とするため:
      - Evo 7B は bf16 でも推定 ~14GB 必要、CPU 推論は非現実的
      - StripedHyena は CUDA カーネル依存で macOS MPS/CPU 未検証
      - fallback 候補の Nucleotide Transformer v2 50M は transformers 5.x と
        複数箇所で互換性破綻（`find_pruneable_heads_and_indices` 廃止、
        `EsmConfig.is_decoder` 属性欠落、他）。NT v2 の custom code は
        transformers 4.x 系を前提としており、本環境の 5.5.4 では動かせない

    代替として **ERNIE-RNA** (multimolecule/ernierna, Yin et al. 2024) を採用する。
    ERNIE-RNA は:
      - 86M パラメータ, hidden_dim=768, CPU で 1配列 5-15秒
      - RNAcentral の ncRNA 配列で pre-training した Transformer 系 foundation model
      - multimolecule ライブラリ経由で transformers 5.x との互換性を維持
      - RNA-FM の "first-generation" に対し ERNIE-RNA は "post-2023 generation" 相当で、
        「foundation model 世代間比較」としての位置付けも明確

    論拠:
      - 本総説の命題A（静的 foundation model は turnover 予測が頭打ち）を問うには
        DNA系 Evo でも RNA系 ERNIE-RNA でも同じ論理が成立する — どちらも
        配列の静的特徴で学習されており、dynamics を学習していない
      - ERNIE-RNA は下流タスク（RNA-RNA interaction, 機能分類）で RNA-FM を
        上回ることが報告されている (Yin et al. 2024)
      - したがって本セッションでは「DNA系 vs RNA系 vs k-mer」の 3軸ではなく、
        「RNA-FM (RNA foundation v1) vs ERNIE-RNA (RNA foundation v2) vs k-mer」
        という RNA 系内での世代比較になる。Evo 7B を元計画通り走らせる版は
        Phase 3 以降の GPU セッションで追加する

出力:
    `evo.npz` (keys: gene_ids, embeddings[N, 768])

実行環境:
    CPU 数分〜十数分（256 配列で想定）。
    依存: torch, multimolecule。
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def chunk_sequence(seq: str, max_len: int = 1022) -> list[str]:
    """Split into non-overlapping chunks; default 1022 leaves room for CLS/SEP."""
    if len(seq) <= max_len:
        return [seq]
    return [seq[i : i + max_len] for i in range(0, len(seq), max_len)]


def extract_ernie_rna(
    entries: list[tuple[str, str]],
    max_chunk_len: int = 1022,
) -> tuple[list[str], np.ndarray, dict]:
    try:
        import torch
        from multimolecule import RnaTokenizer, ErnieRnaModel
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Install: pip install torch multimolecule")
        sys.exit(2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading ERNIE-RNA on {dev}")

    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ernierna")
    model = ErnieRnaModel.from_pretrained("multimolecule/ernierna").to(dev).eval()

    gene_ids, embeddings = [], []
    info = {
        "model": "ERNIE-RNA (multimolecule/ernierna)",
        "n_seq": 0,
        "runtime_s": 0.0,
        "hidden_dim": None,
    }
    t0 = time.time()

    import gc

    with torch.no_grad():
        for i, (gid, seq) in enumerate(entries):
            chunks = chunk_sequence(seq, max_len=max_chunk_len)
            chunk_embs: list[np.ndarray] = []
            for chunk in chunks:
                toks = tokenizer(
                    chunk, return_tensors="pt", max_length=1024, truncation=True
                ).to(dev)
                out = model(**toks)
                hidden = out.last_hidden_state.squeeze(0)  # [L, D]
                chunk_embs.append(hidden.float().mean(dim=0).cpu().numpy())
                del toks, out, hidden
            emb = np.mean(np.stack(chunk_embs, axis=0), axis=0)
            gene_ids.append(gid)
            embeddings.append(emb)
            info["n_seq"] += 1
            if info["hidden_dim"] is None:
                info["hidden_dim"] = int(emb.shape[0])
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info(f"  {i + 1}/{len(entries)} done  elapsed={elapsed:.0f}s")
                gc.collect()

    info["runtime_s"] = time.time() - t0
    log.info(
        f"ERNIE-RNA embeddings: shape=({info['n_seq']}, {info['hidden_dim']}), "
        f"runtime={info['runtime_s']:.1f}s"
    )
    return gene_ids, np.stack(embeddings, axis=0), info


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
        default=Path(__file__).parent.parent / "results" / "embeddings" / "evo.npz",
    )
    parser.add_argument(
        "--compute-log",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "compute_log.md",
    )
    parser.add_argument("--max-chunk-len", type=int, default=1022)
    args = parser.parse_args()

    if not args.input_fa.exists():
        log.error(f"Input FASTA missing: {args.input_fa}")
        sys.exit(1)

    entries = parse_fasta(args.input_fa)
    gene_ids, emb, info = extract_ernie_rna(entries, max_chunk_len=args.max_chunk_len)
    model_tag = "ERNIE-RNA (Evo 7B CPU substitute)"

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, gene_ids=np.array(gene_ids), embeddings=emb)
    log.info(f"Wrote {args.output_npz} using {model_tag}")

    append_compute_log(args.compute_log, model_tag, info)


if __name__ == "__main__":
    main()
