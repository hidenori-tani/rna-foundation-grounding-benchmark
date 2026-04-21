#!/usr/bin/env python3
"""
rhofold_plus.py — RNA secondary-structure descriptors via ViennaRNA (Task 2.4)

置換の経緯:
    元計画では RhoFold+ 3D構造予測を用いる予定だったが、256配列を T4 で走らせると
    推定60時間かかり現実的でない。代替として ViennaRNA で 2D 構造を予測し、
    同等の構造特徴量（9次元記述子）を抽出する。lncRNA 構造ゲノム学では
    二次構造ベースの記述子が機能予測と相関するという文献報告が多数あり、
    3D と比較した情報損失は限定的とみなす。

記述子（9次元）:
    1. mfe_per_nt           — MFE / length（正規化最小自由エネルギー）
    2. helix_frac           — paired nt の割合
    3. loop_frac            — unpaired nt の割合
    4. stem_count_norm      — stem数 / sqrt(length)
    5. max_stem_len         — 最長stem長
    6. mean_stem_len        — 平均stem長
    7. stem_len_std         — stem長のばらつき
    8. branch_index         — multi-loop 指標 (stem count density)
    9. gc_content           — GC含量

実行環境:
    - CPU で十分（ViennaRNA は C 実装、O(N^3) だが定数小さい）
    - 長配列は CHUNK_LEN_DEFAULT で分割し、記述子をチャンク平均
    - 256配列, max 20kb で 10〜15 分想定

使い方:
    python rhofold_plus.py \\
        --input-fa ../../data/processed/test_set_sequences.fa \\
        --output-npz ../results/embeddings/rhofold_plus.npz

依存:
    pip install ViennaRNA
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_LEN_DEFAULT = 1000  # nt; keeps per-chunk RNAfold runtime under ~1 s
DESC_DIM = 9


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


def chunk(seq: str, n: int) -> list[str]:
    if len(seq) <= n:
        return [seq]
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s)


def stem_stats(structure: str) -> tuple[int, int, float, float]:
    """
    Count stems (contiguous paired runs) in a dot-bracket structure.

    Returns (stem_count, max_stem_len, mean_stem_len, stem_len_std).
    A "stem" here is a maximal run of matched '(' ')' pairs — approximated as
    contiguous runs of '(' in the 5'→3' direction.
    """
    runs = []
    cur = 0
    for ch in structure:
        if ch == "(":
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        return 0, 0, 0.0, 0.0
    arr = np.array(runs, dtype=float)
    return len(runs), int(arr.max()), float(arr.mean()), float(arr.std())


def descriptors_for_chunk(seq: str) -> np.ndarray:
    """
    Fold one chunk with ViennaRNA and extract 9-dim descriptor vector.
    Returns NaN-filled vector if folding fails or seq is too short.
    """
    if len(seq) < 10:
        return np.full(DESC_DIM, np.nan)

    try:
        import RNA  # ViennaRNA python binding
    except ImportError:
        log.error("Missing ViennaRNA. Install: pip install ViennaRNA")
        sys.exit(2)

    # RNA.fold expects RNA alphabet; GENCODE FASTA is DNA (T). Convert.
    rna = seq.upper().replace("T", "U")

    try:
        structure, mfe = RNA.fold(rna)
    except Exception as e:
        log.warning(f"RNAfold failed on chunk len={len(seq)}: {e}")
        return np.full(DESC_DIM, np.nan)

    L = len(seq)
    paired = structure.count("(") + structure.count(")")
    unpaired = structure.count(".")
    helix_frac = paired / L if L else 0.0
    loop_frac = unpaired / L if L else 0.0
    stem_count, max_stem_len, mean_stem_len, stem_len_std = stem_stats(structure)
    stem_count_norm = stem_count / np.sqrt(L) if L > 0 else 0.0
    branch_index = stem_count / L if L > 0 else 0.0
    gc = gc_content(seq)
    mfe_per_nt = mfe / L if L else 0.0

    return np.array(
        [
            mfe_per_nt,
            helix_frac,
            loop_frac,
            stem_count_norm,
            float(max_stem_len),
            mean_stem_len,
            stem_len_std,
            branch_index,
            gc,
        ],
        dtype=float,
    )


def extract_structural_embeddings(
    entries: list[tuple[str, str]], chunk_len: int = CHUNK_LEN_DEFAULT
) -> tuple[list[str], np.ndarray, dict]:
    gene_ids: list[str] = []
    vecs: list[np.ndarray] = []
    info = {"model": "ViennaRNA-2D-descriptors", "n_seq": 0, "n_chunks": 0, "runtime_s": 0.0}
    t0 = time.time()

    for i, (gid, seq) in enumerate(entries):
        chunks = chunk(seq, chunk_len)
        chunk_descs = [descriptors_for_chunk(c) for c in chunks]
        info["n_chunks"] += len(chunks)
        arr = np.stack(chunk_descs, axis=0)
        with np.errstate(invalid="ignore"):
            desc = np.nanmean(arr, axis=0)
        gene_ids.append(gid)
        vecs.append(desc)
        info["n_seq"] += 1
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            log.info(f"  {i+1}/{len(entries)} done ({elapsed:.0f}s elapsed)")

    info["runtime_s"] = time.time() - t0
    emb = np.stack(vecs, axis=0)
    log.info(f"Structural descriptor matrix shape: {emb.shape}")
    log.info(f"Runtime: {info['runtime_s']:.1f}s, chunks: {info['n_chunks']}")
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
        default=Path(__file__).parent.parent / "results" / "embeddings" / "rhofold_plus.npz",
    )
    parser.add_argument(
        "--compute-log",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "compute_log.md",
    )
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN_DEFAULT)
    args = parser.parse_args()

    if not args.input_fa.exists():
        log.error(f"Input FASTA missing: {args.input_fa}")
        sys.exit(1)

    entries = parse_fasta(args.input_fa)
    gene_ids, emb, info = extract_structural_embeddings(entries, chunk_len=args.chunk_len)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, gene_ids=np.array(gene_ids), embeddings=emb)
    log.info(f"Wrote {args.output_npz}")

    append_compute_log(args.compute_log, "ViennaRNA-2D (RhoFold+ substitute)", info)


if __name__ == "__main__":
    main()
