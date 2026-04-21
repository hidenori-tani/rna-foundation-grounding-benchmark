#!/usr/bin/env python3
"""
rinalmo.py — Random shallow 1D-CNN proxy for RiNALMo-650M (representation-class substitute)

Why a random-initialised CNN rather than k-mer composition.
    An earlier draft of this benchmark substituted 4-mer composition for the
    RiNALMo-650M embedding. That choice made the proxy identical to the k-mer
    baseline, so any "proxy matches baseline" observation became tautological
    by construction. We therefore replace the proxy with a randomly-initialised,
    non-linear, convolutional feature extractor: it remains CPU-tractable
    (~30 s for 256 sequences) but is architecturally distinguishable from a
    count-based n-gram and captures local composition through learned-shape
    (though untrained) filters rather than exact windowed counts.

    The proxy is a representation-class stand-in, not a performance claim about
    RiNALMo-650M itself. Its purpose is to answer the question: when the
    neural-CNN inductive bias is stripped of large-scale pretraining, does the
    class already reach the lncRNA-turnover ceiling we observe for directly
    evaluated models? The full-parameter RiNALMo replication is carried out in
    benchmark/colab/rinalmo.ipynb.

Architecture.
    One-hot(4) → Conv1d(4→64, k=7) → ReLU →
                 Conv1d(64→128, k=5) → ReLU →
                 Conv1d(128→256, k=3) → ReLU →
                 global mean pool → 256-dim embedding.
    Weights are initialised with torch's default Kaiming-uniform under a fixed
    random seed (42) for full reproducibility. No training is performed.

Reference.
    Random-feature baselines (Rahimi & Recht, NeurIPS 2007) establish that
    untrained non-linear projections form a principled class-level comparator
    against which learned large-scale representations must demonstrate gain.

Output.
    rinalmo.npz (keys: gene_ids, embeddings[N, 256])
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ALPHABET = "ACGT"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
EMB_DIM = 256
SEED = 42


class RandomShallowCNN(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, emb_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        return h.mean(dim=-1)


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


def one_hot_encode(seq: str) -> np.ndarray:
    s = seq.upper().replace("U", "T")
    L = len(s)
    oh = np.zeros((4, L), dtype=np.float32)
    for i, c in enumerate(s):
        idx = CHAR_TO_IDX.get(c)
        if idx is not None:
            oh[idx, i] = 1.0
    return oh


def extract_cnn_embeddings(
    entries: list[tuple[str, str]],
) -> tuple[list[str], np.ndarray, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = RandomShallowCNN().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gene_ids, vecs = [], []
    info = {
        "model": "random-shallow-CNN-proxy",
        "emb_dim": EMB_DIM,
        "seed": SEED,
        "n_seq": 0,
        "runtime_s": 0.0,
    }
    t0 = time.time()
    with torch.no_grad():
        for gid, seq in entries:
            oh = one_hot_encode(seq)
            x = torch.from_numpy(oh).unsqueeze(0)
            emb = model(x).squeeze(0).numpy()
            vecs.append(emb)
            gene_ids.append(gid)
            info["n_seq"] += 1
    info["runtime_s"] = round(time.time() - t0, 2)
    emb_mat = np.stack(vecs, axis=0)
    log.info(
        f"Shallow-CNN embedding matrix shape: {emb_mat.shape}, "
        f"runtime: {info['runtime_s']:.2f}s"
    )
    return gene_ids, emb_mat, info


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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()

    if not args.input_fa.exists():
        log.error(f"Input FASTA missing: {args.input_fa}")
        sys.exit(1)

    entries = parse_fasta(args.input_fa)
    gene_ids, emb, info = extract_cnn_embeddings(entries)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        gene_ids=np.array(gene_ids),
        embeddings=emb,
    )
    log.info(f"Wrote {args.output_npz} (shape {emb.shape})")

    append_compute_log(args.compute_log, "random-shallow-CNN (RiNALMo-650M CPU proxy)", info)


if __name__ == "__main__":
    main()
