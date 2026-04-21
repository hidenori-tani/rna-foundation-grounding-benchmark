#!/usr/bin/env python3
"""
_make_notebooks.py — Generate Phase 2 Colab notebooks.

Each model-specific .ipynb is written with a common layout:
  1. Intro markdown (purpose, GPU requirement)
  2. Drive mount + prerequisites
  3. Package install
  4. Upload test_set_sequences.fa check
  5. Run model extraction
  6. Download embeddings.npz back to local

Usage:
    python _make_notebooks.py

The Colab path to the repository root can be overridden via the
`REPO_PATH` environment variable; by default the repo is assumed to be
mounted at `/content/drive/MyDrive/rna-foundation-grounding-benchmark`.
"""

import json
import os
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent

# Default path when the repository is uploaded to Google Drive.
# Override with the REPO_PATH environment variable when generating notebooks
# for a different layout.
DEFAULT_REPO_PATH = "/content/drive/MyDrive/rna-foundation-grounding-benchmark"
BASE_PATH_IN_COLAB = os.environ.get("REPO_PATH", DEFAULT_REPO_PATH)

MODELS = [
    {
        "name": "rna_fm",
        "display": "RNA-FM",
        "gpu": "T4 (free tier OK)",
        "runtime_hint": "GPU (T4), no need for Pro",
        "extra_install": "multimolecule>=0.0.5 transformers>=4.36",
        "notes": "100M parameters. ~10 min for 100 sequences on free Colab T4.",
    },
    {
        "name": "rinalmo",
        "display": "RiNALMo",
        "gpu": "A100 recommended (Colab Pro)",
        "runtime_hint": "GPU (A100)",
        "extra_install": "multimolecule>=0.0.5 transformers>=4.36",
        "notes": "650M parameters. ~30 min for 100 sequences on A100. T4 risks OOM.",
    },
    {
        "name": "evo",
        "display": "Evo (7B)",
        "gpu": "A100 80GB / H100 (Colab Pro+ or Lambda Labs)",
        "runtime_hint": "GPU (A100 80GB) — use --fallback-nt on OOM",
        "extra_install": "transformers>=4.36 accelerate",
        "notes": "7B StripedHyena. OOM is certain below 40GB. On OOM, run the fallback cell.",
    },
    {
        "name": "rhofold_plus",
        "display": "RhoFold+",
        "gpu": "T4 or A100",
        "runtime_hint": "GPU (T4 is enough)",
        "extra_install": "biopython einops",
        "notes": "Requires a clone of the RhoFold repo. 3D prediction is slow (3-5h / 100 sequences).",
    },
    {
        "name": "deeplncloc",
        "display": "DeepLncLoc",
        "gpu": "CPU is enough",
        "runtime_hint": "CPU (GPU not required)",
        "extra_install": "tensorflow",
        "notes": "CNN-LSTM, ~10M params. Completes in ~5 min.",
    },
]


def cell_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def cell_code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def build_notebook(model: dict) -> dict:
    name = model["name"]
    display = model["display"]
    extra = model["extra_install"]
    notes = model["notes"]
    gpu = model["gpu"]
    runtime = model["runtime_hint"]

    cells = [
        cell_md(
            f"# Phase 2 — {display} embedding extraction\n"
            f"\n"
            f"**Purpose**: Run `test_set_sequences.fa` through {display} and "
            f"write `benchmark/results/embeddings/{name}.npz`.\n"
            f"\n"
            f"- **GPU**: {gpu}\n"
            f"- **Runtime setting**: Runtime → Change runtime type → {runtime}\n"
            f"- **Notes**: {notes}\n"
            f"\n"
            f"This notebook assumes Phase 1 has already produced "
            f"`data/processed/test_set_sequences.fa` and that the repository is "
            f"available on Drive at the path set in the next cell "
            f"(default: `/content/drive/MyDrive/rna-foundation-grounding-benchmark`).\n"
        ),
        cell_md("## 1. Drive mount"),
        cell_code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
        ),
        cell_code(
            f"import os\n"
            f"REPO = os.environ.get('REPO_PATH', '{BASE_PATH_IN_COLAB}')\n"
            f"assert os.path.isdir(REPO), f'Repo not found at {{REPO}}. Upload or adjust REPO_PATH.'\n"
            f"os.chdir(REPO)\n"
            f"print('CWD:', os.getcwd())\n"
        ),
        cell_md("## 2. Install dependencies"),
        cell_code(
            f"!pip install -q numpy pandas scipy scikit-learn torch biopython\n"
            f"!pip install -q {extra}\n"
        ),
        cell_md("## 3. Verify GPU"),
        cell_code(
            "import torch\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('Device:', torch.cuda.get_device_name(0))\n"
            "    print('Memory (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)\n"
        ),
        cell_md("## 4. Verify input exists"),
        cell_code(
            "FA = f'{REPO}/data/processed/test_set_sequences.fa'\n"
            "assert os.path.isfile(FA), f'Missing {FA}. Run Phase 1 first.'\n"
            "n = 0\n"
            "with open(FA) as f:\n"
            "    for line in f:\n"
            "        if line.startswith('>'): n += 1\n"
            "print(f'{n} sequences found in test_set_sequences.fa')\n"
        ),
        cell_md(f"## 5. Run {display} extraction"),
        cell_code(
            f"import sys\n"
            f"sys.path.insert(0, f'{{REPO}}/benchmark')\n"
            f"!python {{REPO}}/benchmark/models/{name}.py\n"
        ),
    ]

    if name == "evo":
        cells.append(cell_md("## 5b. OOM fallback — Nucleotide Transformer 2.5B"))
        cells.append(cell_code(
            "# Run this cell only if Evo 7B fails with OOM\n"
            f"!python {{REPO}}/benchmark/models/{name}.py --fallback-nt\n"
        ))

    cells += [
        cell_md("## 6. Verify output"),
        cell_code(
            f"import numpy as np\n"
            f"OUT = f'{{REPO}}/benchmark/results/embeddings/{name}.npz'\n"
            f"assert os.path.isfile(OUT), f'Extraction failed: {{OUT}} not found'\n"
            f"z = np.load(OUT, allow_pickle=True)\n"
            f"print('keys:', list(z.keys()))\n"
            f"print('n_genes:', len(z['gene_ids']))\n"
            f"print('embedding shape:', z['embeddings'].shape)\n"
        ),
        cell_md(
            "## 7. Download to local (optional)\n"
            "\n"
            "The output is already on Drive; this cell is only for pulling a copy "
            "straight to your local machine.\n"
        ),
        cell_code(
            "from google.colab import files\n"
            f"files.download(f'{{REPO}}/benchmark/results/embeddings/{name}.npz')\n"
        ),
        cell_md(
            f"## Done — next step\n"
            f"\n"
            f"Once `{name}.npz` is generated, either move on to the next model "
            f"notebook, or — once all five are present — run "
            f"`python benchmark/eval.py` locally.\n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for model in MODELS:
        nb = build_notebook(model)
        out = NOTEBOOK_DIR / f"{model['name']}.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
