# Phase 2 Colab Notebooks

Auto-generated Colab notebooks for running five RNA AI models on the test set.

## Suggested run order

| # | Notebook | GPU | Time (100 sequences) | Notes |
|---|---|---|---|---|
| 1 | `rna_fm.ipynb` | T4 (free tier) | ~10 min | Good first sanity-check. |
| 2 | `deeplncloc.ipynb` | CPU | ~5 min | No GPU needed. Can run in parallel. |
| 3 | `rinalmo.ipynb` | A100 (Pro) | ~30 min | A100 is required; T4 OOMs. |
| 4 | `rhofold_plus.ipynb` | T4 | 3-5 h | Enable checkpointing in case of disconnect. |
| 5 | `evo.ipynb` | A100 80GB / H100 | ~1 h | On OOM use the Section 5b fallback (NT 2.5B). |

## Prerequisites

Phase 1 outputs (`data/processed/test_set_sequences.fa` and `test_set_final.csv`)
must already exist on Google Drive at the repository root. By default the
notebooks expect the repository at
`/content/drive/MyDrive/rna-foundation-grounding-benchmark`; override with the
`REPO_PATH` environment variable.

```text
<REPO_PATH>/
├── data/processed/test_set_sequences.fa
├── data/processed/test_set_final.csv
└── benchmark/
    ├── classifiers.py
    ├── eval.py
    └── models/*.py
```

## Common notebook layout

Every `.ipynb` has the same cell sequence:

1. Drive mount
2. Set working directory (reads `REPO_PATH`)
3. pip install (model-specific dependencies)
4. GPU check
5. Input FASTA sanity check
6. Model run (`!python benchmark/models/{name}.py`)
7. Validate output `.npz`
8. Optional local download

## Regenerating the notebooks

Edit `_make_notebooks.py` and rerun:

```bash
cd benchmark/colab
python3 _make_notebooks.py
```

This overwrites all five `.ipynb` files. Do not hand-edit the notebooks; any
manual change is lost on the next regeneration.

## Evo OOM decision tree

- **Colab Pro A100 (40GB)**: Evo 7B OOMs → use the fallback cell (NT 2.5B).
- **Colab Pro+ A100 (80GB)**: Evo 7B fits.
- **Lambda Labs H100 spot**: Evo 7B fits. Estimated cost $30-60.

NT 2.5B is an acceptable substitute for the benchmark comparison.
