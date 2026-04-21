# `benchmark/`

Phase 2 benchmark code for the Perspective *Grounding RNA foundation models in
transcript dynamics*. See the [root README](../README.md) for scientific
context, citation and quick start. This file documents the module layout and
the expected run order.

## Layout

```
benchmark/
├── models/
│   ├── rna_fm.py           # RNA-FM (multimolecule/rnafm, direct)
│   ├── rinalmo.py          # 4-mer composition proxy for RiNALMo 650M
│   ├── evo.py              # ERNIE-RNA 86M proxy for Evo 7B (+ NT 2.5B fallback)
│   ├── rhofold_plus.py     # ViennaRNA 2D descriptors
│   └── deeplncloc.py       # DeepLncLoc (3-mer, direct)
├── classifiers.py          # Logistic Regression + MLP heads
├── eval.py                 # 5-fold stratified CV + leave-one-cell-out CV
├── ablation.py             # Length-stratified analysis
├── interpretability.py     # Integrated Gradients (MLP) + SHAP (LR)
├── colab/                  # GPU notebooks for full-weight replication
└── results/
    ├── embeddings/         # {model}.npz
    ├── structures/         # RhoFold+ PDB outputs (optional)
    ├── feature_importance/ # per-gene interpretability JSON
    ├── metrics_table.csv
    ├── metrics_summary.csv
    ├── ablation_results.csv
    └── compute_log.md
```

## Run order

Phase 1 must have produced `data/processed/test_set_sequences.fa` and
`test_set_final.csv`.

```bash
# Extract embeddings for the five representations
python models/rna_fm.py
python models/rinalmo.py
python models/evo.py
python models/evo.py --fallback-nt          # on OOM, GPU config only
python models/rhofold_plus.py --rhofold-repo ~/RhoFold
python models/deeplncloc.py --deeplncloc-repo ~/DeepLncLoc

# Cross-validated evaluation and downstream analyses
python classifiers.py --self-test
python eval.py
python ablation.py
python interpretability.py
```

## Compute requirements

The CPU-feasible configuration reported in the manuscript completes on a
modern MacBook (M-series CPU) in approximately 5-8 hours end-to-end.
Full-weight replication (RiNALMo 650M / Evo 7B) requires GPUs; see
[colab/README.md](colab/README.md) for per-model estimates.

| Model | CPU-only | GPU | Time (100 seq) | Est. cost |
|---|---|---|---|---|
| RNA-FM | ✓ | T4 | 10 min | $0 |
| RiNALMo (650M) | proxy only | A100 (Pro) | 30 min | ~$10 |
| Evo (7B) | proxy only | A100 80GB / H100 | 1 h | $30-60 |
| RhoFold+ (structure) | proxy only | T4 | 3-5 h | Pro session |
| DeepLncLoc | ✓ | — | 5 min | $0 |

## Dependencies

Install via `pip install -r requirements.txt`, or use the conda environment
file at the repository root (`environment.yml`).

## Troubleshooting

- **Evo OOM on Colab A100 (40GB)**: run `evo.py --fallback-nt` or execute
  Section 5b in `colab/evo.ipynb` to use Nucleotide Transformer 2.5B.
- **AUROC spread < 0.05 across all five models**: rerun `eval.py` after
  switching `test_set_final.csv` to the tertile label column (the script
  emits a warning when this is triggered).
