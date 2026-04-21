# Grounding RNA foundation models in transcript dynamics

Reproducible benchmark and analysis code for the Perspective
*Grounding RNA foundation models in transcript dynamics*
by Hidenori Tani (Yokohama University of Pharmaceutical Sciences).

The benchmark asks whether sequence-only RNA foundation models can predict
long non-coding RNA (lncRNA) turnover across four mammalian cell systems.
Five representative representation classes are evaluated — two directly
(RNA-FM, DeepLncLoc) and three via CPU-feasible proxies (a 4-mer composition
surrogate for RiNALMo 650M, ERNIE-RNA as a surrogate for Evo 7B, and
ViennaRNA 2D descriptors for RhoFold+). All representations plateau below
AUROC 0.70 and perform similarly to simple k-mer baselines, a pattern we
interpret as an *observability gap* between static pretraining corpora and
the dynamic state variables on which lncRNA function depends.

## Citation

> Tani, H. *Grounding RNA foundation models in transcript dynamics.*
> Manuscript submitted. DOI to be updated upon publication.

> Zenodo archive: DOI reserved; link will be added on acceptance.

## Quick start

### Environment

```bash
# Python >= 3.10
python3 --version

# Create an isolated environment (conda shown; venv is equivalent)
conda env create -f environment.yml
conda activate rna-foundation-grounding

# Or, with pip only:
python3 -m venv .venv && source .venv/bin/activate
pip install -r benchmark/requirements.txt
```

### Reproduce Phase 1 (data acquisition, local CPU, ~30-60 min)

```bash
cd data/pipeline
bash run_all.sh
```

This fetches BRIC-seq (Tani 2012), SLAM-seq (Herzog 2017) and TimeLapse-seq
(Schofield 2018) half-life tables, builds a unified merged table, pulls the
GENCODE v44 (human) and vM33 (mouse) lncRNA annotations, and writes
`data/processed/test_set_final.csv` plus `test_set_sequences.fa`.

### Reproduce Phase 2 (model benchmark, ~5-8 h)

#### CPU-feasible configuration (default, reported in the manuscript)

```bash
# Extract embeddings for five models
python benchmark/models/rna_fm.py
python benchmark/models/rinalmo.py      # 4-mer proxy for RiNALMo 650M
python benchmark/models/evo.py          # ERNIE-RNA proxy for Evo 7B
python benchmark/models/rhofold_plus.py # ViennaRNA descriptors
python benchmark/models/deeplncloc.py

# Cross-validated classification + regression
python benchmark/eval.py

# Length-stratified ablation + consensus failure analysis
python benchmark/ablation.py
python benchmark/interpretability.py
```

The CPU proxies are documented in each `models/*.py` docstring.

#### GPU configuration (optional, for full-weight replication)

The five Colab notebooks in `benchmark/colab/` run RNA-FM, RiNALMo 650M,
Evo 7B, RhoFold+ and DeepLncLoc on A100/H100 GPUs. See
[benchmark/colab/README.md](benchmark/colab/README.md) for the suggested
run order and memory requirements. Estimated cost: $50-100 on Colab Pro+
or Lambda Labs spot.

### Regenerate figures

```bash
python figures/fig2_auroc_heatmap.py
python figures/fig3_scatter.py
python figures/fig4_failure_analysis.py
python figures/fig5_framework.py
```

## Repository layout

```
rna-foundation-grounding-benchmark/
├── manuscript/              # Perspective draft (manuscript.md)
├── data/
│   ├── pipeline/            # Phase 1 fetch + merge scripts
│   ├── processed/           # Unified half-life tables + FASTA
│   ├── raw/                 # GEO downloads (gitignored, fetched on demand)
│   └── QC_report.md         # Ground-truth QC summary
├── benchmark/
│   ├── models/              # Five representation extractors
│   ├── classifiers.py       # LR + MLP heads
│   ├── eval.py              # 5-fold stratified + LOCO CV
│   ├── ablation.py          # Length-stratified analysis
│   ├── interpretability.py  # Integrated Gradients + SHAP
│   ├── colab/               # GPU notebooks for full-weight replication
│   └── results/             # metrics_table.csv, embeddings, compute_log.md
├── figures/                 # Publication figures + generators
├── inquiry/                 # Pre-submission correspondence
└── submission/              # Cover letter materials
```

## Data sources

| Dataset | Citation | Accession |
|---|---|---|
| BRIC-seq (HeLa Tet-off) | Tani et al. *Genome Res.* 22, 947 (2012) | GSE30792 |
| SLAM-seq (mESC) | Herzog et al. *Nat. Methods* 14, 1198 (2017) | GSE99970 |
| TimeLapse-seq (K562, MEF) | Schofield et al. *Nat. Methods* 15, 221 (2018) | GSE103493 |
| GENCODE lncRNA (human) | Frankish et al. *Nucleic Acids Res.* 51, D942 (2023) | v44 |
| GENCODE lncRNA (mouse) | (same) | vM33 |

All datasets are public; fetch scripts in `data/pipeline/` reproduce the
downloads and unified annotation.

## Models evaluated

| Label | Full model | Proxy used | Dimension |
|---|---|---|---|
| `rna_fm` | RNA-FM (Chen 2022) | direct | 640 |
| `rinalmo` | RiNALMo 650M (Penić 2025) | 4-mer composition | 256 |
| `evo` | Evo 7B (Nguyen 2024) | ERNIE-RNA 86M (Yin 2025) | 768 |
| `rhofold_plus` | RhoFold+ (Shen 2024) | ViennaRNA 2D descriptors | 9 |
| `deeplncloc` | DeepLncLoc (Zeng 2022) | direct (3-mer) | 64 |

See each `benchmark/models/*.py` docstring for the rationale behind each
proxy and the GPU notebook for full-weight reproduction.

## License

- Code (Python, shell, notebooks): [MIT](LICENSE)
- Data tables and figures: [CC BY 4.0](data/LICENSE)

## Contact

Hidenori Tani, Ph.D.
Associate Professor, Department of Molecular Biology
Yokohama University of Pharmaceutical Sciences
Email: hidenori.tani@yok.hamayaku.ac.jp
ORCID: [0000-0001-6390-4136](https://orcid.org/0000-0001-6390-4136)
