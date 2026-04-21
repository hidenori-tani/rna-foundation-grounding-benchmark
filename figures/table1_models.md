# Table 1. Five RNA AI representation classes benchmarked

Two classes are evaluated directly with their published weights; three are
approximated by CPU-feasible proxies that preserve the representational class
without reproducing the full-parameter inference. The proxy choice is
discussed in Box 1 of the main text; full-weight GPU replication for the
proxied classes is provided in `benchmark/colab/*.ipynb` and will be reported
separately.

| # | Representation class | Direct model (published) | CPU-feasible implementation | Output dim | Licence | Reference |
|---|---|---|---|---|---|---|
| 1 | **RNA-FM** | RNA-FM (100 M params, BERT-like, RNAcentral) | same — `multimolecule/rnafm` | 640 | MIT | Chen et al. 2022 |
| 2 | **RiNALMo-class** | RiNALMo 650 M (33 layers, hidden 1,280) | random-initialised shallow 1-D CNN (3 conv layers, 256-dim mean-pooled, seed 42) | 256 | — | Penić et al. 2025 *Nat. Commun.* |
| 3 | **Evo-class** | Evo 7 B (StripedHyena, 2.7 M prokaryotic genomes) | ERNIE-RNA (`multimolecule/ernierna`, 86 M) | 768 | Apache-2.0 / — | Nguyen et al. 2024 *Science* / Yin et al. 2025 *Nat. Commun.* |
| 4 | **RhoFold+-class** | RhoFold+ (~80 M, 3-D structure predictor) | ViennaRNA 2-D thermodynamic descriptors (9-dim) | 9 | Academic / ViennaRNA GPL | Shen et al. 2024 *Nat. Methods* / Lorenz et al. 2011 |
| 5 | **DeepLncLoc** | DeepLncLoc (CNN-LSTM, 5-class localisation) | same — 3-mer composition head (published implementation) | 64 | Academic / — | Zeng et al. 2022 *Brief. Bioinform.* |

## Rationale for the proxy choices

- **RiNALMo → random shallow CNN**: full-parameter RiNALMo 650 M is not CPU-tractable (fp16 ≳ 10 GB activations; ~5–10 min per sequence → ~20–40 h for 256 sequences). A randomly-initialised 3-layer 1-D CNN (256-dim mean-pooled output, fixed seed) retains the neural-architecture inductive bias — learned-shape non-linear local filters — while being CPU-feasible (<1 s total). Crucially, it is *architecturally distinct from a k-mer count* (non-linear activations, position-dependent filters), so the "all representations match the k-mer baseline" observation remains informative rather than tautological. Random-feature baselines (Rahimi & Recht, NeurIPS 2007) establish the principled comparator role.
- **Evo 7 B → ERNIE-RNA**: Evo 7 B bf16 requires ~14 GB; StripedHyena depends on CUDA kernels and is not verified on macOS MPS/CPU. ERNIE-RNA (Yin et al. 2025, 86 M) runs on CPU at 5–15 s per sequence and represents the "post-2023 RNA foundation model" generation. Nucleotide Transformer v2 (50 M) was excluded due to transformers 5.x API breakage.
- **RhoFold+ → ViennaRNA 2-D descriptors**: RhoFold+ 3-D structure prediction takes minutes per sequence and is CPU-infeasible. The nine ViennaRNA descriptors (MFE, mean pair probability, loop statistics, etc.) capture secondary-structure information at much coarser resolution than the RhoFold+ embedding, and the resulting performance should be read as a lower bound on the structure-aware representation class.

## Scope of the comparison

- Representation classes covered: sequence-only foundation (RNA-FM), neural-feature extractor (shallow CNN as RiNALMo proxy), post-2023 foundation (ERNIE-RNA as Evo proxy), 2-D thermodynamic structure (ViennaRNA as RhoFold+ proxy), localisation-focused published model (DeepLncLoc).
- Parameter spectrum (where applicable): 0 trainable parameters (ViennaRNA, shallow CNN) through 86 M (ERNIE-RNA). The intended 100 M–7 B spectrum for the full-weight comparison is the subject of the GPU replication under `benchmark/colab/`.
- All implementations rely on public weights or open-source libraries and are reproducible via `reproduce.sh`.
