# Table 2. Best-performing representation-classifier combination per evaluation task

**Source**: `benchmark/results/metrics_summary.csv` (Phase 2, CPU-proxy run).
**Test set**: 256 lncRNAs (116 classifiable: stable = 40, unstable = 76; full 256 used for continuous regression).
**CV**: gene-disjoint 5-fold stratified (primary) + leave-one-cell-out (LOCO, diagnostic).
**Representations**: RNA-FM and DeepLncLoc directly evaluated; RiNALMo, Evo and RhoFold+ through CPU-feasible proxies (random shallow 1-D CNN / ERNIE-RNA / ViennaRNA descriptors; see Box 1 and Table 1).

## Classification (binary stable vs unstable)

| Metric | Best combination | 5-fold (mean ± SD) | LOCO (mean ± SD) | Runner-up | Note |
|---|---|---|---|---|---|
| AUROC | **RiNALMo proxy (shallow CNN) + MLP** | **0.694 ± 0.145** | 0.672 ± 0.304 | DeepLncLoc MLP 0.690 ± 0.152 | Four leading MLP combinations (RiNALMo proxy / DeepLncLoc / RNA-FM / Evo proxy) fall within ~1 SD |
| F1 | RNA-FM + LogReg | 0.428 ± 0.168 | — | RhoFold+ proxy LogReg 0.412 ± 0.116 | Pronounced class-imbalance effects |
| MCC | DeepLncLoc + MLP | 0.234 ± 0.228 | — | RNA-FM LogReg 0.194 ± 0.159 | Overall MCC remains low; task is hard |

## Regression (log₂ half-life, hours)

| Metric | Best combination | 5-fold (mean ± SD) | Runner-up | Note |
|---|---|---|---|---|
| Spearman ρ | DeepLncLoc + MLP | 0.186 ± 0.087 | RiNALMo proxy MLP 0.174 ± 0.120 | All representations near chance |
| Pearson r | RNA-FM + MLP | 0.181 ± 0.102 | RiNALMo proxy MLP 0.149 ± 0.119 | — |
| RMSE (log₂ h) | DeepLncLoc + MLP | 1.018 ± 0.092 | RNA-FM MLP 1.025 ± 0.093 | log₂(h) units |
| Spearman ρ (LOCO) | RNA-FM + Ridge | 0.165 ± 0.381 | Evo proxy Ridge 0.159 ± 0.487 | Huge SD from small per-cell N (LOCO diagnostic only) |

## Reading the ranking

- **AUROC spread** across the five representation × three classifier combinations is 0.299 (0.396 RhoFold+ proxy MLP → 0.694 RiNALMo proxy MLP); binary-classification diversification is sufficient for further analysis.
- **No representation's mean AUROC exceeds 0.70** under gene-disjoint 5-fold stratified CV; per-fold variance is 0.08–0.15 and bootstrap 95% CIs overlap across the four leading combinations, so the ranking should be read as approximate.
- **A randomly-initialised shallow 1-D CNN (used as the RiNALMo proxy) matches the directly-evaluated RNA-FM 640-dim embedding within CV variance**, and a classical 3-mer composition baseline (delivered through the DeepLncLoc CPU head) is similarly close — consistent with the benchmark's central reading that representational class, rather than pretraining scale per se, sets the current ceiling under the CPU-proxy conditions tested.
- **Continuous regression is near chance for every representation** (Spearman ρ ≤ 0.19, RMSE clustered 1.02–1.05 in log₂(h) units), consistent with the observability-gap reading that the remaining predictive signal resides in dynamic cellular state rather than in sequence alone.

## LOCO versus 5-fold stratified

- LOCO AUROC SD ranges 0.17–0.30 because per-cell classifiable N is small (HeLa-TetOff has no unstable-class transcripts in the classifiable subset, for example).
- Primary metric throughout the Perspective is 5-fold stratified; LOCO is reported as a diagnostic of cross-cell generalisation, not as a ranking criterion.

## Caveats

- Proxy-to-full-weight AUROC gap for RiNALMo-650M, Evo-7B and RhoFold+ is expected on the basis of independent zero-shot comparisons to be of the order of a few percentage points; GPU replication is in `benchmark/colab/*.ipynb` and will be reported separately.
- Consensus-failure analysis (main-text Fig. 4) uses the same gene-disjoint 5-fold stratified MLP predictions summarised here.
