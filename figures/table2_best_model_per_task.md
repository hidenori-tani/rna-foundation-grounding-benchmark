# Table 2. Best-performing model per evaluation task

**Source**: `benchmark/results/metrics_summary.csv` + `metrics_table.csv` (Phase 2, 2026-04-20, Plan B CPU version)  
**Test set**: 256 sequences (116 classifiable: stable=40, unstable=76; full set used for regression)  
**CV**: 5-fold stratified (classification) / 5-fold (regression) + leave-one-cell-out (LOCO)

## Classification (binary stable vs unstable)

| Metric | Best model | Classifier | 5-fold (mean ± SD) | LOCO (mean ± SD) | Runner-up | Note |
|---|---|---|---|---|---|---|
| AUROC | **rinalmo (4-mer)*** | MLP | **0.695 ± 0.155** | 0.658 ± 0.282 (MLP) | deeplncloc MLP 0.690 | n-gram baseline tops foundation models |
| F1 | rna_fm | LogReg | 0.428 ± 0.169 | — | rhofold_plus LogReg 0.412 | class imbalance effects prominent |
| MCC | deeplncloc (3-mer) | MLP | 0.234 ± 0.228 | — | rinalmo MLP 0.233 | overall MCC low — task is hard |

## Regression (log2 half-life)

| Metric | Best model | Regressor | 5-fold (mean ± SD) | LOCO (mean ± SD) | Runner-up | Note |
|---|---|---|---|---|---|---|
| Spearman ρ | deeplncloc (3-mer) | MLP | 0.186 ± 0.087 | — | rna_fm MLP 0.153 | all models near-chance |
| Pearson r | rna_fm | MLP | 0.181 ± 0.102 | — | deeplncloc MLP 0.146 | RNA-FM leads marginally |
| Spearman ρ (LOCO) | rinalmo (4-mer)* | Ridge | 0.260 ± 0.399 | — | rna_fm Ridge 0.165 | huge SD from small per-cell N |
| RMSE | deeplncloc (3-mer) | MLP | 1.018 ± 0.092 | — | rna_fm MLP 1.025 | log2(h) units |

(*) = Plan B CPU substitute (see Table 1 for details)

## 解釈原則の適用

### 盲点命題 A の実証度（強）

- 5モデルの 5-fold AUROC spread = **0.299** （0.396 rhofold_plus MLP〜0.695 rinalmo MLP）
- **R3 リスク（全モデル 0.05 以内クラスタリング）は不発動** → 通常の二値分類のまま解析継続
- **foundation model は n-gram baseline に勝てない** — k-mer 4-mer MLP（0.695）と k-mer 3-mer MLP（0.690）が RNA-FM MLP（0.672）と ERNIE-RNA MLP（0.655）を上回る
- "black box ≠ better" の定量的実証として §4 benchmarking の主張を裏付け

### 構築命題 C の余地

- 全モデル均等に弱い（AUROC 最高 0.695、Spearman 最高 0.186）→ **dynamic grounding を追加しないと解けない問題** であるという結論を強化
- 単一モデルで AUROC≥0.80 に届くモデルがない → foundation-scaling 戦略では不十分、turnover/localization の動的軸を条件付けする必要

### LOCO vs 5-fold のギャップ

- LOCO の AUROC は 5-fold より高い場合（evo LogReg 0.726）もあるが、SD 0.26-0.4 と極端に広い
- cell line ごとのサンプル数が小さい（HeLa_TetOff は全 unstable でクラシフィカブルなし）ため LOCO 推定は不安定
- 論文中では 5-fold stratified を primary、LOCO を cross-cell generalization の補助として位置付ける
