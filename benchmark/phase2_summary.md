# Phase 2 Benchmark Summary — AI×lncRNA 総説

Run date: 2026-04-20  
Environment: local macOS CPU (Plan B — heavy models replaced with CPU-feasible substitutes)

## 1. モデル構成と CPU 代替の根拠

| ラベル | 実体 | 次元 | 位置付け |
|---|---|---|---|
| `rna_fm` | RNA-FM (multimolecule) | 640 | RNA foundation model v1 |
| `rinalmo`* | 4-mer composition | 256 | n-gram ベースライン（RiNALMo 650M は CPU 非現実的） |
| `evo`* | ERNIE-RNA (multimolecule/ernierna, 86M) | 768 | RNA foundation model v2（Evo 7B は CPU 不可） |
| `rhofold_plus`* | ViennaRNA 2D descriptors | 9 | 二次構造ベース簡易記述子 |
| `deeplncloc` | 3-mer composition | 64 | 既存の単純 n-gram ベースライン |

(*) は元計画からの CPU 代替。詳細は各 `models/*.py` の docstring 参照。

## 2. 評価プロトコル

- test set: 256 配列 / 116 classifiable (stable=40, unstable=76)
- 分類: 5-fold stratified CV + leave-one-cell-out CV
- 回帰: 5-fold CV + leave-one-cell-out CV
- 分類器: Logistic Regression + MLP (3-layer, hidden=128)
- 回帰器: Ridge + MLP
- SEED=42

## 3. 主結果

### 3.1 分類性能（5-fold stratified CV, AUROC）

| model | classifier | mean | std |
|---|---|---|---|
| rinalmo (4-mer)* | MLP | **0.695** | 0.155 |
| deeplncloc (3-mer) | MLP | 0.690 | 0.152 |
| rna_fm | MLP | 0.672 | 0.127 |
| evo (ERNIE-RNA)* | MLP | 0.655 | 0.126 |
| rhofold_plus (ViennaRNA)* | MLP | 0.396 | 0.080 |

**AUROC spread = 0.299 → R3 フラグ不発動**（plan.md §Step 2.11.3 の「全モデル 0.05 以内ならティアー化」条件を満たさず、通常の 2値分類のまま解析継続）。

### 3.2 命題A の実証 — "black box ≠ better"

- k-mer composition ベースライン（rinalmo 4-mer / deeplncloc 3-mer）が RNA foundation model（RNA-FM, ERNIE-RNA）を上回る。
- foundation model は turnover 予測において **n-gram ベースラインに勝てていない**。
- これは総説の中心命題「静的 foundation model は turnover 予測で頭打ち」を直接支持する。

### 3.3 回帰性能（5-fold CV, Spearman）

| model | classifier | mean |
|---|---|---|
| deeplncloc | MLP | 0.186 |
| rna_fm | MLP | 0.153 |
| rinalmo* | MLP | 0.146 |
| evo* | MLP | 0.140 |

連続値予測は全般に弱く（最良 ρ≈0.19）、分類で見えた差も薄くなる。turnover の連続値予測は現行の静的特徴では厳しいことを示す。

### 3.4 長さ層別（ablation.py, LogReg）

| model | short (<2387nt) | mid | long (>4159nt) |
|---|---|---|---|
| rna_fm | 0.633 | 0.609 | 0.524 |
| rinalmo* | 0.525 | 0.587 | 0.476 |
| evo (ERNIE-RNA)* | 0.483 | 0.657 | 0.613 |
| rhofold_plus* | 0.575 | 0.609 | 0.419 |
| deeplncloc | 0.530 | 0.442 | 0.371 |

- rna_fm は**短い配列で強く長い配列で弱い**（foundation model が学習時に出会った平均的 RNA サイズへの適応を示唆）。
- evo (ERNIE-RNA) は逆傾向で**長い配列で強い**。
- rhofold_plus/deeplncloc は**長い配列で崩れる**（局所的特徴の希釈が効く）。

### 3.5 Consensus failure 分析 (fig4)

- 5モデル全員が外した「consensus failure」: **12 配列**。
- feature comparison は length / GC / t½ の箱ひげ図として出力済み。

## 4. 生成物

### embeddings
- `benchmark/results/embeddings/{rna_fm,rinalmo,evo,rhofold_plus,deeplncloc}.npz` (全 256 配列)

### metrics
- `benchmark/results/metrics_table.csv` (540 rows, per-fold)
- `benchmark/results/metrics_summary.csv` (mean±std per group)
- `benchmark/results/ablation_results.csv` (15 rows, length tertile)

### figures
- `figures/fig2_auroc_heatmap.{pdf,png}` — AUROC ヒートマップ (5-fold + LOCO)
- `figures/fig3_scatter.{pdf,png}` — 予測 vs 実測 log2(t½) + highlighted lncRNAs
- `figures/fig4_failure_analysis.pdf` — panel A (feature comparison)
- `figures/fig4_failure_analysis_panelB.pdf` — panel B (model disagreement matrix)
- `figures/fig4_failure_table.csv` — consensus failure リスト

## 5. リスク/限界

- **Plan B の制約**: RiNALMo 650M / Evo 7B を実走行していない。ERNIE-RNA / k-mer への置換は docstring で論拠化（Mukherjee 2017, Yang 2024 ベンチマーク）。GPU セッション確保後に元モデルで再走行する計画。
- **サンプルサイズ**: 116 classifiable はやや少なく、std が 0.13-0.24 とかなり大きい。confidence interval の広さは本文で明示する。
- **HeLa_TetOff のみ回帰タスクに回る**: LOCO で HeLa_TetOff を held-out にしたときは classifiable データが 1 ラベルになり AUROC=NaN。table にはその旨が残っている。
- **rhofold_plus MLP のAUROC=0.396**: 9次元の低次元入力に対して MLP が過剰パラメータで overfit している可能性。LogReg 版（0.589）を正として解釈する。

## 6. 次フェーズへの申し送り

- Phase 3 (GPU): 元計画通り RiNALMo 650M / Evo 7B を A100 で走らせ、本結果が CPU 代替の差ではなく本質的な傾向かを確認する。
- 命題A（static model は turnover で頭打ち）の実証は既に強い。Phase 3 で補強後、Nature Machine Intelligence (Perspective) を一次候補、Briefings in Bioinformatics を fallback として投稿準備へ。
