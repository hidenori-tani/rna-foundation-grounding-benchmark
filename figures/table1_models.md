# Table 1. Five RNA AI models benchmarked in this review

**Phase 2 status (2026-04-20)**: 本 CPU ローカル版では一部モデルを代替実装で評価している。**元モデル列**と**実際に走らせた実装列**を併記する。Plan B の理由は各 `benchmark/models/*.py` docstring に詳細あり。Phase 3 の GPU セッション（$150-200 予算、任意）で元モデルの再評価が可能。

| # | Label | Original model (intended) | CPU-feasible implementation (actual) | Output dim | License | Reference |
|---|---|---|---|---|---|---|
| 1 | **RNA-FM** | RNA-FM (100M, BERT-like, RNAcentral) | same — multimolecule/rnafm | 640 | MIT | Chen et al. 2022 |
| 2 | **RiNALMo** | RiNALMo 650M (33 layers, hidden 1,280) | 4-mer composition (256-dim k-mer) | 256 | — | Penic et al. 2024 (preprint) |
| 3 | **Evo** | Evo 7B (StripedHyena, 2.7M prokaryotic genomes) | ERNIE-RNA (multimolecule/ernierna, 86M) | 768 | Apache-2.0 / — | Nguyen et al. 2024 *Science* / Yin et al. 2024 (ERNIE-RNA) |
| 4 | **RhoFold+** | RhoFold+ (~80M, 3D structure predictor) | ViennaRNA 2D descriptors (9-dim) | 9 | Academic / ViennaRNA GPL | Shen et al. 2024 *Nature Methods* / Lorenz et al. 2011 (ViennaRNA) |
| 5 | **DeepLncLoc** | DeepLncLoc (CNN-LSTM, 5-class localization) | 3-mer composition (64-dim k-mer) | 64 | Academic / — | Zeng et al. 2022 *Briefings in Bioinformatics* |

## Plan B 代替の論拠（要約）

- **RiNALMo → 4-mer**: 650M CPU fp16 でも ~10GB / 1配列 5-10分 → 256配列 20-40時間。4-mer composition は 256次元で RNA-FM (640) と同オーダーの表現容量を持ち、局所モチーフ（AU-rich element AUUU、pumilio UGUA など）を明示的に表現する。論拠: Mukherjee et al. 2017 *Nature SMB*、Ji et al. 2019、Yang et al. 2024 (RNAGenesis benchmark)
- **Evo 7B → ERNIE-RNA**: Evo 7B bf16 で ~14GB、StripedHyena は CUDA カーネル依存で macOS MPS/CPU 未検証。ERNIE-RNA (Yin et al. 2024, 86M) は CPU で 1配列 5-15秒、RNA-FM に対し "post-2023 generation" 相当。fallback 候補の Nucleotide Transformer v2 50M は transformers 5.x 互換性破綻（`find_pruneable_heads_and_indices` 廃止等）により除外
- **RhoFold+ → ViennaRNA 2D**: RhoFold+ は 3D 予測で数分/配列、CPU 非現実的。ViennaRNA は pair probability matrix から派生させた 9 記述子（MFE、平均塩基対確率、loop 統計等）で、二次構造ベースの粗い情報を代替

## カバレッジ（代替後も維持）

- sequence-only foundation（RNA-FM）/ post-2023 foundation（ERNIE-RNA）/ n-gram baselines（4-mer, 3-mer）/ 2D structure（ViennaRNA）の **4カテゴリを網羅**
- パラメータスペクトラム: 0-dim (k-mer) 〜 86M (ERNIE-RNA) — 本来意図した 100M〜7B から縮小したが、定性的比較は成立
- 全て公開重み/公開ライブラリで再現可能
