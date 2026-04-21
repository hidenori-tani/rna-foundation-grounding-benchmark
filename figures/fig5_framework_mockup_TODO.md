# Fig.5 Dynamic grounding framework — 先生の手作業タスク

**Status**: Phase 3 Task 3.5（2026-08 予定）— Phase 2 結果確定後に作成

## 作成目的

1. **本総説 Fig.5**（Briefings §6 / Nat MI §4）の中核図
2. **SPReAD申請書** 研究計画図（Fig.2 SPReAD版）の原案

## ビジュアル設計（3層アーキテクチャ）

### Layer 1: Static AI prediction
- 5モデルのアイコン：RNA-FM / RiNALMo / Evo / RhoFold+ / DeepLncLoc
- 出力：point estimate（確率 or 連続値）

### Layer 2: Dynamic grounding constraint
- **turnover axis**: BRIC-seq / SLAM-seq / TimeLapse-seq 由来の t½ 分布
- **localization axis**: DeepLncLoc または CeFra-seq などの公開局在データ
- 両軸は "biological prior" として Layer 1 の出力を補正・制約する

### Layer 3: Biology-aware output
- Calibrated prediction with uncertainty
- 生物学的に解釈可能な次元（stability bin × localization bin）に射影

### Example lncRNA で flow 例示
- 候補：NEAT1（stable, nuclear）を追跡例として使用
- 入力配列 → 5モデルの raw output → dynamic constraint 適用 → 最終予測
- 各ステップで「何が変わったか」矢印で注釈

## 配色・スタイル

- Fig.1（[`fig1_concept_mockup_TODO.md`](fig1_concept_mockup_TODO.md)）と同系統
- Layer 分離は横帯 or 縦パネルの3段構成
- turnover curve は先生の過去BRIC-seq論文のスタイル踏襲

## エクスポート

- PDF（ベクター）→ `figures/fig5_framework.pdf`
- PNG 300dpi → `figures/fig5_framework.png`

## 作成タイミング

Phase 2 の Task 2.11 完了後（metrics_table.csv と phase2_summary.md 確定後）。
具体例の lncRNA を Phase 2 結果から選ぶため、先行して作成しない。

## 完了後のチェックリスト

- [ ] BioRender で作成
- [ ] PDF export
- [ ] PNG 300dpi export
- [ ] このMDを削除 or 内容を作成ログに更新
- [ ] Briefings §6 で本文と整合確認
- [ ] spread_integration.md の §4 に完成版を反映（SPReAD Fig.2 として転用）
