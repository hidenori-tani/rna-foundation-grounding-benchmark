# Fig.1 コンセプト図 mockup — 先生の手作業タスク

**Status**: 先生のBioRender作業待ち（Task 0.3.2）

## 作成目的

1. **SPReAD申請書** 背景節（2026-05-18締切）に Fig.1 簡略版として使用
2. **本総説** 最終図の下書きとして Phase 3（2026-08）まで叩き台

## ビジュアル設計

### 左パネル: "Black box AI"
- 5モデル（RNA-FM, RiNALMo, Evo, RhoFold+, DeepLncLoc）のアイコン配列
- 入力矢印：lncRNA sequence / structure
- 出力矢印：予測（破線）= 実測とずれるイメージ

### 中央パネル: "Dynamic ground truth"
- BRIC-seq / SLAM-seq の実験模式（標識RNA decay curve）
- turnover軸：short-lived vs long-lived lncRNA の図示
- localization軸：nuclear vs cytoplasmic

### 右パネル: "Biology-aware predictor"
- 同じ5モデル + dynamic grounding layer（turnover / localization constraint）
- 出力：実測と整合する予測

### 配色・スタイル
- BioRenderの Bio Tech テンプレートベース
- TiBS総説 Fig.1 と同じ系統色にしておくと連結時のビジュアル統一感が出る
- 文字は英語（最終図版用）。SPReAD版は日本語キャプションを別途用意

## エクスポート
- PDF（ベクター）
- PNG 300dpi（申請書用）
- 保存先: `~/claude-work/research/review-AI/figures/fig1_concept.pdf` + `fig1_concept.png`

## 完了後のチェックリスト
- [ ] BioRenderで作成
- [ ] PDFエクスポート
- [ ] PNG 300dpi エクスポート  
- [ ] このMD（`fig1_concept_mockup_TODO.md`）を削除 or 内容を作成ログに更新
- [ ] spread_integration.md の §4 に完成版を反映

## 参考として見る既存図

- 先生の過去BRIC-seq論文 Fig.1 系（turnover visualization のスタイル参照）
- Nature Reviews 系のAI × biology レビューの concept figure（例: Jumper et al. AlphaFold paper）

**所要時間の目安**: BioRender に慣れている場合 1-2時間。慣れていない場合 3-4時間。
