# Phase 1 Data Pipeline

Spec: [`../../spec.md`](../../spec.md) §3 / Plan: [`../../plan.md`](../../plan.md) Phase 1 (Tasks 1.1–1.8)

## 実行順序

| 順 | Task | スクリプト | 出力 |
|---|---|---|---|
| 1 | 1.1 | `fetch_bricseq.py` | `../processed/bricseq_halflife.csv` |
| 2 | 1.2 | `fetch_slamseq_herzog.py` | `../processed/slamseq_herzog_halflife.csv` |
| 3 | 1.3 | `fetch_timelapseseq_schofield.py` | `../processed/timelapseseq_schofield_halflife.csv` |
| 4 | 1.4 | `fetch_gencode.py --species both` | `../processed/gencode_v44_lncrna.{gtf,fa}` + `vM33` |
| 5 | 1.5 | `cross_mapping.py --source-csv <each>` | `*_mapped.csv` |
| 6 | 1.6 | `normalize_halflife.py` | `../processed/halflife_merged.csv` |
| 7 | 1.7 | `build_test_set.py` | `../processed/test_set_final.csv` + `.fa` |
| 8 | 1.8 | `qc_report.py` | `../QC_report.md` |

## 初回実行前に要確定

以下の placeholder を先生に確認 or 実サイトから取得して埋める：

- `fetch_bricseq.py` — `TANI2012_SUPP_URLS`（Genome Research supplementary URLs）
- `fetch_slamseq_herzog.py` — `ZENODO_URLS`（halflives.tsv の直接URL）
- `fetch_timelapseseq_schofield.py` — `SUPP_URLS`（Nature Methods Supplementary）

加えて、各 `parse_halflife_table()` の列 rename は実ファイル確認後に確定する。

## データソース・accession 一覧（事実関係）

| データセット | 手法 | Cell line | N (transcripts) | Accession |
|---|---|---|---|---|
| Tani et al. 2012 Genome Res | BRIC-seq (5-EU pulse) | HeLa Tet-off | 11,052 mRNA + 1,418 ncRNA | DDBJ DRA: DRA000345-350, DRA000357-361 + Supp tables |
| Herzog et al. 2017 Nature Methods | SLAM-seq (4sU→T>C, IAA化学) | mESC | 8,405 | GEO: GSE99978 + Zenodo halflives.tsv |
| Schofield et al. 2018 Nature Methods | TimeLapse-seq (4sU→C, OsO4化学) | MEF + K562 | ~数千 | GEO: GSE95854 + Supp tables |

※ SLAM-seq と TimeLapse-seq は**変換機序が異なる**（IAA vs OsO4）が、
  いずれも 4sU metabolic labeling の T>C / U>C mutation 検出方式。
  本総説§3で両手法の相互関係を説明する。

## 依存パッケージ

```
pandas>=2.0
numpy>=1.24
requests>=2.31
openpyxl>=3.1   # Excel supplementary 読み込み用
```

## テスト実行

```bash
# Dry run（未実装パーサー部分は placeholder のまま）
python fetch_gencode.py --species human --raw-dir /tmp/gencode_test --processed-dir /tmp/gencode_test
```
