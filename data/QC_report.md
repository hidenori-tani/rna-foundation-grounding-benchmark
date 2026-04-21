# Phase 1 QC Report

Generated from: `test_set_final.csv`

## 1. Test set size

- Total rows: **256**
- Unique genes: **222**

- `label_binary = excluded`: 140
- `label_binary = unstable`: 76
- `label_binary = stable`: 40

## 2. Cell line coverage

- HeLa_TetOff: 150
- K562: 68
- mESC: 31
- MEF: 7

## 3. Length & GC content

- length median=3052, min=474, max=19245

## 4. Sanity check vs known lncRNAs

- NEAT1: not in test set
- MALAT1: t½=5.31h, label=stable, expected=stable ✅
- KCNQ1OT1: not in test set
- FIRRE: t½=2.25h, label=excluded, expected=unstable-ish ✅
- LINC-PINT: t½=2.35h, label=excluded, expected=unstable-ish ✅

## 5. Risk flags

- R1 OK: classifiable N=116, minority=40
- R1 OK: unmap rate=2.3% ≤ 30%
- R1 OK: max cell-line share=58.6% ≤ 80%
- R1 OK: anchors in set = ['MALAT1', 'FIRRE', 'LINC-PINT']

## 6. Dataset contributions

- `BRIC-seq`: 150
- `TimeLapse-seq`: 75
- `SLAM-seq`: 31

## 7. Decision gate for Phase 2

Phase 2 (GPU spend) は **R1 flag ゼロ** を確認してから実行する。
triggers がある場合は対応策を講じた上で本スクリプトを再実行し、clear を確認すること。
