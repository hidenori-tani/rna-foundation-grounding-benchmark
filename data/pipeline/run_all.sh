#!/usr/bin/env bash
# Phase 1 Data Pipeline — one-shot runner
#
# Usage:
#   cd <repo>/data/pipeline
#   bash run_all.sh                  # run all steps in order
#   bash run_all.sh --dry-run        # print commands only, do not execute
#   bash run_all.sh --from 1.5       # resume from step 1.5
#
# On failure, the script stops at the failing step; rerun from the same
# step after the issue is resolved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-python3}"
PROCESSED="../processed"
DRY_RUN=0
START_FROM="1.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --from) START_FROM="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run() {
    local step="$1"; shift
    local desc="$1"; shift
    # skip if step < START_FROM
    if [[ $(echo -e "$step\n$START_FROM" | sort -V | head -1) != "$START_FROM" ]]; then
        echo "[skip] Step $step: $desc"
        return 0
    fi
    echo ""
    echo "==> Step $step: $desc"
    echo "    $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

# 事前チェック
echo "==> Checking prerequisites..."
$PY -c "import pandas, numpy, requests, openpyxl" 2>/dev/null || {
    echo "Missing Python packages. Run: pip install pandas numpy requests openpyxl"
    exit 1
}
mkdir -p "$PROCESSED"

# ---- Phase 1 ----
run 1.1 "Tani 2012 BRIC-seq (HeLa)"             $PY fetch_bricseq.py --mode supp
run 1.2 "Herzog 2017 SLAM-seq (mESC)"           $PY fetch_slamseq_herzog.py --mode zenodo
run 1.3 "Schofield 2018 TimeLapse-seq (MEF+K562)" $PY fetch_timelapseseq_schofield.py --mode supp
run 1.4 "GENCODE v44 human + vM33 mouse"        $PY fetch_gencode.py --species both
run 1.5.1 "cross_mapping: BRIC-seq"             $PY cross_mapping.py --source-csv "$PROCESSED/bricseq_halflife.csv"
run 1.5.2 "cross_mapping: SLAM-seq"             $PY cross_mapping.py --source-csv "$PROCESSED/slamseq_herzog_halflife.csv"
run 1.5.3 "cross_mapping: TimeLapse-seq"        $PY cross_mapping.py --source-csv "$PROCESSED/timelapseseq_schofield_halflife.csv"
run 1.6 "normalize_halflife (quantile+winsorize)" $PY normalize_halflife.py
run 1.7 "build_test_set (binary + tertile)"     $PY build_test_set.py
run 1.8 "QC report"                             $PY qc_report.py

echo ""
echo "==> Phase 1 complete."
echo "Review: $PROCESSED/test_set_final.csv and ../QC_report.md"
echo ""
echo "Next: Phase 2 — CPU proxies via reproduce.sh, or GPU replication via benchmark/colab/*.ipynb"
