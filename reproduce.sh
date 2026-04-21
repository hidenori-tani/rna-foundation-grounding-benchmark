#!/usr/bin/env bash
# Grounding RNA foundation models in transcript dynamics — reproduction script
#
# Runs Phase 1 (data pipeline) and Phase 2 (CPU-proxy benchmark), then
# regenerates all figures and tables. On a typical laptop (8-core CPU, 16 GB
# RAM) the full pipeline completes in ~45 min. CPU proxies substitute for
# RiNALMo-650M, Evo-7B, and RhoFold+ so the workflow is reviewer-reproducible
# without a GPU; swap in the GPU notebooks under benchmark/colab/ for the
# full-precision replication.
#
# Usage:
#   bash reproduce.sh                    # full end-to-end run
#   bash reproduce.sh --dry-run          # print commands only
#   bash reproduce.sh --skip-phase1      # use existing data/processed/
#   bash reproduce.sh --skip-phase2      # use existing benchmark/results/
#   bash reproduce.sh --skip-figures     # skip figure regeneration
#   bash reproduce.sh -h | --help        # show this header

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-python3}"
DRY_RUN=0
SKIP_PHASE1=0
SKIP_PHASE2=0
SKIP_FIGURES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-phase1) SKIP_PHASE1=1; shift ;;
        --skip-phase2) SKIP_PHASE2=1; shift ;;
        --skip-figures) SKIP_FIGURES=1; shift ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

step() {
    local desc="$1"; shift
    echo ""
    echo "==> $desc"
    echo "    $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

# Prerequisite check
echo "==> Checking Python environment..."
$PY -c "import pandas, numpy, scipy, sklearn, torch, Bio" 2>/dev/null || {
    echo "Missing dependencies. Install with:" >&2
    echo "    conda env create -f environment.yml && conda activate rna-foundation-grounding" >&2
    echo "  or" >&2
    echo "    pip install -r benchmark/requirements.txt" >&2
    exit 1
}

# -------- Phase 1: data pipeline --------
if [[ $SKIP_PHASE1 -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo " Phase 1: Data pipeline"
    echo "=========================================="
    if [[ $DRY_RUN -eq 1 ]]; then
        bash data/pipeline/run_all.sh --dry-run
    else
        bash data/pipeline/run_all.sh
    fi
else
    echo "[skip] Phase 1 (using existing data/processed/)"
    if [[ ! -f data/processed/test_set_final.csv ]]; then
        echo "Error: data/processed/test_set_final.csv missing. Remove --skip-phase1 or run Phase 1 manually." >&2
        exit 1
    fi
fi

# -------- Phase 2: CPU-proxy benchmark --------
if [[ $SKIP_PHASE2 -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo " Phase 2: Embedding extraction (CPU proxies)"
    echo "=========================================="

    step "Model 1/5: RNA-FM"        $PY benchmark/models/rna_fm.py --device cpu
    step "Model 2/5: RiNALMo (4-mer proxy)" $PY benchmark/models/rinalmo.py --device cpu
    step "Model 3/5: Evo (ERNIE-RNA proxy)" $PY benchmark/models/evo.py --device cpu
    step "Model 4/5: RhoFold+ (ViennaRNA proxy)" $PY benchmark/models/rhofold_plus.py
    step "Model 5/5: DeepLncLoc"    $PY benchmark/models/deeplncloc.py --device cpu

    echo ""
    echo "=========================================="
    echo " Phase 2: Evaluation"
    echo "=========================================="
    step "Classifier self-test"     $PY benchmark/classifiers.py --self-test
    step "AUROC/AUPRC metrics"      $PY benchmark/eval.py --classifier both
    step "Ablation study"           $PY benchmark/ablation.py
    step "Interpretability (IG + SHAP)" $PY benchmark/interpretability.py
else
    echo "[skip] Phase 2 (using existing benchmark/results/)"
    if [[ ! -f benchmark/results/metrics_table.csv ]]; then
        echo "Error: benchmark/results/metrics_table.csv missing. Remove --skip-phase2 or run Phase 2 manually." >&2
        exit 1
    fi
fi

# -------- Figures --------
if [[ $SKIP_FIGURES -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo " Figures and tables"
    echo "=========================================="
    step "Fig 2: AUROC heatmap"     $PY figures/fig2_auroc_heatmap.py
    step "Fig 3: Performance scatter" $PY figures/fig3_scatter.py
    step "Fig 4: Failure analysis"  $PY figures/fig4_failure_analysis.py
    step "Fig 5: Framework schematic" $PY figures/fig5_framework.py
fi

echo ""
echo "=========================================="
echo " Reproduction complete"
echo "=========================================="
echo "Outputs:"
echo "  Test set:          data/processed/test_set_final.csv"
echo "  Metrics:           benchmark/results/metrics_table.csv"
echo "  Ablation:          benchmark/results/ablation_results.csv"
echo "  Feature importance: benchmark/results/feature_importance/"
echo "  Figures:           figures/fig{2,3,4,5}_*.{png,pdf}"
echo ""
echo "For GPU replication (RiNALMo-650M, Evo-7B, RhoFold+), see"
echo "benchmark/colab/README.md."
