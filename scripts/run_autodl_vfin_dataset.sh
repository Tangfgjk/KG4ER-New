#!/usr/bin/env bash
# Run the complete V-Fin3 workflow for exactly one dataset on one AutoDL GPU.
#
# First run:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi
# Resume interrupted ER experiments:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi --resume

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin_dataset.sh <dataset> [--resume]"
  exit 2
fi

DATASET="$1"
shift
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
  shift
done

case "$DATASET" in
  Eedi|algebra2005|assist2009-sub|statics2011|XES3G5M-sub-small)
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-disabled}"
SEEDS="2024"
ABLATIONS="full,id_only,no_learner_id,no_learner_relation_id,feature_only"
SEMANTIC_INCLUDE_RUN_ID="${DATASET}_vfin3_semantic_include_test_1seed"
SEMANTIC_EXCLUDE_RUN_ID="${DATASET}_vfin3_semantic_exclude_test_1seed"
COMPARISON_RUN_ID="${DATASET}_vfin3_comparison_1seed"
REPORT_DIR="runs/${DATASET}/${DATASET}_vfin3_pilot_report"

mkdir -p logs
LOG_FILE="logs/${DATASET}_vfin_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== V-Fin3 AutoDL one-seed pilot workflow ====="
echo "dataset=$DATASET"
echo "repository=$REPO_ROOT"
echo "log=$LOG_FILE"
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "===== Step 1/4: validate uploaded ER graph ====="
python codes-New-ConvE/validate_semantic_ready.py \
  --datasets "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph

SEMANTIC_INCLUDE_RESUME=()
if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${SEMANTIC_INCLUDE_RUN_ID}" ]]; then
  SEMANTIC_INCLUDE_RESUME=(--resume)
fi

echo "===== Step 2/4: SemanticConvE with test cognitive triples ====="
python codes-New-ConvE/run_semantic_experiments.py \
  --dataset "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph \
  --run-id "$SEMANTIC_INCLUDE_RUN_ID" \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS" \
  --epochs 25 \
  --bs 1024 \
  --learning-rate 0.001 \
  --negative-ratio 5 \
  --cuda auto \
  --include-test-triples \
  "${SEMANTIC_INCLUDE_RESUME[@]}"

SEMANTIC_EXCLUDE_RESUME=()
if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${SEMANTIC_EXCLUDE_RUN_ID}" ]]; then
  SEMANTIC_EXCLUDE_RESUME=(--resume)
fi

echo "===== Step 3/4: SemanticConvE without test triples ====="
python codes-New-ConvE/run_semantic_experiments.py \
  --dataset "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph \
  --run-id "$SEMANTIC_EXCLUDE_RUN_ID" \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS" \
  --epochs 25 \
  --bs 1024 \
  --learning-rate 0.001 \
  --negative-ratio 5 \
  --cuda auto \
  --exclude-test-triples \
  "${SEMANTIC_EXCLUDE_RESUME[@]}"

COMPARISON_RESUME=()
if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
  COMPARISON_RESUME=(--resume)
fi

echo "===== Step 4/4: all comparison models ====="
python comparison_models/run_v9_comparison_experiments.py \
  --dataset "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph \
  --run-id "$COMPARISON_RUN_ID" \
  --seeds "$SEEDS" \
  --models all \
  --cuda auto \
  "${COMPARISON_RESUME[@]}"

echo "===== Summarize SemanticConvE with test cognitive triples ====="
python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DATASET" \
  --run-id "$SEMANTIC_INCLUDE_RUN_ID" \
  --runs-root runs \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS"

echo "===== Summarize SemanticConvE without test triples ====="
python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DATASET" \
  --run-id "$SEMANTIC_EXCLUDE_RUN_ID" \
  --runs-root runs \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS"

echo "===== Summarize comparison models ====="
python comparison_models/summarize_dataset_results.py \
  --dataset "$DATASET" \
  --run-dir "runs/${DATASET}/${COMPARISON_RUN_ID}"

echo "===== Write combined top-K table and curves ====="
python scripts/report_topk_comparison.py \
  --run-dir "SemanticConvE include-test=runs/${DATASET}/${SEMANTIC_INCLUDE_RUN_ID}" \
  --run-dir "SemanticConvE exclude-test=runs/${DATASET}/${SEMANTIC_EXCLUDE_RUN_ID}" \
  --run-dir "Comparison=runs/${DATASET}/${COMPARISON_RUN_ID}" \
  --output-dir "$REPORT_DIR"

echo "===== Completed: $DATASET ====="
echo "Include-test summary: runs/${DATASET}/${SEMANTIC_INCLUDE_RUN_ID}/summaries"
echo "Exclude-test summary: runs/${DATASET}/${SEMANTIC_EXCLUDE_RUN_ID}/summaries"
echo "Comparison summary: runs/${DATASET}/${COMPARISON_RUN_ID}/summaries"
echo "Combined table and curves: ${REPORT_DIR}"
