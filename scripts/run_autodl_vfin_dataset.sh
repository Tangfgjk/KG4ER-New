#!/usr/bin/env bash
# Run V-Fin4 experiments for one dataset on one AutoDL GPU.
#
# Examples:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi all
#   bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small semantic
#   bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small comparison --resume

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin_dataset.sh <dataset> [all|semantic|comparison] [--resume]"
  exit 2
fi

DATASET="$1"
shift
MODE="all"
RESUME=0

if [[ $# -gt 0 && "$1" != "--resume" ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
  shift
done

case "$DATASET" in
  Eedi|algebra2005|assist2009-sub|statics2011|XES3G5M-sub-small) ;;
  *) echo "Unsupported dataset: $DATASET"; exit 2 ;;
esac
case "$MODE" in
  all|semantic|comparison) ;;
  *) echo "Mode must be one of: all, semantic, comparison"; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-disabled}"
SEEDS="2024,2025,2026,2027,2028"
ABLATIONS="id_only,feature_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,feature_only_no_mastery,feature_only_no_forgetting,feature_only_no_seq"
SEMANTIC_RUN_ID="${DATASET}_vfin4_semantic_5seeds"
COMPARISON_RUN_ID="${DATASET}_vfin4_comparison_5seeds"
REPORT_DIR="runs/${DATASET}/${DATASET}_vfin4_report"

mkdir -p logs
LOG_FILE="logs/${DATASET}_vfin4_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== V-Fin4 five-seed workflow ====="
echo "dataset=$DATASET mode=$MODE"
echo "All training uses triples.txt only; test_triples.txt is evaluation-only."
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "===== Validate uploaded ER graph ====="
python codes-New-ConvE/validate_semantic_ready.py \
  --datasets "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph

if [[ "$MODE" == "all" || "$MODE" == "semantic" ]]; then
  SEMANTIC_RESUME=()
  if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${SEMANTIC_RUN_ID}" ]]; then
    SEMANTIC_RESUME=(--resume)
  fi
  echo "===== SemanticConvE: feature-only primary model plus seven ablations ====="
  python codes-New-ConvE/run_semantic_experiments.py \
    --dataset "$DATASET" \
    --data-root Data_Fin \
    --graph-subdir er_graph \
    --run-id "$SEMANTIC_RUN_ID" \
    --seeds "$SEEDS" \
    --ablations "$ABLATIONS" \
    --epochs 25 \
    --bs 1024 \
    --learning-rate 0.001 \
    --negative-ratio 5 \
    --cuda auto \
    --exclude-test-triples \
    "${SEMANTIC_RESUME[@]}"

  python codes-New-ConvE/summarize_semantic_results.py \
    --dataset "$DATASET" \
    --run-id "$SEMANTIC_RUN_ID" \
    --runs-root runs \
    --seeds "$SEEDS" \
    --ablations "$ABLATIONS"
fi

if [[ "$MODE" == "all" || "$MODE" == "comparison" ]]; then
  COMPARISON_RESUME=()
  if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
    COMPARISON_RESUME=(--resume)
  fi
  echo "===== Comparison models: all baselines except KCP-ER ====="
  python comparison_models/run_v9_comparison_experiments.py \
    --dataset "$DATASET" \
    --data-root Data_Fin \
    --graph-subdir er_graph \
    --run-id "$COMPARISON_RUN_ID" \
    --seeds "$SEEDS" \
    --models all \
    --cuda auto \
    "${COMPARISON_RESUME[@]}"

  python comparison_models/summarize_dataset_results.py \
    --dataset "$DATASET" \
    --run-dir "runs/${DATASET}/${COMPARISON_RUN_ID}"
fi

if [[ -d "runs/${DATASET}/${SEMANTIC_RUN_ID}" && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
  echo "===== Write split top-K tables and readable figures ====="
  python scripts/report_topk_comparison.py \
    --run-dir "SemanticConvE=runs/${DATASET}/${SEMANTIC_RUN_ID}" \
    --run-dir "Comparison=runs/${DATASET}/${COMPARISON_RUN_ID}" \
    --output-dir "$REPORT_DIR"
else
  echo "Combined report deferred: run the other mode on this machine or merge its results first."
fi

echo "===== Completed: $DATASET / $MODE ====="
echo "Semantic summary: runs/${DATASET}/${SEMANTIC_RUN_ID}/summaries"
echo "Comparison summary: runs/${DATASET}/${COMPARISON_RUN_ID}/summaries"
echo "Report: ${REPORT_DIR}"
