#!/usr/bin/env bash
# Run the complete V-Fin workflow for exactly one dataset on one AutoDL GPU.
#
# First run:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi --with-front
# Resume experiments only:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi --resume

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin_dataset.sh <dataset> [--with-front] [--resume]"
  exit 2
fi

DATASET="$1"
shift
WITH_FRONT=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-front)
      WITH_FRONT=1
      ;;
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
SEEDS="2024,2025,2026,2027,2028"
ABLATIONS="full,id_only,no_theta,no_text,no_exercise_ped,no_relation_features,no_mastery,no_forgetting,no_seq,no_type_aware_scoring"
SEMANTIC_RUN_ID="${DATASET}_vfin_semantic_5seeds"
COMPARISON_RUN_ID="${DATASET}_vfin_comparison_5seeds"

mkdir -p logs
LOG_FILE="logs/${DATASET}_vfin_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== V-Fin AutoDL workflow ====="
echo "dataset=$DATASET"
echo "repository=$REPO_ROOT"
echo "log=$LOG_FILE"
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if [[ "$WITH_FRONT" -eq 1 ]]; then
  echo "===== Step 1/3: rebuild front features and ER graph ====="
  python front_pipeline/run_front_pipeline.py \
    --datasets "$DATASET" \
    --data-fin-root Data_Fin \
    --mirt-epochs 70 \
    --mirt-batch-size 1024 \
    --sequence-epochs 30 \
    --sequence-batch-size 32 \
    --ektm-epochs 30 \
    --ektm-batch-size 16 \
    --device cuda \
    --sequence-term one_minus_cos_sq \
    --force
else
  echo "===== Step 1/3: validate existing front features and ER graph ====="
  python front_pipeline/validate_front_pipeline.py \
    --datasets "$DATASET" \
    --data-fin-root Data_Fin \
    --require-graph
fi

SEMANTIC_RESUME=()
if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${SEMANTIC_RUN_ID}" ]]; then
  SEMANTIC_RESUME=(--resume)
fi

echo "===== Step 2/3: SemanticConvE full model and ablations ====="
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
  --include-test-triples \
  "${SEMANTIC_RESUME[@]}"

COMPARISON_RESUME=()
if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
  COMPARISON_RESUME=(--resume)
fi

echo "===== Step 3/3: ID-only comparison models ====="
python comparison_models/run_v9_comparison_experiments.py \
  --dataset "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph \
  --run-id "$COMPARISON_RUN_ID" \
  --seeds "$SEEDS" \
  --models all \
  --cuda auto \
  "${COMPARISON_RESUME[@]}"

echo "===== Summarize SemanticConvE ====="
python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DATASET" \
  --run-id "$SEMANTIC_RUN_ID" \
  --runs-root runs \
  --seeds "$SEEDS" \
  --ablations "$ABLATIONS"

echo "===== Summarize comparison models ====="
python comparison_models/summarize_dataset_results.py \
  --dataset "$DATASET" \
  --run-dir "runs/${DATASET}/${COMPARISON_RUN_ID}"

echo "===== Completed: $DATASET ====="
echo "Semantic summary: runs/${DATASET}/${SEMANTIC_RUN_ID}/summaries"
echo "Comparison summary: runs/${DATASET}/${COMPARISON_RUN_ID}/summaries"
