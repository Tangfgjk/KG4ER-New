#!/usr/bin/env bash
# Run VFin9 KGE comparison models on the sequence-enhanced ER graph.
#
# Usage:
#   bash scripts/run_autodl_vfin9_kge_seq_group.sh group1
#   bash scripts/run_autodl_vfin9_kge_seq_group.sh group2
#   bash scripts/run_autodl_vfin9_kge_seq_group.sh group1 resume
#   bash scripts/run_autodl_vfin9_kge_seq_group.sh group2 summarize-only
#
# Groups:
#   group1 = Eedi, algebra2005, statics2011
#   group2 = assist2009-sub, XES3G5M-sub-small
#
# Environment overrides:
#   SEEDS="2024,2025,2026"
#   MODELS="TransE,TransE-adv,RotatE,DistMult,ComplEx"
#   TOP_KS="5,10,15,...,100"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin9_kge_seq_group.sh <group1|group2|all|dataset1,dataset2> [run|resume|summarize-only]"
  exit 2
fi

GROUP="$1"
MODE="${2:-run}"

SEEDS="${SEEDS:-2024,2025,2026}"
MODELS="${MODELS:-TransE,TransE-adv,RotatE,DistMult,ComplEx}"
TOP_KS="${TOP_KS:-5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100}"
DATA_ROOT="${DATA_ROOT:-Data_Fin}"
GRAPH_SUBDIR="${GRAPH_SUBDIR:-er_graph_with_seq}"
RUN_SUFFIX="${RUN_SUFFIX:-kge_with_seq_3seeds}"
CUDA="${CUDA:-auto}"

case "$MODE" in
  run|resume|summarize-only) ;;
  *) echo "Mode must be one of: run, resume, summarize-only"; exit 2 ;;
esac

case "$GROUP" in
  group1)
    DATASETS=("Eedi" "algebra2005" "statics2011")
    ;;
  group2)
    DATASETS=("assist2009-sub" "XES3G5M-sub-small")
    ;;
  all)
    DATASETS=("Eedi" "algebra2005" "statics2011" "assist2009-sub" "XES3G5M-sub-small")
    ;;
  *)
    IFS=',' read -r -a DATASETS <<< "$GROUP"
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p logs
LOG_FILE="logs/vfin9_kge_seq_${GROUP}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== VFin9 KGE with sequence graph ====="
echo "group=$GROUP mode=$MODE"
echo "datasets=${DATASETS[*]}"
echo "data_root=$DATA_ROOT graph_subdir=$GRAPH_SUBDIR"
echo "models=$MODELS"
echo "seeds=$SEEDS"
echo "top_ks=$TOP_KS"
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

for DATASET in "${DATASETS[@]}"; do
  RUN_ID="${DATASET}_${RUN_SUFFIX}"
  RUN_DIR="runs/${DATASET}/${RUN_ID}"

  echo ""
  echo "================ Dataset ${DATASET} ================"
  echo "run_id=${RUN_ID}"

  echo "----- Validate graph -----"
  python codes-New-ConvE/validate_semantic_ready.py \
    --datasets "$DATASET" \
    --data-root "$DATA_ROOT" \
    --graph-subdir "$GRAPH_SUBDIR"

  if [[ "$MODE" != "summarize-only" ]]; then
    RESUME_ARGS=()
    if [[ "$MODE" == "resume" && -d "$RUN_DIR" ]]; then
      RESUME_ARGS=(--resume)
    fi

    echo "----- Run KGE comparison models -----"
    python comparison_models/run_v9_comparison_experiments.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --graph-subdir "$GRAPH_SUBDIR" \
      --run-id "$RUN_ID" \
      --seeds "$SEEDS" \
      --models "$MODELS" \
      --top-ks "$TOP_KS" \
      --cuda "$CUDA" \
      "${RESUME_ARGS[@]}"
  fi

  echo "----- Summarize KGE comparison results -----"
  python comparison_models/summarize_dataset_results.py \
    --dataset "$DATASET" \
    --run-dir "$RUN_DIR"

  echo "summary -> ${RUN_DIR}/summaries"
done

echo ""
echo "===== Completed VFin9 KGE sequence group: ${GROUP} / ${MODE} ====="
echo "log -> ${LOG_FILE}"
