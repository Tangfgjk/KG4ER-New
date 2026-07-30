#!/usr/bin/env bash
# Run V-Fin8 1000-dim SemanticConvE ablations for one or more datasets.
#
# Examples:
#   bash scripts/run_autodl_vfin_semantic_group.sh Eedi,algebra2005
#   bash scripts/run_autodl_vfin_semantic_group.sh assist2009-sub,statics2011 --resume
#   bash scripts/run_autodl_vfin_semantic_group.sh XES3G5M-sub-small --seeds 2024,2025,2026

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin_semantic_group.sh <dataset[,dataset...]> [--seeds 2024,2025,2026] [--resume]"
  exit 2
fi

DATASETS_ARG="$1"
shift
SEEDS="2024,2025,2026"
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --seeds)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--seeds requires a comma-separated value"
        exit 2
      fi
      SEEDS="$1"
      ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IFS=',' read -r -a DATASETS <<< "$DATASETS_ARG"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets were supplied"
  exit 2
fi

COMMON_ARGS=(--seeds "$SEEDS")
if [[ "$RESUME" -eq 1 ]]; then
  COMMON_ARGS+=(--resume)
fi

echo "===== V-Fin8 1000-dim SemanticConvE group run ====="
echo "datasets=${DATASETS_ARG}"
echo "seeds=${SEEDS}"
echo "resume=${RESUME}"

for dataset in "${DATASETS[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  if [[ -z "$dataset" ]]; then
    continue
  fi
  echo ""
  echo "================ ${dataset} ================"
  bash scripts/run_autodl_vfin_dataset.sh "$dataset" semantic "${COMMON_ARGS[@]}"
done

echo ""
echo "===== Group run completed ====="
