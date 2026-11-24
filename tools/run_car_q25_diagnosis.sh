#!/usr/bin/env bash
# Run from the project root after syncing code with Git. Does not train or alter weights.
set -euo pipefail
cd "$(dirname "$0")/.."
experiment="${1:-runs/mdmt_car_q25_author}"
output="${2:-runs/car_q25_diagnosis_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python tools/diagnose.py \
  --checkpoints "$experiment/best.pt" "$experiment/last.pt" \
  --device cuda --batch-size 4 --workers 0 --limit 240 --visualize 6 \
  --output "$output"
echo "Diagnostic reports and images: $output"
