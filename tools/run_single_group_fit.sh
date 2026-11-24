#!/usr/bin/env bash
# Two independent debug fits, starting from identical trained weights.
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:-runs/mdmt_car_q25_author/last.pt}"
steps="${2:-500}"
output="${3:-runs/single_group_fit_$(date +%Y%m%d_%H%M%S)}"
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "Choose a new output directory: $output" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python tools/fit_single_group.py --checkpoint "$checkpoint" \
  --split train --index 0 --steps "$steps" --log-every 25 \
  --device cuda --output "$output/train_index0"
python tools/fit_single_group.py --checkpoint "$checkpoint" \
  --split val --index 0 --steps "$steps" --log-every 25 \
  --device cuda --output "$output/val_index0"
echo "Independent single-group debug fits completed: $output"
