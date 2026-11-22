#!/usr/bin/env bash
# Run from the project root. New runs, identical data and seed; no resume.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
for mode in trainable frozen; do
  output="runs/localization_${mode}"
  torchrun --nproc-per-node=2 --rdzv-backend=static \
    --master-addr=127.0.0.1 --master-port=29643 train.py \
    --config "configs/localization_controls/${mode}.json" \
    --device cuda --workers 4 --epochs 5 --progress bar --output "$output"
  # Same-sequence validation is evaluated each epoch during training.
  python tools/diagnose.py --checkpoint "$output/last.pt" \
    --device cuda --limit 64 --batch-size 4 --output "$output/diagnosis.json"
  # "test" in these diagnostic configs is sparse sequence64, NOT official MDMT.
  python evaluate.py --checkpoint "$output/last.pt" --split test \
    --device cuda --workers 0 --batch-size 4 --output "$output/unseen_sequence64.json"
done
