#!/usr/bin/env bash
# Execute from the project root. This is a new architecture/loss run.
set -euo pipefail
if [[ ! -f data/processed/MDMT/localization_controls/train.jsonl || \
      ! -f data/processed/MDMT/localization_controls/val.jsonl || \
      ! -f configs/localization_controls/trainable.json ]]; then
  python tools/prepare_localization_controls.py \
    --config "${CONTROL_SOURCE_CONFIG:-runs/mdmt_sparse_q25_no_dropout/config.json}"
fi
if [[ ! -f configs/mdmt_resnet50_spatial.json ]]; then
  python tools/prepare_spatial_config.py
fi
python tools/prepare_center_config.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
output=runs/mdmt_resnet50_center
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29643}" train.py \
  --config configs/mdmt_resnet50_center.json --device cuda \
  --batch-size 4 --workers 4 --epochs 15 --progress bar --output "$output"
python tools/diagnose.py --checkpoint "$output/last.pt" --device cuda \
  --limit 64 --batch-size 4 --output "$output/diagnosis.json"
python evaluate.py --checkpoint "$output/best.pt" --device cuda --split test \
  --batch-size 4 --workers 0 --output "$output/unseen_sequence64_best.json"
