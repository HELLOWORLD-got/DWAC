#!/usr/bin/env bash
# Full MDMT training; execute from the project root.
set -euo pipefail
python tools/prepare_full_center.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
output=runs/mdmt_resnet50_center_full
check_output=runs/mdmt_resnet50_center_full_gpu_check
if [[ "${SMOKE:-0}" != 1 && "${RESUME:-0}" != 1 && -f "$output/last.pt" ]]; then
  echo "Run exists. Continue with RESUME=1 EPOCHS=30 bash tools/run_full_center_training.sh" >&2
  exit 2
fi
if [[ ! -f "$check_output/last.pt" ]]; then
  echo "Checking 256 queries on 16 dense training pairs and 8 dense validation pairs ..."
  torchrun --nproc-per-node=2 --rdzv-backend=static \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29643}" train.py \
    --config configs/mdmt_resnet50_center_full_smoke.json --device cuda \
    --batch-size 4 --workers 0 --epochs 1 --progress bar --output "$check_output"
fi
if [[ "${SMOKE:-0}" == 1 ]]; then
  echo "Dense GPU check complete; full training not started."
  exit 0
fi
config=configs/mdmt_resnet50_center_full.json
resume=()
if [[ "${RESUME:-0}" == 1 ]]; then
  if [[ ! -f "$output/last.pt" || ! -f "$output/config.json" ]]; then
    echo "No complete full-data checkpoint/config to resume: $output" >&2
    exit 2
  fi
  config="$output/config.json"
  resume=(--resume "$output/last.pt")
fi
# The dense check is NOT resumed here; full training starts from ImageNet.
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29643}" train.py \
  --config "$config" --device cuda --batch-size 4 --workers "${WORKERS:-4}" \
  --epochs "${EPOCHS:-15}" --progress bar --output "$output" "${resume[@]}"
cp data/processed/MDMT/full_center_audit.json "$output/data_audit.json"
python tools/diagnose.py --checkpoint "$output/last.pt" --device cuda \
  --limit 64 --batch-size 4 --output "$output/diagnosis.json"
echo "Full train/validation stage complete. Official test is reserved for final evaluation."
