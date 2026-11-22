#!/usr/bin/env bash
# Run from the project root; optional positional arguments select variants.
set -euo pipefail
if [[ ! -f data/processed/MDMT/localization_controls/train.jsonl || \
      ! -f data/processed/MDMT/localization_controls/val.jsonl || \
      ! -f configs/localization_controls/trainable.json ]]; then
  python tools/prepare_localization_controls.py \
    --config "${CONTROL_SOURCE_CONFIG:-runs/mdmt_sparse_q25_no_dropout/config.json}"
fi
python tools/prepare_architecture_controls.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
variants=(resnet_baseline resnet_pe vit_baseline)
if (( $# )); then variants=("$@"); fi
for mode in "${variants[@]}"; do
  case "$mode" in resnet_baseline|resnet_pe|vit_baseline) ;; *) echo "Unknown variant: $mode" >&2; exit 2;; esac
  extra=(); eval_extra=(); diagnose_limit=64; epochs=5; output="runs/architecture_${mode}"
  if [[ "${SMOKE:-0}" == 1 ]]; then
    extra=(--limit-train 16 --limit-val 8); eval_extra=(--limit 8)
    diagnose_limit=8; epochs=1; output="${output}_smoke"
  fi
  torchrun --nproc-per-node=2 --rdzv-backend=static \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29643}" train.py \
    --config "configs/architecture_controls/${mode}.json" \
    --device cuda --batch-size 4 --workers 4 --epochs "$epochs" \
    --progress bar --output "$output" "${extra[@]}"
  python tools/diagnose.py --checkpoint "$output/last.pt" --device cuda \
    --limit "$diagnose_limit" --batch-size 4 --output "$output/diagnosis.json"
  # Dedicated config test split = sparse sequence64, NOT official MDMT test.
  python evaluate.py --checkpoint "$output/last.pt" --split test --device cuda \
    --workers 0 --batch-size 4 --output "$output/unseen_sequence64.json" "${eval_extra[@]}"
done
