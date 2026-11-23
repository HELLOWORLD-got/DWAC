#!/usr/bin/env bash
# Fork a recorded optimization-recovery stage; original run stays intact.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
output=runs/mdmt_resnet50_center_full_stable
source_checkpoint="${SOURCE_CHECKPOINT:-runs/mdmt_resnet50_center_full/best.pt}"
if [[ -f "$output/last.pt" ]]; then source_checkpoint="$output/last.pt"; fi
python - "$source_checkpoint" <<'PY'
import json
from pathlib import Path
import sys
import torch
path=Path(sys.argv[1]);state=torch.load(path,map_location='cpu',weights_only=True)
bad=[name for name,value in state['model'].items() if not bool(torch.isfinite(value).all())]
for key,entry in state['optimizer']['state'].items():
    for name,value in entry.items():
        if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):bad.append(f'optimizer/{key}/{name}')
if bad:raise ValueError(f'Checkpoint contains nonfinite state; recovery stopped: {bad[:5]}')
config=state['config'];config['training'].update(lr=5e-5,clip_grad=1.0)
Path('configs/mdmt_resnet50_center_full_recovery.json').write_text(json.dumps(config,indent=2)+'\n')
print(f'Recovering from {path}, completed epoch={state["epoch"]}; LR=5e-5, clip=1.0',flush=True)
PY
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29643}" train.py \
  --config configs/mdmt_resnet50_center_full_recovery.json \
  --resume "$source_checkpoint" --allow-optimization-change --save-failure-batch \
  --device cuda --workers "${WORKERS:-4}" --epochs "${EPOCHS:-15}" \
  --progress bar --output "$output"
python tools/diagnose.py --checkpoint "$output/last.pt" --device cuda \
  --limit 64 --batch-size 4 --output "$output/diagnosis.json"
