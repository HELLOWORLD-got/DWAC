"""Replay one captured local batch with anomaly tracing; no optimizer update."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training.engine import build_model, build_criterion, forward_loss, freeze_training_bn
from training.gradients import gradient_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    state=torch.load(args.snapshot,map_location='cpu',weights_only=True)
    device=torch.device(args.device);net=build_model(state['config']).to(device)
    net.load_state_dict(state['model'],strict=True);net.train();freeze_training_bn(net,state['config'])
    if state.get('rng'):
        torch.set_rng_state(state['rng']['torch_rng'])
        if device.type=='cuda' and state['rng']['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['rng']['cuda_rng'],device)
    print(f'Local replay: rank={state["rank"]}, epoch={state["epoch"]}, batch={state["batch"]}. '
          'This does not replay DDP gradient reduction.',flush=True)
    with torch.autograd.detect_anomaly(check_nan=True):
        _,losses,_=forward_loss(net,build_criterion(net,state['config']),state['images'],state['targets'],device)
        print({key:float(value.detach()) for key,value in losses.items()},flush=True)
        losses['loss'].backward()
    report=gradient_report(net.named_parameters())
    output=Path(args.snapshot).with_suffix('.replay.json')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(dict(all_gradients_finite=report['all_gradients_finite'],total_norm_fp64=report['total_norm_fp64'],report=str(output)))


if __name__=='__main__':main()
