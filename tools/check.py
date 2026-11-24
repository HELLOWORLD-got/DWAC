"""Check pinned source, full MDMT capacity, environment, and optional real-data forward/backward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torchvision
from training.data import MultiViewDataset, collate
from training.engine import build_model, build_criterion, forward_loss, freeze_training_bn, load_config


def verify_source(root):
    provenance=json.loads((root/'AUTHOR_SOURCE.json').read_text())
    expected=provenance['files']
    actual={str(p.relative_to(root)) for p in (root/'model').glob('*.py')}
    if actual!=set(expected):raise ValueError('Unexpected/missing author model files')
    for name,record in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Author source changed: {name}')
    return provenance['commit']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/mdmt_resnet50.json')
    p.add_argument('--device',default='cpu')
    p.add_argument('--forward',action='store_true')
    p.add_argument('--backward',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    config=load_config(args.config);source=verify_source(root)
    report=dict(python=sys.version.split()[0],torch=torch.__version__,torchvision=torchvision.__version__,
                cuda_available=torch.cuda.is_available(),author_commit=source,model_files_verified=12,
                num_queries=config['model']['num_queries'],splits={})
    datasets={};sequences={}
    for split in ('train','val','test'):
        ds=MultiViewDataset(config['data'][split],size=(config['data']['height'],config['data']['width']),
                            num_views=config['model']['num_views'],num_queries=config['model']['num_queries'],
                            class_mode=config['data'].get('class_mode'),
                            num_classes=config['model']['num_classes'],augment=False)
        seq=set();dense=0;maximum=-1
        for i in range(len(ds)):
            r=ds.record(i);seq.add(str(r['sequence']))
            if len(r['labels'])>maximum:dense=i;maximum=len(r['labels'])
        datasets[split]=(ds,dense);sequences[split]=seq
        report['splits'][split]=dict(samples=len(ds),maximum_instances=maximum,sequences=len(seq))
    if any(sequences[a]&sequences[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Sequence leakage between full dataset splits')
    for name in ('resnet50-11ad3fa6.pth','vit_b_16-c867db91.pth'):
        report.setdefault('weights',{})[name]=(root/'weights'/name).is_file()
    if args.forward or args.backward:
        device=torch.device(args.device);ds,index=datasets['train'];images,targets=collate([ds[index]])
        net=build_model(config,initialize_weights=True).to(device)
        net.train(args.backward);freeze_training_bn(net,config)
        with torch.set_grad_enabled(args.backward):
            output,losses,_=forward_loss(net,build_criterion(net,config),images,targets,device)
            if args.backward:losses['loss'].backward()
        report['real_sample']=dict(index=index,sequence=targets[0]['sequence'],frame=targets[0]['frame'],
                                   instances=len(targets[0]['labels']),boxes_shape=list(output['boxes'].shape),
                                   losses={k:float(v.detach()) for k,v in losses.items()},backward=args.backward)
        if args.backward:
            bad=[n for n,v in net.named_parameters() if v.requires_grad and (v.grad is None or not bool(torch.isfinite(v.grad).all()))]
            if bad:raise ValueError(f'Missing/nonfinite gradients: {bad[:5]}')
            report['real_sample']['all_trainable_gradients_finite']=True
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
