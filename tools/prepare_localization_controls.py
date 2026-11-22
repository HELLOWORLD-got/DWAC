"""Prepare paired localization controls with a temporal holdout and frame gap."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path

def prepare(config_path, output, config_dir, gap=20, fraction=.2):
    base=json.loads(Path(config_path).read_text())
    source=Path(base['data']['train']).resolve()
    groups={}
    for line in source.read_text().splitlines():
        sample=json.loads(line)
        groups.setdefault(str(sample['sequence']),[]).append(sample)
    partitions={'train':[], 'val':[]};details={}
    for sequence,samples in sorted(groups.items()):
        samples.sort(key=lambda s:int(s['frame']))
        holdout=max(1,math.ceil(len(samples)*fraction))
        threshold=int(samples[-holdout]['frame'])
        train=[s for s in samples if int(s['frame'])<threshold-gap]
        val=samples[-holdout:]
        if not train:raise ValueError(f'No training frames remain for sequence {sequence}')
        assert max(int(s['frame']) for s in train)<min(int(s['frame']) for s in val)-gap
        partitions['train'].extend(train);partitions['val'].extend(val)
        details[sequence]={'train':len(train),'val':len(val),'excluded':len(samples)-len(train)-len(val),'last_train_frame':int(train[-1]['frame']),'first_val_frame':threshold}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if source in [output/'train.jsonl',output/'val.jsonl']:raise ValueError('Output must differ from source')
    for split,samples in partitions.items():
        with (output/f'{split}.jsonl').open('w') as f:
            for sample in samples:
                sample=deepcopy(sample)
                sample['images']=[os.path.relpath(source.parent/name,output) for name in sample['images']]
                f.write(json.dumps(sample,separators=(',',':'))+'\n')
    base['data'].update(train=os.path.relpath(output/'train.jsonl'),val=os.path.relpath(output/'val.jsonl'),augment=False)
    # In these diagnostic configs "test" means the original held-out sequence,
    # not the official MDMT test split. Label this explicitly in the report.
    base['data']['test']=json.loads(Path(config_path).read_text())['data']['val']
    base['training'].update(epochs=5,batch_size=4,seed=42,workers=4,background_weight=.1,freeze_bn_stats=True)
    for key in ('limit_train','limit_val','clip_grad'):base['training'][key]=None
    base['model'].update(num_queries=25,dropout=0.,pretrained=True)
    config_dir=Path(config_dir);config_dir.mkdir(parents=True,exist_ok=True)
    paths=[]
    for label,freeze in [('trainable',False),('frozen',True)]:
        cfg=deepcopy(base);cfg['model']['freeze_backbone']=False
        cfg['training']['freeze_backbone_stem']=freeze
        path=config_dir/f'{label}.json';path.write_text(json.dumps(cfg,indent=2)+'\n');paths.append(str(path))
    report={'source':os.path.relpath(source),'gap_original_frame_indices':gap,'validation_fraction':fraction,'per_sequence':details,'counts':{k:len(v) for k,v in partitions.items()},'configs':paths,'test_split_meaning':'Original sparse held-out sequence 64, NOT official MDMT test','frozen_scope':'Pretrained ResNet stem/layer1-4 frozen; 1x1 projection remains trainable','initialization':'Both runs start fresh from ImageNet, not epoch5/15; identical seed and split'}
    (output/'split_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='runs/mdmt_sparse_q25_no_dropout_epoch15/config.json')
    p.add_argument('--output',default='data/processed/MDMT/localization_controls')
    p.add_argument('--config-dir',default='configs/localization_controls')
    p.add_argument('--gap',type=int,default=20)
    p.add_argument('--val-fraction',type=float,default=.2)
    a=p.parse_args()
    if a.gap<0 or not 0<a.val_fraction<1:p.error('gap must be nonnegative and fraction in (0,1)')
    prepare(a.config,a.output,a.config_dir,a.gap,a.val_fraction)
if __name__=='__main__':main()
