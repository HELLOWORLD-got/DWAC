"""Prepare matched ResNet / MVOQ-position / ViT pilot configurations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='configs/localization_controls/trainable.json')
    p.add_argument('--output',default='configs/architecture_controls')
    a=p.parse_args();base=json.loads(Path(a.base).read_text())
    base['data'].update(height=416,width=720,augment=False)
    base['model'].update(num_queries=25,dropout=0.,pretrained=True,freeze_backbone=False)
    base['training'].update(epochs=5,batch_size=4,seed=42,workers=4,background_weight=.1,freeze_bn_stats=True,freeze_backbone_stem=False,limit_train=None,limit_val=None,clip_grad=None)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    variants = [('resnet_baseline','resnet50','none',False),
                ('resnet_pe','resnet50','keys_values',False),
                ('vit_baseline','vit_b16','none',False),
                ('resnet_residual','resnet50','none',True),
                ('resnet_pe_residual','resnet50','keys_values',True)]
    for name,backbone,position,residual in variants:
        cfg=deepcopy(base);cfg['model'].update(backbone=backbone,mvoq_position_encoding=position,
                                              mvoq_instance_residual=residual)
        path=output/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n');print(path)
if __name__=='__main__':main()
