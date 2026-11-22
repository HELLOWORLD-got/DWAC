"""Compare localization changes on identical, evenly sampled MDMT images."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from training.data import MultiViewDataset, collate
from training.engine import build_model
from training.metrics import xyxy, decode

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs=2, required=True)
    p.add_argument('--output', default='runs/epoch5_vs15')
    p.add_argument('--limit',type=int,default=16)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.limit<2:p.error('--limit must be at least 2')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    all_results=[];previous={};pictures={};config=None
    for path in args.checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True);cfg=state['config']
        if config is not None:
            for key in ('model','data','evaluation'):
                if config[key]!=cfg[key]:raise ValueError(f'Checkpoints differ in {key}')
        config=cfg;net=build_model(cfg).to(args.device).eval();net.load_state_dict(state['model'])
        result={'checkpoint':path,'epoch':state['epoch'],'splits':{}}
        for split in ('train','val'):
            ds=MultiViewDataset(cfg['data'][split],size=(cfg['data']['height'],cfg['data']['width']))
            count=min(args.limit,len(ds));indices=[i*(len(ds)-1)//(count-1) for i in range(count)]
            rows=[];hit_vectors=[];ious=[];boxes_all=[];entropies=[]
            for start in range(0,count,4):
                images,targets=collate([ds[i] for i in indices[start:start+4]])
                out=net([x.to(args.device) for x in images]);preds=decode(out)
                if start==0:
                    captured=[]
                    hooks=[l.attn.register_forward_pre_hook(lambda m,a:captured.append((m,a[:3]))) for l in net.mvoq.layers]
                    net.mvoq(out['instance_features'],out['view_features'])
                    for hook in hooks:hook.remove()
                    for module,inputs in captured:
                        _,weights=module(*inputs,need_weights=True,average_attn_weights=False)
                        prob=weights.clamp_min(1e-20)
                        entropies.append(float((-(prob*prob.log()).sum(-1)/__import__('math').log(prob.shape[-1])).mean()))
                boxes_all.append(out['boxes'].cpu())
                for target,pred in zip(targets,preds):
                    row={'sequence':target['sequence'],'frame':target['frame'],'active_queries':int((pred['present'].any(0)).sum())}
                    hit=total=0
                    for view in range(2):
                        gt=target['boxes'][view,target['visible'][view]]
                        overlaps=box_iou(xyxy(pred['boxes'][view]),xyxy(gt))
                        best=overlaps.max(0).values
                        hit_vectors.append(best>=.5);ious.append(best);hit+=int((best>=.5).sum());total+=len(gt)
                    row.update(hits=hit,gt=total);rows.append(row)
                    if not pictures.get((state['epoch'],split)):
                        pictures[(state['epoch'],split)]=(target,pred,ds.path.parent)
            hits=torch.cat(hit_vectors);best=torch.cat(ious);boxes=torch.cat(boxes_all)
            summary={'samples':count,'raw_box_hits':int(hits.sum()),'visible_gt_boxes':len(hits),'median_best_iou':float(best.median()),'mvoq_attention_entropy':entropies,'per_sample':rows}
            if split in previous:
                old,old_boxes=previous[split]
                summary.update(newly_hit=int((hits&~old).sum()),lost_hits=int((~hits&old).sum()),retained_hits=int((hits&old).sum()),mean_abs_box_change=float((boxes-old_boxes).abs().mean()))
            previous[split]=(hits,boxes)
            result['splits'][split]=summary
            print(state['epoch'],split,{k:v for k,v in summary.items() if k!='per_sample'},flush=True)
        all_results.append(result)
        del net,state
    (root/'comparison.json').write_text(json.dumps(all_results,indent=2)+'\n')
    for split in ('train','val'):
        canvas=Image.new('RGB',(1440,810))
        for row,r in enumerate(all_results):
            target,pred,parent=pictures[(r['epoch'],split)]
            for view in range(2):
                with Image.open(parent/target['images'][view]) as source:im=source.convert('RGB').resize((720,405))
                draw=ImageDraw.Draw(im);scale=torch.tensor([720,405,720,405])
                for box in xyxy(target['boxes'][view,target['visible'][view]])*scale:draw.rectangle(box.tolist(),outline='lime',width=2)
                for box in xyxy(pred['boxes'][view,pred['present'][view]])*scale:draw.rectangle(box.tolist(),outline='red',width=2)
                draw.text((8,8),f'Epoch {r["epoch"]} {split} seq{target["sequence"]} frame{target["frame"]} view{view+1} GREEN=GT RED=prediction',fill='white')
                canvas.paste(im,(720*view,405*row))
        canvas.save(root/f'{split}.jpg')
if __name__=='__main__':main()
