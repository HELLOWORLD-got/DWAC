"""Audit instance/spatial paths, localization gradients, and fixed-sample matching.

Does not change production architecture or overwrite training checkpoints.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import pad_targets
from training.data import MultiViewDataset, collate
from training.engine import build_model,build_criterion,freeze_training_bn,seed_all
from training.metrics import xyxy,AssociationMetrics,decode
from torchvision.ops import box_iou

def pack(targets,n):
    return pad_targets([t['labels'] for t in targets],[t['boxes'] for t in targets],[t['visible'] for t in targets],n,1)

def mapping(assignments,targets):
    return torch.cat([a[:,0][a[:,1].argsort()][:int(n)] for a,n in zip(assignments,targets['num_instances'])])

def matcher_of(criterion):
    return criterion.author.matcher if hasattr(criterion,'author') else criterion.matcher

def grad_stats(module):
    params=[p for p in module.parameters() if p.requires_grad]
    grads=[p.grad for p in params if p.grad is not None]
    return {'trainable_tensors':len(params),'gradient_tensors':len(grads),'nonfinite_tensors':sum(not bool(torch.isfinite(g).all()) for g in grads),'l2':math.sqrt(sum(float(g.square().sum()) for g in grads)),'rms':math.sqrt(sum(float(g.square().sum()) for g in grads)/max(1,sum(g.numel() for g in grads)))}

@torch.no_grad()
def coverage(out,targets):
    hits=total=0;assoc=AssociationMetrics()
    for b,(t,p) in enumerate(zip(targets,decode(out))):
        assoc.update(p,t)
        for k in range(len(t['visible'])):
            gt=t['boxes'][k,t['visible'][k]]
            if len(gt):
                hits+=int((box_iou(xyxy(out['boxes'][b,k]),xyxy(gt)).max(0).values>=.5).sum());total+=len(gt)
    return {'raw_hits':hits,'visible_gt':total,'true_associations':assoc.compute()['true_associations']}

def inspect(net,criterion,images,targets,cfg):
    net.eval();net.zero_grad(set_to_none=True)
    out=net(images);padded=pack(targets,out['boxes'].shape[2]);losses=criterion(out,padded)
    nodes=[out['instance_features'],out['global_feature'],*out['view_features']]
    grads=torch.autograd.grad(losses['loss_reg'],nodes,retain_graph=True,allow_unused=True)
    node_stats={name:{'activation_rms':float(x.detach().square().mean().sqrt()),'gradient_rms':None if g is None else float(g.square().mean().sqrt()),'gradient_l2':None if g is None else float(g.norm())} for name,x,g in zip(['instance_features','global_feature','view0','view1'],nodes,grads)}
    losses['loss_reg'].backward()
    modules={name:grad_stats(getattr(net,name)) for name in ['backbone','mvff','giq','mvoq']}
    modules['giq_class_head']=grad_stats(net.giq.class_head)
    layers=[]
    for layer in net.mvoq.layers:
        g=layer.attn.in_proj_weight.grad;d=layer.attn.embed_dim
        layers.append({'instance_projection':grad_stats(layer.instance_projection),'qkv_weight_gradient_l2':{n:float(x.norm()) for n,x in zip(['Q','K','V'],g.split(d))}})
    with torch.no_grad():
        ins=out['instance_features'].detach();views=[v.detach() for v in out['view_features']];boxes=out['boxes'].detach()
        base=net.mvoq(ins,views)
        changed=net.mvoq(ins.flip(1),views)['boxes']
        permuted=[v.flatten(2).flip(-1).reshape_as(v) for v in views]
        spatial=net.mvoq(ins,permuted)['boxes']
        identity={'instance_only_slot_reverse_mean_box_change':float((changed-boxes).abs().mean()),'view_spatial_permutation_max_box_change':float((spatial-boxes).abs().max())}
        # Jointly permuting instance slots and object-query slots should permute
        # outputs exactly: confirms indexing, not semantic correctness.
        old=net.mvoq.object_queries.detach().clone()
        net.mvoq.object_queries.copy_(old.flip(1))
        joint=net.mvoq(ins.flip(1),views)['boxes'].flip(2)
        net.mvoq.object_queries.copy_(old)
        identity['joint_slot_permutation_max_error']=float((joint-boxes).abs().max())
    # The constant-spatial-token limit exposes whether a direct instance path exists.
    test_ins=ins.clone().requires_grad_();constant=[v.mean((2,3),keepdim=True).expand_as(v) for v in views]
    predicted=net.mvoq(test_ins,constant)['boxes'];weights=torch.linspace(.1,1.,predicted.numel()).reshape_as(predicted)
    derivative,=torch.autograd.grad((predicted*weights).sum(),test_ins)
    with torch.no_grad():
        constant_swapped=net.mvoq(test_ins.flip(1),constant)['boxes']
    identity.update(constant_tokens_instance_gradient_max=float(derivative.abs().max()),constant_tokens_instance_permutation_max_box_change=float((constant_swapped-predicted).detach().abs().max()))
    # Same weights/input, no update: matching should be deterministic at dropout=0.
    with torch.no_grad():
        a=matcher_of(criterion)(out['class_logits'],out['boxes'],padded)
        repeat=net(images);b=matcher_of(criterion)(repeat['class_logits'],repeat['boxes'],padded)
    return {'coverage':coverage(out,targets),'losses':{k:float(v.detach()) for k,v in losses.items()},'node_gradients':node_stats,'module_regression_gradients':modules,'mvoq_layers':layers,'identity_and_space':identity,'same_weights_matching_change_rate':float((mapping(a,padded)!=mapping(b,padded)).float().mean())}

class FixedMatcher(torch.nn.Module):
    def __init__(self,assignments):super().__init__();self.assignments=assignments
    def forward(self,*args):return self.assignments

def matching_run(base,features,targets,cfg,steps,fixed):
    seed_all(42);net=deepcopy(base);criterion=build_criterion(net,cfg);matcher=matcher_of(criterion);padded=pack(targets,cfg['model']['num_queries'])
    optimizer=torch.optim.AdamW([p for n,p in net.named_parameters() if not n.startswith('backbone.')],lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    def forward():
        giq=net.giq(net.mvff(features));out=net.mvoq(giq['instance_features'],features);out['class_logits']=giq['class_logits'];return out
    net.eval()
    with torch.no_grad():
        initial=forward();assignments=matcher(initial['class_logits'],initial['boxes'],padded);previous=mapping(assignments,padded)
    if fixed:
        if hasattr(criterion,'author'):criterion.author.matcher=FixedMatcher(assignments)
        else:criterion.matcher=FixedMatcher(assignments)
    records=[];changed=compared=0
    for step in range(steps+1):
        # dropout=0: eval keeps numerical path aligned with path inspection.
        net.eval();optimizer.zero_grad(set_to_none=True);out=forward()
        with torch.no_grad():
            optimal=matcher(out['class_logits'],out['boxes'],padded);current=mapping(optimal,padded)
            rate=float((current!=previous).float().mean()) if step else 0.
            if step:changed+=int((current!=previous).sum());compared+=current.numel()
            previous=current
        losses=criterion(out,padded)
        record={'step':step,'loss':float(losses['loss'].detach()),'reg':float(losses['loss_reg'].detach()),'optimal_match_changed_fraction':rate}
        if step%10==0 or step==steps:record.update(coverage(out,targets));print('fixed' if fixed else 'dynamic',record,flush=True)
        records.append(record)
        if step==steps:break
        losses['loss'].backward();optimizer.step()
    return {'fixed_supervision':fixed,'note':'Optimal matcher still probed when supervision fixed; changes refer to same GTs across updates','mean_optimal_matching_switch_fraction':changed/max(compared,1),'records':records}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',default=['runs/mdmt_sparse_q25_no_dropout/last.pt','runs/mdmt_sparse_q25_no_dropout_epoch15/last.pt']);p.add_argument('--steps',type=int,default=40);p.add_argument('--output',default='runs/formula_audit_20261122');args=p.parse_args()
    if args.steps<1:p.error('steps must be positive')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);report=[]
    for path in args.checkpoints:
        ck=torch.load(path,map_location='cpu',weights_only=True);cfg=ck['config'];net=build_model(cfg).eval();net.load_state_dict(ck['model']);criterion=build_criterion(net,cfg)
        train_ds=MultiViewDataset('data/processed/MDMT/train.jsonl',size=(405,720));train_images,train_targets=collate([train_ds[i] for i in [249,4140,7209,7955]])
        row={'checkpoint':path,'epoch':ck['epoch'],'config':cfg,'paths':{}}
        val_ds=MultiViewDataset(cfg['data']['val'],size=(405,720));val_images,val_targets=collate([val_ds[i] for i in [0,120]])
        for split,images,targets in [('train',[v[:2] for v in train_images],train_targets[:2]),('val',val_images,val_targets)]:
            row['paths'][split]=inspect(net,criterion,images,targets,cfg);print('path',ck['epoch'],split,row['paths'][split],flush=True)
        with torch.no_grad():features=[net.backbone(v).detach() for v in train_images]
        row['matching_dynamic']=matching_run(net,features,train_targets,cfg,args.steps,False)
        row['matching_fixed']=matching_run(net,features,train_targets,cfg,args.steps,True)
        report.append(row);(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',root/'audit.json',flush=True)
if __name__=='__main__':main()
