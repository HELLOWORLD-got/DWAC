"""Contracts for the unmodified author model and the training peripherals."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torchvision.ops import box_convert, generalized_box_iou
from model import ModelConfig, pad_targets
from model.box_ops import box_loss,box_loss_pairwise
from model.loss import SetCriterion
from model.matcher import HungarianMatcher
from training.data import MultiViewDataset,collate
from training.engine import build_model,build_criterion,forward_loss,freeze_training_bn,load_config
from training.metrics import AssociationMetrics,DetectionMetrics,decode
from training.distributed import capture_rng,per_rank_batch_size
from tools.check import verify_source
from train import run,resume_stage_metadata


def target(empty=False):
    m=0 if empty else 1
    return dict(labels=torch.zeros(m,dtype=torch.long),boxes=torch.tensor([[[.4,.4,.2,.2]],[[0.,0.,0.,0.]]])[:,:m],
                visible=torch.tensor([[True],[False]])[:,:m],sizes=[[64,64],[64,64]],sequence='synthetic',frame=0)


def pack(ts,n=3):
    return pad_targets([t['labels'] for t in ts],[t['boxes'] for t in ts],[t['visible'] for t in ts],n,1)


class AuthorTrainingTests(unittest.TestCase):
    def config(self,root):
        path=root/'config.json';manifest=root/'samples.json'
        Image.new('RGB',(64,64),'gray').save(root/'image.png')
        manifest.write_text(json.dumps([dict(images=['image.png']*2,boxes=[[[.4,.4,.2,.2]],[[0.,0.,0.,0.]]],
            visible=[[True],[False]],labels=[0],ids=[7],sizes=[[64,64]]*2,sequence='synthetic',frame=0)]))
        path.write_text(json.dumps(dict(model=ModelConfig(pretrained=False,num_views=2,num_classes=1,
            num_queries=3,hidden_dim=32,num_heads=4,mvff_layers=1,giq_layers=1,mvoq_layers=1,dropout=0.).to_dict(),
            data=dict(train=str(manifest),val=str(manifest),test=str(manifest),height=64,width=64,augment=False),
            training=dict(epochs=1,batch_size=1,workers=0,clip_grad=1.),
            evaluation=dict(confidence_threshold=.5,iou_threshold=.5,ucr_threshold=.6))))
        return load_config(path)

    def test_source_is_exactly_pinned(self):
        self.assertEqual(verify_source(Path.cwd()),'e2076afc7f59d9f82e17bc31cd8ddcf9a047fb5a')

    def test_capacity_and_invisible_padding(self):
        t=target();p=pack([t]);self.assertEqual(p['labels'].tolist(),[[0,1,1]])
        self.assertEqual(int(p['visibility'].sum()),1)
        with self.assertRaises(ValueError):pack([dict(t,labels=torch.zeros(4,dtype=torch.long),boxes=torch.ones(2,4,4),visible=torch.ones(2,4,dtype=torch.bool))])

    def test_original_loss_and_background_regression_mask(self):
        t=target();p=pack([t]);boxes=p['boxes'].clone().requires_grad_()
        out=dict(class_logits=torch.tensor([[[8.,-8.],[-8.,8.],[-8.,8.]]]),boxes=boxes,
                 confidences=p['visibility']*.98+.01)
        loss=SetCriterion(1,HungarianMatcher())(out,p)
        self.assertAlmostEqual(float(loss['loss_reg'].detach()),0.,places=6)
        grad,=torch.autograd.grad(loss['loss_reg'],boxes)
        self.assertEqual(float(grad[0,1].abs().sum()),0.)
        self.assertEqual(float(grad[:,:,1:].abs().sum()),0.)

    def test_box_math_matches_independent_torchvision_reference(self):
        a=torch.tensor([[.3,.4,.2,.1],[.7,.6,.1,.2]]);b=a+.01
        expected=5*(a-b).abs().sum(-1)+2*(1-generalized_box_iou(box_convert(a,'cxcywh','xyxy'),box_convert(b,'cxcywh','xyxy')).diag())
        torch.testing.assert_close(box_loss(a,b),expected)
        torch.testing.assert_close(box_loss_pairwise(a,b).diag(),expected)

    def test_oracle_decoding_and_metrics(self):
        t=target();p=pack([t]);out=dict(class_logits=torch.tensor([[[8.,-8.],[-8.,8.],[-8.,8.]]]),boxes=p['boxes'],confidences=p['visibility']*.98+.01)
        pred=decode(out)[0];self.assertFalse(bool(pred['present'][:,1:].any()))
        a=AssociationMetrics();a.update(pred,t);self.assertEqual(a.compute()['UCR'],1.)
        d=DetectionMetrics(1);d.update(pred,t);self.assertAlmostEqual(d.compute()['AP50'],1.)

    def test_all_parameters_receive_finite_gradients_in_mixed_empty_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.config(Path(tmp));net=build_model(cfg).train();freeze_training_bn(net,cfg)
            criterion=build_criterion(net,cfg);self.assertIs(type(criterion),SetCriterion)
            outputs,losses,_=forward_loss(net,criterion,[torch.randn(2,3,64,64) for _ in range(2)],
                                         [target(),target(True)],torch.device('cpu'))
            self.assertEqual(set(losses),{'loss','loss_cls','loss_reg','loss_conf'})
            self.assertEqual(set(outputs),{'class_logits','boxes','confidences'})
            losses['loss'].backward()
            for name,p in net.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)

    def test_ddp_regression_scaling_matches_global_batch(self):
        ts=[target(),target()];ts[1]['visible']=torch.tensor([[True],[True]])
        ts[1]['boxes'][1]=ts[1]['boxes'][0]
        boxes=torch.tensor([[[[.41,.41,.2,.2],[.8,.8,.1,.1],[.2,.2,.1,.1]]]*2]*2,requires_grad=True)
        logits=torch.tensor([[[8.,-8.],[-8.,8.],[-8.,8.]]]*2);conf=torch.full((2,2,3),.5)
        class Net:
            def __init__(self,o):self.o=o
            def __call__(self,images):return self.o
        all_out=dict(boxes=boxes,class_logits=logits,confidences=conf)
        _,global_loss,_=forward_loss(Net(all_out),SetCriterion(1,HungarianMatcher()),[],ts,torch.device('cpu'))
        global_loss['loss_reg'].backward()
        for i in range(2):
            local_box=boxes[i:i+1].detach().requires_grad_();o=dict(boxes=local_box,class_logits=logits[i:i+1],confidences=conf[i:i+1])
            with patch('training.engine.dist.all_reduce',side_effect=lambda x:x.fill_(3)),patch('training.engine.dist.get_world_size',return_value=2):
                _,loss,_=forward_loss(Net(o),SetCriterion(1,HungarianMatcher()),[],[ts[i]],torch.device('cpu'),True)
            loss['loss_reg'].backward();torch.testing.assert_close(local_box.grad/2,boxes.grad[i:i+1])

    def test_train_save_resume_and_fork(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.config(root)
            args=SimpleNamespace(output=str(root/'run'),resume=None,progress='log',allow_optimization_change=False,
                                 limit_train=None,limit_val=None,clip_grad=1.,save_failure_batch=False)
            with contextlib.redirect_stdout(io.StringIO()):run(args,cfg,torch.device('cpu'),0,1)
            first=torch.load(root/'run'/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(first['epoch'],1);self.assertTrue((root/'run'/'best.pt').is_file())
            cfg['training']['epochs']=2;args.resume=str(root/'run'/'last.pt')
            with contextlib.redirect_stdout(io.StringIO()):run(args,cfg,torch.device('cpu'),0,1)
            state=torch.load(root/'run'/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(state['epoch'],2);self.assertEqual(len((root/'run'/'history.jsonl').read_text().splitlines()),2)
            best,lineage=resume_stage_metadata(state,cfg['training'],args.resume,root/'fork',False)
            self.assertEqual(best,(-1.,-1.));self.assertEqual(lineage['source_epoch'],2)

    def test_improved_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.config(root);cfg['training']['center_loss_weight']=.25
            p=root/'bad.json';p.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Modified losses'):load_config(p)

    def test_batch_size_is_global(self):
        self.assertEqual(per_rank_batch_size(4,2),2)
        with self.assertRaises(ValueError):per_rank_batch_size(4,8)


if __name__=='__main__':unittest.main()
