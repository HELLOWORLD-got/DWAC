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
from training.data import MultiViewDataset,collate,read_tracks
from tools.prepare_data import audit_native_categories, prepare, prepare_official_test
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

    def test_xml_categories_are_preserved_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'labels.xml'
            p.write_text('<annotations><track id="7" label="car"><box frame="0" outside="0" xtl="2" ytl="2" xbr="8" ybr="8"/></track><track id="9" label="person"><box frame="0" outside="0" xtl="10" ytl="2" xbr="15" ybr="8"/></track></annotations>')
            frames,_,_=read_tracks(p,20,20)
            self.assertEqual(frames[0][7]['label'],1)
            self.assertEqual(frames[0][9]['label'],0)

    def test_native_conflicts_fail_before_manifest_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw';out=Path(tmp)/'out';out.mkdir()
            for split in ('train','val','test'):
                for v in (1,2):(root/split/str(v)).mkdir(parents=True)
            for v,label in [(1,'person'),(2,'car')]:
                (root/'train'/str(v)/f'23-{v}').mkdir()
                xml=root/'new_xml'/str(v)/f'23-{v}.xml';xml.parent.mkdir(parents=True)
                xml.write_text(f'<annotations><track id="7" label="{label}"><box frame="0" outside="0" xtl="1" ytl="1" xbr="4" ybr="4"/></track></annotations>')
            r=audit_native_categories(root);self.assertEqual(r['conflict_count'],1)
            self.assertEqual(r['simultaneous_conflict_count'],1)
            (out/'train.jsonl').write_text('existing-data')
            with self.assertRaisesRegex(ValueError,'Native labels conflict'):prepare(root,out,class_mode='native')
            self.assertEqual((out/'train.jsonl').read_text(),'existing-data')

    def test_four_class_author_loss_uses_background_index_four(self):
        boxes=torch.tensor([[[.4,.4,.2,.2]],[[.4,.4,.2,.2]]])
        p=pad_targets([torch.tensor([1])],[boxes],[torch.ones(2,1,dtype=torch.bool)],2,4)
        self.assertEqual(p['labels'].tolist(),[[1,4]])
        logits=torch.full((1,2,5),-8.);logits[0,0,1]=8.;logits[0,1,4]=8.
        out=dict(class_logits=logits,boxes=p['boxes'],confidences=p['visibility']*.98+.01)
        loss=SetCriterion(4,HungarianMatcher())(out,p)
        self.assertLess(float(loss['loss_cls']),1e-5)
        self.assertAlmostEqual(float(loss['loss_reg']),0.,places=6)
        pred=decode(out)[0];self.assertEqual(int(pred['labels'][0]),1)
        self.assertFalse(bool(pred['present'][:,1].any()))

    def test_car_only_parser_drops_non_car_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'labels.xml'
            p.write_text('<annotations>'+''.join(
                f'<track id="{i}" label="{label}"><box frame="0" outside="0" xtl="2" ytl="2" xbr="8" ybr="8"/></track>'
                for i,label in enumerate(['car','person','bus','bicycle']))+'</annotations>')
            frames,labels,_=read_tracks(p,20,20,class_mode='car_only')
            self.assertEqual(set(frames[0]),{0});self.assertEqual(labels,{0:'car'})
            self.assertEqual(frames[0][0]['label'],0)

    def test_car_filter_keeps_complete_groups_and_empty_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw';out=Path(tmp)/'prepared'
            def track(identity,label,frames):
                return f'<track id="{identity}" label="{label}">'+''.join(
                    f'<box frame="{f}" outside="0" xtl="2" ytl="2" xbr="8" ybr="8"/>' for f in frames)+'</track>'
            for split,seq in [('train','1'),('val','2'),('test','3')]:
                for v in (1,2):
                    images=root/split/str(v)/f'{seq}-{v}';images.mkdir(parents=True)
                    for i in range(1,5):Image.new('RGB',(20,20)).save(images/f'{i:08}.jpg')
                    tracks=track(7,'car',[0])+track(99,'person',[0,1,2,3])+track(100,'bus',[1])
                    tracks+=track(13,'car' if v==1 else 'person',[0])
                    tracks+=track(8 if v==1 else 9,'car',[1])
                    tracks+=track(10 if v==1 else 12,'car',[2])
                    if v==1:tracks+=track(11,'car',[2])
                    xml=root/'new_xml'/str(v)/f'{seq}-{v}.xml';xml.parent.mkdir(parents=True,exist_ok=True)
                    xml.write_text('<annotations>'+tracks+'</annotations>')
            with contextlib.redirect_stdout(io.StringIO()):r=prepare(root,out,class_mode='car_only',max_instances=2)
            rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
            self.assertEqual([s['frame'] for s in rows],[0,1,3])
            self.assertEqual(rows[0]['ids'],[7,13]);self.assertEqual(rows[0]['visible'],[[True,True],[True,False]])
            self.assertEqual(rows[1]['ids'],[8,9])
            self.assertEqual(rows[2]['ids'],[])
            self.assertEqual(r['splits']['train']['excluded_over_query_limit'],1)
            self.assertEqual(r['splits']['train']['empty_target_frames'],1)
            self.assertFalse(r['ground_truth_truncated'])
            ds=MultiViewDataset(out/'train.jsonl',size=(20,20),num_classes=1,num_queries=2,class_mode='car_only')
            self.assertEqual(len(ds[2][1]['labels']),0)

    def test_official_txt_cannot_replace_car_only_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'car.jsonl'
            manifest.write_text(json.dumps(dict(class_mode='car_only',labels=[0]))+'\n')
            with self.assertRaisesRegex(ValueError,'discard native classes'):
                prepare_official_test(manifest,root/'gt',root/'official.jsonl')

    def test_car_only_export_rejects_all_object_official_scoring(self):
        from tools.score_official_mda import score
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'_metadata.json').write_text(json.dumps(dict(class_mode='car_only')))
            with self.assertRaisesRegex(ValueError,'all-object official TXT'):
                score(root,root/'gt','reference/mdmt_official/mango_eval.py')


if __name__=='__main__':unittest.main()
