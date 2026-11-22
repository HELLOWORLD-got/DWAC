"""Independent checks for supervision and decoding; no datasets or downloads.

Run from the repository root: python -m unittest discover -s tests -v
These checks do not establish convergence or paper-level reproduction.
"""

import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model import ModelConfig, build_model, pad_targets
from model.box_ops import box_loss, box_loss_pairwise
from model.loss import SetCriterion
from model.matcher import HungarianMatcher
from model.association import ConflictFreeMultiViewAssociation
from training.background_loss import BackgroundWeightedLoss
from training.engine import forward_loss, freeze_training_bn
from training.metrics import AssociationMetrics, DetectionMetrics, decode
from tools.overfit import MatchingProbe
from training.data import MultiViewDataset, image_tensor


def scene():
    return dict(
        labels=torch.tensor([0, 0]),
        boxes=torch.tensor([[[.2, .3, .1, .2], [.7, .6, .2, .1]],
                            [[.4, .5, .1, .2], [0., 0., 0., 0.]]]),
        visible=torch.tensor([[True, True], [True, False]]),
        sizes=[[720, 405], [720, 405]], sequence="synthetic", frame=0,
    )


def pack(target):
    return pad_targets([target["labels"]], [target["boxes"]],
                       [target["visible"]], 3, 1)


def oracle(target):
    padded = pack(target)
    # Prediction order deliberately differs from GT order.
    order = torch.tensor([1, 2, 0])
    logits = torch.tensor([[[8., -8.], [-8., 8.], [8., -8.]]])
    return dict(class_logits=logits, class_probs=logits.softmax(-1),
                boxes=padded["boxes"][:, :, order].clone(),
                confidences=padded["visibility"][:, :, order] * .98 + .01)


class TrainingContracts(unittest.TestCase):
    def test_frozen_stem_keeps_projection_trainable(self):
        from training.engine import build_model as build_training_model, load_config
        config = load_config("configs/localization_controls/frozen.json")
        config["model"]["pretrained"] = False
        model = build_training_model(config)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.stem.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.projection.parameters()))
        model.train()
        freeze_training_bn(model, config)
        self.assertFalse(model.backbone.stem.training)
        self.assertTrue(model.backbone.projection.training)

    def test_crop_and_flip_keep_pixels_and_boxes_aligned(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            im = Image.new("RGB", (20, 10))
            ImageDraw.Draw(im).rectangle([4, 2, 7, 5], fill="white")
            im.save(folder / "image.png")
            manifest = folder / "sample.json"
            manifest.write_text(json.dumps([dict(
                images=["image.png", "image.png"], labels=[0], ids=[42],
                boxes=[[[.3, .4, .2, .4]], [[.3, .4, .2, .4]]],
                visible=[[True], [True]], sizes=[[20, 10], [20, 10]],
            )]))
            dataset = MultiViewDataset(manifest, size=(10, 20), augment=True)
            dataset.jitter = lambda image: image
            cases = [([1., 0.], None, [.7, .4, .2, .4], im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
                     ([0., 1.], [.8, .2, 0.], [.125, .5, .25, .5], im.crop((4, 0, 20, 8)))]
            for draws, uniforms, expected_box, expected_image in cases:
                with patch("training.data.random.random", side_effect=draws), \
                     patch("training.data.random.uniform", side_effect=uniforms):
                    images, target = dataset[0]
                torch.testing.assert_close(target["boxes"], torch.tensor([[expected_box], [expected_box]]))
                torch.testing.assert_close(images[0], image_tensor(expected_image, (10, 20)))
                self.assertEqual(target["ids"].tolist(), [42])

    def test_padding_and_capacity(self):
        target = scene()
        padded = pack(target)
        self.assertEqual(padded["labels"].tolist(), [[0, 0, 1]])
        self.assertEqual(padded["num_instances"].tolist(), [2])
        self.assertEqual(int(padded["visibility"].sum()), 3)
        with self.assertRaises(ValueError):
            pad_targets([target["labels"]], [target["boxes"]], [target["visible"]], 1, 1)

    def test_matcher_and_loss_against_exact_answer(self):
        target = scene()
        outputs = oracle(target)
        criterion = SetCriterion(1, HungarianMatcher())
        assignment = criterion.matcher(outputs["class_logits"], outputs["boxes"], pack(target))[0]
        self.assertEqual(assignment.tolist(), [[0, 1], [1, 2], [2, 0]])
        loss = criterion(outputs, pack(target))
        self.assertAlmostEqual(float(loss["loss_reg"]), 0., places=6)
        self.assertAlmostEqual(float(loss["loss_conf"]), -math.log(.99), places=6)
        self.assertAlmostEqual(float(loss["loss_cls"]), math.log1p(math.exp(-16)), places=6)
        weighted = BackgroundWeightedLoss(criterion, 1.)(outputs, pack(target))
        for key in loss:
            torch.testing.assert_close(loss[key], weighted[key])
        # Permuting real annotations must not change supervision.
        shuffled = dict(target)
        for key in ("boxes", "visible"):
            shuffled[key] = target[key][:, [1, 0]]
        shuffled["labels"] = target["labels"][[1, 0]]
        for weight in (1., .1):
            loss_fn = BackgroundWeightedLoss(criterion, weight)
            torch.testing.assert_close(loss_fn(outputs, pack(target))["loss"],
                                       loss_fn(outputs, pack(shuffled))["loss"])

    def test_invisible_and_padded_boxes_have_no_regression_gradient(self):
        target = scene()
        outputs = oracle(target)
        outputs["boxes"] = (outputs["boxes"] + .005).requires_grad_()
        for weight in (1., .1):
            criterion = BackgroundWeightedLoss(SetCriterion(1, HungarianMatcher()), weight)
            gradient, = torch.autograd.grad(criterion(outputs, pack(target))["loss_reg"],
                                           outputs["boxes"], retain_graph=True)
            self.assertEqual(float(gradient[:, :, 1].abs().sum()), 0.)  # padded query
            self.assertEqual(float(gradient[:, 1, 0].abs().sum()), 0.)  # real but absent
            self.assertGreater(float(gradient[:, 0, 0].abs().sum()), 0.)

    def test_pairwise_and_matched_geometry(self):
        from torchvision.ops import generalized_box_iou, box_convert
        a = torch.tensor([[.3, .4, .2, .1], [.7, .6, .1, .2]])
        b = a + torch.tensor([.02, -.03, .01, -.01])
        expected = 5 * (a - b).abs().sum(-1) + 2 * (1 - generalized_box_iou(
            box_convert(a, "cxcywh", "xyxy"), box_convert(b, "cxcywh", "xyxy")).diag())
        torch.testing.assert_close(box_loss(a, b), expected)
        torch.testing.assert_close(box_loss_pairwise(a, b).diag(), expected)

    def test_oracle_metrics_and_broken_association(self):
        target = scene()
        pred = decode(oracle(target))[0]
        association, detection = AssociationMetrics(), DetectionMetrics(1)
        association.update(pred, target)
        detection.update(pred, target)
        self.assertEqual(association.compute()["MDA"], 1.)
        self.assertAlmostEqual(detection.compute()["mAP"], 1.)
        # Keep the detections identical but change the ID in camera 2.
        pred["boxes"][1] = pred["boxes"][1, [2, 1, 0]]
        pred["present"][1] = pred["present"][1, [2, 1, 0]]
        association = AssociationMetrics()
        association.update(pred, target)
        self.assertEqual(association.compute()["true_associations"], 0)

    def test_predict_threshold_never_admits_background(self):
        class FakeModel:
            config = ModelConfig(num_classes=1)
            def forward(self, images):
                result = oracle(scene())
                result["confidences"].fill_(.99)
                return result
        result = ConflictFreeMultiViewAssociation.predict(FakeModel(), None, class_threshold=.5)
        self.assertFalse(bool(result[0]["valid"][:, 1].any()))

    def test_matching_probe_tracks_all_scenes(self):
        first = [torch.tensor([[0, 0], [1, 1], [2, 2]]) for _ in range(2)]
        second = [first[0], torch.tensor([[0, 1], [1, 0], [2, 2]])]
        class Matcher(torch.nn.Module):
            def forward(self, *args):
                return first
        probe = MatchingProbe(Matcher(), fixed=False)
        targets = {"num_instances": torch.tensor([2, 2])}
        probe(None, None, targets)
        with patch.object(probe.matcher, "forward", return_value=second):
            probe(None, None, targets)
        self.assertEqual(probe.sample_change_rate(), .5)

    def test_actual_model_gradient_and_batch_independence(self):
        torch.manual_seed(42)
        cfg = ModelConfig(pretrained=False, hidden_dim=32, num_views=2, num_classes=1,
                          num_queries=3, mvff_layers=1, giq_layers=1, mvoq_layers=1, dropout=0.)
        net = build_model(cfg)
        images = [torch.randn(2, 3, 64, 64) for _ in range(2)]
        settings = {"model": {"freeze_backbone": False}, "training": {"freeze_bn_stats": True}}
        net.train()
        freeze_training_bn(net, settings)
        out, losses, _ = forward_loss(net, net.build_criterion(), images, [scene(), scene()], torch.device("cpu"))
        losses["loss"].backward()
        for name, module in (("backbone", net.backbone), ("mvff", net.mvff), ("giq", net.giq), ("mvoq", net.mvoq)):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0., name)
        net.eval()
        with torch.no_grad():
            single = net([x[:1] for x in images])
            repeated = net(images)
        for key in ("boxes", "class_logits", "confidences"):
            torch.testing.assert_close(out[key], repeated[key], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(single[key], repeated[key][:1], atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
