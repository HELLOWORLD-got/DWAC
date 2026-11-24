"""Check real-GT assignments, error units and an actual fixed-group update."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image
import torch

from model import ModelConfig
from training.engine import build_model
from tools.fit_single_group import (real_gt_to_query, switch_fraction,
                                     matched_geometry, run)


class SingleGroupTests(unittest.TestCase):
    def test_real_mapping_ignores_padding_switches(self):
        first = real_gt_to_query(torch.tensor([[0, 2], [1, 0], [2, 1], [3, 3]]), 2)
        second = real_gt_to_query(torch.tensor([[0, 3], [1, 0], [2, 1], [3, 2]]), 2)
        self.assertEqual(first, [1, 2])
        self.assertEqual(switch_fraction(first, second), 0.)
        self.assertEqual(switch_fraction(first, [2, 1]), 1.)

    def test_geometry_uses_joint_mapping_and_visible_boxes_only(self):
        target = dict(boxes=torch.tensor([[[.5, .5, .2, .2]], [[0., 0., 0., 0.]]]),
                      visible=torch.tensor([[True], [False]]))
        pred = dict(boxes=torch.tensor([[[.1, .1, .1, .1], [.6, .5, .3, .2]],
                                       [[.1, .1, .1, .1], [.9, .9, .9, .9]]]))
        report = matched_geometry(pred, target, [1], 100, 200)
        self.assertAlmostEqual(report['mean_abs_center_error_xy_pixels'][0], 10., places=4)
        self.assertAlmostEqual(report['mean_abs_size_error_wh_pixels'][0], 10., places=4)
        self.assertEqual(report['visible_gt_boxes'], 1)

    def test_real_update_preserves_model_and_records_regression_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (64, 64), 'gray').save(root / 'image.png')
            manifest = root / 'data.json'
            manifest.write_text(json.dumps([dict(images=['image.png'] * 2,
                boxes=[[[.4, .4, .2, .2]]] * 2, visible=[[True], [True]],
                labels=[0], ids=[7], sequence='debug', frame=0, sizes=[[64,64]] * 2)]))
            config = dict(model=ModelConfig(pretrained=False, num_views=2, num_classes=1,
                num_queries=3, hidden_dim=32, num_heads=4, mvff_layers=1,
                giq_layers=1, mvoq_layers=1, dropout=.1).to_dict(),
                data=dict(train=str(manifest), val=str(manifest), height=64, width=64, augment=True),
                training=dict(seed=42, lr=1e-4, weight_decay=1e-5, optimizer='adamw',
                              freeze_bn_stats=True, clip_grad=1.),
                evaluation=dict(confidence_threshold=.5, iou_threshold=.5, ucr_threshold=.6))
            checkpoint = root / 'last.pt'
            torch.save(dict(model=build_model(config).state_dict(), config=config, epoch=15), checkpoint)
            args = SimpleNamespace(checkpoint=str(checkpoint), config=None, split='train',
                index=0, steps=2, log_every=1, lr=None, device='cpu', output=str(root / 'fit'))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result['final']['step'], 2)
            self.assertFalse(result['experiment']['config']['data']['augment'])
            self.assertEqual(result['experiment']['config']['model'], config['model'])
            probe = result['final']['regression_gradients']
            for group in ('backbone', 'mvff', 'giq', 'mvoq.reg_heads'):
                self.assertTrue(probe[group]['finite'])
                self.assertGreater(probe[group]['nonzero'], 0)
            self.assertEqual(probe['mvoq.conf_heads']['connected'], 0)
            saved = torch.load(root / 'fit' / 'last.pt', weights_only=True)
            self.assertEqual(saved['debug_step'], 2)
            self.assertEqual(len((root / 'fit' / 'assignments.jsonl').read_text().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, 'empty output'):
                run(args)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
