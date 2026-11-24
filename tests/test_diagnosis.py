"""Check diagnostics distinguish localization coverage, identity, and filtering."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from model import ModelConfig
from training.engine import build_model
from tools.diagnose import main, raw_diagnostics, sample_indices


class DiagnosisTests(unittest.TestCase):
    def test_sampling_covers_manifest_and_is_unique(self):
        indices = sample_indices(3066, 240)
        self.assertEqual((indices[0], indices[-1]), (0, 3065))
        self.assertEqual(len(set(indices)), 240)
        self.assertEqual(sample_indices(95, 240), list(range(95)))

    def test_separates_localization_identity_and_filtering(self):
        a, b = [.2, .2, .1, .1], [.8, .8, .1, .1]
        target = dict(boxes=torch.tensor([[a], [b]]), visible=torch.ones(2, 1, dtype=torch.bool))
        pred = dict(boxes=torch.tensor([[a, b], [a, b]]), present=torch.ones(2, 2, dtype=torch.bool))
        counts, values, _ = raw_diagnostics(pred, target, .5)
        self.assertEqual(counts['raw_box_hits'], 2)
        self.assertEqual(counts['shared_gt_with_raw_hit_in_every_view'], 1)
        self.assertEqual(counts['raw_same_query_hits'], 0)
        self.assertEqual(values, [1., 1.])
        pred['boxes'][1] = torch.tensor([b, a])
        pred['present'][1, 0] = False
        counts, _, _ = raw_diagnostics(pred, target, .5)
        self.assertEqual(counts['raw_same_query_hits'], 1)
        self.assertEqual(counts['kept_same_query_hits'], 0)

    def test_empty_gt(self):
        target = dict(boxes=torch.empty(2, 0, 4), visible=torch.empty(2, 0, dtype=torch.bool))
        pred = dict(boxes=torch.ones(2, 2, 4) * .2, present=torch.ones(2, 2, dtype=torch.bool))
        counts, values, _ = raw_diagnostics(pred, target, .5)
        self.assertEqual(counts['raw_box_hits'], 0)
        self.assertEqual(values, [])

    def test_end_to_end_checkpoint_report_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (64, 64), 'gray').save(root / 'image.png')
            rows = [dict(images=['image.png'] * 2, boxes=[[[.4, .4, .2, .2]]] * 2,
                         visible=[[True], [True]], labels=[0], ids=[9], sizes=[[64, 64]] * 2,
                         sequence=str(i + 1), frame=i) for i in range(2)]
            manifest = root / 'samples.json'
            manifest.write_text(json.dumps(rows))
            config = dict(model=ModelConfig(pretrained=False, num_views=2, num_classes=1,
                          num_queries=3, hidden_dim=32, num_heads=4, mvff_layers=1,
                          giq_layers=1, mvoq_layers=1, dropout=0.).to_dict(),
                          data=dict(train=str(manifest), val=str(manifest), height=64,
                                    width=64, augment=True),
                          training={}, evaluation=dict(confidence_threshold=.5,
                                    iou_threshold=.5, ucr_threshold=.6))
            checkpoint = root / 'last.pt'
            net = build_model(config)
            torch.save(dict(model=net.state_dict(), config=config, epoch=15), checkpoint)
            output = root / 'diagnosis'
            argv = ['diagnose.py', '--checkpoints', str(checkpoint), '--limit', '2',
                    '--visualize', '1', '--output', str(output)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            summary = json.loads((output / 'comparison.json').read_text())
            self.assertEqual(len(summary), 2)
            report = json.loads(next(output.glob('*/report.json')).read_text())
            self.assertFalse(report['train']['augmentation'])
            self.assertEqual(report['train']['samples'], 2)
            self.assertEqual(set(report['train']['per_sequence']), {'1', '2'})
            self.assertEqual(len(list(output.glob('*/*.jpg'))), 2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
