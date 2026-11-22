import unittest
import torch
from tools.diagnose import geometry_counts


class GeometryDiagnosisTests(unittest.TestCase):
    def test_translation_and_scale_errors_are_distinguished(self):
        gt = torch.tensor([[.5, .5, .2, .2]])
        visible = torch.tensor([True])
        shifted = geometry_counts(torch.tensor([[.1, .1, .2, .2]]), gt, visible, .5)
        self.assertEqual(shifted["raw_box_hits"], 0)
        self.assertEqual(shifted["oracle_center_box_hits"], 1)
        self.assertEqual(shifted["oracle_size_box_hits"], 0)
        small = geometry_counts(torch.tensor([[.5, .5, .02, .02]]), gt, visible, .5)
        self.assertEqual(small["raw_box_hits"], 0)
        self.assertEqual(small["oracle_center_box_hits"], 0)
        self.assertEqual(small["oracle_size_box_hits"], 1)

    def test_invisible_targets_are_excluded(self):
        gt = torch.tensor([[.5, .5, .2, .2]])
        result = geometry_counts(gt, gt, torch.tensor([False]), .5)
        self.assertTrue(all(value == 0 for value in result.values()))

    def test_clipped_geometry_matches_existing_iou(self):
        from torchvision.ops import box_iou
        from training.metrics import xyxy
        torch.manual_seed(42)
        pred, gt = torch.rand(25, 4), torch.rand(20, 4)
        expected = int((box_iou(xyxy(pred), xyxy(gt)) >= .5).any(0).sum())
        result = geometry_counts(pred, gt, torch.ones(20, dtype=torch.bool), .5)
        self.assertEqual(result["raw_box_hits"], expected)


if __name__ == "__main__":
    unittest.main()
