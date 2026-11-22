"""Spatial sensitivity and legacy behavior for optional MVOQ encoding."""
import unittest
import torch
from model.mvoq import MultiViewObjectQuery


class MVOQPositionTests(unittest.TestCase):
    def test_instance_residual_survives_constant_view_tokens(self):
        torch.manual_seed(42)
        for pe in ("none", "keys_values"):
            model = MultiViewObjectQuery(dim=32, num_heads=4, num_views=2,
                                        num_queries=3, num_layers=2, dropout=0.,
                                        position_encoding=pe, instance_residual=True).eval()
            instance = torch.randn(2, 3, 32, requires_grad=True)
            views = [torch.randn(2, 32, 1, 1).expand(-1, -1, 3, 5) for _ in range(2)]
            boxes = model(instance, views)["boxes"]
            shuffled = model(instance.flip(1), views)["boxes"]
            self.assertGreater(float((boxes-shuffled).detach().abs().max()), 1e-4)
            weights = torch.linspace(.1, 1., boxes.numel()).reshape_as(boxes)
            gradient, = torch.autograd.grad((boxes*weights).sum(), instance)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().max()), 1e-5)

    def test_coordinates_break_spatial_permutation_invariance(self):
        torch.manual_seed(42)
        old = MultiViewObjectQuery(dim=32, num_heads=4, num_views=2,
                                   num_queries=3, num_layers=1, dropout=0.).eval()
        new = MultiViewObjectQuery(dim=32, num_heads=4, num_views=2,
                                   num_queries=3, num_layers=1, dropout=0.,
                                   position_encoding="keys_values").eval()
        new.load_state_dict(old.state_dict(), strict=True)
        instance = torch.randn(2, 3, 32, requires_grad=True)
        views = [torch.randn(2, 32, 3, 5, requires_grad=True) for _ in range(2)]
        shuffled = [v.flatten(2).flip(-1).reshape_as(v) for v in views]
        torch.testing.assert_close(old(instance, views)["boxes"],
                                   old(instance, shuffled)["boxes"], atol=1e-6, rtol=1e-6)
        output = new(instance, views)
        changed = new(instance, shuffled)
        self.assertGreater(float((output["boxes"]-changed["boxes"]).detach().abs().max()), 1e-5)
        loss = output["boxes"].square().sum() + output["confidences"].sum()
        loss.backward()
        for tensor in [instance, *views]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiViewObjectQuery(position_encoding="invalid")


if __name__ == "__main__":
    unittest.main()
