import math
import unittest
from unittest.mock import patch
import torch
from model.mvoq import MultiViewObjectQuery, spatial_grid
from model import pad_targets
from model.loss import SetCriterion
from model.matcher import HungarianMatcher
from training.background_loss import BackgroundWeightedLoss
from training.center_loss import CenterSupervisionLoss


class CenterReferenceTests(unittest.TestCase):
    def sample(self, empty=False):
        n = 0 if empty else 1
        target = pad_targets([torch.zeros(n, dtype=torch.long)],
                             [torch.tensor([[[.5, .25, .2, .2]], [[0., 0., 0., 0.]]])[:, :n]],
                             [torch.tensor([[True], [False]])[:, :n]], 3, 1)
        grid = spatial_grid(2, 3, device='cpu', dtype=torch.float32)
        boxes = torch.tensor([[[[.1, .1, .1, .1], [.5, .25, .2, .2], [.9, .9, .1, .1]],
                               [[.1, .1, .1, .1], [.5, .25, .2, .2], [.9, .9, .1, .1]]]])
        output = dict(class_logits=torch.tensor([[[-8., 8.], [8., -8.], [-8., 8.]]]),
                      boxes=boxes, confidences=torch.full((1, 2, 3), .5),
                      center_attention=torch.full((1, 2, 3, 6), 1/6, requires_grad=True),
                      center_grid=grid, center_grid_scale=torch.tensor([3., 2.]))
        return output, target

    def test_grid_and_zero_offset_geometry(self):
        grid = spatial_grid(2, 3, device='cpu', dtype=torch.float32)
        torch.testing.assert_close(grid[0], torch.tensor([1/6, .25]))
        torch.testing.assert_close(grid[-1], torch.tensor([5/6, .75]))
        torch.manual_seed(42)
        model = MultiViewObjectQuery(dim=32, num_heads=4, num_views=2, num_queries=3,
                                    dropout=0., position_encoding='normalized_keys_values',
                                    instance_residual=True, center_mode='attention_reference')
        output = model(torch.randn(2, 3, 32), [torch.randn(2, 32, 2, 3) for _ in range(2)])
        torch.testing.assert_close(output['boxes'][..., :2], output['reference_points'])
        torch.testing.assert_close(output['center_attention'].sum(-1), torch.ones(2, 2, 3))
        self.assertTrue(bool(torch.isfinite(output['boxes']).all()))

    def test_supervision_uses_same_assignment_and_masks_absent_slots(self):
        for background in (1., .1):
            output, target = self.sample()
            base = SetCriterion(1, HungarianMatcher())
            if background != 1.:
                base = BackgroundWeightedLoss(base, background)
            loss = CenterSupervisionLoss(base, weight=.25)
            with patch.object(loss.author.matcher, 'forward', wraps=loss.author.matcher.forward) as calls:
                values = loss(output, target)
                self.assertEqual(calls.call_count, 1)
            self.assertAlmostEqual(float(values['loss_center'].detach()), .25*math.log(6), places=6)
            values['loss_center'].backward()
            gradient = output['center_attention'].grad
            self.assertGreater(float(gradient[0, 0, 1].abs().sum()), 0.)
            self.assertEqual(float(gradient[0, 1].abs().sum()), 0.)
            self.assertEqual(float(gradient[0, 0, [0, 2]].abs().sum()), 0.)

    def test_empty_scene_has_finite_zero_center_loss(self):
        output, target = self.sample(empty=True)
        criterion = CenterSupervisionLoss(SetCriterion(1, HungarianMatcher()))
        loss = criterion(output, target)['loss_center']
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(output['center_attention'].grad.abs().sum()), 0.)

    def test_center_supervision_reaches_attention_query_projection(self):
        torch.manual_seed(42)
        model = MultiViewObjectQuery(dim=32, num_heads=4, num_views=2, num_queries=3,
                                    dropout=0., position_encoding='normalized_keys_values',
                                    instance_residual=True, center_mode='attention_reference')
        output = model(torch.randn(1, 3, 32), [torch.randn(1, 32, 2, 3) for _ in range(2)])
        output['class_logits'] = torch.tensor([[[-8., 8.], [8., -8.], [-8., 8.]]])
        _, target = self.sample()
        loss = CenterSupervisionLoss(SetCriterion(1, HungarianMatcher()))(output, target)['loss_center']
        loss.backward()
        q_gradient = model.layers[-1].attn.in_proj_weight.grad[:32]
        self.assertTrue(bool(torch.isfinite(q_gradient).all()))
        self.assertGreater(float(q_gradient.abs().sum()), 0.)

    def test_distributed_visible_count_scaling_matches_global_gradient(self):
        from training.engine import forward_loss
        output0, target = self.sample()
        output1, _ = self.sample()
        t0 = dict(labels=torch.tensor([0]), boxes=target['boxes'][0, :, :1],
                  visible=torch.tensor([[True], [False]]))
        t1 = dict(labels=torch.tensor([0]), boxes=t0['boxes'].clone(),
                  visible=torch.tensor([[True], [True]]))
        t1['boxes'][1] = t1['boxes'][0]
        combined = {k: torch.cat([output0[k], output1[k]], 0)
                    for k in ('class_logits', 'boxes', 'confidences', 'center_attention')}
        combined['center_attention'] = combined['center_attention'].detach().requires_grad_()
        combined.update(center_grid=output0['center_grid'], center_grid_scale=output0['center_grid_scale'])

        class Net:
            def __init__(self, output): self.output = output
            def __call__(self, images): return self.output

        def criterion(): return CenterSupervisionLoss(SetCriterion(1, HungarianMatcher()))
        _, global_loss, _ = forward_loss(Net(combined), criterion(), [], [t0, t1], torch.device('cpu'))
        global_loss['loss_center'].backward()
        local_values = []
        for rank, (output, target) in enumerate([(output0, t0), (output1, t1)]):
            with patch('training.engine.dist.all_reduce', side_effect=lambda tensor: tensor.fill_(3)), \
                 patch('training.engine.dist.get_world_size', return_value=2):
                _, loss, _ = forward_loss(Net(output), criterion(), [], [target], torch.device('cpu'), True)
            local_values.append(loss['loss_center'].detach())
            loss['loss_center'].backward()
            torch.testing.assert_close(output['center_attention'].grad / 2,
                                       combined['center_attention'].grad[rank:rank+1])
        torch.testing.assert_close(sum(local_values)/2, global_loss['loss_center'].detach())


if __name__ == '__main__':
    unittest.main()
