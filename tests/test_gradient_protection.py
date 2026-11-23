import json
import math
import unittest
import torch
from training.gradients import clip_gradients, GradientFailure


class GradientProtectionTests(unittest.TestCase):
    def parameter(self, gradient):
        p = torch.nn.Parameter(torch.zeros(len(gradient)))
        p.grad = torch.tensor(gradient, dtype=p.dtype)
        return p

    def test_finite_clipping(self):
        p = self.parameter([3., 4.])
        norm, overflow = clip_gradients([('weight', p)], 1.)
        self.assertAlmostEqual(norm, 5.)
        self.assertFalse(overflow)
        self.assertLessEqual(float(p.grad.norm()), 1.)

    def test_overflow_of_finite_fp32_norm_is_clipped_safely(self):
        p = self.parameter([1e25, -1e25])
        norm, overflow = clip_gradients([('weight', p)], 1.)
        self.assertTrue(overflow)
        self.assertTrue(math.isfinite(norm))
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertAlmostEqual(float(p.grad.norm()), 1., places=5)

    def test_real_nonfinite_gradients_are_never_suppressed(self):
        for value in (float('nan'), float('inf')):
            p = self.parameter([value, 1.])
            with self.assertRaises(GradientFailure) as error:
                clip_gradients([('bad_weight', p)], 1.)
            report = error.exception.diagnostics
            self.assertEqual(report['classification'], 'nonfinite_gradient_entries')
            self.assertEqual(report['nonfinite_parameters'], ['bad_weight'])
            json.dumps(report, allow_nan=False)

    def test_overflow_requires_finite_clipping_threshold(self):
        p = self.parameter([1e25, -1e25])
        with self.assertRaises(GradientFailure) as error:
            clip_gradients([('weight', p)], float('inf'))
        self.assertTrue(error.exception.diagnostics['all_gradients_finite'])
        self.assertEqual(error.exception.diagnostics['classification'],
                         'finite_gradients_fp32_norm_overflow')

    def test_legacy_no_clipping_preserves_finite_gradient(self):
        p = self.parameter([3., 4.])
        before = p.grad.clone()
        clip_gradients([('weight', p)], float('inf'))
        torch.testing.assert_close(before, p.grad, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
