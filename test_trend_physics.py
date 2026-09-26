"""Focused checks for cut pairing, gradients, source-only delta, and raw audit."""

import unittest

import numpy as np
import torch

from trend_physics import adjacent_positions, physical_audit, pava_nondecreasing, source_delta, trend_loss


class TrendPhysicsChecks(unittest.TestCase):
    def test_pairs_ignore_shuffle_cross_tool_gap_and_duplicate(self):
        tools = ["a", "a", "b", "a", "a", "b", "a"]
        cuts = [3, 1, 2, 2, 5, 3, 2]
        left, right = adjacent_positions(tools, cuts)
        # a:2 is duplicated, so neither a:1->2 nor a:2->3 is admissible.
        self.assertEqual((left, right), ([2], [5]))

    def test_loss_and_zero_gradient(self):
        pred = torch.tensor([3., 5., 2.], requires_grad=True)
        loss, pairs = trend_loss(pred, ["a"] * 3, [2, 1, 4], .5)
        self.assertEqual(pairs, 1)
        self.assertAlmostEqual(loss.item(), 1.5**2)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.tensor([-3., 3., 0.])))
        zero, count = trend_loss(pred, ["a", "b", "a"], [1, 2, 4], .5)
        self.assertEqual(count, 0)
        self.assertEqual(zero.item(), 0.)

    def test_delta_uses_only_passed_source_labels(self):
        self.assertAlmostEqual(source_delta([1., 2., 4.], [1, 2, 3])[0], .375)

    def test_zero_delta_gradient_and_pava_least_squares(self):
        pred = torch.tensor([4., 2., 3.], requires_grad=True)
        loss, pairs = trend_loss(pred, ["a"] * 3, [1, 2, 3], 0.0)
        self.assertEqual(pairs, 2)
        self.assertAlmostEqual(loss.item(), 2.0)
        self.assertGreater(torch.autograd.grad(loss, pred)[0].norm().item(), 0)
        np.testing.assert_allclose(pava_nondecreasing([4., 2., 3.]), [3., 3., 3.])
        from sklearn.isotonic import IsotonicRegression
        random_values = np.random.default_rng(42).normal(size=100)
        reference = IsotonicRegression(increasing=True).fit_transform(np.arange(100), random_values)
        np.testing.assert_allclose(pava_nondecreasing(random_values), reference, atol=1e-12)

    def test_physical_audit_preserves_predictions_and_reports_gaps(self):
        pred = np.asarray([-1., 2., np.nan, 8., 9.])
        original = pred.copy()
        frame, report = physical_audit([3, 1, 2, 3, 5], pred, .2, 5, expected=range(1, 6))
        self.assertEqual(report["missing_cuts"], [4])
        self.assertEqual(report["duplicate_cuts"], [3])
        self.assertEqual(report["nonfinite_count"], 1)
        self.assertEqual(report["negative_count"], 1)
        self.assertEqual(report["above_source_max_oor_count"], 2)
        self.assertEqual(report["adjacent_pair_count"], 0)
        self.assertTrue(np.array_equal(pred, original, equal_nan=True))
        self.assertEqual(frame.cut_index.tolist(), [1, 2, 3, 3, 5])


if __name__ == "__main__":
    unittest.main()
