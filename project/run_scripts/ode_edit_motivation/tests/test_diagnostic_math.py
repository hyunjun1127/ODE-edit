import unittest

import torch

from project.run_scripts.ode_edit_motivation.contracts import LowRankFactor
from project.run_scripts.ode_edit_motivation.diagnostic_math import (
    CallableMetricOperator,
    NativeMemitSolver,
    WoodburyMemitSolver,
    c_cosine_similarity,
    c_inner_product,
    c_norm,
    exact_top1_stop,
    finite_difference_calibration,
    joint_additivity,
    rank_turnover,
    reroutability,
    smooth_target_utility,
)


def factor(name, left, right):
    return LowRankFactor(
        weight_name=name,
        left=torch.as_tensor(left, dtype=torch.float64),
        right=torch.as_tensor(right, dtype=torch.float64),
        expected_weight_sha256="0" * 64,
    )


class LowRankMathTests(unittest.TestCase):
    def test_c_inner_and_norm_match_dense_reference(self):
        first = factor(
            "w",
            [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]],
            [[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0], [0.5, 0.25]],
        )
        second = factor(
            "w",
            [[0.0, 1.0], [2.0, 0.5], [-1.0, 1.5]],
            [[1.0, 1.0], [0.0, 2.0], [2.0, -1.0], [1.0, 0.0]],
        )
        raw = torch.tensor(
            [
                [2.0, 0.2, 0.0, 0.0],
                [0.2, 1.5, 0.1, 0.0],
                [0.0, 0.1, 1.0, 0.3],
                [0.0, 0.0, 0.3, 1.2],
            ],
            dtype=torch.float64,
        )
        metric = raw @ raw.T
        dense_first = first.left @ first.right.T
        dense_second = second.left @ second.right.T
        expected_inner = torch.trace(dense_first @ metric @ dense_second.T)
        expected_norm = torch.sqrt(torch.trace(dense_first @ metric @ dense_first.T))

        torch.testing.assert_close(c_inner_product(first, second, metric), expected_inner)
        torch.testing.assert_close(
            c_inner_product(
                first,
                second,
                CallableMetricOperator(lambda value: metric @ value),
            ),
            expected_inner,
        )
        torch.testing.assert_close(c_norm(first, metric), expected_norm)
        torch.testing.assert_close(c_cosine_similarity(first, first, metric), torch.tensor(1.0, dtype=torch.float64))

    def test_native_and_woodbury_solvers_agree(self):
        torch.manual_seed(4)
        basis = torch.randn(7, 7, dtype=torch.float64)
        covariance = basis @ basis.T + 0.5 * torch.eye(7, dtype=torch.float64)
        keys = torch.randn(7, 3, dtype=torch.float64)
        native = NativeMemitSolver().adjusted_keys(covariance, keys, 2.75)
        woodbury = WoodburyMemitSolver().adjusted_keys(covariance, keys, 2.75)
        torch.testing.assert_close(native, woodbury, rtol=1e-10, atol=1e-10)

    def test_exact_stop_is_detached_from_smooth_utility(self):
        logits = torch.tensor(
            [[0.0, 4.0, 1.0], [3.0, 2.5, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        decision = exact_top1_stop(logits, torch.tensor([1, 1]), margin_threshold=0.2)
        self.assertEqual(decision.satisfied, (True, False))
        self.assertEqual(decision.predicted_ids, (1, 0))
        utility = smooth_target_utility(logits, torch.tensor([1, 1]), temperature=0.5)
        self.assertTrue(utility.requires_grad)
        utility.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_metric_helpers_have_known_answers(self):
        calibration = finite_difference_calibration(
            lambda scale: 2.0 + 3.0 * scale + scale**3,
            predicted_slope=3.0,
            steps=[0.1, 0.01],
        )
        self.assertLess(calibration.samples[-1].absolute_error, 0.001)

        additivity = joint_additivity(
            baseline=10.0,
            individual_values=[12.0, 13.0],
            joint_value=14.0,
        )
        self.assertEqual(additivity.predicted_joint, 15.0)
        self.assertEqual(additivity.interaction, -1.0)

        turnover = rank_turnover(
            {"a": 3.0, "b": 2.0, "c": 1.0},
            {"a": 1.0, "b": 2.0, "c": 3.0},
            k=2,
        )
        self.assertEqual(turnover.turnover_fraction, 0.5)

        rerouted = reroutability(
            baseline=1.0,
            direct_value=5.0,
            rerouted_values={"layer4": 2.0, "layer5": 4.0},
        )
        self.assertEqual(rerouted.best_route, "layer5")
        self.assertEqual(rerouted.recovered_fraction, 0.75)


if __name__ == "__main__":
    unittest.main()
