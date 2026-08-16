from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.bg_soft_diagnostics import (
    FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION,
    HeldoutRequestResidualActivationOverlay,
    NOT_DEFINED,
    TARGET_HOLD_ACTIVE,
    signed_slope_comparison_receipt,
    single_write_audit_summary,
    target_hold_clock_receipt,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash


class _HeldoutLayer(torch.nn.Module):
    def __init__(self, *, sequence_first: bool) -> None:
        super().__init__()
        self.sequence_first = sequence_first

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        if self.sequence_first:
            return activation.transpose(0, 1).contiguous()
        return activation


class _HeldoutModel(torch.nn.Module):
    def __init__(self, *, sequence_first: bool) -> None:
        super().__init__()
        self.target = _HeldoutLayer(sequence_first=sequence_first)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.target(activation)


def _residual() -> torch.Tensor:
    return torch.tensor(
        [[float(index + 1) for index in range(10)], [float(101 + index) for index in range(10)]],
        dtype=torch.float32,
    )


def _source(
    forward_index: int, *, rows: int = 4, sequence_length: int = 5
) -> torch.Tensor:
    return (
        torch.arange(rows * sequence_length * 2, dtype=torch.float32)
        .reshape(rows, sequence_length, 2)
        .add_(float(1_000 * forward_index))
    )


class BgSoftDiagnosticsTests(unittest.TestCase):
    def test_signed_slope_receipt_has_stable_sign_rank_and_typed_classification(self) -> None:
        receipt = signed_slope_comparison_receipt(
            torch.tensor((-0.5, 0.0, -0.25, -1.0, -0.25), dtype=torch.float32),
            (0.2, 0.9, 0.1, -0.4, 0.5),
            0.0,
            1.75,
        )
        self.assertEqual(receipt["overlay_positive_mask"], [False] * 5)
        self.assertEqual(receipt["overlay_positive_count"], 0)
        self.assertEqual(receipt["overlay_rank_order"], [5, 6, 8, 4, 7])
        self.assertEqual(receipt["no_hook_positive_mask"], [True, True, True, False, True])
        self.assertEqual(receipt["no_hook_positive_count"], 4)
        self.assertEqual(receipt["no_hook_rank_order"], [5, 8, 4, 6, 7])
        self.assertEqual(
            receipt["classification"],
            FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION,
        )
        self.assertTrue(receipt["false_field_collapse_overlay_objective_saturation"])
        self.assertEqual(receipt["classification_decision_influence_count"], 0)
        self.assertEqual(len(receipt["overlay_signed_slopes_sha256"]), 64)
        self.assertEqual(len(receipt["overlay"]["signed_slopes_tensor_sha256"]), 64)
        self.assertEqual(
            receipt["identity_sha256"],
            canonical_hash(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "identity_sha256"
                }
            ),
        )
        positive_overlay = signed_slope_comparison_receipt(
            (0.5, -0.1, 0.0, -0.2, -0.3),
            (0.1, 0.2, 0.3, 0.4, 0.5),
            0.0,
            0.25,
        )
        self.assertEqual(positive_overlay["overlay_positive_count"], 1)
        self.assertEqual(
            positive_overlay["classification"],
            FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION,
        )

    def test_signed_slope_receipt_fails_closed_for_malformed_or_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "shape"):
            signed_slope_comparison_receipt((1.0, 2.0), (1.0,) * 5, 1.0, 1.0)
        with self.assertRaisesRegex(ODEBFContractError, "finite"):
            signed_slope_comparison_receipt(
                (1.0,) * 5,
                (1.0, 2.0, float("nan"), 4.0, 5.0),
                1.0,
                1.0,
            )
        with self.assertRaisesRegex(ODEBFContractError, "nonnegative"):
            signed_slope_comparison_receipt((1.0,) * 5, (1.0,) * 5, -1.0, 1.0)
        with self.assertRaisesRegex(ODEBFContractError, "layer order"):
            signed_slope_comparison_receipt(
                (1.0,) * 5,
                (1.0,) * 5,
                1.0,
                1.0,
                layer_order=(4, 5, 6, 7, 9),
            )

    def test_single_write_audit_reports_additivity_cancellation_and_grams(self) -> None:
        first = torch.tensor(((1.0, 0.0), (0.0, 0.0)))
        second = torch.tensor(((-1.0, 0.0), (0.0, 0.0)))
        third = torch.tensor(((0.0, 1.0), (0.0, 0.0)))
        fourth = torch.tensor(((0.0, 0.0), (0.0, 1.0)))
        fifth = torch.zeros((2, 2), dtype=torch.float32)
        all_layer = first + second + third + fourth + fifth
        receipt = single_write_audit_summary(
            all_layer,
            (first, second, third, fourth, fifth),
            all_layer,
        )
        self.assertEqual(receipt["layer_order"], [4, 5, 6, 7, 8])
        self.assertEqual(receipt["interaction_error"], 0.0)
        self.assertEqual(receipt["interaction_error_raw_norm"], 0.0)
        self.assertEqual(receipt["interaction_error_epsilon"], 1.0e-12)
        self.assertEqual(receipt["raw_gram"][0][1], -1.0)
        self.assertEqual(receipt["raw_gram"][1][0], -1.0)
        self.assertEqual(receipt["normalized_cosine_gram"][0][1], -1.0)
        self.assertEqual(receipt["normalized_cosine_gram"][4][4], NOT_DEFINED)
        self.assertEqual(receipt["per_layer"][4]["cosine_to_intended"], NOT_DEFINED)
        self.assertEqual(receipt["per_layer"][4]["norm_gain_to_intended"], 0.0)
        zero = torch.zeros((1, 1), dtype=torch.float32)
        nonadditive = single_write_audit_summary(
            torch.ones((1, 1), dtype=torch.float32),
            (zero, zero, zero, zero, zero),
            torch.full((1, 1), 2.0, dtype=torch.float32),
        )
        self.assertEqual(nonadditive["interaction_error_raw_norm"], 2.0)
        self.assertEqual(
            nonadditive["interaction_error"],
            2.0 / (2.0 + nonadditive["interaction_error_epsilon"]),
        )
        self.assertEqual(
            nonadditive["interaction_error_normalized"],
            nonadditive["interaction_error"],
        )

    def test_single_write_audit_types_zero_target_metrics_and_rejects_bad_matrices(self) -> None:
        zero = torch.zeros((2, 2), dtype=torch.float32)
        delta = torch.ones((2, 2), dtype=torch.float32)
        receipt = single_write_audit_summary(
            zero,
            (delta, delta, delta, delta, delta),
            5.0 * delta,
        )
        self.assertEqual(receipt["target_metric_status"], NOT_DEFINED)
        self.assertEqual(
            receipt["per_layer_norm_gain_to_intended"], [NOT_DEFINED] * 5
        )
        self.assertEqual(receipt["interaction_error"], 0.0)
        with self.assertRaisesRegex(ODEBFContractError, "finite"):
            single_write_audit_summary(
                torch.tensor(((float("nan"),),), dtype=torch.float32),
                (torch.ones((1, 1)),) * 5,
                torch.ones((1, 1)),
            )
        with self.assertRaisesRegex(ODEBFContractError, "shape"):
            single_write_audit_summary(
                torch.ones((1, 1)),
                (torch.ones((1, 1)),) * 4 + (torch.ones((2, 1)),),
                torch.ones((1, 1)),
            )

    def test_target_hold_clock_requires_exact_k8_invariance(self) -> None:
        bootstrap = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        states = tuple(bootstrap.clone() for _ in range(8))
        receipt = target_hold_clock_receipt(bootstrap, states)
        self.assertEqual(receipt["status"], TARGET_HOLD_ACTIVE)
        self.assertEqual(receipt["weight_step_count"], 8)
        self.assertEqual(receipt["weight_h_fraction"], {"numerator": 1, "denominator": 8})
        self.assertEqual(receipt["tau_W"], 1.0)
        self.assertEqual(receipt["target_advance_count"], 0)
        self.assertEqual(receipt["scientific_retry_count"], 0)
        self.assertTrue(receipt["target_hash_invariant"])
        self.assertEqual(
            receipt["target_sha256_by_weight_step"],
            [receipt["bootstrap_target_sha256"]] * 8,
        )

        changed = list(states)
        changed[3] = changed[3] + 1.0
        with self.assertRaisesRegex(ODEBFContractError, "invariant"):
            target_hold_clock_receipt(bootstrap, changed)
        with self.assertRaisesRegex(ODEBFContractError, "state count"):
            target_hold_clock_receipt(bootstrap, states[:-1])
        with self.assertRaisesRegex(ODEBFContractError, "weight h"):
            target_hold_clock_receipt(bootstrap, states, h=0.25)
        with self.assertRaisesRegex(ODEBFContractError, "target advance"):
            target_hold_clock_receipt(bootstrap, states, target_advance_count=1)
        with self.assertRaisesRegex(ODEBFContractError, "scientific retry"):
            target_hold_clock_receipt(bootstrap, states, scientific_retry_count=1)

    def test_heldout_overlay_batch_first_uses_request_columns_and_leaves_locality_unhooked(self) -> None:
        model = _HeldoutModel(sequence_first=False)
        positions = tuple((1, -1, 0, 2) for _ in range(10))
        overlay = HeldoutRequestResidualActivationOverlay(
            model,
            "target",
            _residual(),
            positions,
            (2,) * 10,
        )
        with overlay:
            for request_ordinal in range(10):
                source = _source(request_ordinal)
                result = model(source)
                for row, position in enumerate((1, 4)):
                    torch.testing.assert_close(
                        result[row, position, :],
                        source[row, position, :] + _residual()[:, request_ordinal],
                    )
                torch.testing.assert_close(result[2:, :, :], source[2:, :, :])
        self.assertFalse(overlay.hook_active)
        overlay.assert_complete()
        receipt = overlay.raw_free_payload()
        self.assertEqual(receipt["hook_call_count"], 10)
        self.assertEqual(receipt["layouts"], ["BATCH_FIRST"] * 10)
        self.assertEqual(receipt["patched_row_count"], 20)
        self.assertEqual(receipt["locality_no_hook_row_count"], 20)
        self.assertEqual(receipt["absolute_replacement_count"], 0)
        self.assertEqual(receipt["maximum_authoritative_assignment_error"], 0.0)

    def test_heldout_overlay_default_batch_policy_handles_square_layout(self) -> None:
        model = _HeldoutModel(sequence_first=False)
        positions = tuple((0, 1, 2, 3) for _ in range(10))
        residual = _residual()
        overlay = HeldoutRequestResidualActivationOverlay(
            model,
            "target",
            residual,
            positions,
            (2,) * 10,
        )
        with overlay:
            for request_ordinal in range(10):
                source = _source(request_ordinal, sequence_length=4)
                result = model(source)
                torch.testing.assert_close(
                    result[0, 0, :], source[0, 0, :] + residual[:, request_ordinal]
                )
        receipt = overlay.raw_free_payload()
        self.assertEqual(receipt["activation_layout_policy"], "BATCH_FIRST")
        self.assertEqual(receipt["layouts"], ["BATCH_FIRST"] * 10)

    def test_heldout_overlay_supports_sequence_first_layout(self) -> None:
        model = _HeldoutModel(sequence_first=True)
        positions = tuple((0, 3, 1, -1) for _ in range(10))
        residual = _residual()
        overlay = HeldoutRequestResidualActivationOverlay(
            model,
            "target",
            residual,
            positions,
            (2,) * 10,
            activation_layout_policy="SEQUENCE_FIRST",
        )
        with overlay:
            for request_ordinal in range(10):
                source = _source(request_ordinal)
                result = model(source)
                expected = source.transpose(0, 1).contiguous()
                for row, position in enumerate((0, 3)):
                    torch.testing.assert_close(
                        result[position, row, :],
                        expected[position, row, :] + residual[:, request_ordinal],
                    )
                torch.testing.assert_close(result[:, 2:, :], expected[:, 2:, :])
        self.assertEqual(overlay.raw_free_payload()["layouts"], ["SEQUENCE_FIRST"] * 10)

    def test_heldout_overlay_removes_its_hook_when_a_forward_fails(self) -> None:
        model = _HeldoutModel(sequence_first=False)
        positions = [(99, 1, 0, 2)] + [(1, -1, 0, 2)] * 9
        overlay = HeldoutRequestResidualActivationOverlay(
            model,
            "target",
            _residual(),
            positions,
            (2,) * 10,
        )
        source = _source(0)
        with self.assertRaisesRegex(ODEBFContractError, "out of range"):
            with overlay:
                model(source)
        self.assertFalse(overlay.hook_active)
        torch.testing.assert_close(model(source), source)


if __name__ == "__main__":
    unittest.main()
