import math
import unittest

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.direct_z_fidelity import (
    BatchPositionPatch,
    DirectZFidelityError,
    aggregate_sequence_spill,
    aggregate_shared_delta_fidelity,
    combine_unit_c_single_layer_factors,
    combine_unit_c_single_layer_proposals,
    rewrap_layer_output,
    solve_five_direction_ridge,
    solve_nonnegative_l2_ball_ridge,
    unwrap_layer_output,
)
from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot


def _assert_tensor_free(test: unittest.TestCase, value) -> None:
    test.assertNotIsInstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_tensor_free(test, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_tensor_free(test, child)


class _FiveLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(2, 2, bias=False) for _ in range(5)
        )


class LayerOutputPatchTests(unittest.TestCase):
    def test_tensor_add_patch_clones_and_supports_negative_positions(self) -> None:
        activation = torch.zeros(2, 3, 2)
        original = activation.clone()
        patch = BatchPositionPatch(
            layer_name="layer.8",
            positions=(0, -1),
            values=torch.tensor([1.5, -2.0]),
            mode="add",
        )

        updated = patch(activation, "layer.8")

        self.assertIsInstance(updated, torch.Tensor)
        self.assertIsNot(updated, activation)
        torch.testing.assert_close(activation, original)
        torch.testing.assert_close(updated[0, 0], torch.tensor([1.5, -2.0]))
        torch.testing.assert_close(updated[1, 2], torch.tensor([1.5, -2.0]))
        torch.testing.assert_close(updated[0, 1], torch.zeros(2))

    def test_tuple_replace_patch_preserves_trailing_payload(self) -> None:
        activation = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        cache = object()
        output = (activation, cache, "metadata")
        values = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        patch = BatchPositionPatch(
            layer_name="layer.8",
            positions=(1, 0),
            values=values,
            mode="replace",
        )

        updated = patch(output, "layer.8")

        self.assertIsInstance(updated, tuple)
        self.assertIs(updated[1], cache)
        self.assertEqual(updated[2], "metadata")
        torch.testing.assert_close(updated[0][0, 1], values[0])
        torch.testing.assert_close(updated[0][1, 0], values[1])
        torch.testing.assert_close(
            activation,
            torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        )

    def test_nonmatching_layer_returns_same_object(self) -> None:
        output = (torch.zeros(1, 2, 3), "cache")
        patch = BatchPositionPatch(
            layer_name="wanted",
            positions=(0,),
            values=torch.ones(3),
        )

        self.assertIs(patch(output, "other"), output)

    def test_sequence_first_layout_is_supported(self) -> None:
        activation = torch.zeros(3, 2, 2)
        patch = BatchPositionPatch(
            layer_name="layer",
            positions=(2, 1),
            values=torch.tensor([3.0, 4.0]),
            batch_dim=1,
            sequence_dim=0,
        )

        updated = patch(activation, "layer")

        torch.testing.assert_close(updated[2, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(updated[1, 1], torch.tensor([3.0, 4.0]))

    def test_unwrap_and_rewrap_fail_closed(self) -> None:
        activation = torch.zeros(1, 2, 3)
        self.assertIs(unwrap_layer_output((activation, "cache")), activation)
        replacement = torch.ones_like(activation)
        self.assertIs(rewrap_layer_output(activation, replacement), replacement)
        with self.assertRaises(DirectZFidelityError):
            unwrap_layer_output(("not-a-tensor",))
        with self.assertRaises(DirectZFidelityError):
            rewrap_layer_output(activation, torch.zeros(2, 3))


class FidelityAggregationTests(unittest.TestCase):
    def test_shared_delta_decomposes_gain_parallel_and_orthogonal_error(self) -> None:
        base = torch.zeros(3, 2)
        delta = torch.tensor([2.0, 0.0])
        edited = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ]
        )

        result = aggregate_shared_delta_fidelity(base, edited, delta)

        self.assertEqual(result.context_count, 3)
        self.assertEqual(result.generated_count, 2)
        self.assertAlmostEqual(result.canonical_target_error_l2, math.sqrt(2.0))
        self.assertAlmostEqual(result.canonical_gain, 0.5)
        self.assertAlmostEqual(result.canonical_cosine, 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(result.canonical_parallel_error_l2, 1.0)
        self.assertAlmostEqual(result.canonical_orthogonal_error_l2, 1.0)
        self.assertAlmostEqual(result.generated_mean_target_error_l2, 1.0)
        self.assertAlmostEqual(result.generated_worst_target_error_l2, 2.0)
        self.assertAlmostEqual(result.generated_mean_gain, 0.5)
        self.assertAlmostEqual(result.generated_worst_absolute_gain_error, 1.0)
        self.assertAlmostEqual(result.generated_mean_cosine, 0.5)
        self.assertAlmostEqual(result.generated_worst_cosine, 0.0)
        self.assertAlmostEqual(result.generated_mean_parallel_error_l2, 1.0)
        self.assertAlmostEqual(result.generated_worst_parallel_error_l2, 2.0)
        self.assertAlmostEqual(result.generated_mean_orthogonal_error_l2, 0.0)
        _assert_tensor_free(self, result.to_dict())

    def test_explicit_absolute_canonical_target_is_used(self) -> None:
        base = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
        delta = torch.tensor([2.0, 0.0])
        edited = base + delta

        result = aggregate_shared_delta_fidelity(
            base,
            edited,
            delta,
            canonical_absolute_target=torch.tensor([4.0, 1.0]),
        )

        # The realized canonical update is exactly delta (gain 1), while its
        # explicit absolute target is one unit away.
        self.assertAlmostEqual(result.canonical_gain, 1.0)
        self.assertAlmostEqual(result.canonical_target_error_l2, 1.0)

    def test_sequence_aggregation_separates_subject_miss_and_off_token_spill(self) -> None:
        base = torch.zeros(3, 3, 2)
        delta = torch.tensor([1.0, 0.0])
        subject_positions = (1, 0, 2)
        edited = base.clone()
        for context, position in enumerate(subject_positions):
            edited[context, position] += delta
        edited[0, 1, 0] += 0.5
        edited[0, 0, 1] += 2.0
        edited[2, 0, 1] += 1.0

        result = aggregate_sequence_spill(
            base,
            edited,
            delta,
            subject_positions,
        )

        self.assertAlmostEqual(result.canonical_subject_miss_l2, 0.5)
        self.assertAlmostEqual(result.generated_mean_subject_miss_l2, 0.0)
        self.assertAlmostEqual(result.canonical_off_token_spill_rms, math.sqrt(2.0))
        self.assertAlmostEqual(
            result.generated_mean_off_token_spill_rms,
            1.0 / (2.0 * math.sqrt(2.0)),
        )
        self.assertAlmostEqual(
            result.generated_worst_off_token_spill_rms,
            1.0 / math.sqrt(2.0),
        )
        self.assertAlmostEqual(result.global_subject_miss_rms, 0.5 / math.sqrt(3.0))
        self.assertAlmostEqual(result.global_off_token_spill_rms, math.sqrt(5.0 / 6.0))
        self.assertAlmostEqual(
            result.full_sequence_intervention_error_rms,
            math.sqrt(7.0 / 12.0),
        )
        _assert_tensor_free(self, result.to_dict())

    def test_sequence_mask_excludes_padding(self) -> None:
        base = torch.zeros(2, 3, 1)
        edited = base.clone()
        delta = torch.tensor([1.0])
        edited[:, 0, 0] = 1.0
        edited[:, 2, 0] = 100.0
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

        result = aggregate_sequence_spill(
            base,
            edited,
            delta,
            (0, 0),
            attention_mask=mask,
        )

        self.assertEqual(result.valid_token_count, 4)
        self.assertEqual(result.off_token_count, 2)
        self.assertAlmostEqual(result.global_off_token_spill_rms, 0.0)
        self.assertAlmostEqual(result.full_sequence_intervention_error_rms, 0.0)

    def test_zero_delta_and_invalid_subject_mask_fail_closed(self) -> None:
        with self.assertRaises(DirectZFidelityError):
            aggregate_shared_delta_fidelity(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                torch.zeros(2),
            )
        with self.assertRaises(DirectZFidelityError):
            aggregate_sequence_spill(
                torch.zeros(2, 2, 1),
                torch.zeros(2, 2, 1),
                torch.ones(1),
                (0, 0),
                attention_mask=torch.tensor([[0, 1], [1, 1]]),
            )


class ProjectedRidgeTests(unittest.TestCase):
    def test_nonnegative_solution_clips_infeasible_identity_coordinates(self) -> None:
        design = torch.eye(5, dtype=torch.float64)
        target = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.0], dtype=torch.float64)

        result = solve_five_direction_ridge(
            design,
            target,
            radius=10.0,
            ridge=0.0,
        )

        self.assertTrue(result.converged)
        torch.testing.assert_close(
            torch.tensor(result.coefficients),
            torch.tensor([1.0, 0.0, 3.0, 0.5, 0.0]),
            rtol=1e-8,
            atol=1e-8,
        )
        self.assertAlmostEqual(result.residual_l2, math.sqrt(5.0))
        self.assertEqual(result.active_count, 3)
        _assert_tensor_free(self, result.to_dict())

    def test_l2_ball_is_active(self) -> None:
        result = solve_five_direction_ridge(
            torch.eye(5),
            torch.ones(5),
            radius=1.0,
            ridge=0.0,
        )

        expected = torch.full((5,), 1.0 / math.sqrt(5.0))
        torch.testing.assert_close(
            torch.tensor(result.coefficients), expected, rtol=1e-7, atol=1e-7
        )
        self.assertAlmostEqual(result.coefficient_l2, 1.0, places=8)
        self.assertTrue(result.converged)

    def test_ridge_shrinks_unconstrained_identity_solution(self) -> None:
        result = solve_five_direction_ridge(
            torch.eye(5),
            torch.tensor([2.0, 4.0, 0.0, 6.0, 8.0]),
            radius=20.0,
            ridge=1.0,
        )

        torch.testing.assert_close(
            torch.tensor(result.coefficients),
            torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0]),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_generic_solver_supports_other_dimensions(self) -> None:
        result = solve_nonnegative_l2_ball_ridge(
            torch.eye(3),
            torch.tensor([1.0, 2.0, 3.0]),
            radius=10.0,
            ridge=0.0,
        )

        self.assertEqual(result.coefficients, (1.0, 2.0, 3.0))
        with self.assertRaises(DirectZFidelityError):
            solve_five_direction_ridge(
                torch.eye(4),
                torch.ones(4),
                radius=1.0,
            )


class ProposalCombinationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _FiveLayerModel()
        self.model.eval()
        request = EditRequest.from_mapping(
            {
                "case_id": "case-1",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        contexts = ContextManifest.freeze([["{}"]], source="unit-test")
        names = tuple(f"layers.{index}.weight" for index in range(5))
        snapshot = capture_snapshot(
            self.model,
            model_id="toy-five-layer",
            requests=(request,),
            context_id=contexts.manifest_id,
            hparams={"layers": list(range(5))},
            weight_names=names,
        )
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.tensor([[float(index + 1)], [float(index + 2)]]),
                right=torch.tensor([[1.0], [2.0]]),
                expected_weight_sha256=snapshot.parameter(name).sha256,
            )
            for index, name in enumerate(names)
        )
        self.reference = MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="unit-test/unit-c-reference",
            residual_denominator=5,
        )
        self.singles = tuple(
            MemitFactorProposal(
                snapshot=snapshot,
                factors=(factor,),
                semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
                solver_name=f"unit-test/{factor.weight_name}",
                residual_denominator=5,
            )
            for factor in factors
        )

    def test_combination_aligns_coefficients_by_weight_and_reference_order(self) -> None:
        reversed_singles = tuple(reversed(self.singles))
        coefficients = (1.0, 2.0, 3.0, 4.0, 5.0)

        combined = combine_unit_c_single_layer_proposals(
            self.reference,
            reversed_singles,
            coefficients,
        )

        self.assertEqual(
            tuple(factor.weight_name for factor in combined.factors),
            tuple(factor.weight_name for factor in self.reference.factors),
        )
        for index, (actual, original) in enumerate(
            zip(combined.factors, self.reference.factors, strict=True)
        ):
            expected_coefficient = float(5 - index)
            torch.testing.assert_close(
                actual.left,
                original.left * expected_coefficient,
            )
            torch.testing.assert_close(actual.right, original.right)
        self.assertEqual(combined.snapshot_id, self.reference.snapshot_id)
        self.assertTrue(combined.is_synchronous)
        self.assertEqual(combined.residual_denominator, 5)

    def test_zero_factor_is_retained_and_negative_coefficient_is_rejected(self) -> None:
        scaled = combine_unit_c_single_layer_factors(
            self.reference.factors,
            (0.0, 1.0, 1.0, 1.0, 1.0),
        )

        torch.testing.assert_close(scaled[0].left, torch.zeros_like(scaled[0].left))
        self.assertEqual(len(scaled), 5)
        with self.assertRaises(DirectZFidelityError):
            combine_unit_c_single_layer_factors(
                self.reference.factors,
                (-1.0, 1.0, 1.0, 1.0, 1.0),
            )

    def test_missing_single_layer_proposal_fails_closed(self) -> None:
        with self.assertRaises(DirectZFidelityError):
            combine_unit_c_single_layer_proposals(
                self.reference,
                self.singles[:-1],
                (1.0, 1.0, 1.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
