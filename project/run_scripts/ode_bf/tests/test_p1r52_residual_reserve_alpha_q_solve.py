from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_alpha_q_solve as q_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_alpha_q_solve import (
    ALPHA_Q_SOLVE_EXPRESSION,
    ALPHA_Q_SOLVE_REFERENCE,
    NATIVE_DENSE_BYTE_EQUIVALENCE,
    solve_residual_independent_alpha_q_fp32,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    UpdateMatchOrientation,
    build_prepared_low_rank_update,
)


class ResidualReserveAlphaQOnlySolveTests(unittest.TestCase):
    @staticmethod
    def fixture(
        *,
        dimension: int = 4,
        batch: int = 2,
        covariance_scale: float = 0.1,
    ):
        generator = torch.Generator().manual_seed(104)
        raw = torch.randn(
            (dimension, dimension),
            dtype=torch.float32,
            generator=generator,
        )
        projector = torch.eye(dimension, dtype=torch.float32) + 0.03 * raw
        keys = torch.randn(
            (dimension, batch),
            dtype=torch.float32,
            generator=generator,
        )
        covariance = covariance_scale * torch.eye(
            dimension,
            dtype=torch.float32,
        )
        regularization = torch.tensor(0.25, dtype=torch.float32)
        return projector, keys, covariance, regularization

    @staticmethod
    def solve(inputs, **kwargs):
        projector, keys, covariance, regularization = inputs
        return solve_residual_independent_alpha_q_fp32(
            projector,
            keys,
            covariance,
            regularization,
            layer=kwargs.pop("layer", 4),
            residual_tolerance=kwargs.pop("residual_tolerance", 1.0e-6),
            **kwargs,
        )

    def test_direct_expression_exact_and_certificate(self) -> None:
        projector, keys, covariance, regularization = self.fixture()
        expected_system = projector @ (keys @ keys.T + covariance) + (
            regularization * torch.eye(4, dtype=torch.float32)
        )
        expected_rhs = projector @ keys
        expected_q = torch.linalg.solve(expected_system, expected_rhs)
        result = self.solve((projector, keys, covariance, regularization))
        self.assertTrue(torch.equal(result.q32, expected_q))
        self.assertEqual(result.receipt.system_sha256, tensor_sha256(expected_system))
        self.assertEqual(result.receipt.rhs_sha256, tensor_sha256(expected_rhs))
        self.assertEqual(result.receipt.q_sha256, tensor_sha256(expected_q))
        self.assertLessEqual(
            result.receipt.relative_solve_residual,
            result.receipt.residual_tolerance,
        )
        self.assertEqual(result.receipt.reference, ALPHA_Q_SOLVE_REFERENCE)
        self.assertEqual(
            result.receipt.native_dense_byte_equivalence,
            NATIVE_DENSE_BYTE_EQUIVALENCE,
        )
        self.assertEqual(result.receipt.expression_order, ALPHA_Q_SOLVE_EXPRESSION)
        self.assertEqual(result.receipt.logical_q_solve_count, 1)

    def test_zero_and_nonzero_committed_covariance_are_finite_and_distinct(self) -> None:
        zero_inputs = self.fixture(covariance_scale=0.0)
        nonzero_inputs = (
            zero_inputs[0],
            zero_inputs[1],
            0.4 * torch.eye(4, dtype=torch.float32),
            zero_inputs[3],
        )
        zero = self.solve(zero_inputs)
        nonzero = self.solve(nonzero_inputs)
        self.assertTrue(torch.isfinite(zero.q32).all())
        self.assertTrue(torch.isfinite(nonzero.q32).all())
        self.assertFalse(torch.equal(zero.q32, nonzero.q32))
        self.assertEqual(zero.receipt.observed_key_rank, 2)
        self.assertEqual(zero.receipt.key_column_count, 2)
        self.assertEqual(zero.receipt.minimum_required_key_rank, 2)

    def test_partially_dependent_three_column_keys_rank_two_pass(self) -> None:
        projector, keys, covariance, regularization = self.fixture()
        partially_dependent = torch.stack(
            (keys[:, 0], keys[:, 1], keys[:, 0]),
            dim=1,
        )
        result = self.solve(
            (projector, partially_dependent, covariance, regularization)
        )
        self.assertEqual(result.receipt.observed_key_rank, 2)
        self.assertEqual(result.receipt.key_column_count, 3)
        self.assertEqual(result.receipt.minimum_required_key_rank, 2)
        self.assertLessEqual(
            result.receipt.relative_solve_residual,
            result.receipt.residual_tolerance,
        )

    def test_residual_certificate_tolerance_fail_closes(self) -> None:
        inputs = self.fixture()
        ordinary = self.solve(inputs)
        self.assertGreater(ordinary.receipt.relative_solve_residual, 0.0)
        with self.assertRaises(ODEBFContractError):
            self.solve(inputs, residual_tolerance=1.0e-30)

    def test_q_feeds_single_construction_direct_and_transpose_without_cast(self) -> None:
        for dimension, left_rows, orientation in (
            (4, 3, UpdateMatchOrientation.DIRECT.value),
            (3, 4, UpdateMatchOrientation.TRANSPOSE_VIEW.value),
        ):
            with self.subTest(orientation=orientation):
                result = self.solve(self.fixture(dimension=dimension))
                q_pointer = int(result.q32.data_ptr())
                q_version = int(result.q32._version)
                left = torch.arange(
                    1,
                    1 + left_rows * 2,
                    dtype=torch.float32,
                ).reshape(left_rows, 2)
                construction = build_prepared_low_rank_update(
                    construction_id=f"q-feed-{orientation}",
                    layer=4,
                    weight_name="model.layers.4.mlp.down_proj.weight",
                    parameter_shape=(3, 4),
                    beta_applied_left32=left,
                    q_side32=result.q32,
                )
                expected_raw = left @ result.q32.T
                expected = (
                    expected_raw
                    if orientation == UpdateMatchOrientation.DIRECT.value
                    else expected_raw.T
                )
                self.assertEqual(construction.receipt.matched_orientation, orientation)
                self.assertTrue(torch.equal(construction.matched_update32, expected))
                self.assertEqual(int(result.q32.data_ptr()), q_pointer)
                self.assertEqual(int(result.q32._version), q_version)
                self.assertEqual(
                    construction.receipt.raw_right_sha256,
                    result.receipt.q_sha256,
                )

    def test_residual_is_not_an_api_or_source_dependency(self) -> None:
        signature = inspect.signature(solve_residual_independent_alpha_q_fp32)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "projector32",
                "joint_keys32",
                "committed_covariance32",
                "regularization32",
                "layer",
                "residual_tolerance",
                "condition_max_dimension",
            ),
        )
        source = inspect.getsource(solve_residual_independent_alpha_q_fp32)
        for forbidden in (
            "target_residual",
            "terminal_residual",
            "residual32",
            "target_error",
            "dense_update",
        ):
            self.assertNotIn(forbidden, source)

    def test_inputs_pointer_version_hash_immutable_and_receipt_raw_free(self) -> None:
        inputs = self.fixture()
        guards = tuple(
            (int(item.data_ptr()), int(item._version), tensor_sha256(item))
            for item in inputs
        )
        result = self.solve(inputs)
        observed = tuple(
            (int(item.data_ptr()), int(item._version), tensor_sha256(item))
            for item in inputs
        )
        self.assertEqual(observed, guards)
        self.assertTrue(
            result.receipt.input_pointer_version_hash_immutability_verified
        )
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in result.receipt.raw_free_payload().values()
            )
        )
        with self.assertRaises(FrozenInstanceError):
            result.receipt.observed_key_rank = 0

    def test_invalid_dtype_device_shape_rank_nonfinite_and_controls_fail(self) -> None:
        projector, keys, covariance, regularization = self.fixture()
        cases = (
            (projector.to(torch.float64), keys, covariance, regularization),
            (
                projector.detach().clone().requires_grad_(True),
                keys,
                covariance,
                regularization,
            ),
            (projector, keys.to(device="meta"), covariance, regularization),
            (projector[:3], keys, covariance, regularization),
            (projector, keys, covariance[:3], regularization),
            (projector, keys, covariance, regularization.reshape(1)),
            (projector, keys, covariance, regularization.to(torch.float64)),
            (projector, keys, covariance, torch.tensor(0.0, dtype=torch.float32)),
        )
        for values in cases:
            with self.subTest(shapes=tuple(tuple(item.shape) for item in values)):
                with self.assertRaises(ODEBFContractError):
                    self.solve(values)

        rank_deficient = keys.clone()
        rank_deficient[:, 1] = rank_deficient[:, 0]
        with self.assertRaises(ODEBFContractError):
            self.solve((projector, rank_deficient, covariance, regularization))
        zero_rank = torch.zeros_like(keys)
        with self.assertRaises(ODEBFContractError):
            self.solve((projector, zero_rank, covariance, regularization))
        nonfinite = covariance.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            self.solve((projector, keys, nonfinite, regularization))
        for tolerance in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ODEBFContractError):
                self.solve(
                    (projector, keys, covariance, regularization),
                    residual_tolerance=tolerance,
                )
        for maximum in (-1, True):
            with self.assertRaises(ODEBFContractError):
                self.solve(
                    (projector, keys, covariance, regularization),
                    condition_max_dimension=maximum,
                )

    def test_compute_and_prohibited_path_contract(self) -> None:
        source = inspect.getsource(q_module)
        for prohibited in (
            "canonical_alpha_fp32_solve",
            "solve_alpha_woodbury",
            "assemble_effective_bf16",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "CumulativeBF16FunctionalTrial",
            "p1r52_pir",
            "p1r52_fpiq",
            "p1r52_sequential_contract",
            "easyeditor",
        ):
            self.assertNotIn(prohibited, source)
        result = self.solve(self.fixture())
        receipt = result.receipt
        self.assertEqual(receipt.dense_writer_update_construction_count, 0)
        self.assertEqual(receipt.fp64_algorithm_tensor_count, 0)
        self.assertGreaterEqual(receipt.fp64_scalar_reduction_count, 2)
        self.assertEqual(receipt.model_forward_count, 0)
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.semantic_backward_count, 0)
        self.assertEqual(receipt.slope_backward_count, 0)
        self.assertEqual(receipt.materialization_count, 0)
        self.assertTrue(math.isfinite(receipt.q_norm))


if __name__ == "__main__":
    unittest.main()
