from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_layer_step as step_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_geometry import (
    build_residual_reserve_geometry,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_layer_step import (
    LAYER_STEP_BETA_PROOF_STATUS,
    NOMINAL_FACTOR_FP32_RTOL,
    NOMINAL_REFERENCE_RECEIPT_NAME,
    SHADOW_APPLICATION_PROOF_STATUS,
    derive_nominal_reference_factor,
    plan_residual_reserve_layer_step,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    LowRankFP32Factor,
    LowRankFactorSource,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    INPUT_BETA_APPLICATION_PROOF_STATUS,
    UpdateMatchOrientation,
)


class ResidualReserveLayerStepTests(unittest.TestCase):
    @staticmethod
    def geometry():
        return build_residual_reserve_geometry(
            torch.tensor([0.05, 0.1, 0.2, 0.25, 0.4], dtype=torch.float32)
        )

    @staticmethod
    def inputs(*, q_dimension: int = 4, residual_rows: int = 3, batch: int = 2):
        generator = torch.Generator().manual_seed(211)
        target = torch.randn(
            (residual_rows, batch),
            generator=generator,
            dtype=torch.float32,
        )
        terminal = torch.randn(
            (residual_rows, batch),
            generator=generator,
            dtype=torch.float32,
        )
        keys = torch.randn(
            (q_dimension, batch),
            generator=generator,
            dtype=torch.float32,
        )
        projector = torch.eye(q_dimension, dtype=torch.float32)
        covariance = 0.2 * torch.eye(q_dimension, dtype=torch.float32)
        regularization = torch.tensor(0.3, dtype=torch.float32)
        return target, terminal, keys, projector, covariance, regularization

    def plan(
        self,
        *,
        layer: int = 4,
        transpose: bool = False,
        inputs=None,
        geometry=None,
    ):
        values = (
            self.inputs(q_dimension=3, residual_rows=4)
            if transpose
            else self.inputs()
        ) if inputs is None else inputs
        return plan_residual_reserve_layer_step(
            *values,
            self.geometry() if geometry is None else geometry,
            layer=layer,
            weight_name=f"model.layers.{layer}.mlp.down_proj.weight",
            parameter_shape=(3, 4),
            construction_id=f"layer-step-{layer}-{'t' if transpose else 'd'}",
            q_residual_tolerance=1.0e-6,
        )

    def test_all_layers_nonuniform_pi_beta_once_and_no_last_dump(self) -> None:
        geometry = self.geometry()
        for layer in range(4, 9):
            with self.subTest(layer=layer):
                plan = self.plan(layer=layer, geometry=geometry)
                index = layer - 4
                expected_left = (
                    geometry.beta[index] * plan.residual32
                ).contiguous()
                self.assertTrue(
                    torch.equal(plan.beta_applied_left32, expected_left)
                )
                self.assertEqual(
                    plan.receipt.beta_applied_left_sha256,
                    tensor_sha256(expected_left),
                )
                self.assertEqual(plan.receipt.beta_application_count, 1)
                self.assertEqual(plan.receipt.internal_beta_reapplication_count, 0)
                self.assertEqual(
                    plan.receipt.beta_proof_status,
                    LAYER_STEP_BETA_PROOF_STATUS,
                )
                self.assertEqual(
                    plan.receipt.construction_input_beta_proof_status,
                    INPUT_BETA_APPLICATION_PROOF_STATUS,
                )
                self.assertLessEqual(float(geometry.beta[index]), float(geometry.mass))
                self.assertLess(float(geometry.beta[index]), 1.0)

    def test_direct_and_transpose_official_orientations(self) -> None:
        direct = self.plan(layer=4)
        transpose = self.plan(layer=8, transpose=True)
        self.assertEqual(
            direct.receipt.construction_orientation,
            UpdateMatchOrientation.DIRECT.value,
        )
        self.assertEqual(
            transpose.receipt.construction_orientation,
            UpdateMatchOrientation.TRANSPOSE_VIEW.value,
        )
        for plan in (direct, transpose):
            self.assertEqual(plan.prepared_update.matched_update32.shape, (3, 4))
            self.assertEqual(plan.receipt.q_solve_count, 1)
            self.assertEqual(plan.receipt.dense_construction_count, 1)
            self.assertEqual(plan.receipt.second_dense_construction_count, 0)

    def test_current_terminal_changes_residual_but_not_q(self) -> None:
        values = self.inputs()
        first = self.plan(inputs=values)
        changed = list(values)
        changed[1] = changed[1] + 0.125
        second = self.plan(inputs=tuple(changed))
        self.assertFalse(torch.equal(first.residual32, second.residual32))
        self.assertEqual(first.receipt.q_sha256, second.receipt.q_sha256)
        self.assertTrue(torch.equal(first.q32, second.q32))
        self.assertNotEqual(
            first.receipt.beta_applied_left_sha256,
            second.receipt.beta_applied_left_sha256,
        )

    def test_nominal_factor_quota_conversion_matches_test_only_dense(self) -> None:
        geometry = self.geometry()
        for layer in range(4, 9):
            with self.subTest(layer=layer):
                plan = self.plan(layer=layer, geometry=geometry)
                result = derive_nominal_reference_factor(
                    plan,
                    geometry,
                    layer=layer,
                )
                source = plan.prepared_update.factor
                source_dense = source.left @ source.right.T
                quota = geometry.mass * geometry.pi[layer - 4]
                expected = source_dense / quota
                observed = result.factor.left @ result.factor.right.T
                torch.testing.assert_close(
                    observed,
                    expected,
                    rtol=NOMINAL_FACTOR_FP32_RTOL,
                    atol=0.0,
                )
                self.assertIs(
                    result.factor.source,
                    LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
                )
                self.assertEqual(result.receipt.name, NOMINAL_REFERENCE_RECEIPT_NAME)
                self.assertEqual(result.receipt.division_count, 1)
                self.assertEqual(result.receipt.dense_factor_materialization_count, 0)
                self.assertEqual(result.receipt.dense_update_construction_count, 0)
                self.assertFalse(
                    result.receipt.authoritative_transaction_committed_claim
                )
                self.assertEqual(
                    result.receipt.shadow_application_proof_status,
                    SHADOW_APPLICATION_PROOF_STATUS,
                )
                self.assertEqual(
                    result.receipt.layer_step_receipt_identity,
                    plan.receipt.identity_sha256,
                )
                self.assertEqual(
                    result.receipt.geometry_receipt_identity,
                    plan.receipt.geometry_receipt_identity,
                )
                self.assertTrue(
                    result.receipt.quota_byte_exact_to_geometry_omega
                )
                self.assertEqual(
                    result.receipt.quota_tensor_sha256,
                    result.receipt.omega_tensor_sha256,
                )
                self.assertEqual(
                    result.receipt.test_only_dense_equivalence_rtol,
                    NOMINAL_FACTOR_FP32_RTOL,
                )

    def test_inputs_pointer_version_hash_immutable_and_intermediates_nonalias(self) -> None:
        values = self.inputs()
        geometry = self.geometry()
        tensors = values + (
            geometry.pi,
            geometry.omega,
            geometry.suffix_retention,
            geometry.beta,
            geometry.rho,
            geometry.mass,
        )
        guards = tuple(
            (int(item.data_ptr()), int(item._version), tensor_sha256(item))
            for item in tensors
        )
        plan = self.plan(inputs=values, geometry=geometry)
        observed = tuple(
            (int(item.data_ptr()), int(item._version), tensor_sha256(item))
            for item in tensors
        )
        self.assertEqual(observed, guards)
        self.assertTrue(
            plan.receipt.input_pointer_version_hash_immutability_verified
        )
        self.assertTrue(plan.receipt.intermediate_nonalias_verified)
        nominal = derive_nominal_reference_factor(
            plan,
            geometry,
            layer=4,
        )
        self.assertNotEqual(
            int(nominal.factor.left.data_ptr()),
            int(plan.prepared_update.factor.left.data_ptr()),
        )
        self.assertNotEqual(
            int(nominal.factor.right.data_ptr()),
            int(plan.prepared_update.factor.right.data_ptr()),
        )
        with self.assertRaises(FrozenInstanceError):
            plan.receipt.beta_application_count = 2

    def test_invalid_layer_dtype_shape_axis_device_nonfinite_fail_closed(self) -> None:
        values = self.inputs()
        for layer in (3, 9, True):
            with self.assertRaises(ODEBFContractError):
                self.plan(layer=layer, inputs=values)
        bad_dtype = list(values)
        bad_dtype[0] = bad_dtype[0].to(torch.float64)
        with self.assertRaises(ODEBFContractError):
            self.plan(inputs=tuple(bad_dtype))
        bad_shape = list(values)
        bad_shape[1] = bad_shape[1][:2]
        with self.assertRaises(ODEBFContractError):
            self.plan(inputs=tuple(bad_shape))
        bad_axis = list(values)
        bad_axis[0] = torch.ones((3, 3), dtype=torch.float32)
        bad_axis[1] = torch.zeros((3, 3), dtype=torch.float32)
        with self.assertRaises(ODEBFContractError):
            self.plan(inputs=tuple(bad_axis))
        bad_device = list(values)
        bad_device[2] = bad_device[2].to(device="meta")
        with self.assertRaises(ODEBFContractError):
            self.plan(inputs=tuple(bad_device))
        nonfinite = list(values)
        nonfinite[0] = nonfinite[0].clone()
        nonfinite[0][0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            self.plan(inputs=tuple(nonfinite))

    def test_invalid_nominal_quota_and_provenance_fail_closed(self) -> None:
        zero_geometry = build_residual_reserve_geometry(
            torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        )
        zero_plan = self.plan(layer=4, geometry=zero_geometry)
        with self.assertRaises(ODEBFContractError):
            derive_nominal_reference_factor(
                zero_plan,
                zero_geometry,
                layer=4,
            )

        geometry = self.geometry()
        plan = self.plan(geometry=geometry)
        source = plan.prepared_update.factor
        nominal_source = LowRankFP32Factor(
            source.layer,
            source.parameter_shape,
            source.left,
            source.right,
            LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
        )
        invalid_update = replace(plan.prepared_update, factor=nominal_source)
        invalid = replace(plan, prepared_update=invalid_update)
        with self.assertRaises((ODEBFContractError, ODEBFStateError)):
            derive_nominal_reference_factor(invalid, geometry, layer=4)
        with self.assertRaises((ODEBFContractError, ODEBFStateError)):
            derive_nominal_reference_factor(plan, geometry, layer=5)

    def test_geometry_and_full_plan_provenance_mismatch_fail_closed(self) -> None:
        geometry = self.geometry()
        plan = self.plan(geometry=geometry)
        uniform = build_residual_reserve_geometry(
            torch.full((5,), 0.2, dtype=torch.float32)
        )
        with self.assertRaises(ODEBFStateError):
            derive_nominal_reference_factor(plan, uniform, layer=4)

        forged_receipt = replace(
            plan.receipt,
            geometry_receipt_identity="forged-geometry",
        )
        with self.assertRaises(ODEBFStateError):
            derive_nominal_reference_factor(
                replace(plan, receipt=forged_receipt),
                geometry,
                layer=4,
            )

        other = plan_residual_reserve_layer_step(
            *self.inputs(),
            geometry,
            layer=4,
            weight_name="model.layers.4.mlp.down_proj.weight",
            parameter_shape=(3, 4),
            construction_id="other-single-construction",
            q_residual_tolerance=1.0e-6,
        )
        with self.assertRaises(ODEBFStateError):
            derive_nominal_reference_factor(
                replace(plan, prepared_update=other.prepared_update),
                geometry,
                layer=4,
            )
        with self.assertRaises(ODEBFContractError):
            derive_nominal_reference_factor(
                plan.prepared_update,
                geometry,
                layer=4,
            )

    def test_autograd_inputs_fail_closed_and_outputs_are_detached(self) -> None:
        for field in ("pi", "beta", "mass"):
            with self.subTest(field=field):
                geometry = self.geometry()
                getattr(geometry, field).requires_grad_(True)
                with self.assertRaises(ODEBFContractError):
                    self.plan(geometry=geometry)

        for index in (0, 1):
            with self.subTest(input_index=index):
                values = list(self.inputs())
                values[index].requires_grad_(True)
                with self.assertRaises(ODEBFContractError):
                    self.plan(inputs=tuple(values))

        plan = self.plan()
        outputs = (
            plan.residual32,
            plan.q32,
            plan.beta_applied_left32,
            plan.prepared_update.matched_update32,
            plan.prepared_update.factor.left,
            plan.prepared_update.factor.right,
        )
        self.assertFalse(any(value.requires_grad for value in outputs))
        self.assertEqual(
            plan.receipt.algorithm_tensor_requires_grad_count,
            0,
        )
        self.assertEqual(len(plan.receipt.input_identities), 12)
        self.assertFalse(
            any(item.requires_grad for item in plan.receipt.input_identities)
        )

    def test_compute_receipt_and_prohibited_paths(self) -> None:
        source = inspect.getsource(step_module)
        for prohibited in (
            "assemble_effective_bf16",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "apply_prepared_low_rank_update",
            "OfficialStyleFP32SequentialTransaction",
            "CommittedGrossLoadLedger",
            "p1r52_residual_reserve_committed_state",
            "p1r52_residual_reserve_pc_router",
            "p1r52_sequential_runtime",
            "p1r52_pir",
            "p1r52_fpiq",
            "compute_z",
            "apply_AlphaEdit",
        ):
            self.assertNotIn(prohibited, source)
        plan = self.plan()
        receipt = plan.receipt
        self.assertEqual(receipt.model_forward_count, 0)
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.semantic_backward_count, 0)
        self.assertEqual(receipt.slope_backward_count, 0)
        self.assertEqual(receipt.apply_count, 0)
        self.assertEqual(receipt.commit_count, 0)
        self.assertEqual(receipt.materialization_count, 0)
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in receipt.raw_free_payload().values()
            )
        )


if __name__ == "__main__":
    unittest.main()
