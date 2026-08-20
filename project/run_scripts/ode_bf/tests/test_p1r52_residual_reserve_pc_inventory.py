from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_pc_inventory as inventory_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    LayerPCInventoryInput,
    LowRankFP32Factor,
    LowRankFactorSource,
    RESIDUAL_RESERVE_INVENTORY_LAYERS,
    SealedPrevalidatedCovariance,
    build_residual_reserve_pc_inventory,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_router import (
    SOLVER_PRIMAL_TOLERANCE,
    evaluate_quadratic_proxy,
    solve_residual_reserve_pc_router,
)


class ResidualReservePCInventoryTests(unittest.TestCase):
    MASS = torch.tensor(0.25, dtype=torch.float32)
    PARAMETER_SHAPE = (3, 4)

    @staticmethod
    def nominal_factor(layer: int, scale: float = 1.0) -> LowRankFP32Factor:
        ordinal = layer - 3
        left = scale * torch.tensor(
            [
                [1.0 + 0.1 * ordinal, -0.25],
                [0.5, 0.75 + 0.05 * ordinal],
                [-0.4, 0.3],
            ],
            dtype=torch.float32,
        )
        right = torch.tensor(
            [
                [0.8, -0.2],
                [0.1 * ordinal, 0.6],
                [-0.5, 0.4],
                [0.3, 0.2 + 0.03 * ordinal],
            ],
            dtype=torch.float32,
        )
        return LowRankFP32Factor(
            layer,
            ResidualReservePCInventoryTests.PARAMETER_SHAPE,
            left,
            right,
            LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
        )

    @staticmethod
    def history_factor(
        layer: int,
        ordinal: int = 0,
    ) -> LowRankFP32Factor:
        left = torch.tensor(
            [
                [0.2 + 0.03 * layer + 0.01 * ordinal],
                [-0.1 + 0.02 * ordinal],
                [0.05 * layer - 0.01 * ordinal],
            ],
            dtype=torch.float32,
        )
        right = torch.tensor(
            [
                [0.4 - 0.02 * ordinal],
                [-0.3 + 0.01 * layer],
                [0.15 + 0.03 * ordinal],
                [0.2 - 0.01 * layer],
            ],
            dtype=torch.float32,
        )
        return LowRankFP32Factor(
            layer,
            ResidualReservePCInventoryTests.PARAMETER_SHAPE,
            left,
            right,
            LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
        )

    @staticmethod
    def covariance(layer: int) -> SealedPrevalidatedCovariance:
        diagonal = torch.tensor(
            [1.0, 1.25, 1.5, 1.75], dtype=torch.float32
        ) + 0.01 * (layer - 4)
        return SealedPrevalidatedCovariance(
            layer,
            torch.diag(diagonal),
            f"sealed-wikipedia-covariance-layer-{layer}",
        )

    def layer_input(
        self,
        layer: int,
        *,
        history_count: int = 1,
        cumulative_lambda: float | None = None,
        p_constant_shift: float = 0.0,
    ) -> LayerPCInventoryInput:
        nominal = self.nominal_factor(layer)
        history = tuple(
            self.history_factor(layer, ordinal) for ordinal in range(history_count)
        )
        covariance = self.covariance(layer)
        dense_drift = torch.zeros(self.PARAMETER_SHAPE, dtype=torch.float32)
        for factor in history:
            dense_drift = dense_drift + factor.left @ factor.right.T
        p_constant = float(
            torch.trace(dense_drift @ covariance.value @ dense_drift.T)
        ) + p_constant_shift
        return LayerPCInventoryInput(
            layer=layer,
            nominal_factor=nominal,
            committed_drift_factors=history,
            covariance=covariance,
            committed_p_constant=p_constant,
            cumulative_precast_lambda=(
                0.15 * (layer - 3)
                if cumulative_lambda is None
                else cumulative_lambda
            ),
            pretrained_weight_norm_squared=12.0 + layer,
        )

    def inputs(self, **kwargs) -> tuple[LayerPCInventoryInput, ...]:
        return tuple(
            self.layer_input(layer, **kwargs)
            for layer in RESIDUAL_RESERVE_INVENTORY_LAYERS
        )

    @staticmethod
    def dense_values(
        layers: tuple[LayerPCInventoryInput, ...],
        mass: float,
        pi: torch.Tensor,
    ) -> tuple[float, float]:
        p_value = 0.0
        c_value = 0.0
        for ordinal, item in enumerate(layers):
            nominal = item.nominal_factor.left @ item.nominal_factor.right.T
            drift = torch.zeros(item.nominal_factor.parameter_shape, dtype=torch.float32)
            for factor in item.committed_drift_factors:
                drift = drift + factor.left @ factor.right.T
            candidate = drift + mass * float(pi[ordinal]) * nominal
            p_value += float(
                torch.trace(candidate @ item.covariance.value @ candidate.T)
            )
            applied = mass * float(pi[ordinal]) * nominal
            c_value += (
                (1.0 + item.cumulative_precast_lambda)
                * float(torch.sum(applied * applied))
                / item.pretrained_weight_norm_squared
            )
        return p_value, c_value

    def test_low_rank_proxy_matches_direct_dense_formulas(self) -> None:
        layers = self.inputs(history_count=2)
        result = build_residual_reserve_pc_inventory(layers, mass=self.MASS)
        allocations = (
            torch.full((5,), 0.2, dtype=torch.float32),
            torch.tensor([0.5, 0.1, 0.15, 0.2, 0.05], dtype=torch.float32),
            torch.tensor([0.05, 0.35, 0.1, 0.15, 0.35], dtype=torch.float32),
        )
        for pi in allocations:
            with self.subTest(pi=tuple(float(item) for item in pi)):
                dense_p, dense_c = self.dense_values(layers, float(self.MASS), pi)
                proxy_p = evaluate_quadratic_proxy(result.p_proxy, pi)
                proxy_c = evaluate_quadratic_proxy(result.c_proxy, pi)
                self.assertAlmostEqual(proxy_p, dense_p, places=5)
                self.assertAlmostEqual(proxy_c, dense_c, places=6)

    def test_zero_history_has_zero_p_constant_and_linear(self) -> None:
        result = build_residual_reserve_pc_inventory(
            self.inputs(history_count=0), mass=self.MASS
        )
        self.assertEqual(result.p_proxy.constant, 0.0)
        self.assertTrue(torch.equal(result.p_proxy.linear, torch.zeros(5)))
        self.assertTrue(bool((torch.diag(result.p_proxy.quadratic) > 0.0).all()))
        self.assertTrue(bool((torch.diag(result.c_proxy.quadratic) > 0.0).all()))
        self.assertTrue(
            all(item.committed_factor_count == 0 for item in result.receipt.layer_receipts)
        )

    def test_multi_history_cross_and_supplied_constant(self) -> None:
        layers = self.inputs(history_count=2)
        result = build_residual_reserve_pc_inventory(layers, mass=self.MASS)
        expected_constant = sum(item.committed_p_constant for item in layers)
        self.assertAlmostEqual(result.p_proxy.constant, expected_constant, places=12)
        for item, receipt in zip(layers, result.receipt.layer_receipts, strict=True):
            nominal = item.nominal_factor.left @ item.nominal_factor.right.T
            drift = sum(
                (
                    factor.left @ factor.right.T
                    for factor in item.committed_drift_factors
                ),
                torch.zeros(item.nominal_factor.parameter_shape, dtype=torch.float32),
            )
            expected_cross = float(
                torch.trace(drift @ item.covariance.value @ nominal.T)
            )
            self.assertAlmostEqual(receipt.cross_term, expected_cross, places=6)
            self.assertEqual(receipt.committed_factor_count, 2)

    def test_lambda_only_changes_c_and_p_constant_shift_is_route_invariant(self) -> None:
        base_layers = self.inputs(history_count=1, cumulative_lambda=0.0)
        lambda_layers = self.inputs(history_count=1, cumulative_lambda=3.0)
        base = build_residual_reserve_pc_inventory(base_layers, mass=self.MASS)
        changed = build_residual_reserve_pc_inventory(lambda_layers, mass=self.MASS)
        self.assertEqual(base.p_proxy.raw_free_payload(), changed.p_proxy.raw_free_payload())
        self.assertTrue(torch.equal(base.c_proxy.linear, changed.c_proxy.linear))
        self.assertFalse(torch.equal(base.c_proxy.quadratic, changed.c_proxy.quadratic))

        shifted_layers = tuple(
            replace(item, committed_p_constant=item.committed_p_constant + 1.0e16)
            for item in base_layers
        )
        shifted = build_residual_reserve_pc_inventory(shifted_layers, mass=self.MASS)
        self.assertTrue(torch.equal(base.p_proxy.linear, shifted.p_proxy.linear))
        self.assertTrue(torch.equal(base.p_proxy.quadratic, shifted.p_proxy.quadratic))
        self.assertNotEqual(base.p_proxy.constant, shifted.p_proxy.constant)
        base_route = solve_residual_reserve_pc_router(base.p_proxy, base.c_proxy)
        shifted_route = solve_residual_reserve_pc_router(
            shifted.p_proxy, shifted.c_proxy
        )
        self.assertTrue(torch.equal(base_route.pi_balanced, shifted_route.pi_balanced))
        self.assertEqual(base_route.receipt.p_scale, shifted_route.receipt.p_scale)
        self.assertEqual(base_route.receipt.selected_status, shifted_route.receipt.selected_status)

    def test_mass_scales_linear_and_diagonals(self) -> None:
        layers = self.inputs(history_count=1)
        small = build_residual_reserve_pc_inventory(
            layers, mass=torch.tensor(0.125, dtype=torch.float32)
        )
        large = build_residual_reserve_pc_inventory(
            layers, mass=torch.tensor(0.25, dtype=torch.float32)
        )
        torch.testing.assert_close(
            large.p_proxy.linear,
            2.0 * small.p_proxy.linear,
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            large.p_proxy.quadratic,
            4.0 * small.p_proxy.quadratic,
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            large.c_proxy.quadratic,
            4.0 * small.c_proxy.quadratic,
            rtol=2.0e-6,
            atol=1.0e-7,
        )

    def test_factor_inputs_are_immutable_nonaliased_and_dense_path_absent(self) -> None:
        left = torch.ones((3, 1), dtype=torch.float32)
        right = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        pointers = (int(left.data_ptr()), int(right.data_ptr()))
        versions = (int(left._version), int(right._version))
        hashes = (tensor_sha256(left), tensor_sha256(right))
        factor = LowRankFP32Factor(
            4,
            self.PARAMETER_SHAPE,
            left,
            right,
            LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
        )
        self.assertEqual(pointers, (int(left.data_ptr()), int(right.data_ptr())))
        self.assertEqual(versions, (int(left._version), int(right._version)))
        self.assertEqual(hashes, (tensor_sha256(left), tensor_sha256(right)))
        self.assertNotEqual(int(factor.left.data_ptr()), pointers[0])
        self.assertNotEqual(int(factor.right.data_ptr()), pointers[1])

        layers = self.inputs(history_count=2)
        guarded = []
        for item in layers:
            guarded.extend(
                (item.nominal_factor.left, item.nominal_factor.right)
            )
            for historical in item.committed_drift_factors:
                guarded.extend((historical.left, historical.right))
            guarded.append(item.covariance.value)
        before = tuple(
            (int(value.data_ptr()), int(value._version), tensor_sha256(value))
            for value in guarded
        )
        result = build_residual_reserve_pc_inventory(layers, mass=self.MASS)
        after = tuple(
            (int(value.data_ptr()), int(value._version), tensor_sha256(value))
            for value in guarded
        )
        self.assertEqual(before, after)
        input_pointers = {item[0] for item in before}
        for output in (
            result.p_proxy.linear,
            result.p_proxy.quadratic,
            result.c_proxy.linear,
            result.c_proxy.quadratic,
        ):
            self.assertNotIn(int(output.data_ptr()), input_pointers)
        self.assertTrue(result.receipt.input_immutability_verified)
        source = inspect.getsource(inventory_module)
        self.assertEqual(source.count("covariance.value @ nominal.right"), 1)
        for prohibited_dense_expression in (
            "nominal.left @ nominal.right.T",
            "historical.left @ historical.right.T",
            "torch.outer",
            "torch.einsum",
        ):
            self.assertNotIn(prohibited_dense_expression, source)
        self.assertTrue(
            all(
                item.covariance_right_matmul_count == 1
                and item.nominal_redivision_count == 0
                and item.dense_drift_materialization_count == 0
                and item.dense_nominal_materialization_count == 0
                and item.dense_candidate_materialization_count == 0
                for item in result.receipt.layer_receipts
            )
        )

    def test_output_is_fp32_cpu_diagonal_and_router_consumable(self) -> None:
        result = build_residual_reserve_pc_inventory(
            self.inputs(history_count=1), mass=self.MASS
        )
        for tensor in (
            result.p_proxy.linear,
            result.p_proxy.quadratic,
            result.c_proxy.linear,
            result.c_proxy.quadratic,
        ):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, "cpu")
        for quadratic in (result.p_proxy.quadratic, result.c_proxy.quadratic):
            off_diagonal = quadratic - torch.diag(torch.diag(quadratic))
            self.assertTrue(torch.equal(off_diagonal, torch.zeros_like(off_diagonal)))
        route = solve_residual_reserve_pc_router(result.p_proxy, result.c_proxy)
        self.assertEqual(route.pi_balanced.dtype, torch.float32)
        self.assertAlmostEqual(float(route.pi_balanced.sum()), 1.0, places=6)
        self.assertEqual(result.receipt.simplex_degrees_of_freedom, 4)
        self.assertIsNotNone(result.receipt.uniform_gradient_cosine)
        self.assertIn(
            result.receipt.uniform_gradient_alignment,
            ("NEGATIVE", "ZERO", "POSITIVE"),
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        nominal = self.nominal_factor(4)
        covariance = self.covariance(4)
        historical = self.history_factor(4)
        prepared_historical = LowRankFP32Factor(
            historical.layer,
            historical.parameter_shape,
            historical.left,
            historical.right,
            LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32,
        )
        invalid_constructors = (
            lambda: LowRankFP32Factor(
                4,
                self.PARAMETER_SHAPE,
                torch.ones((2, 1), dtype=torch.float32),
                torch.ones((4, 1), dtype=torch.float32),
                LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
            ),
            lambda: LowRankFP32Factor(
                4,
                self.PARAMETER_SHAPE,
                torch.ones((3, 1), dtype=torch.float64),
                torch.ones((4, 1), dtype=torch.float64),
                LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
            ),
            lambda: LowRankFP32Factor(
                4,
                self.PARAMETER_SHAPE,
                torch.ones((3, 1), dtype=torch.float32),
                torch.ones((4, 1), dtype=torch.float32, device="meta"),
                LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
            ),
            lambda: LowRankFP32Factor(
                4,
                self.PARAMETER_SHAPE,
                torch.full((3, 1), float("nan"), dtype=torch.float32),
                torch.ones((4, 1), dtype=torch.float32),
                LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
            ),
            lambda: SealedPrevalidatedCovariance(
                4, torch.eye(4, dtype=torch.float32), ""
            ),
            lambda: SealedPrevalidatedCovariance(
                4, torch.ones((4, 3), dtype=torch.float32), "sealed"
            ),
            lambda: LayerPCInventoryInput(
                4,
                nominal,
                (),
                SealedPrevalidatedCovariance(
                    4, torch.eye(3, dtype=torch.float32), "sealed-wrong-dimension"
                ),
                0.0,
                0.0,
                1.0,
            ),
            lambda: LayerPCInventoryInput(
                4, nominal, (), covariance, 0.0, -1.0, 1.0
            ),
            lambda: LayerPCInventoryInput(
                4, nominal, (), covariance, 0.0, 0.0, 0.0
            ),
            lambda: LayerPCInventoryInput(
                4, nominal, (nominal,), covariance, 0.0, 0.0, 1.0
            ),
            lambda: LayerPCInventoryInput(
                4,
                nominal,
                (prepared_historical,),
                covariance,
                0.0,
                0.0,
                1.0,
            ),
            lambda: LayerPCInventoryInput(
                4, nominal, [], covariance, 0.0, 0.0, 1.0
            ),
            lambda: LayerPCInventoryInput(
                4, nominal, (), covariance, -1.0, 0.0, 1.0
            ),
        )
        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ODEBFContractError):
                    constructor()

        roundoff = LayerPCInventoryInput(
            4,
            nominal,
            (),
            covariance,
            -0.5 * SOLVER_PRIMAL_TOLERANCE,
            0.0,
            1.0,
        )
        self.assertEqual(roundoff.committed_p_constant, 0.0)
        self.assertEqual(
            roundoff.p_constant_numerical_tolerance,
            SOLVER_PRIMAL_TOLERANCE,
        )

        with self.assertRaises(ODEBFContractError):
            build_residual_reserve_pc_inventory(
                self.inputs(), mass=torch.tensor(0.0, dtype=torch.float32)
            )

    def test_prohibited_imports_and_compute_receipt(self) -> None:
        source = inspect.getsource(inventory_module)
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "CumulativeBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "PIR-U",
            "PIRU",
            "FPIQ",
            "p1r52_sequential_contract",
            "torch.linalg.eig",
            "numpy.linalg.eig",
        ):
            self.assertNotIn(prohibited, source)
        result = build_residual_reserve_pc_inventory(
            self.inputs(history_count=1), mass=self.MASS
        )
        self.assertEqual(result.receipt.model_forward_count, 0)
        self.assertEqual(result.receipt.model_backward_count, 0)
        self.assertEqual(result.receipt.dense_materialization_count, 0)
        self.assertEqual(result.receipt.post_storage_decision_influence_count, 0)


if __name__ == "__main__":
    unittest.main()
