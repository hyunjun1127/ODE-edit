from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_route_assembly as assembly_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadState,
    CommittedLayerState,
    LayerCommittedStateAnchor,
    initialize_committed_gross_load_state,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_fp32_transaction import (
    LayerParameterBinding,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_nominal_shadow import (
    ShadowAlphaLayerContext,
    build_prefix_observation,
    run_uniform_nominal_shadow_probe,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    LowRankFP32Factor,
    LowRankFactorSource,
    SealedPrevalidatedCovariance,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_router import (
    evaluate_quadratic_proxy,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_route_assembly import (
    ROUTE_ASSEMBLY_STATUS,
    UNIFORM_DECISION_OFF_STATUS,
    assemble_residual_reserve_pc_route,
    observe_uniform_route_without_resolve,
)


LAYERS = RESIDUAL_RESERVE_LAYER_ORDER


class _PrefixProvider:
    def __init__(self, bindings, target, *, zero_residual: bool = False) -> None:
        self.bindings = bindings
        self.target = target
        self.zero_residual = zero_residual

    def __call__(self, layer, pre_state):
        if self.zero_residual:
            terminal = self.target.detach().clone()
        else:
            aggregate = sum(
                float(binding.parameter.detach().float().sum())
                for binding in self.bindings.values()
            )
            terminal = torch.tensor(
                [[0.15, -0.05], [0.25, 0.10], [-0.20, 0.30]],
                dtype=torch.float32,
            ) + aggregate * 1.0e-4
        keys = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
            dtype=torch.float32,
        )
        return build_prefix_observation(
            layer=layer,
            pre_observation_weight_state=pre_state,
            current_terminal32=terminal,
            joint_keys32=keys,
        )


class ResidualReserveRouteAssemblyTests(unittest.TestCase):
    def shadow(self, *, target_shift: float = 0.0, zero_residual: bool = False):
        generator = torch.Generator().manual_seed(701)
        bindings = {}
        contexts = {}
        for layer in LAYERS:
            shape = (3, 4) if layer in (4, 6, 8) else (4, 3)
            parameter = torch.nn.Parameter(
                torch.randn(shape, generator=generator, dtype=torch.float32).to(
                    dtype=torch.bfloat16
                ),
                requires_grad=False,
            )
            weight_name = f"model.layers.{layer}.mlp.down_proj.weight"
            bindings[layer] = LayerParameterBinding(layer, weight_name, parameter)
            contexts[layer] = ShadowAlphaLayerContext(
                layer=layer,
                weight_name=weight_name,
                parameter_shape=shape,
                projector32=torch.eye(4, dtype=torch.float32),
                committed_covariance32=0.2 * torch.eye(4, dtype=torch.float32),
                regularization32=torch.tensor(0.3, dtype=torch.float32),
                q_residual_tolerance=1.0e-6,
                q_condition_max_dimension=32,
                construction_id=f"route-assembly-shadow-{target_shift}-{layer}",
            )
        target = torch.tensor(
            [[0.9, 0.7], [0.4, 0.6], [-0.1, 0.2]],
            dtype=torch.float32,
        ) + target_shift
        result = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            contexts,
            _PrefixProvider(bindings, target, zero_residual=zero_residual),
            transaction_id=f"route-assembly-shadow-{target_shift}-{zero_residual}",
        )
        return result

    @staticmethod
    def covariances(shadow, *, scale: float = 1.0, suffix: str = "base"):
        return tuple(
            SealedPrevalidatedCovariance(
                factor.layer,
                scale * torch.eye(factor.parameter_shape[1], dtype=torch.float32),
                f"sealed-route-covariance-{suffix}-{factor.layer}",
            )
            for factor in shadow.factors
        )

    @staticmethod
    def zero_state(covariances, *, state_id: str = "route-zero"):
        anchors = tuple(
            LayerCommittedStateAnchor(
                covariance.layer,
                covariance,
                20.0 + covariance.layer,
            )
            for covariance in covariances
        )
        return initialize_committed_gross_load_state(anchors, state_id=state_id)

    @staticmethod
    def nonzero_state(shadow, covariances, *, scale: float = 0.2):
        transaction_id = f"committed-route-{scale}"
        layers = []
        for nominal, covariance in zip(
            shadow.factors, covariances, strict=True
        ):
            factor = LowRankFP32Factor(
                nominal.layer,
                nominal.parameter_shape,
                scale * nominal.left,
                nominal.right,
                LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
            )
            dense = factor.left @ factor.right.T
            p_constant = float(
                torch.trace(dense @ covariance.value @ dense.T)
            )
            pretrained_norm = 20.0 + nominal.layer
            precast_lambda = float(torch.sum(dense * dense)) / pretrained_norm
            layers.append(
                CommittedLayerState(
                    layer=nominal.layer,
                    covariance=covariance,
                    pretrained_weight_norm_squared=pretrained_norm,
                    committed_precast_factors=(factor,),
                    structural_p_constant=p_constant,
                    cumulative_precast_lambda=precast_lambda,
                    cumulative_actual_post_storage_load_observation=(
                        1.1 * precast_lambda
                    ),
                    version=1,
                    committed_transaction_ids=(transaction_id,),
                    committed_construction_receipt_ids=(
                        f"committed-construction-{scale}-{nominal.layer}",
                    ),
                )
            )
        return CommittedGrossLoadState(
            state_id=f"route-history-{scale}",
            layers=tuple(layers),
            version=1,
            commit_count=1,
            committed_transaction_ids=(transaction_id,),
            last_transaction_id=transaction_id,
        )

    @staticmethod
    def tensor_guards(shadow, state, covariances):
        tensors = [
            shadow.geometry.pi,
            shadow.geometry.omega,
            shadow.geometry.beta,
            shadow.geometry.mass,
        ]
        for factor in shadow.factors:
            tensors.extend((factor.left, factor.right))
        for layer in state.layers:
            tensors.append(layer.covariance.value)
            for factor in layer.committed_precast_factors:
                tensors.extend((factor.left, factor.right))
        tensors.extend(covariance.value for covariance in covariances)
        return tuple(
            (
                tensor,
                int(tensor.data_ptr()),
                int(tensor._version),
                tensor_sha256(tensor),
            )
            for tensor in tensors
        )

    def assert_guards(self, guards):
        for tensor, pointer, version, sha256 in guards:
            self.assertEqual(int(tensor.data_ptr()), pointer)
            self.assertEqual(int(tensor._version), version)
            self.assertEqual(tensor_sha256(tensor), sha256)

    def test_zero_history_builds_one_bound_route_and_selected_geometry(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        state = self.zero_state(covariances)
        guards = self.tensor_guards(shadow, state, covariances)
        original_solver = assembly_module.solve_residual_reserve_pc_router
        with mock.patch.object(
            assembly_module,
            "solve_residual_reserve_pc_router",
            wraps=original_solver,
        ) as solver:
            result = assemble_residual_reserve_pc_route(
                shadow, state, covariances
            )
            self.assertEqual(solver.call_count, 1)
        self.assert_guards(guards)
        self.assertEqual(result.receipt.status, ROUTE_ASSEMBLY_STATUS)
        self.assertEqual(result.receipt.committed_state_version, 0)
        self.assertEqual(len(result.receipt.layer_bindings), 5)
        self.assertTrue(
            all(item.committed_factor_count == 0 for item in result.receipt.layer_bindings)
        )
        self.assertEqual(result.receipt.ledger.router_call_count, 1)
        self.assertEqual(
            result.receipt.ledger.router_decision_to_execution_pi_copy_count,
            1,
        )
        self.assertEqual(result.receipt.ledger.cross_device_pi_copy_count, 0)
        self.assertEqual(result.receipt.ledger.covariance_right_matmul_count, 5)
        self.assertEqual(result.receipt.ledger.model_forward_count, 0)
        self.assertEqual(result.receipt.ledger.model_backward_count, 0)
        self.assertEqual(result.receipt.ledger.q_solve_count, 0)
        self.assertEqual(result.receipt.ledger.native_apply_count, 0)
        self.assertEqual(result.receipt.ledger.transaction_count, 0)
        self.assertEqual(result.receipt.ledger.materialization_count, 0)
        self.assertEqual(result.receipt.ledger.weight_mutation_count, 0)
        self.assertIs(result.selected_pi.dtype, torch.float32)
        self.assertEqual(
            result.selected_pi.device,
            shadow.geometry.pi.device,
        )
        self.assertEqual(result.receipt.router_decision_device, "cpu")
        self.assertEqual(
            result.receipt.selected_execution_device,
            str(shadow.geometry.pi.device),
        )
        self.assertEqual(
            result.receipt.router_selected_pi_sha256,
            result.receipt.selected_pi_sha256,
        )
        self.assertEqual(
            result.receipt.selected_pi_sha256,
            result.receipt.selected_geometry_pi_sha256,
        )
        self.assertTrue(result.receipt.device_independent_pi_bytes_exact)
        self.assertNotEqual(
            int(result.selected_pi.data_ptr()),
            int(result.routing.pi_balanced.data_ptr()),
        )
        self.assertNotEqual(
            int(result.selected_pi.data_ptr()),
            int(result.selected_geometry.pi.data_ptr()),
        )
        self.assertAlmostEqual(float(result.selected_pi.sum()), 1.0, places=6)
        self.assertTrue(bool((result.selected_pi >= 0.0).all()))
        self.assertAlmostEqual(
            float(result.selected_geometry.omega.sum()),
            float(result.selected_geometry.mass),
            places=6,
        )
        self.assertLess(result.selected_geometry.receipt.last_beta, 1.0)
        pointers = [
            int(tensor.data_ptr())
            for factor in shadow.factors
            for tensor in (factor.left, factor.right)
        ]
        self.assertEqual(len(pointers), len(set(pointers)))

    def test_nonzero_successful_history_and_deterministic_identity(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        state = self.nonzero_state(shadow, covariances)
        first = assemble_residual_reserve_pc_route(shadow, state, covariances)
        second = assemble_residual_reserve_pc_route(shadow, state, covariances)
        self.assertEqual(first.receipt.identity_sha256, second.receipt.identity_sha256)
        self.assertTrue(torch.equal(first.selected_pi, second.selected_pi))
        self.assertEqual(first.receipt.committed_state_version, 1)
        self.assertTrue(
            all(item.committed_factor_count == 1 for item in first.receipt.layer_bindings)
        )
        self.assertTrue(
            all(
                factor.source
                is LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
                for layer in state.layers
                for factor in layer.committed_precast_factors
            )
        )
        with self.assertRaises(FrozenInstanceError):
            first.receipt.committed_state_version = 2

    def test_flat_axes_and_uniform_decision_off_do_not_resolve_again(self):
        shadow = self.shadow(zero_residual=True)
        covariances = self.covariances(shadow)
        state = self.zero_state(covariances)
        original_solver = assembly_module.solve_residual_reserve_pc_router
        with mock.patch.object(
            assembly_module,
            "solve_residual_reserve_pc_router",
            wraps=original_solver,
        ) as solver:
            result = assemble_residual_reserve_pc_route(
                shadow, state, covariances
            )
            observation = observe_uniform_route_without_resolve(result)
            self.assertEqual(solver.call_count, 1)
        self.assertEqual(
            result.routing.receipt.selected_status,
            "BOTH_AXES_FLAT_UNIFORM",
        )
        self.assertTrue(torch.equal(result.selected_pi, result.routing.pi_uniform))
        self.assertEqual(observation.status, UNIFORM_DECISION_OFF_STATUS)
        self.assertEqual(observation.router_call_count, 0)
        self.assertEqual(observation.decision_influence_count, 0)
        self.assertEqual(
            observation.execution_device,
            str(result.selected_pi.device),
        )
        self.assertEqual(
            observation.router_pi_sha256,
            tensor_sha256(observation.pi),
        )
        self.assertTrue(observation.device_independent_pi_bytes_exact)
        self.assertEqual(
            observation.router_decision_to_execution_pi_copy_count,
            1,
        )
        self.assertTrue(torch.equal(observation.pi, result.routing.pi_uniform))
        self.assertEqual(
            observation.p_value,
            evaluate_quadratic_proxy(result.inventory.p_proxy, observation.pi),
        )
        self.assertEqual(
            observation.c_value,
            evaluate_quadratic_proxy(result.inventory.c_proxy, observation.pi),
        )
        self.assertEqual(
            observation.geometry.receipt.raw_free_payload()["identity_sha256"],
            result.selected_geometry.receipt.raw_free_payload()["identity_sha256"],
        )

    def test_route_identity_binds_state_covariance_and_nominal_factor(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        zero = self.zero_state(covariances)
        history = self.nonzero_state(shadow, covariances, scale=0.35)
        base = assemble_residual_reserve_pc_route(shadow, zero, covariances)
        changed_state = assemble_residual_reserve_pc_route(
            shadow, history, covariances
        )
        self.assertNotEqual(
            base.receipt.identity_sha256,
            changed_state.receipt.identity_sha256,
        )

        changed_covariances = self.covariances(
            shadow, scale=1.4, suffix="changed"
        )
        changed_cov_state = self.zero_state(
            changed_covariances, state_id="route-covariance-change"
        )
        changed_cov = assemble_residual_reserve_pc_route(
            shadow, changed_cov_state, changed_covariances
        )
        self.assertNotEqual(
            base.receipt.identity_sha256,
            changed_cov.receipt.identity_sha256,
        )

        changed_shadow = self.shadow(target_shift=0.125)
        changed_nominal = assemble_residual_reserve_pc_route(
            changed_shadow,
            self.zero_state(self.covariances(changed_shadow)),
            self.covariances(changed_shadow),
        )
        self.assertNotEqual(
            tuple(item.identity_sha256 for item in shadow.factors),
            tuple(item.identity_sha256 for item in changed_shadow.factors),
        )
        self.assertNotEqual(
            base.receipt.nominal_scientific_identity,
            changed_nominal.receipt.nominal_scientific_identity,
        )
        self.assertNotEqual(
            base.receipt.identity_sha256,
            changed_nominal.receipt.identity_sha256,
        )

    def test_mixed_stale_components_and_candidate_provenance_fail_close(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        state = self.zero_state(covariances)
        stale_covariances = self.covariances(
            shadow, scale=1.1, suffix="stale"
        )
        with self.assertRaises(ODEBFStateError):
            assemble_residual_reserve_pc_route(
                shadow, state, stale_covariances
            )
        with self.assertRaises(ODEBFContractError):
            assemble_residual_reserve_pc_route(
                shadow, state, covariances[:-1]
            )
        replacement = LowRankFP32Factor(
            4,
            shadow.factors[0].parameter_shape,
            1.1 * shadow.factors[0].left,
            shadow.factors[0].right,
            LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
        )
        forged_shadow = replace(
            shadow,
            factors=(replacement,) + shadow.factors[1:],
        )
        with self.assertRaises(ODEBFStateError):
            assemble_residual_reserve_pc_route(
                forged_shadow, state, covariances
            )
        prepared_candidate = LowRankFP32Factor(
            4,
            shadow.factors[0].parameter_shape,
            shadow.factors[0].left,
            shadow.factors[0].right,
            LowRankFactorSource.AUTHORITATIVE_PREPARED_PRECAST_FP32,
        )
        with self.assertRaises(ODEBFContractError):
            CommittedLayerState(
                layer=4,
                covariance=covariances[0],
                pretrained_weight_norm_squared=24.0,
                committed_precast_factors=(prepared_candidate,),
                structural_p_constant=0.0,
                cumulative_precast_lambda=0.0,
                cumulative_actual_post_storage_load_observation=0.0,
                version=1,
                committed_transaction_ids=("candidate",),
                committed_construction_receipt_ids=("candidate",),
            )

    def test_mixed_inventory_is_rejected_by_exact_layer_provenance(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        zero_state = self.zero_state(covariances)
        history_state = self.nonzero_state(shadow, covariances, scale=0.31)
        mixed_inventory = assemble_residual_reserve_pc_route(
            shadow,
            history_state,
            covariances,
        ).inventory
        with mock.patch.object(
            assembly_module,
            "build_residual_reserve_pc_inventory",
            return_value=mixed_inventory,
        ):
            with self.assertRaises(ODEBFStateError):
                assemble_residual_reserve_pc_route(
                    shadow,
                    zero_state,
                    covariances,
                )

    def test_mixed_router_and_replaced_balanced_pi_fail_close(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        zero_state = self.zero_state(covariances)
        history_state = self.nonzero_state(shadow, covariances, scale=0.42)
        mixed_routing = assemble_residual_reserve_pc_route(
            shadow,
            history_state,
            covariances,
        ).routing
        with mock.patch.object(
            assembly_module,
            "solve_residual_reserve_pc_router",
            return_value=mixed_routing,
        ):
            with self.assertRaises(ODEBFStateError):
                assemble_residual_reserve_pc_route(
                    shadow,
                    zero_state,
                    covariances,
                )

        valid = assemble_residual_reserve_pc_route(
            shadow,
            zero_state,
            covariances,
        )
        forged = replace(
            valid.routing,
            pi_balanced=torch.tensor(
                [0.1, 0.1, 0.1, 0.1, 0.6],
                dtype=torch.float32,
            ),
        )
        with mock.patch.object(
            assembly_module,
            "solve_residual_reserve_pc_router",
            return_value=forged,
        ):
            with self.assertRaises(ODEBFStateError):
                assemble_residual_reserve_pc_route(
                    shadow,
                    zero_state,
                    covariances,
                )

    def test_inventory_receipt_exactly_binds_every_m3b_input(self):
        shadow = self.shadow()
        covariances = self.covariances(shadow)
        state = self.nonzero_state(shadow, covariances, scale=0.27)
        result = assemble_residual_reserve_pc_route(
            shadow,
            state,
            covariances,
        )
        self.assertEqual(
            result.inventory.receipt.mass_sha256,
            tensor_sha256(shadow.geometry.mass),
        )
        for nominal, state_layer, covariance, receipt in zip(
            shadow.factors,
            state.layers,
            covariances,
            result.inventory.receipt.layer_receipts,
            strict=True,
        ):
            self.assertEqual(receipt.nominal_factor_identity, nominal.identity_sha256)
            self.assertEqual(
                receipt.committed_factor_identities,
                tuple(
                    factor.identity_sha256
                    for factor in state_layer.committed_precast_factors
                ),
            )
            self.assertEqual(
                receipt.committed_factor_count,
                len(state_layer.committed_precast_factors),
            )
            self.assertEqual(receipt.covariance_identity, covariance.identity_sha256)
            self.assertEqual(
                receipt.covariance_artifact_identity,
                covariance.artifact_identity,
            )
            self.assertEqual(
                receipt.committed_p_constant,
                state_layer.structural_p_constant,
            )
            self.assertEqual(
                receipt.cumulative_precast_lambda,
                state_layer.cumulative_precast_lambda,
            )
            self.assertEqual(
                receipt.pretrained_weight_norm_squared,
                state_layer.pretrained_weight_norm_squared,
            )

    def test_prohibited_imports_and_route_only_compute_boundary(self):
        source = inspect.getsource(assembly_module)
        prohibited = (
            "p1r52_residual_reserve_fp32_transaction import",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_pir",
            "p1r52_fpiq",
            "OfficialStyleFP32SequentialTransaction",
            "apply_layer_fp32",
            "commit_outer",
            "commit_prepared",
            "stage_authoritative_commit",
        )
        for symbol in prohibited:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)
        self.assertIn("device=execution_device", source)
        self.assertNotIn('selected_pi.device.type != "cpu"', source)


if __name__ == "__main__":
    unittest.main()
