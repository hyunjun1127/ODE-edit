from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_nominal_shadow as shadow_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_fp32_transaction import (
    FP32TransactionMode,
    LayerParameterBinding,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_layer_step import (
    SHADOW_APPLICATION_PROOF_STATUS,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_nominal_shadow import (
    SHADOW_PROOF_STATUS,
    UNIFORM_NOMINAL_PI,
    PrefixObservation,
    ShadowAlphaLayerContext,
    build_prefix_observation,
    run_uniform_nominal_shadow_probe,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    LowRankFactorSource,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    UpdateMatchOrientation,
)


LAYERS = (4, 5, 6, 7, 8)


class DeterministicPrefixProvider:
    def __init__(
        self,
        bindings,
        *,
        fail_layer=None,
        mutate_layer=None,
        wrong_count_layer=None,
        stale_layer=None,
        wrong_receipt_layer=None,
    ) -> None:
        self.bindings = bindings
        self.fail_layer = fail_layer
        self.mutate_layer = mutate_layer
        self.wrong_count_layer = wrong_count_layer
        self.stale_layer = stale_layer
        self.wrong_receipt_layer = wrong_receipt_layer
        self.call_layers = []
        self.first_state = None

    def __call__(self, layer, pre_state):
        self.call_layers.append(layer)
        if self.first_state is None:
            self.first_state = pre_state
        if layer == self.fail_layer:
            raise RuntimeError(f"injected callback failure at layer {layer}")
        with torch.no_grad():
            aggregate = sum(
                float(
                    binding.parameter.detach()
                    .to(dtype=torch.float32)
                    .sum()
                )
                for binding in self.bindings.values()
            )
            terminal = torch.tensor(
                [[0.15, -0.05], [0.25, 0.10], [-0.20, 0.30]],
                dtype=torch.float32,
            ) + aggregate * 1.0e-4
            keys = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
                dtype=torch.float32,
            ) + aggregate * 1.0e-5
        state = self.first_state if layer == self.stale_layer else pre_state
        observation = build_prefix_observation(
            layer=layer,
            pre_observation_weight_state=state,
            current_terminal32=terminal,
            joint_keys32=keys,
        )
        if layer == self.wrong_count_layer:
            observation = replace(
                observation,
                receipt=replace(
                    observation.receipt,
                    model_backward_count=1,
                ),
            )
        if layer == self.wrong_receipt_layer:
            observation = replace(
                observation,
                receipt=replace(observation.receipt, layer=layer + 1),
            )
        if layer == self.mutate_layer:
            with torch.no_grad():
                self.bindings[4].parameter.add_(1.0)
        return observation


class ResidualReserveNominalShadowTests(unittest.TestCase):
    def fixture(self, *, storage_dtype=torch.bfloat16):
        generator = torch.Generator().manual_seed(307)
        bindings = {}
        contexts = {}
        for layer in LAYERS:
            shape = (3, 4) if layer in (4, 6, 8) else (4, 3)
            parameter = torch.nn.Parameter(
                torch.randn(shape, generator=generator, dtype=torch.float32).to(
                    dtype=storage_dtype
                )
            )
            weight_name = f"model.layers.{layer}.mlp.down_proj.weight"
            bindings[layer] = LayerParameterBinding(
                layer,
                weight_name,
                parameter,
            )
            contexts[layer] = ShadowAlphaLayerContext(
                layer=layer,
                weight_name=weight_name,
                parameter_shape=shape,
                projector32=torch.eye(4, dtype=torch.float32),
                committed_covariance32=(
                    0.2 * torch.eye(4, dtype=torch.float32)
                ),
                regularization32=torch.tensor(0.3, dtype=torch.float32),
                q_residual_tolerance=1.0e-6,
                q_condition_max_dimension=32,
                construction_id=f"nominal-shadow-layer-{layer}",
            )
        target = torch.tensor(
            [[0.9, 0.7], [0.4, 0.6], [-0.1, 0.2]],
            dtype=torch.float32,
        )
        return target, bindings, contexts

    @staticmethod
    def parameter_guards(bindings):
        return tuple(
            (
                layer,
                int(bindings[layer].parameter.data_ptr()),
                tensor_sha256(bindings[layer].parameter),
            )
            for layer in LAYERS
        )

    def execute(
        self,
        *,
        transaction_id="shadow-test",
        provider=None,
        dtype=torch.bfloat16,
    ):
        target, bindings, contexts = self.fixture(storage_dtype=dtype)
        callback = provider(bindings) if callable(provider) else provider
        if callback is None:
            callback = DeterministicPrefixProvider(bindings)
        guards = self.parameter_guards(bindings)
        result = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            contexts,
            callback,
            transaction_id=transaction_id,
        )
        return result, target, bindings, contexts, callback, guards

    def test_order_prefix_visibility_counts_and_exact_restore(self) -> None:
        result, _, bindings, _, callback, guards = self.execute()
        self.assertEqual(callback.call_layers, list(LAYERS))
        self.assertEqual(self.parameter_guards(bindings), guards)
        self.assertEqual(result.receipt.uniform_pi, UNIFORM_NOMINAL_PI)
        self.assertEqual(result.receipt.transaction_mode, "SHADOW")
        self.assertEqual(result.transaction_receipt.mode, "SHADOW")
        self.assertEqual(result.receipt.terminal_forward_count, 5)
        self.assertEqual(result.receipt.key_forward_count, 5)
        self.assertEqual(result.receipt.q_solve_count, 5)
        self.assertEqual(result.receipt.dense_update_construction_count, 5)
        self.assertEqual(result.receipt.temporary_native_apply_count, 5)
        self.assertEqual(result.receipt.native_storage_assignment_count, 5)
        self.assertEqual(result.receipt.restore_count, 1)
        self.assertEqual(result.receipt.logical_outer_commit_count, 0)
        self.assertEqual(result.receipt.persistent_commit_count, 0)
        self.assertEqual(result.receipt.external_materializer_count, 0)
        first_post = result.layer_results[0].application.m3a_layer_receipt
        layer5_pre = result.layer_results[1].observation.receipt
        layer4_state = next(
            item
            for item in layer5_pre.pre_observation_weight_state
            if item.layer == 4
        )
        self.assertEqual(
            layer4_state.sha256,
            first_post.post_storage_parameter_sha256,
        )
        for index, layer_result in enumerate(result.layer_results):
            self.assertEqual(layer_result.receipt.layer, LAYERS[index])
            self.assertEqual(
                layer_result.application.m3a_layer_receipt.layer,
                LAYERS[index],
            )

    def test_shadow_proof_closes_only_in_b2b1(self) -> None:
        result, *_ = self.execute()
        self.assertEqual(result.receipt.shadow_proof_status, SHADOW_PROOF_STATUS)
        for layer_result in result.layer_results:
            self.assertEqual(
                layer_result.nominal.receipt.shadow_application_proof_status,
                SHADOW_APPLICATION_PROOF_STATUS,
            )
            self.assertEqual(
                layer_result.receipt.shadow_proof_status,
                SHADOW_PROOF_STATUS,
            )
            self.assertEqual(
                layer_result.receipt.b2a_input_shadow_proof_status,
                SHADOW_APPLICATION_PROOF_STATUS,
            )
            self.assertEqual(
                layer_result.receipt.exact_construction_apply_count,
                1,
            )
            self.assertEqual(
                layer_result.receipt.authoritative_transaction_commit_claim_count,
                0,
            )
        self.assertEqual(result.transaction_receipt.logical_outer_commit_count, 0)
        self.assertEqual(result.transaction_receipt.persistent_commit_count, 0)

    def test_direct_transpose_nominal_factor_equivalence_and_nonalias(self) -> None:
        result, *_ = self.execute(dtype=torch.float32)
        orientations = []
        left_pointers = []
        right_pointers = []
        for layer_result in result.layer_results:
            plan = layer_result.plan
            nominal = layer_result.nominal
            quota = result.geometry.omega[plan.receipt.layer_index]
            expected = plan.prepared_update.matched_update32 / quota
            observed = nominal.factor.left @ nominal.factor.right.T
            torch.testing.assert_close(observed, expected, rtol=1.0e-6, atol=0.0)
            orientations.append(plan.receipt.construction_orientation)
            left_pointers.append(int(nominal.factor.left.data_ptr()))
            right_pointers.append(int(nominal.factor.right.data_ptr()))
            self.assertIs(
                nominal.factor.source,
                LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
            )
        self.assertIn(UpdateMatchOrientation.DIRECT.value, orientations)
        self.assertIn(UpdateMatchOrientation.TRANSPOSE_VIEW.value, orientations)
        self.assertEqual(len(set(left_pointers)), 5)
        self.assertEqual(len(set(right_pointers)), 5)
        self.assertEqual(len(set(left_pointers + right_pointers)), 10)

    def test_repeated_same_entry_scientific_identity_is_stable(self) -> None:
        target, bindings, contexts = self.fixture()
        first = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            contexts,
            DeterministicPrefixProvider(bindings),
            transaction_id="process-local-a",
        )
        factor_hashes = tuple(item.identity_sha256 for item in first.factors)
        second = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            contexts,
            DeterministicPrefixProvider(bindings),
            transaction_id="process-local-b",
        )
        self.assertEqual(
            first.receipt.scientific_identity_sha256,
            second.receipt.scientific_identity_sha256,
        )
        self.assertEqual(
            factor_hashes,
            tuple(item.identity_sha256 for item in second.factors),
        )
        self.assertIn(
            "tensor_identity.pointer",
            first.receipt.process_local_identity_exclusions,
        )

    def test_callback_failures_at_4_6_8_restore_and_return_no_result(self) -> None:
        for layer in (4, 6, 8):
            with self.subTest(layer=layer):
                target, bindings, contexts = self.fixture()
                guards = self.parameter_guards(bindings)
                provider = DeterministicPrefixProvider(
                    bindings,
                    fail_layer=layer,
                )
                with self.assertRaises(RuntimeError):
                    run_uniform_nominal_shadow_probe(
                        target,
                        bindings,
                        contexts,
                        provider,
                        transaction_id=f"failure-{layer}",
                    )
                self.assertEqual(self.parameter_guards(bindings), guards)

    def test_callback_mutation_wrong_counts_and_stale_state_restore(self) -> None:
        scenarios = (
            {"mutate_layer": 6},
            {"wrong_count_layer": 6},
            {"stale_layer": 6},
            {"wrong_receipt_layer": 6},
        )
        for index, scenario in enumerate(scenarios):
            with self.subTest(scenario=scenario):
                target, bindings, contexts = self.fixture()
                guards = self.parameter_guards(bindings)
                provider = DeterministicPrefixProvider(bindings, **scenario)
                with self.assertRaises((ODEBFContractError, ODEBFStateError)):
                    run_uniform_nominal_shadow_probe(
                        target,
                        bindings,
                        contexts,
                        provider,
                        transaction_id=f"bad-observation-{index}",
                    )
                self.assertEqual(self.parameter_guards(bindings), guards)

    def test_nonfinite_and_context_mutation_fail_closed(self) -> None:
        target, bindings, contexts = self.fixture()
        guards = self.parameter_guards(bindings)

        def nonfinite_provider(layer, state):
            terminal = torch.zeros((3, 2), dtype=torch.float32)
            terminal[0, 0] = float("nan")
            keys = torch.eye(4, 2, dtype=torch.float32)
            return build_prefix_observation(
                layer=layer,
                pre_observation_weight_state=state,
                current_terminal32=terminal,
                joint_keys32=keys,
            )

        with self.assertRaises(ODEBFContractError):
            run_uniform_nominal_shadow_probe(
                target,
                bindings,
                contexts,
                nonfinite_provider,
                transaction_id="nonfinite",
            )
        self.assertEqual(self.parameter_guards(bindings), guards)

        target, bindings, contexts = self.fixture()
        guards = self.parameter_guards(bindings)

        class ContextMutator(DeterministicPrefixProvider):
            def __call__(self, layer, state):
                result = super().__call__(layer, state)
                if layer == 4:
                    self.contexts[5].projector32[0, 0] += 1.0
                return result

        provider = ContextMutator(bindings)
        provider.contexts = contexts
        with self.assertRaises(ODEBFStateError):
            run_uniform_nominal_shadow_probe(
                target,
                bindings,
                contexts,
                provider,
                transaction_id="context-mutation",
            )
        self.assertEqual(self.parameter_guards(bindings), guards)

    def test_observation_builder_and_receipts_are_immutable_raw_free(self) -> None:
        result, *_ = self.execute()
        payload = result.receipt.raw_free_payload()
        self.assertFalse(
            any(isinstance(value, torch.Tensor) for value in payload.values())
        )
        with self.assertRaises(FrozenInstanceError):
            result.receipt.restore_count = 2
        with self.assertRaises(ODEBFContractError):
            build_prefix_observation(
                layer=4,
                pre_observation_weight_state=result.receipt.entry_weight_state,
                current_terminal32=torch.zeros((3, 2), dtype=torch.float32),
                joint_keys32=torch.eye(4, 2, dtype=torch.float32),
                terminal_forward_count=2,
            )

    def test_prohibited_paths_and_compute_ledger(self) -> None:
        source = inspect.getsource(shadow_module)
        for prohibited in (
            "p1r52_residual_reserve_pc_router",
            "p1r52_residual_reserve_pc_inventory_builder",
            "p1r52_residual_reserve_committed_state",
            "FP32TransactionMode.AUTHORITATIVE",
            "prepare_authoritative_commit",
            "commit_prepared",
            "assemble_effective_bf16",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "p1r52_sequential_runtime",
            "p1r52_pir",
            "p1r52_fpiq",
            "compute_z",
            "apply_AlphaEdit",
        ):
            self.assertNotIn(prohibited, source)
        result, *_ = self.execute()
        receipt = result.receipt
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.semantic_backward_count, 0)
        self.assertEqual(receipt.slope_backward_count, 0)
        self.assertEqual(receipt.router_call_count, 0)
        self.assertEqual(receipt.heldout_evaluator_count, 0)
        self.assertEqual(receipt.ledger_append_count, 0)
        self.assertEqual(receipt.history_append_count, 0)
        self.assertEqual(receipt.post_storage_decision_influence_count, 0)
        self.assertIs(
            FP32TransactionMode.SHADOW,
            FP32TransactionMode.SHADOW,
        )


if __name__ == "__main__":
    unittest.main()
