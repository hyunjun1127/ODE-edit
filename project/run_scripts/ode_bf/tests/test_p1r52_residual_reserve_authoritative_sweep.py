from __future__ import annotations

from dataclasses import replace
import inspect
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_authoritative_sweep as sweep_module,
)
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_layer_step as layer_step_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_authoritative_sweep import (
    AUTHORITATIVE_SWEEP_STATUS,
    run_authoritative_residual_reserve_sweep,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
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
from project.run_scripts.ode_bf.p1r52_residual_reserve_route_assembly import (
    assemble_residual_reserve_pc_route,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    EXACT_SINGLE_CONSTRUCTION_STATUS,
)


LAYERS = RESIDUAL_RESERVE_LAYER_ORDER


class _PrefixProvider:
    def __init__(self, bindings, *, fail_layer=None) -> None:
        self.bindings = bindings
        self.fail_layer = fail_layer
        self.call_layers = []

    def __call__(self, layer, pre_state):
        self.call_layers.append(layer)
        if layer == self.fail_layer:
            raise RuntimeError(f"injected observation failure at layer {layer}")
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
        ) + aggregate * 1.0e-5
        return build_prefix_observation(
            layer=layer,
            pre_observation_weight_state=pre_state,
            current_terminal32=terminal,
            joint_keys32=keys,
        )


class ResidualReserveAuthoritativeSweepTests(unittest.TestCase):
    def base_fixture(self, *, history: bool = False, dtype=torch.bfloat16):
        generator = torch.Generator().manual_seed(811)
        bindings = {}
        nominal_contexts = {}
        for layer in LAYERS:
            shape = (3, 4) if layer in (4, 6, 8) else (4, 3)
            parameter = torch.nn.Parameter(
                torch.randn(shape, generator=generator, dtype=torch.float32).to(
                    dtype=dtype
                ),
                requires_grad=False,
            )
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            bindings[layer] = LayerParameterBinding(layer, name, parameter)
            nominal_contexts[layer] = ShadowAlphaLayerContext(
                layer=layer,
                weight_name=name,
                parameter_shape=shape,
                projector32=torch.eye(4, dtype=torch.float32),
                committed_covariance32=0.2 * torch.eye(4, dtype=torch.float32),
                regularization32=torch.tensor(0.3, dtype=torch.float32),
                q_residual_tolerance=1.0e-6,
                q_condition_max_dimension=32,
                construction_id=f"nominal-authoritative-fixture-{layer}",
            )
        target = torch.tensor(
            [[0.9, 0.7], [0.4, 0.6], [-0.1, 0.2]],
            dtype=torch.float32,
        )
        nominal = run_uniform_nominal_shadow_probe(
            target,
            bindings,
            nominal_contexts,
            _PrefixProvider(bindings),
            transaction_id="authoritative-fixture-nominal-shadow",
        )
        covariances = tuple(
            SealedPrevalidatedCovariance(
                factor.layer,
                torch.eye(factor.parameter_shape[1], dtype=torch.float32),
                f"authoritative-sealed-covariance-{factor.layer}",
            )
            for factor in nominal.factors
        )
        state = self.committed_state(nominal, covariances) if history else (
            initialize_committed_gross_load_state(
                tuple(
                    LayerCommittedStateAnchor(
                        covariance.layer,
                        covariance,
                        30.0 + covariance.layer,
                    )
                    for covariance in covariances
                ),
                state_id="authoritative-fixture-state",
            )
        )
        ledger = CommittedGrossLoadLedger(state)
        route = assemble_residual_reserve_pc_route(
            nominal,
            ledger.state,
            covariances,
        )
        authoritative_contexts = {
            layer: replace(
                context,
                construction_id=f"authoritative-fixture-{layer}",
            )
            for layer, context in nominal_contexts.items()
        }
        return (
            target,
            bindings,
            nominal,
            authoritative_contexts,
            ledger,
            route,
        )

    @staticmethod
    def committed_state(nominal, covariances):
        transaction_id = "prior-successful-commit"
        layers = []
        for factor, covariance in zip(
            nominal.factors,
            covariances,
            strict=True,
        ):
            committed = LowRankFP32Factor(
                factor.layer,
                factor.parameter_shape,
                0.15 * factor.left,
                factor.right,
                LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
            )
            dense = committed.left @ committed.right.T
            p_constant = float(
                torch.trace(dense @ covariance.value @ dense.T)
            )
            anchor = 30.0 + factor.layer
            precast_load = float(torch.sum(dense * dense)) / anchor
            layers.append(
                CommittedLayerState(
                    layer=factor.layer,
                    covariance=covariance,
                    pretrained_weight_norm_squared=anchor,
                    committed_precast_factors=(committed,),
                    structural_p_constant=p_constant,
                    cumulative_precast_lambda=precast_load,
                    cumulative_actual_post_storage_load_observation=(
                        1.2 * precast_load
                    ),
                    version=1,
                    committed_transaction_ids=(transaction_id,),
                    committed_construction_receipt_ids=(
                        f"prior-construction-{factor.layer}",
                    ),
                )
            )
        return CommittedGrossLoadState(
            state_id="authoritative-fixture-state",
            layers=tuple(layers),
            version=1,
            commit_count=1,
            committed_transaction_ids=(transaction_id,),
            last_transaction_id=transaction_id,
        )

    @staticmethod
    def parameter_snapshot(bindings):
        return {
            layer: (
                int(binding.parameter.data_ptr()),
                tensor_sha256(binding.parameter),
                binding.parameter.detach().clone(),
            )
            for layer, binding in bindings.items()
        }

    def assert_snapshot(self, bindings, snapshot):
        for layer, binding in bindings.items():
            pointer, sha256, value = snapshot[layer]
            self.assertEqual(int(binding.parameter.data_ptr()), pointer)
            self.assertEqual(tensor_sha256(binding.parameter), sha256)
            self.assertTrue(torch.equal(binding.parameter, value))

    def run_success(self, *, history=False, dtype=torch.bfloat16):
        fixture = self.base_fixture(history=history, dtype=dtype)
        target, bindings, nominal, contexts, ledger, route = fixture
        entry = self.parameter_snapshot(bindings)
        provider = _PrefixProvider(bindings)
        result = run_authoritative_residual_reserve_sweep(
            target,
            nominal,
            route,
            bindings,
            contexts,
            provider,
            ledger,
            transaction_id=f"authoritative-success-{history}-{dtype}",
        )
        return result, fixture, provider, entry

    def test_success_exact_prefix_commit_and_derived_ledger(self):
        result, fixture, provider, entry = self.run_success()
        _, bindings, _, _, ledger, route = fixture
        self.assertEqual(provider.call_layers, list(LAYERS))
        self.assertEqual(result.receipt.status, AUTHORITATIVE_SWEEP_STATUS)
        self.assertEqual(result.receipt.route_assembly_identity, route.receipt.identity_sha256)
        self.assertEqual(result.receipt.committed_state_before_version, 0)
        self.assertEqual(result.receipt.committed_state_after_version, 1)
        self.assertEqual(ledger.state.version, 1)
        self.assertEqual(result.receipt.ledger.terminal_forward_count, 5)
        self.assertEqual(result.receipt.ledger.key_forward_count, 5)
        self.assertEqual(result.receipt.ledger.q_solve_count, 5)
        self.assertEqual(result.receipt.ledger.dense_update_construction_count, 5)
        self.assertEqual(result.receipt.ledger.exact_apply_count, 5)
        self.assertEqual(result.receipt.ledger.native_storage_assignment_count, 5)
        self.assertEqual(result.receipt.ledger.storage_cast_boundary_count, 5)
        self.assertEqual(result.receipt.ledger.logical_outer_commit_count, 1)
        self.assertEqual(result.receipt.ledger.ledger_commit_count, 1)
        self.assertEqual(result.receipt.ledger.successful_factor_append_count, 5)
        self.assertEqual(result.receipt.ledger.router_call_count, 0)
        self.assertEqual(result.receipt.ledger.inventory_build_count, 0)
        self.assertEqual(result.receipt.ledger.model_backward_count, 0)
        self.assertEqual(result.receipt.ledger.external_materializer_count, 0)
        for layer, layer_result in zip(LAYERS, result.layer_results, strict=True):
            expected = entry[layer][2].clone()
            with torch.no_grad():
                expected[...] = (
                    expected + layer_result.plan.prepared_update.matched_update32.float()
                )
            self.assertTrue(torch.equal(bindings[layer].parameter, expected))
            self.assertEqual(
                layer_result.receipt.factor_update_equivalence_status,
                EXACT_SINGLE_CONSTRUCTION_STATUS,
            )
            self.assertEqual(
                ledger.state.layers[layer - 4].committed_precast_factors[-1].source,
                LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
            )
        for index in range(1, 5):
            previous = result.layer_results[index - 1]
            current = result.layer_results[index]
            prior_state = next(
                item
                for item in current.observation.receipt.pre_observation_weight_state
                if item.layer == previous.receipt.layer
            )
            self.assertEqual(
                prior_state.sha256,
                previous.receipt.post_storage_parameter_sha256,
            )

    def test_nonzero_history_advances_exactly_once(self):
        result, fixture, _, _ = self.run_success(history=True, dtype=torch.float32)
        _, _, _, _, ledger, _ = fixture
        self.assertEqual(result.receipt.committed_state_before_version, 1)
        self.assertEqual(result.receipt.committed_state_after_version, 2)
        self.assertEqual(ledger.state.version, 2)
        self.assertTrue(
            all(len(layer.committed_precast_factors) == 2 for layer in ledger.state.layers)
        )
        self.assertTrue(
            all(
                item.source
                is LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
                for layer in ledger.state.layers
                for item in layer.committed_precast_factors
            )
        )

    def test_observation_q_plan_apply_failures_restore_at_layers_4_6_8(self):
        for failure_kind in ("observation", "q", "plan", "apply"):
            for fail_layer in (4, 6, 8):
                with self.subTest(kind=failure_kind, layer=fail_layer):
                    target, bindings, nominal, contexts, ledger, route = (
                        self.base_fixture()
                    )
                    snapshot = self.parameter_snapshot(bindings)
                    state_identity = ledger.state.identity_sha256
                    provider = _PrefixProvider(
                        bindings,
                        fail_layer=(
                            fail_layer if failure_kind == "observation" else None
                        ),
                    )

                    original_q = (
                        layer_step_module.solve_residual_independent_alpha_q_fp32
                    )
                    original_plan = sweep_module.plan_residual_reserve_layer_step
                    original_apply = sweep_module.apply_prepared_low_rank_update

                    def q_side_effect(*args, **kwargs):
                        if kwargs["layer"] == fail_layer:
                            raise RuntimeError("injected q failure")
                        return original_q(*args, **kwargs)

                    def plan_side_effect(*args, **kwargs):
                        if kwargs["layer"] == fail_layer:
                            raise RuntimeError("injected plan failure")
                        return original_plan(*args, **kwargs)

                    def apply_side_effect(transaction, construction):
                        if construction.receipt.layer == fail_layer:
                            raise RuntimeError("injected apply failure")
                        return original_apply(transaction, construction)

                    patches = []
                    if failure_kind == "q":
                        patches.append(
                            mock.patch.object(
                                layer_step_module,
                                "solve_residual_independent_alpha_q_fp32",
                                side_effect=q_side_effect,
                            )
                        )
                    if failure_kind == "plan":
                        patches.append(
                            mock.patch.object(
                                sweep_module,
                                "plan_residual_reserve_layer_step",
                                side_effect=plan_side_effect,
                            )
                        )
                    if failure_kind == "apply":
                        patches.append(
                            mock.patch.object(
                                sweep_module,
                                "apply_prepared_low_rank_update",
                                side_effect=apply_side_effect,
                            )
                        )
                    for patcher in patches:
                        patcher.start()
                    try:
                        with self.assertRaises(RuntimeError):
                            run_authoritative_residual_reserve_sweep(
                                target,
                                nominal,
                                route,
                                bindings,
                                contexts,
                                provider,
                                ledger,
                                transaction_id=(
                                    f"failure-{failure_kind}-{fail_layer}"
                                ),
                            )
                    finally:
                        for patcher in reversed(patches):
                            patcher.stop()
                    self.assert_snapshot(bindings, snapshot)
                    self.assertEqual(ledger.state.identity_sha256, state_identity)
                    self.assertEqual(ledger.state.version, 0)

    def test_pre_and_post_stage_corruption_restore_w_and_ledger(self):
        for location in ("pre-stage", "post-stage"):
            with self.subTest(location=location):
                target, bindings, nominal, contexts, ledger, route = (
                    self.base_fixture()
                )
                snapshot = self.parameter_snapshot(bindings)
                state_identity = ledger.state.identity_sha256
                provider = _PrefixProvider(bindings)
                if location == "pre-stage":
                    original = ledger.stage_authoritative_commit

                    def corrupt(*args, **kwargs):
                        with torch.no_grad():
                            bindings[4].parameter.add_(1.0)
                        return original(*args, **kwargs)

                    patcher = mock.patch.object(
                        ledger,
                        "stage_authoritative_commit",
                        side_effect=corrupt,
                    )
                else:
                    original = ledger.commit_staged_with_transaction

                    def corrupt(*args, **kwargs):
                        with torch.no_grad():
                            bindings[4].parameter.add_(1.0)
                        return original(*args, **kwargs)

                    patcher = mock.patch.object(
                        ledger,
                        "commit_staged_with_transaction",
                        side_effect=corrupt,
                    )
                with patcher:
                    with self.assertRaises(ODEBFStateError):
                        run_authoritative_residual_reserve_sweep(
                            target,
                            nominal,
                            route,
                            bindings,
                            contexts,
                            provider,
                            ledger,
                            transaction_id=f"corruption-{location}",
                        )
                self.assert_snapshot(bindings, snapshot)
                self.assertEqual(ledger.state.identity_sha256, state_identity)
                self.assertEqual(ledger.state.version, 0)

    def test_nested_compute_receipt_mutation_fails_before_commit(self):
        for mutation in ("plan_backward", "q_materialization"):
            with self.subTest(mutation=mutation):
                target, bindings, nominal, contexts, ledger, route = (
                    self.base_fixture()
                )
                snapshot = self.parameter_snapshot(bindings)
                state_identity = ledger.state.identity_sha256
                original_plan = sweep_module.plan_residual_reserve_layer_step

                def mutate(*args, **kwargs):
                    plan = original_plan(*args, **kwargs)
                    if kwargs["layer"] != 6:
                        return plan
                    if mutation == "plan_backward":
                        return replace(
                            plan,
                            receipt=replace(
                                plan.receipt,
                                model_backward_count=1,
                            ),
                        )
                    return replace(
                        plan,
                        q_solve=replace(
                            plan.q_solve,
                            receipt=replace(
                                plan.q_solve.receipt,
                                materialization_count=1,
                            ),
                        ),
                    )

                with mock.patch.object(
                    sweep_module,
                    "plan_residual_reserve_layer_step",
                    side_effect=mutate,
                ):
                    with self.assertRaises(ODEBFStateError):
                        run_authoritative_residual_reserve_sweep(
                            target,
                            nominal,
                            route,
                            bindings,
                            contexts,
                            _PrefixProvider(bindings),
                            ledger,
                            transaction_id=f"nested-mutation-{mutation}",
                        )
                self.assert_snapshot(bindings, snapshot)
                self.assertEqual(ledger.state.identity_sha256, state_identity)
                self.assertEqual(ledger.state.version, 0)

    def test_route_and_target_provenance_fail_before_native_mutation(self):
        target, bindings, nominal, contexts, ledger, route = self.base_fixture()
        snapshot = self.parameter_snapshot(bindings)
        forged_route = replace(
            route,
            selected_pi=torch.tensor(
                [0.1, 0.1, 0.1, 0.1, 0.6], dtype=torch.float32
            ),
        )
        with self.assertRaises(ODEBFStateError):
            run_authoritative_residual_reserve_sweep(
                target,
                nominal,
                forged_route,
                bindings,
                contexts,
                _PrefixProvider(bindings),
                ledger,
                transaction_id="forged-route",
            )
        with self.assertRaises(ODEBFStateError):
            run_authoritative_residual_reserve_sweep(
                target.detach().clone(),
                nominal,
                route,
                bindings,
                contexts,
                _PrefixProvider(bindings),
                ledger,
                transaction_id="forged-target-pointer",
            )
        self.assert_snapshot(bindings, snapshot)
        self.assertEqual(ledger.state.version, 0)

    def test_precommit_ledger_derivation_failure_restores_w_and_ledger(self):
        target, bindings, nominal, contexts, ledger, route = self.base_fixture()
        snapshot = self.parameter_snapshot(bindings)
        before = ledger.state
        before_identity = before.identity_sha256
        with mock.patch.object(
            sweep_module,
            "_derive_ledger",
            side_effect=RuntimeError("injected precommit ledger derivation failure"),
        ):
            with self.assertRaises(RuntimeError):
                run_authoritative_residual_reserve_sweep(
                    target,
                    nominal,
                    route,
                    bindings,
                    contexts,
                    _PrefixProvider(bindings),
                    ledger,
                    transaction_id="derive-ledger-failure",
                )
        self.assert_snapshot(bindings, snapshot)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.identity_sha256, before_identity)
        self.assertEqual(ledger.state.version, 0)

    def test_postcommit_has_no_ledger_derivation_or_hash_work(self):
        target, bindings, nominal, contexts, ledger, route = self.base_fixture()
        committed = False
        original_commit = ledger.commit_staged_with_transaction
        original_derive = sweep_module._derive_ledger
        original_hash = sweep_module.canonical_hash

        def commit_then_mark(*args, **kwargs):
            nonlocal committed
            result = original_commit(*args, **kwargs)
            committed = True
            return result

        def derive_before_commit(*args, **kwargs):
            self.assertFalse(committed)
            return original_derive(*args, **kwargs)

        def hash_before_commit(*args, **kwargs):
            self.assertFalse(committed)
            return original_hash(*args, **kwargs)

        with (
            mock.patch.object(
                ledger,
                "commit_staged_with_transaction",
                side_effect=commit_then_mark,
            ),
            mock.patch.object(
                sweep_module,
                "_derive_ledger",
                side_effect=derive_before_commit,
            ),
            mock.patch.object(
                sweep_module,
                "canonical_hash",
                side_effect=hash_before_commit,
            ),
        ):
            result = run_authoritative_residual_reserve_sweep(
                target,
                nominal,
                route,
                bindings,
                contexts,
                _PrefixProvider(bindings),
                ledger,
                transaction_id="postcommit-zero-fallible",
            )
        self.assertTrue(committed)
        self.assertEqual(result.receipt.committed_state_after_version, 1)

    def test_final_context_mutations_after_layer8_fail_before_prepare(self):
        for context_field in (
            "projector32",
            "committed_covariance32",
            "regularization32",
        ):
            with self.subTest(field=context_field):
                target, bindings, nominal, contexts, ledger, route = (
                    self.base_fixture()
                )
                snapshot = self.parameter_snapshot(bindings)
                before = ledger.state
                before_identity = before.identity_sha256
                original_apply = sweep_module.apply_prepared_low_rank_update

                def mutate_after_layer8(transaction, construction):
                    application = original_apply(transaction, construction)
                    if construction.receipt.layer == 8:
                        with torch.no_grad():
                            getattr(contexts[8], context_field).add_(1.0)
                    return application

                with mock.patch.object(
                    sweep_module,
                    "apply_prepared_low_rank_update",
                    side_effect=mutate_after_layer8,
                ):
                    with self.assertRaises(ODEBFStateError):
                        run_authoritative_residual_reserve_sweep(
                            target,
                            nominal,
                            route,
                            bindings,
                            contexts,
                            _PrefixProvider(bindings),
                            ledger,
                            transaction_id=f"final-context-{context_field}",
                        )
                self.assert_snapshot(bindings, snapshot)
                self.assertIs(ledger.state, before)
                self.assertEqual(ledger.state.identity_sha256, before_identity)
                self.assertEqual(ledger.state.version, 0)

    def test_forged_wrapper_telemetry_fails_precommit_and_rolls_back(self):
        for field in (
            "terminal_sha256",
            "actual_post_storage_delta_energy",
            "construction_orientation",
        ):
            with self.subTest(field=field):
                target, bindings, nominal, contexts, ledger, route = (
                    self.base_fixture()
                )
                snapshot = self.parameter_snapshot(bindings)
                before = ledger.state
                before_identity = before.identity_sha256
                receipt_type = sweep_module.AuthoritativeSweepLayerReceipt

                def forge_receipt(**kwargs):
                    receipt = receipt_type(**kwargs)
                    if kwargs["layer"] != 6:
                        return receipt
                    if field == "terminal_sha256":
                        return replace(receipt, terminal_sha256="0" * 64)
                    if field == "actual_post_storage_delta_energy":
                        return replace(
                            receipt,
                            actual_post_storage_delta_energy=(
                                receipt.actual_post_storage_delta_energy + 1.0
                            ),
                        )
                    replacement = (
                        "TRANSPOSE_VIEW"
                        if receipt.construction_orientation == "DIRECT"
                        else "DIRECT"
                    )
                    return replace(
                        receipt,
                        construction_orientation=replacement,
                    )

                with mock.patch.object(
                    sweep_module,
                    "AuthoritativeSweepLayerReceipt",
                    side_effect=forge_receipt,
                ):
                    with self.assertRaises(ODEBFStateError):
                        run_authoritative_residual_reserve_sweep(
                            target,
                            nominal,
                            route,
                            bindings,
                            contexts,
                            _PrefixProvider(bindings),
                            ledger,
                            transaction_id=f"forged-wrapper-{field}",
                        )
                self.assert_snapshot(bindings, snapshot)
                self.assertIs(ledger.state, before)
                self.assertEqual(ledger.state.identity_sha256, before_identity)
                self.assertEqual(ledger.state.version, 0)

    def test_prohibited_paths_and_authoritative_compute_boundary(self):
        source = inspect.getsource(sweep_module)
        prohibited = (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_pir",
            "p1r52_fpiq",
            "solve_residual_reserve_pc_router",
            "build_residual_reserve_pc_inventory",
        )
        for symbol in prohibited:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)


if __name__ == "__main__":
    unittest.main()
