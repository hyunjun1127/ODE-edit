from __future__ import annotations

from dataclasses import replace
import inspect
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_authoritative_sweep as sweep_module,
)
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_phase_a_adapter as adapter_module,
)
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_route_assembly as assembly_module,
)
from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24TargetStep
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    P1R52_INSTRUCTION_ID,
    P1R52_METHOD_ID,
    P1R52_REPAIR_REASON,
    P1R52_REPAIR_REVISION,
    P1R52SelectedTarget,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
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
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    SealedPrevalidatedCovariance,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_phase_a_adapter import (
    PHASE_A_ADAPTER_STATUS,
    ResidualReservePhaseAArm,
    freeze_residual_reserve_phase_a_action,
    run_residual_reserve_phase_a_outer,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    ScalableObjectiveResult,
)


LAYERS = RESIDUAL_RESERVE_LAYER_ORDER


class _PrefixProvider:
    def __init__(self, bindings, *, fail_call: int | None = None) -> None:
        self.bindings = bindings
        self.fail_call = fail_call
        self.calls = 0

    def __call__(self, layer, pre_state):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected Phase-A prefix failure")
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


class ResidualReservePhaseAAdapterTests(unittest.TestCase):
    @staticmethod
    def selected_target() -> P1R52SelectedTarget:
        target = torch.tensor(
            [[0.9, 0.7], [0.4, 0.6], [-0.1, 0.2]],
            dtype=torch.float32,
        )
        displacement = torch.tensor(
            [[0.1, 0.2], [0.2, 0.1], [0.05, -0.05]],
            dtype=torch.float32,
        )
        required = target - torch.tensor(
            [[0.1, 0.0], [0.2, 0.1], [-0.2, 0.1]],
            dtype=torch.float32,
        )
        payload = {
            "schema": "phase-a-selected-target-fixture/v1",
            "instruction_id": P1R52_INSTRUCTION_ID,
            "method_id": P1R52_METHOD_ID,
            "repair_revision": P1R52_REPAIR_REVISION,
            "repair_reason": P1R52_REPAIR_REASON,
            "k": 0,
            "target_next_sha256": tensor_sha256(target),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        step = P1R24TargetStep(
            target_next=target,
            target_displacement=displacement,
            required_displacement=required,
            write_velocity=(required / 0.125).contiguous(),
            nll_gradient=torch.ones_like(target, dtype=torch.float64),
            combined_gradient=2.0 * torch.ones_like(target, dtype=torch.float64),
            rho_write_signed=1.0,
            rho_write=1.0,
            alpha_target_signed=1.0,
            frozen_mask=(False, False),
            receipt=payload,
        )
        endpoint_identity = canonical_hash({"endpoint": "selected"})
        endpoint = ScalableObjectiveResult(
            loss=0.5,
            per_request_values=(0.4, 0.6),
            request_order_sha256=canonical_hash(["request-0", "request-1"]),
            target_span_sha256=canonical_hash(["span-0", "span-1"]),
            target_span_identities=(
                canonical_hash(["span-0"]),
                canonical_hash(["span-1"]),
            ),
            suffix_token_counts=(1, 1),
            model_forward_count=1,
            backward_count=0,
            processed_token_count=2,
            padded_token_count=2,
            target_gradient=None,
            coefficient_gradient=None,
            plan_sha256=canonical_hash(["plan"]),
            model_state_sha256=canonical_hash(["model-state"]),
            identity_sha256=endpoint_identity,
        )
        state = P1R51ControllerState(None, (0.1, 0.2))
        return P1R52SelectedTarget(step, endpoint, state, payload)

    @staticmethod
    def fixture():
        generator = torch.Generator().manual_seed(971)
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
                committed_covariance32=0.2 * torch.eye(4, dtype=torch.float32),
                regularization32=torch.tensor(0.3, dtype=torch.float32),
                q_residual_tolerance=1.0e-6,
                q_condition_max_dimension=32,
                construction_id=f"phase-a-adapter-{layer}",
            )
        covariances = tuple(
            SealedPrevalidatedCovariance(
                layer,
                torch.eye(
                    bindings[layer].parameter.shape[1],
                    dtype=torch.float32,
                ),
                f"phase-a-covariance-{layer}",
            )
            for layer in LAYERS
        )
        state = initialize_committed_gross_load_state(
            tuple(
                LayerCommittedStateAnchor(
                    covariance.layer,
                    covariance,
                    30.0 + covariance.layer,
                )
                for covariance in covariances
            ),
            state_id="phase-a-adapter-state",
        )
        return bindings, contexts, covariances, CommittedGrossLoadLedger(state)

    @staticmethod
    def snapshot(bindings):
        return {
            layer: (
                binding.parameter,
                int(binding.parameter.data_ptr()),
                tensor_sha256(binding.parameter),
                binding.parameter.detach().clone(),
            )
            for layer, binding in bindings.items()
        }

    def assert_snapshot(self, bindings, snapshot):
        for layer, binding in bindings.items():
            parameter, pointer, sha256, value = snapshot[layer]
            self.assertIs(binding.parameter, parameter)
            self.assertEqual(int(binding.parameter.data_ptr()), pointer)
            self.assertEqual(tensor_sha256(binding.parameter), sha256)
            self.assertTrue(torch.equal(binding.parameter, value))

    def run_arm(self, selected, arm, *, execution_id, action_freeze=None):
        bindings, contexts, covariances, ledger = self.fixture()
        if action_freeze is None:
            action_freeze = freeze_residual_reserve_phase_a_action(
                selected,
                bindings,
            )
        result = run_residual_reserve_phase_a_outer(
            selected,
            action_freeze,
            arm,
            bindings,
            contexts,
            _PrefixProvider(bindings),
            ledger,
            covariances,
            execution_id=execution_id,
        )
        return result, bindings, ledger

    def test_same_selected_target_and_entry_feed_independent_arm_modes(self):
        selected = self.selected_target()
        freeze_bindings, _, _, _ = self.fixture()
        shared_freeze = freeze_residual_reserve_phase_a_action(
            selected,
            freeze_bindings,
        )
        original_solver = assembly_module.solve_residual_reserve_pc_router
        with mock.patch.object(
            assembly_module,
            "solve_residual_reserve_pc_router",
            wraps=original_solver,
        ) as solver:
            uniform, uniform_bindings, uniform_ledger = self.run_arm(
                selected,
                ResidualReservePhaseAArm.RR_UNIFORM,
                execution_id="phase-a-uniform",
                action_freeze=shared_freeze,
            )
            self.assertEqual(solver.call_count, 0)
            pcsoft, pcsoft_bindings, pcsoft_ledger = self.run_arm(
                selected,
                ResidualReservePhaseAArm.RR_PCSOFT,
                execution_id="phase-a-pcsoft",
                action_freeze=shared_freeze,
            )
            self.assertEqual(solver.call_count, 1)
        self.assertEqual(uniform.receipt.status, PHASE_A_ADAPTER_STATUS)
        self.assertEqual(
            uniform.receipt.selected_target_binding.identity_sha256,
            pcsoft.receipt.selected_target_binding.identity_sha256,
        )
        self.assertEqual(
            uniform.receipt.action_freeze_identity,
            pcsoft.receipt.action_freeze_identity,
        )
        self.assertEqual(
            uniform.receipt.entry_weight_scientific_identity,
            pcsoft.receipt.entry_weight_scientific_identity,
        )
        self.assertEqual(uniform.receipt.route_solver_call_count, 0)
        self.assertEqual(pcsoft.receipt.route_solver_call_count, 1)
        self.assertEqual(
            pcsoft.route.receipt.router_selected_status,
            pcsoft.route.routing.receipt.selected_status,
        )
        self.assertEqual(
            pcsoft.route.receipt.selected_pi,
            pcsoft.route.receipt.router_selected_pi,
        )
        self.assertEqual(
            tensor_sha256(pcsoft.route.selected_pi),
            tensor_sha256(pcsoft.route.routing.pi_balanced),
        )
        self.assertEqual(
            tuple(float(item) for item in uniform.route.selected_pi),
            (float(torch.tensor(0.2, dtype=torch.float32)),) * 5,
        )
        self.assertEqual(uniform_ledger.state.version, 1)
        self.assertEqual(pcsoft_ledger.state.version, 1)
        self.assertTrue(
            all(
                len(layer.committed_precast_factors) == 1
                for layer in uniform_ledger.state.layers
            )
        )
        self.assertTrue(
            all(
                uniform_bindings[layer].parameter is not pcsoft_bindings[layer].parameter
                for layer in LAYERS
            )
        )

    def test_compute_action_freeze_and_no_history_or_heldout(self):
        result, _, _ = self.run_arm(
            self.selected_target(),
            ResidualReservePhaseAArm.RR_PCSOFT,
            execution_id="phase-a-compute",
        )
        compute = result.receipt.compute
        self.assertEqual(compute.shadow_terminal_forward_count, 5)
        self.assertEqual(compute.shadow_key_forward_count, 5)
        self.assertEqual(compute.shadow_q_solve_count, 5)
        self.assertEqual(compute.shadow_temporary_native_apply_count, 5)
        self.assertEqual(compute.shadow_restore_count, 1)
        self.assertEqual(compute.authoritative_terminal_forward_count, 5)
        self.assertEqual(compute.authoritative_key_forward_count, 5)
        self.assertEqual(compute.authoritative_q_solve_count, 5)
        self.assertEqual(compute.authoritative_dense_update_construction_count, 5)
        self.assertEqual(compute.authoritative_exact_apply_count, 5)
        self.assertEqual(compute.authoritative_native_storage_assignment_count, 5)
        self.assertEqual(compute.authoritative_storage_cast_boundary_count, 5)
        self.assertEqual(compute.authoritative_logical_outer_commit_count, 1)
        self.assertEqual(compute.authoritative_ledger_commit_count, 1)
        self.assertEqual(compute.route_solver_call_count, 1)
        self.assertEqual(compute.model_backward_count, 0)
        self.assertEqual(compute.semantic_backward_count, 0)
        self.assertEqual(compute.slope_backward_count, 0)
        self.assertEqual(compute.heldout_evaluator_count, 0)
        self.assertEqual(compute.alpha_history_consume_count, 0)
        self.assertEqual(compute.alpha_history_append_count, 0)
        self.assertEqual(compute.alpha_history_finalize_count, 0)
        self.assertEqual(compute.target_recompute_count, 0)
        self.assertEqual(result.receipt.target_controller_decision_influence_count, 0)

    def test_shadow_route_authoritative_and_precommit_failures_restore(self):
        for failure in ("shadow", "route", "authoritative", "precommit"):
            with self.subTest(failure=failure):
                selected = self.selected_target()
                bindings, contexts, covariances, ledger = self.fixture()
                snapshot = self.snapshot(bindings)
                before = ledger.state
                provider = _PrefixProvider(
                    bindings,
                    fail_call=8 if failure == "authoritative" else None,
                )
                patches = []
                if failure == "shadow":
                    def fail_shadow(*args, **kwargs):
                        with torch.no_grad():
                            bindings[4].parameter.add_(1.0)
                        raise RuntimeError("injected shadow failure")

                    patches.append(
                        mock.patch.object(
                            adapter_module,
                            "run_uniform_nominal_shadow_probe",
                            side_effect=fail_shadow,
                        )
                    )
                if failure == "route":
                    def fail_route(*args, **kwargs):
                        with torch.no_grad():
                            bindings[5].parameter.add_(1.0)
                        raise RuntimeError("injected route failure")

                    patches.append(
                        mock.patch.object(
                            adapter_module,
                            "assemble_residual_reserve_uniform_route",
                            side_effect=fail_route,
                        )
                    )
                if failure == "precommit":
                    patches.append(
                        mock.patch.object(
                            sweep_module,
                            "_derive_ledger",
                            side_effect=RuntimeError("injected precommit failure"),
                        )
                    )
                for patcher in patches:
                    patcher.start()
                try:
                    with self.assertRaises(RuntimeError):
                        run_residual_reserve_phase_a_outer(
                            selected,
                            freeze_residual_reserve_phase_a_action(
                                selected,
                                bindings,
                            ),
                            ResidualReservePhaseAArm.RR_UNIFORM,
                            bindings,
                            contexts,
                            provider,
                            ledger,
                            covariances,
                            execution_id=f"phase-a-failure-{failure}",
                        )
                finally:
                    for patcher in reversed(patches):
                        patcher.stop()
                self.assert_snapshot(bindings, snapshot)
                self.assertIs(ledger.state, before)
                self.assertEqual(ledger.state.version, 0)

    def test_target_controller_mutation_during_shadow_fails_and_restores(self):
        selected = self.selected_target()
        bindings, contexts, covariances, ledger = self.fixture()
        snapshot = self.snapshot(bindings)
        before = ledger.state
        base = _PrefixProvider(bindings)

        def mutate_target(layer, pre_state):
            observation = base(layer, pre_state)
            if layer == 5:
                selected.receipt["k"] = 1
            return observation

        with self.assertRaises(ODEBFStateError):
            run_residual_reserve_phase_a_outer(
                selected,
                freeze_residual_reserve_phase_a_action(selected, bindings),
                ResidualReservePhaseAArm.RR_UNIFORM,
                bindings,
                contexts,
                mutate_target,
                ledger,
                covariances,
                execution_id="phase-a-target-mutation",
            )
        self.assert_snapshot(bindings, snapshot)
        self.assertIs(ledger.state, before)

    def test_cross_arm_receipt_and_stale_target_fail_before_authoritative(self):
        selected = self.selected_target()
        bindings, contexts, covariances, ledger = self.fixture()
        snapshot = self.snapshot(bindings)
        before = ledger.state
        original_uniform = adapter_module.assemble_residual_reserve_uniform_route

        def mixed_route(nominal, state, sealed_covariances):
            uniform = original_uniform(nominal, state, sealed_covariances)
            pcsoft = assembly_module.assemble_residual_reserve_pc_route(
                nominal,
                state,
                sealed_covariances,
            )
            return replace(uniform, receipt=pcsoft.receipt)

        with mock.patch.object(
            adapter_module,
            "assemble_residual_reserve_uniform_route",
            side_effect=mixed_route,
        ):
            with self.assertRaises(ODEBFStateError):
                run_residual_reserve_phase_a_outer(
                    selected,
                    freeze_residual_reserve_phase_a_action(selected, bindings),
                    ResidualReservePhaseAArm.RR_UNIFORM,
                    bindings,
                    contexts,
                    _PrefixProvider(bindings),
                    ledger,
                    covariances,
                    execution_id="phase-a-cross-arm",
                )
        self.assert_snapshot(bindings, snapshot)
        self.assertIs(ledger.state, before)

    def test_stale_entry_action_freeze_rejects_before_shadow(self):
        selected = self.selected_target()
        bindings, contexts, covariances, ledger = self.fixture()
        action_freeze = freeze_residual_reserve_phase_a_action(
            selected,
            bindings,
        )
        with torch.no_grad():
            bindings[4].parameter.add_(1.0)
        stale_snapshot = self.snapshot(bindings)
        before = ledger.state
        with mock.patch.object(
            adapter_module,
            "run_uniform_nominal_shadow_probe",
            side_effect=AssertionError("stale entry must fail before shadow"),
        ) as shadow:
            with self.assertRaises(ODEBFStateError):
                run_residual_reserve_phase_a_outer(
                    selected,
                    action_freeze,
                    ResidualReservePhaseAArm.RR_UNIFORM,
                    bindings,
                    contexts,
                    _PrefixProvider(bindings),
                    ledger,
                    covariances,
                    execution_id="phase-a-stale-entry",
                )
        self.assertEqual(shadow.call_count, 0)
        self.assert_snapshot(bindings, stale_snapshot)
        self.assertIs(ledger.state, before)

    def test_stale_selected_target_rejects_before_shadow(self):
        selected = self.selected_target()
        bindings, contexts, covariances, ledger = self.fixture()
        action_freeze = freeze_residual_reserve_phase_a_action(
            selected,
            bindings,
        )
        stale_step = replace(
            selected.target_step,
            target_next=(selected.target_step.target_next + 1.0).contiguous(),
        )
        stale = replace(selected, target_step=stale_step)
        snapshot = self.snapshot(bindings)
        before = ledger.state
        with mock.patch.object(
            adapter_module,
            "run_uniform_nominal_shadow_probe",
            side_effect=AssertionError("stale target must fail before shadow"),
        ) as shadow:
            with self.assertRaises(ODEBFStateError):
                run_residual_reserve_phase_a_outer(
                    stale,
                    action_freeze,
                    ResidualReservePhaseAArm.RR_UNIFORM,
                    bindings,
                    contexts,
                    _PrefixProvider(bindings),
                    ledger,
                    covariances,
                    execution_id="phase-a-stale-target",
                )
        self.assertEqual(shadow.call_count, 0)
        self.assert_snapshot(bindings, snapshot)
        self.assertIs(ledger.state, before)

    def test_prohibited_paths_and_adapter_only_compute_boundary(self):
        source = inspect.getsource(adapter_module)
        prohibited = (
            "P1HistoryLedger",
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_pir",
            "p1r52_fpiq",
            "heldout_evaluator(",
        )
        for symbol in prohibited:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, source)


if __name__ == "__main__":
    unittest.main()
