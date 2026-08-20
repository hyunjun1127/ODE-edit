from __future__ import annotations

import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p1r39_normalized_gradient_target import (
    _full_current_residual_step,
)
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    prepare_p1r52_rescue_proposal,
    prepare_p1r52_target_proposal,
    select_p1r52_target_proposal,
)
from project.run_scripts.ode_bf.p1r52_independent_runtime import (
    expected_p1r52_result_name,
)
from project.run_scripts.ode_bf.p1r52_target_depth import (
    P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS,
    P1R52_TARGET_DEPTH_INNER_COUNTS,
    P1R52TargetDepth,
    P1R52TargetDepthInner,
    reassemble_p1r52_outer_target,
    requestwise_byte_identical_mask,
    run_p1r52_target_depth_scheduler,
)
from project.run_scripts.ode_bf.p1r52_target_depth_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r52_target_depth_extension import (
    EXTENSION_DEPTHS,
    extension_cells,
    extension_result_name,
    validate_extension_depths,
)
from project.run_scripts.ode_bf.p1r52_target_depth_extension_panel import (
    LOCK_FILE as EXTENSION_LOCK_FILE,
    load_and_validate_lock as load_and_validate_extension_lock,
)
from project.run_scripts.ode_bf.p1r52_target_depth_target_only_runtime import (
    expected_p1r52_target_depth_result_name,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(
    values: tuple[float, ...], gradient: torch.Tensor | None
) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values),
        values,
        "1" * 64,
        "2" * 64,
        tuple("3" * 64 for _ in values),
        tuple(1 for _ in values),
        1,
        int(gradient is not None),
        10,
        10,
        gradient,
        None,
        "4" * 64,
        "5" * 64,
        "6" * 64,
    )


def kl_result(values: tuple[float, ...], gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(
        sum(values) / len(values),
        values,
        gradient,
        1,
        1,
        len(values),
        len(values),
        "7" * 64,
    )


class P1R52TargetDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float32)
        self.terminal = self.entry.clone()
        self.gradient = torch.tensor(
            [[0.125, -0.25], [0.25, 0.125]], dtype=torch.float32
        )

    def proposal(self, state: P1R51ControllerState, *, step_index: int):
        return prepare_p1r52_target_proposal(
            self.entry,
            self.terminal,
            self.entry,
            objective((1.0, 2.0), self.gradient),
            kl_result((0.0, 0.0), torch.zeros_like(self.gradient)),
            P1R24AliasTargetLock.for_alias("llama3-8b-inst"),
            state,
            alias="llama3-8b-inst",
            step_index=step_index,
            shared_speed=0.4,
            kl_teacher_input_sha256="8" * 64,
        )

    def inner(
        self,
        *,
        inner_index: int,
        target: torch.Tensor,
        entry: torch.Tensor,
        state: P1R51ControllerState,
        outer_step: int = 0,
        depth: int = 3,
    ) -> P1R52TargetDepthInner:
        step = _full_current_residual_step(
            target_next=target,
            current_target=entry,
            current_terminal=self.terminal,
            nll_gradient=self.gradient,
            aggregate_gradient=self.gradient,
            held=tuple(False for _ in range(target.shape[1])),
            receipt={
                "selection_by_request": ["PRIMARY"] * target.shape[1],
                "entry_norm_calibration_count": int(inner_index == 0),
            },
        )
        return P1R52TargetDepthInner(
            inner_index,
            outer_step * depth + inner_index,
            step,
            objective((0.9, 1.8), None),
            state,
        )

    def reassemble(self, depth: P1R52TargetDepth, inners):
        return reassemble_p1r52_outer_target(
            outer_step_index=0,
            depth=depth,
            outer_entry_target=self.entry,
            current_terminal=self.terminal,
            inner_steps=inners,
            outer_entry_nll_gradient=self.gradient,
            outer_entry_aggregate_gradient=self.gradient,
            physical_state_sha256_before="a" * 64,
            physical_state_sha256_after_inners="a" * 64,
            teacher_sha256_before="b" * 64,
            teacher_sha256_after_inners="b" * 64,
            history_cache_sha256_before="c" * 64,
            history_cache_sha256_after_inners="c" * 64,
            factor_inventory_sha256_before="d" * 64,
            factor_inventory_sha256_after_inners="d" * 64,
        )

    def test_depth_policy_is_locked_to_original_and_extension_values(self) -> None:
        self.assertEqual(P1R52_TARGET_DEPTH_INNER_COUNTS, (1, 3, 8, 10, 15))
        self.assertEqual(P1R52TargetDepth.from_inner_count(1), P1R52TargetDepth.IL1)
        self.assertEqual(P1R52TargetDepth.from_inner_count(3), P1R52TargetDepth.IL3_FULL)
        self.assertEqual(P1R52TargetDepth.from_inner_count(5), P1R52TargetDepth.IL5_FULL)
        self.assertEqual(P1R52TargetDepth.from_inner_count(8), P1R52TargetDepth.IL8_FULL)
        self.assertEqual(P1R52TargetDepth.from_inner_count(10), P1R52TargetDepth.IL10_FULL)
        self.assertEqual(P1R52TargetDepth.from_inner_count(15), P1R52TargetDepth.IL15_FULL)
        self.assertEqual(P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS, (5,))

    def test_extension_list_has_no_il5_and_exact_result_names(self) -> None:
        self.assertEqual(validate_extension_depths(EXTENSION_DEPTHS), EXTENSION_DEPTHS)
        self.assertEqual([item.inner_count for item in EXTENSION_DEPTHS], [8, 10, 15])
        self.assertEqual(
            [item["depth"] for item in extension_cells()],
            ["IL8-FULL", "IL10-FULL", "IL15-FULL"],
        )
        self.assertEqual(
            extension_result_name(P1R52TargetDepth.IL15_FULL),
            "s05-p1r52-target-depth-atomic-b10x10-llama3-8b-inst-soft-il15full-extension-inner-telemetry-tech-r1-v1",
        )
        self.assertEqual(
            extension_result_name(P1R52TargetDepth.IL15_FULL),
            expected_p1r52_result_name(
                "llama3-8b-inst",
                "soft",
                target_depth=P1R52TargetDepth.IL15_FULL,
                attempt_suffix="extension-inner-telemetry-tech-r1",
            ),
        )
        with self.assertRaises(ODEBFContractError):
            validate_extension_depths((P1R52TargetDepth.IL8_FULL,))

    def test_extension_numerical_lock(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock, file_sha = load_and_validate_extension_lock(
            root / "locks" / EXTENSION_LOCK_FILE
        )
        self.assertEqual(
            lock["depth_policies"],
            {"IL8-FULL": 8, "IL10-FULL": 10, "IL15-FULL": 15},
        )
        self.assertEqual(lock["il5_action_count"], 0)
        self.assertEqual(
            lock["expected_inner_rows_per_case"],
            {"IL8-FULL": 64, "IL10-FULL": 80, "IL15-FULL": 120},
        )
        self.assertEqual(lock["expected_outer_rows_per_case"], 8)
        self.assertEqual(lock["inner_added_backward_count"], 0)
        self.assertEqual(lock["inner_added_generation_count"], 0)
        self.assertEqual(len(file_sha), 64)

    def test_numerical_lock_and_target_only_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock, file_sha = load_and_validate_lock(root / "locks" / LOCK_FILE)
        self.assertEqual(lock["depth_policies"], {"IL1": 1, "IL3-FULL": 3})
        self.assertEqual(lock["inner_writer_materialization_count"], 0)
        self.assertEqual(len(file_sha), 64)
        self.assertEqual(
            expected_p1r52_target_depth_result_name(
                "llama3-8b-inst", depth="IL3-FULL", case_count=10
            ),
            "s05-p1r52-target-depth-target-only-b10x10-llama3-8b-inst-il3full-v1",
        )

    def test_actual_runtime_routes_through_reusable_scheduler(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scalable = (root / "p1_scalable_batched_experiment.py").read_text(
            encoding="utf-8"
        )
        target_only = (root / "p1r52_target_depth_target_only_runtime.py").read_text(
            encoding="utf-8"
        )
        scheduler = (root / "p1r52_target_depth.py").read_text(encoding="utf-8")
        runtime = (root / "p1_runtime.py").read_text(encoding="utf-8")
        self.assertIn("run_p1r52_target_depth_scheduler(", scalable)
        self.assertIn("run_p1r52_target_depth_scheduler(", target_only)
        self.assertIn("run_p1r52_target_depth_target_only(", runtime)
        self.assertNotIn("h / 3", scheduler)
        self.assertNotIn("writer_materialize(", scheduler)
        self.assertNotIn("evaluate_heldout", scheduler)
        self.assertIn('"writer_materialization_count": 0', target_only)
        materialize = scalable.index("materializer.materialize(")
        state_commit = scalable.index("p1r52_state = p1r52_pending_state", materialize)
        self.assertLess(materialize, state_commit)
        self.assertIn("fixed_state_identities=fixed_depth_state", scalable)

    def test_t1_outer_reassembly_preserves_core_target_step(self) -> None:
        state = P1R51ControllerState.zero(self.entry)
        target = self.entry + 0.125
        inner = self.inner(inner_index=0, target=target, entry=self.entry, state=state, depth=1)
        result = self.reassemble(P1R52TargetDepth.IL1, [inner])
        self.assertTrue(torch.equal(result.target_step.target_next, inner.target_step.target_next))
        self.assertTrue(
            torch.equal(result.target_step.target_displacement, inner.target_step.target_displacement)
        )
        self.assertTrue(
            torch.equal(result.target_step.required_displacement, inner.target_step.required_displacement)
        )
        self.assertTrue(torch.equal(result.target_step.write_velocity, inner.target_step.write_velocity))
        self.assertEqual(result.target_step.rho_write_signed, inner.target_step.rho_write_signed)
        self.assertEqual(result.receipt["executed_inner_count"], 1)

    def test_t1_scheduler_matches_direct_existing_operator(self) -> None:
        target_result = objective((1.0, 2.0), self.gradient)
        kl = kl_result((0.0, 0.0), torch.zeros_like(self.gradient))
        endpoint = objective((0.1, 0.2), None)
        state = P1R51ControllerState.zero(self.entry)
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        direct_proposal = prepare_p1r52_target_proposal(
            self.entry,
            self.terminal,
            self.entry,
            target_result,
            kl,
            lock,
            state,
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
            kl_teacher_input_sha256="8" * 64,
        )
        direct_rescue = prepare_p1r52_rescue_proposal(
            direct_proposal,
            self.entry,
            self.terminal,
            target_result,
            endpoint,
            step_index=0,
        )
        direct = select_p1r52_target_proposal(
            direct_proposal,
            direct_rescue,
            self.entry,
            self.terminal,
            target_result,
            endpoint,
            None,
            step_index=0,
        )
        wrapped = run_p1r52_target_depth_scheduler(
            outer_step_index=0,
            depth=P1R52TargetDepth.IL1,
            current_target=self.entry,
            current_terminal=self.terminal,
            target_origin=self.entry,
            state=state,
            lock=lock,
            alias="llama3-8b-inst",
            shared_speed=0.4,
            teacher_sha256="8" * 64,
            evaluate_target=lambda _: target_result,
            evaluate_kl=lambda _: kl,
            evaluate_endpoint=lambda _target, _role: endpoint,
            fixed_state_identities=lambda: ("a" * 64, "b" * 64, "c" * 64),
        )
        self.assertTrue(
            torch.equal(wrapped.target_step.target_next, direct.target_step.target_next)
        )
        self.assertEqual(
            wrapped.inner_steps[0].target_step.receipt["identity_sha256"],
            direct.target_step.receipt["identity_sha256"],
        )
        self.assertEqual(
            wrapped.selected_endpoint.identity_sha256,
            direct.selected_endpoint.identity_sha256,
        )
        self.assertEqual(
            wrapped.next_state.entry_semantic_gradient_norm,
            direct.next_state.entry_semantic_gradient_norm,
        )
        self.assertEqual(
            wrapped.next_state.cumulative_accepted_activation_path,
            direct.next_state.cumulative_accepted_activation_path,
        )

    def test_t3_reassembles_outer_not_last_inner_displacement(self) -> None:
        state = P1R51ControllerState.zero(self.entry)
        first = self.entry + 0.125
        second = first + 0.25
        third = second - 0.0625
        inners = [
            self.inner(inner_index=0, target=first, entry=self.entry, state=state),
            self.inner(inner_index=1, target=second, entry=first, state=state),
            self.inner(inner_index=2, target=third, entry=second, state=state),
        ]
        result = self.reassemble(P1R52TargetDepth.IL3_FULL, inners)
        self.assertTrue(torch.equal(result.target_step.target_displacement, third - self.entry))
        self.assertTrue(torch.equal(result.target_step.required_displacement, third - self.terminal))
        self.assertNotEqual(
            result.receipt["last_inner_target_displacement_sha256"],
            result.receipt["outer_net_target_displacement_sha256"],
        )
        self.assertEqual(result.receipt["inner_rho_sum_count"], 0)
        self.assertEqual(result.receipt["inner_writer_materialization_count"], 0)
        self.assertEqual(result.receipt["finite_demand_builder_expected_count_per_outer"], 1)

    def test_full_tensor_and_requestwise_byte_identity(self) -> None:
        right = self.entry.clone()
        right[0, 1] = torch.nextafter(right[0, 1], torch.tensor(float("inf")))
        self.assertEqual(requestwise_byte_identical_mask(self.entry, right), (True, False))

    def test_calibration_uses_state_transition_not_step_index(self) -> None:
        zero = P1R51ControllerState.zero(self.entry)
        calibrated = self.proposal(zero, step_index=5)
        self.assertEqual(calibrated.receipt["entry_norm_calibration_count"], 1)
        self.assertEqual(
            calibrated.receipt["entry_norm_calibration_trigger"], "STATE_NONE_TO_VALUE"
        )
        repeated = self.proposal(calibrated.next_state, step_index=0)
        self.assertEqual(repeated.receipt["entry_norm_calibration_count"], 0)

    def test_fixed_outer_state_mismatch_fails_closed(self) -> None:
        state = P1R51ControllerState.zero(self.entry)
        inner = self.inner(inner_index=0, target=self.entry + 0.1, entry=self.entry, state=state, depth=1)
        with self.assertRaises(ODEBFContractError):
            reassemble_p1r52_outer_target(
                outer_step_index=0,
                depth=P1R52TargetDepth.IL1,
                outer_entry_target=self.entry,
                current_terminal=self.terminal,
                inner_steps=[inner],
                outer_entry_nll_gradient=self.gradient,
                outer_entry_aggregate_gradient=self.gradient,
                physical_state_sha256_before="a" * 64,
                physical_state_sha256_after_inners="e" * 64,
                teacher_sha256_before="b" * 64,
                teacher_sha256_after_inners="b" * 64,
                history_cache_sha256_before="c" * 64,
                history_cache_sha256_after_inners="c" * 64,
                factor_inventory_sha256_before="d" * 64,
                factor_inventory_sha256_after_inners="d" * 64,
            )

    def test_scheduler_runs_three_full_operators_at_same_outer_index(self) -> None:
        calls = {"target": 0, "kl": 0, "endpoint": 0, "fixed": 0}

        def evaluate_target(target: torch.Tensor):
            calls["target"] += 1
            values = (1.0 - 0.1 * calls["target"], 2.0 - 0.1 * calls["target"])
            return objective(values, self.gradient)

        def evaluate_kl(target: torch.Tensor):
            calls["kl"] += 1
            return kl_result((0.0, 0.0), torch.zeros_like(target))

        def evaluate_endpoint(target: torch.Tensor, role: str):
            calls["endpoint"] += 1
            return objective((0.1, 0.2), None)

        def fixed_state():
            calls["fixed"] += 1
            return "a" * 64, "b" * 64, "c" * 64

        result = run_p1r52_target_depth_scheduler(
            outer_step_index=4,
            depth=P1R52TargetDepth.IL3_FULL,
            current_target=self.entry,
            current_terminal=self.terminal,
            target_origin=self.entry,
            state=P1R51ControllerState.zero(self.entry),
            lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9),
            alias="llama3-8b-inst",
            shared_speed=0.4,
            teacher_sha256="8" * 64,
            evaluate_target=evaluate_target,
            evaluate_kl=evaluate_kl,
            evaluate_endpoint=evaluate_endpoint,
            fixed_state_identities=fixed_state,
        )
        self.assertEqual(calls, {"target": 3, "kl": 3, "endpoint": 3, "fixed": 2})
        self.assertEqual(
            [item.global_target_update_ordinal for item in result.inner_steps],
            [12, 13, 14],
        )
        self.assertEqual(
            [item.target_step.receipt["k"] for item in result.inner_steps], [4, 4, 4]
        )
        self.assertEqual(result.receipt["entry_norm_calibration_count"], 1)
        self.assertEqual(result.receipt["inner_writer_materialization_count"], 0)

    def test_il15_runs_120_updates_per_case_without_inner_writer(self) -> None:
        calls = {"target": 0, "kl": 0, "endpoint": 0, "fixed": 0}

        def evaluate_target(target: torch.Tensor):
            calls["target"] += 1
            value = 2.0 - 0.01 * calls["target"]
            return objective((value, value + 1.0), self.gradient)

        def evaluate_kl(target: torch.Tensor):
            calls["kl"] += 1
            return kl_result((0.0, 0.0), torch.zeros_like(target))

        def evaluate_endpoint(target: torch.Tensor, role: str):
            calls["endpoint"] += 1
            return objective((0.1, 0.2), None)

        def fixed_state():
            calls["fixed"] += 1
            return "a" * 64, "b" * 64, "c" * 64

        result = run_p1r52_target_depth_scheduler(
            outer_step_index=7,
            depth=P1R52TargetDepth.IL15_FULL,
            current_target=self.entry,
            current_terminal=self.terminal,
            target_origin=self.entry,
            state=P1R51ControllerState.zero(self.entry),
            lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9),
            alias="llama3-8b-inst",
            shared_speed=0.4,
            teacher_sha256="8" * 64,
            evaluate_target=evaluate_target,
            evaluate_kl=evaluate_kl,
            evaluate_endpoint=evaluate_endpoint,
            fixed_state_identities=fixed_state,
        )
        self.assertEqual(calls, {"target": 15, "kl": 15, "endpoint": 15, "fixed": 2})
        self.assertEqual(len(result.inner_steps), 15)
        self.assertEqual(
            [item.global_target_update_ordinal for item in result.inner_steps],
            list(range(105, 120)),
        )
        self.assertEqual(
            [item.target_step.receipt["k"] for item in result.inner_steps], [7] * 15
        )
        self.assertEqual(result.receipt["configured_inner_count"], 15)
        self.assertEqual(result.receipt["inner_writer_materialization_count"], 0)
        self.assertEqual(8 * result.receipt["configured_inner_count"], 120)

    def test_inner_telemetry_is_request_bound_and_observation_only(self) -> None:
        calls = {"target": 0, "kl": 0, "endpoint": 0, "observe": 0}

        def evaluate_target(target: torch.Tensor):
            calls["target"] += 1
            return objective((1.0, 2.0), self.gradient)

        def evaluate_kl(target: torch.Tensor):
            calls["kl"] += 1
            return kl_result((0.0, 0.0), torch.zeros_like(target))

        def evaluate_endpoint(target: torch.Tensor, role: str):
            calls["endpoint"] += 1
            return objective((0.1, 0.2), None)

        def observe_inner(**values):
            calls["observe"] += 1
            self.assertEqual(values["outer_step_index"], 2)
            self.assertEqual(values["configured_inner_count"], 8)
            return {
                "identity_sha256": f"{calls['observe']:064x}",
                "added_model_forward_count": 10,
                "added_processed_token_count": 20,
                "added_backward_count": 0,
                "added_generation_count": 0,
                "controller_action_influence_count": 0,
                "duplicate_evaluation_count": 0,
            }

        result = run_p1r52_target_depth_scheduler(
            outer_step_index=2,
            depth=P1R52TargetDepth.IL8_FULL,
            current_target=self.entry,
            current_terminal=self.terminal,
            target_origin=self.entry,
            state=P1R51ControllerState.zero(self.entry),
            lock=P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9),
            alias="llama3-8b-inst",
            shared_speed=0.4,
            teacher_sha256="8" * 64,
            evaluate_target=evaluate_target,
            evaluate_kl=evaluate_kl,
            evaluate_endpoint=evaluate_endpoint,
            fixed_state_identities=lambda: ("a" * 64, "b" * 64, "c" * 64),
            observe_inner=observe_inner,
        )
        self.assertEqual(calls, {"target": 8, "kl": 8, "endpoint": 8, "observe": 8})
        self.assertEqual(result.receipt["inner_heldout_access_count"], 8)
        self.assertEqual(result.receipt["inner_observation_pass_count"], 8)
        self.assertEqual(
            result.receipt["inner_observation_added_model_forward_count"], 80
        )
        self.assertEqual(result.receipt["inner_observation_added_backward_count"], 0)
        self.assertEqual(result.receipt["inner_observation_added_generation_count"], 0)
        self.assertEqual(result.receipt["inner_observation_action_influence_count"], 0)
        self.assertEqual(result.receipt["inner_observation_duplicate_evaluation_count"], 0)
        self.assertTrue(
            all(
                item["accepted_z_observation"] is not None
                for item in result.receipt["inner_trajectory"]
            )
        )

    def test_atomic_result_identity_is_depth_specific(self) -> None:
        self.assertEqual(
            expected_p1r52_result_name(
                "llama3-8b-inst", "soft", target_depth=P1R52TargetDepth.IL3_FULL
            ),
            "s05-p1r52-target-depth-atomic-b10x10-llama3-8b-inst-soft-il3full-v1",
        )
        with self.assertRaises(ODEBFContractError):
            expected_p1r52_result_name(
                "llama3-8b-inst", "pir-j0", target_depth=P1R52TargetDepth.IL3_FULL
            )
        self.assertEqual(
            expected_p1r52_result_name(
                "llama3-8b-inst", "soft", target_depth=P1R52TargetDepth.IL15_FULL
            ),
            "s05-p1r52-target-depth-atomic-b10x10-llama3-8b-inst-soft-il15full-v1",
        )

    def test_atomic_adapter_plumbs_depth_without_writer_variant(self) -> None:
        root = Path(__file__).resolve().parents[4]
        independent = (root / "project/run_scripts/ode_bf/p1r52_independent_runtime.py").read_text()
        shared = (root / "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py").read_text()
        sbatch = (root / "project/run_scripts/session05_ode_bf_p1r52_target_depth_atomic.sbatch").read_text()
        self.assertIn("p1r52_target_depth=depth_policy.inner_count", independent)
        self.assertIn("p1r52_target_depth=p1r52_target_depth", shared)
        self.assertIn('readonly DEPTH="IL3-FULL"', sbatch)
        self.assertIn('readonly ARMS=(neutral soft neutral soft)', sbatch)
        self.assertNotIn("h/3", sbatch)
        self.assertNotIn("writer_policy", sbatch)

    def test_extension_launcher_has_exact_depths_and_no_h_rescale(self) -> None:
        root = Path(__file__).resolve().parents[4]
        sbatch = (
            root
            / "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic.sbatch"
        ).read_text()
        dry_plan = (
            root
            / "project/run_scripts/session05_ode_bf_p1r52_target_depth_extension_atomic_dry_plan.py"
        ).read_text()
        submitter = (
            root
            / "project/run_scripts/session05_ode_bf_submit_p1r52_target_depth_extension_atomic.py"
        ).read_text()
        joined = "\n".join((sbatch, dry_plan, submitter))
        self.assertIn("DEPTHS=(IL8-FULL IL10-FULL IL15-FULL)", sbatch)
        self.assertIn("#SBATCH --array=0-2%3", sbatch)
        self.assertIn("PROJECT_GPU_CAP - active", submitter)
        self.assertIn("extension-inner-telemetry-tech-r1", sbatch)
        self.assertNotIn("IL5", joined)
        self.assertNotIn("h/3", joined)
        self.assertNotIn("h/depth", joined)

    def test_inner_telemetry_source_has_exact_no_influence_counters(self) -> None:
        root = Path(__file__).resolve().parents[4]
        observer = (
            root
            / "project/run_scripts/ode_bf/p1r52_target_depth_inner_telemetry.py"
        ).read_text()
        runtime = (
            root / "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py"
        ).read_text()
        self.assertIn('"target_objective_nll_by_request"', observer)
        self.assertIn('"accepted_z_rewrite_nll_by_request"', observer)
        self.assertIn('"accepted_z_rephrase_nll_by_request"', observer)
        self.assertIn('"inner_step_movement_norm_by_request"', observer)
        self.assertIn('"writer_w_rewrite_nll_by_request"', observer)
        self.assertIn('"rewrite_w_minus_z_nll_gap_by_request"', observer)
        self.assertIn('"added_backward_count": 0', observer)
        self.assertIn('"added_generation_count": 0', observer)
        self.assertIn('"controller_action_influence_count": 0', observer)
        self.assertIn('"duplicate_evaluation_count": 0', observer)
        self.assertIn('canonical_hash(geometry["residual"])', observer)
        self.assertNotIn('geometry["residual"]["identity_sha256"]', observer)
        self.assertIn("terminal_target_depth_telemetry_four_panel", runtime)

    def test_actual_runtime_activation_uses_the_locked_depth_count_registry(self) -> None:
        root = Path(__file__).resolve().parents[4]
        scalable = (
            root / "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py"
        ).read_text()
        self.assertIn("allowed_target_depth_counts", scalable)
        self.assertIn("p1r52_target_depth not in allowed_target_depth_counts", scalable)
        self.assertIn("P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS", scalable)
        self.assertNotIn("p1r52_target_depth not in (1, 3)", scalable)


if __name__ == "__main__":
    unittest.main()
