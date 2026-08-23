from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
    P1R24_H,
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
from project.run_scripts.ode_bf.p1r52_target_timescale import (
    SCHEDULES,
    TargetTimescaleCell,
    reconstruct_target_subcycle_final_selected,
    run_target_subcycle_scheduler,
    schedule_for_cell,
)
from project.run_scripts.ode_bf.p1r52_target_timescale_b100 import (
    TECHNICAL_ATTEMPT_SUFFIX,
    expected_result_name,
    role_for_cell,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(values: tuple[float, ...], gradient: torch.Tensor | None) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, int(gradient is not None), 10, 10, gradient, None,
        "4" * 64, "5" * 64, "6" * 64,
    )


def kl_result(values: tuple[float, ...], gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(
        sum(values) / len(values), values, gradient, 1, 1,
        len(values), len(values), "7" * 64,
    )


class TargetTimescaleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float32)
        self.terminal = self.entry.clone()
        self.gradient = torch.tensor(
            [[0.125, -0.25], [0.25, 0.125]], dtype=torch.float32
        )
        self.nll = objective((1.0, 2.0), self.gradient)
        self.kl = kl_result((0.0, 0.0), torch.zeros_like(self.gradient))
        self.endpoint = objective((0.1, 0.2), None)
        self.lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")

    def run_schedule(self, cell: TargetTimescaleCell, *, fixed=None):
        return run_target_subcycle_scheduler(
            outer_step_index=0,
            schedule=schedule_for_cell(cell),
            current_target=self.entry,
            current_terminal=self.terminal,
            target_origin=self.entry,
            state=P1R51ControllerState.zero(self.entry),
            lock=self.lock,
            alias="llama3-8b-inst",
            shared_speed=0.4,
            teacher_sha256="8" * 64,
            evaluate_target=lambda _target: self.nll,
            evaluate_kl=lambda _target: self.kl,
            evaluate_endpoint=lambda _target, _role: self.endpoint,
            fixed_state_identities=(
                fixed
                if fixed is not None
                else lambda: ("a" * 64, "b" * 64, "c" * 64)
            ),
            first_target_result=self.nll,
            first_kl_result=self.kl,
        )

    def test_exact_schedule_algebra(self) -> None:
        self.assertEqual(
            [
                (
                    item.cell.value,
                    float(item.target_horizon),
                    item.microsteps_per_outer,
                    float(item.target_dt),
                    item.total_field_evaluations,
                )
                for item in SCHEDULES
            ],
            [
                ("Z0-COARSE", 1.0, 1, 0.125, 8),
                ("Z1-REFINE", 1.0, 2, 0.0625, 16),
                ("Z15", 1.5, 3, 0.0625, 24),
                ("Z20", 2.0, 4, 0.0625, 32),
                ("Z30", 3.0, 6, 0.0625, 48),
            ],
        )
        for item in SCHEDULES:
            self.assertEqual(
                8 * item.microsteps_per_outer * item.target_dt,
                item.target_horizon,
            )

    def test_technical_retry_uses_distinct_create_once_namespace(self) -> None:
        self.assertEqual(TECHNICAL_ATTEMPT_SUFFIX, "tech-r1")
        self.assertEqual(
            expected_result_name(role_for_cell(0)),
            "s05-p1r52-target-timescale-b100-z0-coarse-tech-r1-v1",
        )

    def test_z0_matches_legacy_p1r52_target_and_selection(self) -> None:
        state = P1R51ControllerState.zero(self.entry)
        legacy = prepare_p1r52_target_proposal(
            self.entry, self.terminal, self.entry, self.nll, self.kl,
            self.lock, state, alias="llama3-8b-inst", step_index=0,
            shared_speed=0.4, kl_teacher_input_sha256="8" * 64,
        )
        explicit = prepare_p1r52_target_proposal(
            self.entry, self.terminal, self.entry, self.nll, self.kl,
            self.lock, state, alias="llama3-8b-inst", step_index=0,
            shared_speed=0.4, kl_teacher_input_sha256="8" * 64,
            target_dt=P1R24_H,
        )
        self.assertTrue(torch.equal(legacy.primary_step.target_next, explicit.primary_step.target_next))
        self.assertEqual(legacy.receipt["identity_sha256"], explicit.receipt["identity_sha256"])
        legacy_rescue = prepare_p1r52_rescue_proposal(
            legacy, self.entry, self.terminal, self.nll, self.endpoint, step_index=0
        )
        legacy_selected = select_p1r52_target_proposal(
            legacy, legacy_rescue, self.entry, self.terminal, self.nll,
            self.endpoint, None, step_index=0,
        )
        observed = self.run_schedule(TargetTimescaleCell.Z0_COARSE)
        self.assertTrue(torch.equal(
            observed.target_step.target_next,
            legacy_selected.target_step.target_next,
        ))
        self.assertEqual(
            observed.microsteps[0].selected.receipt["identity_sha256"],
            legacy_selected.receipt["identity_sha256"],
        )

    def test_all_microsteps_execute_and_clocks_advance(self) -> None:
        zero = torch.zeros_like(self.gradient)
        self.nll = objective((0.0, 0.0), zero)
        self.kl = kl_result((0.0, 0.0), zero)
        self.endpoint = objective((0.0, 0.0), None)
        observed = self.run_schedule(TargetTimescaleCell.Z30)
        self.assertEqual(len(observed.microsteps), 6)
        self.assertEqual(
            [item.global_field_evaluation_ordinal for item in observed.microsteps],
            list(range(6)),
        )
        self.assertTrue(observed.receipt["all_configured_microsteps_executed"])
        self.assertEqual(observed.receipt["early_break_count"], 0)
        self.assertAlmostEqual(observed.receipt["target_time_after"], 0.375)

    def test_final_bridge_rejects_stale_target(self) -> None:
        observed = self.run_schedule(TargetTimescaleCell.Z15)
        selected, bridge = reconstruct_target_subcycle_final_selected(
            observed, step_index=0
        )
        self.assertTrue(torch.equal(selected.target_step.target_next, observed.target_step.target_next))
        self.assertEqual(bridge.intermediate_writer_authority_count, 0)
        stale_step = replace(observed.target_step, target_next=self.entry)
        with self.assertRaisesRegex(ODEBFStateError, "stale target"):
            reconstruct_target_subcycle_final_selected(
                replace(observed, target_step=stale_step), step_index=0
            )

    def test_inner_fixed_state_mutation_fails_closed(self) -> None:
        calls = 0

        def fixed():
            nonlocal calls
            calls += 1
            return (
                ("d" if calls >= 3 else "a") * 64,
                "b" * 64,
                "c" * 64,
            )

        with self.assertRaisesRegex(ODEBFStateError, "fixed state changed"):
            self.run_schedule(TargetTimescaleCell.Z15, fixed=fixed)

    def test_source_has_no_shared_speed_dt_emulation_or_early_break(self) -> None:
        source = inspect.getsource(run_target_subcycle_scheduler)
        self.assertIn("target_dt=target_dt", source)
        self.assertNotIn("shared_speed *", source)
        self.assertNotIn("\n            break", source)
        self.assertNotIn("materialize", source)
        self.assertNotIn("evaluate_heldout", source)

    def test_nonfinite_field_fails_closed(self) -> None:
        self.nll = objective(
            (1.0, 2.0),
            torch.tensor([[float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32),
        )
        with self.assertRaisesRegex(ODEBFContractError, "nonfinite"):
            self.run_schedule(TargetTimescaleCell.Z1_REFINE)


if __name__ == "__main__":
    unittest.main()
