from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import P1R51ControllerState
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import P1R52AmplitudeContext, prepare_p1r52_target_proposal
from project.run_scripts.ode_bf.p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from project.run_scripts.ode_bf.p1r54_pdz_ablation import (
    ClockedPDZAmplitudePolicy,
    PDZAblationArm,
    PDZAblationSelection,
    P1R54ScientificBoundaryNTSM,
    TargetMicrostepClock,
    run_direct_euler_subcycle_scheduler,
    run_pdz_ablation_subcycle_scheduler,
    schedule_for_arm,
)
from project.run_scripts.ode_bf.p1r54_pdz_ablation_analysis import group_microsteps_by_outer
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def objective(values: tuple[float, ...], gradient: torch.Tensor | None) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, int(gradient is not None), 10, 10, gradient, None,
        "4" * 64, "5" * 64, "6" * 64,
    )


def kl_result(count: int, gradient: torch.Tensor) -> P1R24KLResult:
    return P1R24KLResult(0.0, tuple(0.0 for _ in range(count)), gradient, 1, 1, count, count, "7" * 64)


def context(step: int = 0) -> P1R52AmplitudeContext:
    semantic = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    norm = torch.linalg.vector_norm(semantic, dim=0)
    return P1R52AmplitudeContext(
        step, 9.0, semantic, norm, norm + 1.0,
        torch.tensor([0.25, 1.5], dtype=torch.float64),
        torch.ones(2, dtype=torch.bool), -semantic / norm.unsqueeze(0),
        torch.ones(2, dtype=torch.float64), 99.0,
        torch.tensor([1.5, 2.5], dtype=torch.float64),
    )


class P1R54PDZAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float32)
        self.terminal = self.entry.clone()
        self.gradient = torch.tensor([[0.125, -0.25], [0.25, 0.125]], dtype=torch.float32)
        self.nll = objective((1.0, 2.0), self.gradient)
        self.kl = kl_result(2, torch.zeros_like(self.gradient))
        self.good_endpoint = objective((0.5, 1.0), None)
        self.lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9)

    def _run(self, arm: PDZAblationArm, outer: int, policy: ClockedPDZAmplitudePolicy, *, fixed=None, endpoint=None):
        return run_pdz_ablation_subcycle_scheduler(
            outer_step_index=outer,
            schedule=schedule_for_arm(arm),
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
            evaluate_endpoint=lambda _target, _role: endpoint or self.good_endpoint,
            fixed_state_identities=fixed or (lambda: ("a" * 64, "b" * 64, "c" * 64)),
            first_target_result=self.nll,
            first_kl_result=self.kl,
            amplitude_policy=policy,
        )

    def test_clock_and_four_exact_arm_schedules(self) -> None:
        expected = {
            PDZAblationArm.PDZ_T2_PRC: (2, 2.0, 16, 15),
            PDZAblationArm.PDZ_T3_PRC: (3, 3.0, 24, 23),
            PDZAblationArm.PDZ_T5_PRC: (5, 5.0, 40, 39),
            PDZAblationArm.PDZ_T1_DIRECT: (1, 1.0, 8, 7),
        }
        for arm, (width, horizon, evaluations, ordinal) in expected.items():
            with self.subTest(arm=arm):
                schedule = schedule_for_arm(arm)
                clock = TargetMicrostepClock(
                    7, width - 1, ordinal, width, schedule.target_dt
                )
                self.assertEqual(clock.time_after, horizon)
                self.assertEqual(
                    (
                        schedule.microsteps_per_outer,
                        float(schedule.target_dt),
                        float(schedule.target_horizon),
                        schedule.total_field_evaluations,
                    ),
                    (width, 0.125, horizon, evaluations),
                )
        t2 = schedule_for_arm(PDZAblationArm.PDZ_T2_PRC)
        with self.assertRaises(ODEBFContractError):
            TargetMicrostepClock(0, 1, 0, 2, t2.target_dt)

    def test_legacy_pdz_m1_amplitude_and_proposal_are_byte_exact(self) -> None:
        old = EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.PDZ, request_count=2)(context())
        new = ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2)(context())
        self.assertTrue(torch.equal(old.amplitude, new.amplitude))
        common = dict(alias="llama3-8b-inst", step_index=0, shared_speed=0.4, kl_teacher_input_sha256="8" * 64, target_dt=0.125)
        old_proposal = prepare_p1r52_target_proposal(self.entry, self.terminal, self.entry, self.nll, self.kl, self.lock, P1R51ControllerState.zero(self.entry), amplitude_policy=EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.PDZ, request_count=2), **common)
        new_proposal = prepare_p1r52_target_proposal(self.entry, self.terminal, self.entry, self.nll, self.kl, self.lock, P1R51ControllerState.zero(self.entry), amplitude_policy=ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2), **common)
        self.assertTrue(torch.equal(old_proposal.primary_step.target_next, new_proposal.primary_step.target_next))
        self.assertEqual(old_proposal.receipt["primary_delta_sha256"], new_proposal.receipt["primary_delta_sha256"])

    def test_prc_chain_clock_selector_counts_rho_and_freeze(self) -> None:
        expected = {
            PDZAblationArm.PDZ_T2_PRC: 16,
            PDZAblationArm.PDZ_T3_PRC: 24,
            PDZAblationArm.PDZ_T5_PRC: 40,
        }
        for arm, count in expected.items():
            with self.subTest(arm=arm):
                policy = ClockedPDZAmplitudePolicy(arm, request_count=2)
                all_micro = []
                current = self.entry
                state = P1R51ControllerState.zero(self.entry)
                for outer in range(8):
                    observed = run_pdz_ablation_subcycle_scheduler(
                        outer_step_index=outer,
                        schedule=schedule_for_arm(arm),
                        current_target=current,
                        current_terminal=self.terminal,
                        target_origin=self.entry,
                        state=state,
                        lock=self.lock,
                        alias="llama3-8b-inst",
                        shared_speed=0.4,
                        teacher_sha256="8" * 64,
                        evaluate_target=lambda _target: self.nll,
                        evaluate_kl=lambda _target: self.kl,
                        evaluate_endpoint=lambda _target, _role: self.good_endpoint,
                        fixed_state_identities=lambda: (
                            "a" * 64,
                            "b" * 64,
                            "c" * 64,
                        ),
                        first_target_result=self.nll,
                        first_kl_result=self.kl,
                        amplitude_policy=policy,
                    )
                    for micro in range(1, len(observed.microsteps)):
                        self.assertEqual(
                            observed.microsteps[micro].receipt[
                                "entry_target_sha256"
                            ],
                            observed.microsteps[micro - 1].receipt[
                                "selected_target_sha256"
                            ],
                        )
                    all_micro.extend(item.receipt for item in observed.microsteps)
                    current = observed.target_step.target_next
                    state = observed.next_state
                groups = group_microsteps_by_outer(arm, all_micro)
                self.assertEqual((len(groups), len(groups[0])), (8, count // 8))
                terminal = policy.terminal_receipt()
                self.assertEqual(
                    (
                        terminal["field_gradient_count"],
                        terminal["primary_evaluation_count"],
                        terminal["selector_count"],
                    ),
                    (count, count, count),
                )
                self.assertEqual(
                    (
                        terminal["rho_capture_event_count"],
                        terminal["rho_calibration_count"],
                        terminal["rho_refresh_count"],
                    ),
                    (1, 2, 0),
                )

        calls = 0
        def changed():
            nonlocal calls
            calls += 1
            return (("d" if calls >= 3 else "a") * 64, "b" * 64, "c" * 64)
        with self.assertRaises(ODEBFStateError):
            self._run(PDZAblationArm.PDZ_T2_PRC, 0, ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T2_PRC, request_count=2), fixed=changed)

    def test_direct_is_unconditionally_direct_without_selector_or_rescue(self) -> None:
        policy = ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2)
        observed = self._run(PDZAblationArm.PDZ_T1_DIRECT, 0, policy)
        micro = observed.microsteps[0].receipt
        self.assertEqual(micro["selection_by_request"], [PDZAblationSelection.DIRECT_EULER.value] * 2)
        selection = micro["selection_receipt"]
        for key in ("rescue_proposal_count", "rescue_evaluation_count", "selector_call_count", "current_candidate_count", "hold_count", "rollback_count", "retry_count", "line_search_count", "adaptive_count", "semantic_acceptance_helper_count"):
            self.assertEqual(selection[key], 0)
        self.assertTrue(torch.equal(observed.target_step.target_next, observed.microsteps[0].selected.target_step.target_next))
        source = inspect.getsource(run_direct_euler_subcycle_scheduler)
        for forbidden in ("prepare_p1r52_rescue_proposal", "select_p1r52_target_proposal", "RESCUE", "CURRENT"):
            self.assertNotIn(forbidden, source)

    def test_direct_keeps_request_negative_progress_without_contraction(self) -> None:
        endpoint = objective((1.5, 1.0), None)
        observed = self._run(
            PDZAblationArm.PDZ_T1_DIRECT,
            0,
            ClockedPDZAmplitudePolicy(
                PDZAblationArm.PDZ_T1_DIRECT, request_count=2
            ),
            endpoint=endpoint,
        )
        receipt = observed.microsteps[0].receipt[
            "external_amplitude_selected_observation"
        ]
        self.assertEqual(receipt["negative_progress_by_request"], [True, False])
        self.assertEqual(receipt["negative_progress_count"], 1)
        self.assertEqual(observed.receipt["target_hold_mask"], [False, False])
        self.assertEqual(observed.receipt["selector_rescue_current_count"], 0)

    def test_direct_ntsm_is_typed_and_pre_write(self) -> None:
        endpoint = objective((2.0, 4.0), None)
        with self.assertRaises(P1R54ScientificBoundaryNTSM) as raised:
            self._run(PDZAblationArm.PDZ_T1_DIRECT, 0, ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2), endpoint=endpoint)
        receipt = raised.exception.receipt
        self.assertEqual(receipt["status"], "SCIENTIFIC_BOUNDARY_NTSM")
        self.assertGreater(receipt["aggregate_worsening"], receipt["numerical_tolerance"])
        self.assertEqual(receipt["writer_update_count"], 0)
        self.assertEqual(receipt["fallback_zero_write_rescue_contraction_exclusion_skip_count"], 0)

    def test_rho_refresh_nonfinite_and_wrong_clock_fail_closed(self) -> None:
        policy = ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T2_PRC, request_count=2)
        policy(context(0))
        with self.assertRaises(ODEBFStateError):
            policy(replace(context(0), target_origin_norm=torch.tensor([1.5, 2.6], dtype=torch.float64)))
        with self.assertRaises(ODEBFContractError):
            ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T2_PRC, request_count=2)(context(1))
        bad = replace(context(0), target_new_nll=torch.tensor([float("nan"), 1.0], dtype=torch.float64))
        with self.assertRaises(ODEBFContractError):
            ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2)(bad)

    def test_no_alias_or_low_precision_output(self) -> None:
        source = context()
        policy = ClockedPDZAmplitudePolicy(PDZAblationArm.PDZ_T1_DIRECT, request_count=2)
        result = policy(source)
        self.assertEqual(result.amplitude.dtype, torch.float64)
        self.assertNotEqual(result.amplitude.data_ptr(), source.target_origin_norm.data_ptr())
        self.assertEqual(tensor_sha256(source.target_origin_norm), tensor_sha256(torch.tensor([1.5, 2.5], dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
