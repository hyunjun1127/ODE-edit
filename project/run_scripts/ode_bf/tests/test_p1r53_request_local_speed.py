from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    P1R52AmplitudeContext,
    prepare_p1r52_target_proposal,
)
from project.run_scripts.ode_bf.p1r53_request_local_speed import (
    RequestLocalSpeedArm,
    RequestLocalSpeedPolicy,
)
from project.run_scripts.ode_bf.p1r53_request_local_speed_b100 import (
    HELDOUT_K_INDICES,
    expected_result_name,
    role_for_cell,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def _context(
    *,
    step: int = 0,
    nll: tuple[float, ...] = (1.0, 2.0, 4.0),
    semantic_scale: float = 1.0,
) -> P1R52AmplitudeContext:
    semantic = semantic_scale * torch.tensor(
        [[1.0, -2.0, 0.5], [2.0, 1.0, -1.5]], dtype=torch.float64
    )
    norm = torch.linalg.vector_norm(semantic, dim=0)
    direction = -semantic / norm.unsqueeze(0)
    entry = torch.tensor([2.25, 2.50, 1.75], dtype=torch.float64)
    shared = 0.4
    parent = shared * norm / entry
    return P1R52AmplitudeContext(
        step,
        shared,
        semantic,
        norm,
        entry,
        torch.tensor(nll, dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
        direction,
        parent,
        float(torch.sum(torch.square(parent))),
    )


def _objective(values: tuple[float, ...], gradient: torch.Tensor | None) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, int(gradient is not None), 10, 10, gradient, None,
        "4" * 64, "5" * 64, "6" * 64,
    )


class P1R53RequestLocalSpeedTests(unittest.TestCase):
    def test_roles_and_terminal_only_heldout(self) -> None:
        self.assertEqual(HELDOUT_K_INDICES, (7,))
        self.assertEqual(
            [expected_result_name(role_for_cell(index)) for index in range(2)],
            [
                "s05-p1r53-request-local-speed-llama-b100-lp-s-tech-r1-v1",
                "s05-p1r53-request-local-speed-llama-b100-lfd-e-tech-r1-v1",
            ],
        )

    def test_lp_s_is_local_under_frozen_shared_scale(self) -> None:
        direct = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)
        changed = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)
        left = direct(_context())
        right = changed(_context(nll=(1.0, 2000.0, 0.0001)))
        self.assertTrue(torch.equal(left.amplitude, right.amplitude))
        self.assertTrue(torch.equal(left.amplitude, _context().local_parent_amplitude))
        self.assertEqual(left.receipt["p1r53_cross_request_nll_decision_count"], 0)
        self.assertEqual(left.receipt["p1r53_batch_nll_denominator_decision_count"], 0)
        self.assertLessEqual(left.receipt["p1r53_relative_energy_identity_error"], 1e-12)

    def test_lp_s_permutation_and_duplicate_invariance_with_frozen_scale(self) -> None:
        base = _context()
        permutation = torch.tensor([2, 0, 1])
        permuted = P1R52AmplitudeContext(
            0,
            base.shared_speed,
            base.semantic_gradient[:, permutation],
            base.semantic_gradient_norm[permutation],
            base.entry_semantic_gradient_norm[permutation],
            base.target_new_nll[permutation],
            base.active_mask[permutation],
            base.kdc_direction[:, permutation],
            base.local_parent_amplitude[permutation],
            base.counterfactual_global_reference_energy,
        )
        observed = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)(permuted)
        inverse = torch.argsort(permutation)
        expected = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)(base)
        self.assertTrue(torch.equal(observed.amplitude[inverse], expected.amplitude))

        duplicate = P1R52AmplitudeContext(
            0,
            base.shared_speed,
            torch.cat((base.semantic_gradient, base.semantic_gradient[:, :1]), dim=1),
            torch.cat((base.semantic_gradient_norm, base.semantic_gradient_norm[:1])),
            torch.cat((base.entry_semantic_gradient_norm, base.entry_semantic_gradient_norm[:1])),
            torch.cat((base.target_new_nll, base.target_new_nll[:1])),
            torch.cat((base.active_mask, base.active_mask[:1])),
            torch.cat((base.kdc_direction, base.kdc_direction[:, :1]), dim=1),
            torch.cat((base.local_parent_amplitude, base.local_parent_amplitude[:1])),
            float(torch.sum(torch.square(torch.cat((base.local_parent_amplitude, base.local_parent_amplitude[:1]))))),
        )
        duplicated = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=4)(duplicate)
        self.assertTrue(torch.equal(duplicated.amplitude[:3], expected.amplitude))

    def test_lfd_e_k0_identity_frozen_kappa_and_nominal_rate(self) -> None:
        policy = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LFD_E, request_count=3)
        entry = _context()
        first = policy(entry)
        self.assertTrue(torch.equal(first.amplitude, entry.local_parent_amplitude))
        self.assertEqual(policy.kappa_calibration_count, 3)
        self.assertEqual(policy.kappa_refresh_count, 0)
        second_context = _context(step=1, nll=(0.8, 1.4, 3.0), semantic_scale=0.7)
        second = policy(second_context)
        sigma = -torch.sum(
            second_context.semantic_gradient * second_context.kdc_direction, dim=0
        )
        kappa = torch.tensor(second.receipt["p1r53_local_kappa_by_request"], dtype=torch.float64)
        expected = kappa * second_context.target_new_nll / sigma
        self.assertTrue(torch.allclose(second.amplitude, expected, atol=0.0, rtol=1e-15))
        self.assertLessEqual(
            second.receipt["p1r53_nominal_rate_identity_max_abs_residual"], 1e-15
        )
        self.assertEqual(policy.kappa_calibration_count, 3)
        self.assertEqual(policy.kappa_refresh_count, 0)

    def test_policy_clock_finishes_at_exact_k8(self) -> None:
        for arm in RequestLocalSpeedArm:
            policy = RequestLocalSpeedPolicy(arm, request_count=3)
            for step in range(8):
                context = _context(step=step, semantic_scale=1.0 - step * 0.03)
                decision = policy(context)
                field_receipt = {
                    **decision.receipt,
                    "allocation_amplitude_by_request": [
                        float(item) for item in decision.amplitude
                    ],
                    "post_cast_energy_by_request": [
                        float(item * item) for item in decision.amplitude
                    ],
                    "clamp_hit_by_request": [False, False, False],
                }
                policy.observe_selected(
                    step_index=step,
                    current_nll=tuple(float(item) for item in context.target_new_nll),
                    selected_nll=tuple(
                        float(item * 0.9) for item in context.target_new_nll
                    ),
                    field_receipt=field_receipt,
                    selection=("PRIMARY", "PRIMARY", "PRIMARY"),
                    target_dt=0.125,
                )
            terminal = policy.terminal_receipt()
            self.assertEqual(terminal["amplitude_call_count"], 8)
            self.assertEqual(terminal["additional_model_forward_count"], 0)
            self.assertEqual(terminal["additional_backward_count"], 0)
            self.assertEqual(terminal["per_request_backward_loop_count"], 0)
            if arm is RequestLocalSpeedArm.LFD_E:
                self.assertTrue(terminal["k0_lp_lfd_amplitude_identity"])

    def test_k0_lp_lfd_full_proposal_identity(self) -> None:
        current = torch.tensor(
            [[2.0, 3.0, 4.0], [1.0, 2.0, 1.5]], dtype=torch.float32
        )
        gradient = torch.tensor(
            [[0.2, -0.1, 0.15], [0.1, 0.2, -0.05]], dtype=torch.float32
        ) / 3.0
        nll = _objective((1.0, 2.0, 4.0), gradient)
        kl = P1R24KLResult(
            0.0,
            (0.0, 0.0, 0.0),
            torch.zeros_like(gradient),
            1,
            1,
            3,
            3,
            "7" * 64,
        )
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9)
        common = dict(
            alias="llama3-8b-inst",
            step_index=0,
            shared_speed=0.4,
            kl_teacher_input_sha256="8" * 64,
            target_dt=0.125,
        )
        lp = prepare_p1r52_target_proposal(
            current, current, current, nll, kl, lock,
            P1R51ControllerState.zero(current),
            amplitude_policy=RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3),
            **common,
        )
        lfd = prepare_p1r52_target_proposal(
            current, current, current, nll, kl, lock,
            P1R51ControllerState.zero(current),
            amplitude_policy=RequestLocalSpeedPolicy(RequestLocalSpeedArm.LFD_E, request_count=3),
            **common,
        )
        self.assertTrue(torch.equal(lp.kdc_direction, lfd.kdc_direction))
        self.assertTrue(torch.equal(lp.raw_velocity, lfd.raw_velocity))
        self.assertTrue(torch.equal(lp.primary_step.target_next, lfd.primary_step.target_next))
        self.assertEqual(lp.receipt["added_model_forward_count"], 0)
        self.assertEqual(lfd.receipt["added_backward_count"], 0)


if __name__ == "__main__":
    unittest.main()
