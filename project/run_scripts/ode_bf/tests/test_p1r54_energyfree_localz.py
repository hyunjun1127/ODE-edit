from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
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
from project.run_scripts.ode_bf.p1r54_energyfree_localz import (
    EnergyFreeLocalZArm,
    EnergyFreeLocalZPolicy,
)
from project.run_scripts.ode_bf.p1r54_energyfree_localz_b100 import (
    HELDOUT_K_INDICES,
    expected_result_name,
    role_for_cell,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectiveResult


def _context(
    *,
    step: int = 0,
    nll: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
    semantic: torch.Tensor | None = None,
) -> P1R52AmplitudeContext:
    if nll is None:
        nll = torch.tensor([0.0, 0.25, 2.0], dtype=torch.float64)
    count = nll.numel()
    if semantic is None:
        semantic = torch.stack(
            (
                torch.linspace(1.0, 2.0, count, dtype=torch.float64),
                torch.linspace(2.5, 1.5, count, dtype=torch.float64),
            )
        )
    norm = torch.linalg.vector_norm(semantic, dim=0)
    direction = -semantic / norm.unsqueeze(0)
    if rho is None:
        rho = torch.linspace(1.0, 3.0, count, dtype=torch.float64)
    return P1R52AmplitudeContext(
        step,
        777.0,
        semantic,
        norm,
        123.0 * torch.ones(count, dtype=torch.float64),
        nll,
        torch.ones(count, dtype=torch.bool),
        direction,
        456.0 * torch.ones(count, dtype=torch.float64),
        999999.0,
        rho,
    )


def _objective(values: tuple[float, ...], gradient: torch.Tensor) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        sum(values) / len(values), values, "1" * 64, "2" * 64,
        tuple("3" * 64 for _ in values), tuple(1 for _ in values),
        1, 1, 10, 10, gradient, None,
        "4" * 64, "5" * 64, "6" * 64,
    )


class P1R54EnergyFreeLocalZTests(unittest.TestCase):
    def test_roles_and_terminal_only_heldout(self) -> None:
        self.assertEqual(HELDOUT_K_INDICES, (7,))
        self.assertEqual(
            [expected_result_name(role_for_cell(index)) for index in range(2)],
            [
                "s05-p1r54-energyfree-localz-llama-b100-fz-tech-r1-v1",
                "s05-p1r54-energyfree-localz-llama-b100-pdz-tech-r1-v1",
            ],
        )

    def test_fz_velocity_norm_is_request_local_rho(self) -> None:
        context = _context()
        decision = EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.FZ, request_count=3)(context)
        self.assertTrue(torch.equal(decision.amplitude, context.target_origin_norm))
        self.assertFalse(decision.enforce_reference_energy_identity)
        receipt = decision.receipt
        self.assertEqual(receipt["p1r54_velocity_amplitude_max_abs_residual"], 0.0)
        for key in (
            "p1r54_shared_scale_decision_access_count",
            "p1r54_global_reference_energy_decision_access_count",
            "p1r54_batch_nll_denominator_decision_access_count",
            "p1r54_entry_gradient_denominator_decision_access_count",
            "p1r54_gradient_ratio_decision_access_count",
            "p1r54_kappa_decision_access_count",
            "p1r54_inverse_slope_decision_access_count",
            "p1r54_unused_energy_redistribution_count",
            "p1r54_added_model_forward_count",
            "p1r54_added_backward_count",
        ):
            self.assertEqual(receipt[key], 0)

    def test_pdz_deficit_stability_bounds_and_monotonicity(self) -> None:
        nll = torch.tensor([0.0, 1.0e-12, 0.5, 1000.0], dtype=torch.float64)
        rho = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
        decision = EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.PDZ, request_count=4)(
            _context(nll=nll, rho=rho)
        )
        expected = rho * (-torch.expm1(-nll))
        self.assertTrue(torch.equal(decision.amplitude, expected))
        self.assertEqual(float(decision.amplitude[0]), 0.0)
        self.assertAlmostEqual(float(decision.amplitude[1]), 2.0e-12, places=23)
        self.assertTrue(torch.all(decision.amplitude[1:] > 0.0))
        self.assertTrue(torch.all(decision.amplitude <= rho))
        self.assertTrue(torch.all(decision.amplitude[1:] >= decision.amplitude[:-1]))

    def test_batch_permutation_add_remove_duplicate_and_b10_b100_invariance(self) -> None:
        base_nll = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64)
        base_rho = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float64)
        base_context = _context(nll=base_nll, rho=base_rho)
        for arm in EnergyFreeLocalZArm:
            expected = EnergyFreeLocalZPolicy(arm, request_count=3)(base_context).amplitude
            permutation = torch.tensor([2, 0, 1])
            permuted_context = P1R52AmplitudeContext(
                0, -999.0,
                base_context.semantic_gradient[:, permutation],
                base_context.semantic_gradient_norm[permutation],
                base_context.entry_semantic_gradient_norm[permutation] * 999.0,
                base_context.target_new_nll[permutation],
                base_context.active_mask[permutation],
                base_context.kdc_direction[:, permutation],
                base_context.local_parent_amplitude[permutation] * 777.0,
                -1.0,
                base_context.target_origin_norm[permutation],
            )
            observed = EnergyFreeLocalZPolicy(arm, request_count=3)(permuted_context).amplitude
            self.assertTrue(torch.equal(observed[torch.argsort(permutation)], expected))

            for count in (2, 4, 10, 100):
                indices = torch.arange(count) % 3
                expanded = P1R52AmplitudeContext(
                    0, float(count),
                    base_context.semantic_gradient[:, indices],
                    base_context.semantic_gradient_norm[indices],
                    torch.full((count,), float(count), dtype=torch.float64),
                    base_context.target_new_nll[indices],
                    torch.ones(count, dtype=torch.bool),
                    base_context.kdc_direction[:, indices],
                    torch.full((count,), 12345.0, dtype=torch.float64),
                    float(count * count),
                    base_context.target_origin_norm[indices],
                )
                output = EnergyFreeLocalZPolicy(arm, request_count=count)(expanded).amplitude
                self.assertTrue(torch.equal(output, expected[indices]))

    def test_rho_is_k0_immutable_and_never_refreshed(self) -> None:
        policy = EnergyFreeLocalZPolicy(EnergyFreeLocalZArm.PDZ, request_count=3)
        first = _context()
        policy(first)
        second = _context(step=1, nll=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
        policy(second)
        changed = _context(
            step=2,
            nll=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
            rho=torch.tensor([1.0, 2.0, 3.01], dtype=torch.float64),
        )
        with self.assertRaises(ODEBFContractError):
            policy(changed)

    def test_full_proposal_reuses_direction_barrier_and_adds_no_fb(self) -> None:
        current = torch.tensor(
            [[2.0, 3.0, 4.0], [1.0, 2.0, 1.5]], dtype=torch.float32
        )
        gradient = torch.tensor(
            [[0.2, -0.1, 0.15], [0.1, 0.2, -0.05]], dtype=torch.float32
        ) / 3.0
        nll = _objective((1.0, 2.0, 4.0), gradient)
        kl = P1R24KLResult(
            0.0, (0.0, 0.0, 0.0), torch.zeros_like(gradient),
            1, 1, 3, 3, "7" * 64,
        )
        lock = P1R24AliasTargetLock("llama3-8b-inst", 0.0, 0.0, 1.0e9)
        common = dict(
            alias="llama3-8b-inst", step_index=0, shared_speed=0.4,
            kl_teacher_input_sha256="8" * 64, target_dt=0.125,
        )
        rows = []
        for arm in EnergyFreeLocalZArm:
            rows.append(
                prepare_p1r52_target_proposal(
                    current, current, current, nll, kl, lock,
                    P1R51ControllerState.zero(current),
                    amplitude_policy=EnergyFreeLocalZPolicy(arm, request_count=3),
                    **common,
                )
            )
        self.assertTrue(torch.equal(rows[0].kdc_direction, rows[1].kdc_direction))
        for row in rows:
            self.assertTrue(torch.all(torch.sum(row.semantic_gradient * row.kdc_direction, dim=0) < 0.0))
            self.assertEqual(row.receipt["p1r54_added_model_forward_count"], 0)
            self.assertEqual(row.receipt["p1r54_added_backward_count"], 0)
            self.assertEqual(row.receipt["clamp_energy_redistribution_count"], 0)

    def test_terminal_clock_and_observation_have_no_decision_authority(self) -> None:
        for arm in EnergyFreeLocalZArm:
            policy = EnergyFreeLocalZPolicy(arm, request_count=3)
            for step in range(8):
                context = _context(
                    step=step,
                    nll=torch.tensor([0.1 + step, 0.2 + step, 0.3 + step], dtype=torch.float64),
                )
                decision = policy(context)
                field = {
                    **decision.receipt,
                    "allocation_amplitude_by_request": [float(item) for item in decision.amplitude],
                    "post_cast_energy_by_request": [float(item * item) for item in decision.amplitude],
                    "clamp_hit_by_request": [False, False, False],
                }
                observed = policy.observe_selected(
                    step_index=step,
                    current_nll=tuple(float(item) for item in context.target_new_nll),
                    selected_nll=tuple(float(item * 0.9) for item in context.target_new_nll),
                    field_receipt=field,
                    selection=("PRIMARY", "PRIMARY", "PRIMARY"),
                    target_dt=0.125,
                )
                self.assertEqual(observed["selected_observation_decision_influence_count"], 0)
            terminal = policy.terminal_receipt()
            self.assertEqual(terminal["amplitude_call_count"], 8)
            self.assertEqual(terminal["rho_refresh_count"], 0)
            self.assertEqual(terminal["forbidden_decision_access_count"], 0)

    def test_p1r53_existing_policy_ignores_new_optional_context_member(self) -> None:
        old = _context(nll=torch.tensor([0.1, 0.25, 2.0], dtype=torch.float64))
        left = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)(old)
        changed = P1R52AmplitudeContext(
            old.step_index, old.shared_speed, old.semantic_gradient,
            old.semantic_gradient_norm, old.entry_semantic_gradient_norm,
            old.target_new_nll, old.active_mask, old.kdc_direction,
            old.local_parent_amplitude, old.counterfactual_global_reference_energy,
            old.target_origin_norm * 987.0,
        )
        right = RequestLocalSpeedPolicy(RequestLocalSpeedArm.LP_S, request_count=3)(changed)
        self.assertTrue(torch.equal(left.amplitude, right.amplitude))
        self.assertEqual(left.receipt, right.receipt)


if __name__ == "__main__":
    unittest.main()
