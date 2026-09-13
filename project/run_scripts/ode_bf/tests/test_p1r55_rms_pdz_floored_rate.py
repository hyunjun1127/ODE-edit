from __future__ import annotations

import math
from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import P1R52AmplitudeContext
from project.run_scripts.ode_bf.p1r54_energyfree_localz import (
    EnergyFreeLocalZArm,
    EnergyFreeLocalZPolicy,
)
from project.run_scripts.ode_bf.p1r55_objective_risk import (
    ObjectiveRiskPolicy,
    reduce_context_risk,
)
from project.run_scripts.ode_bf.p1r55_pdz_floored_rate import (
    AmplitudePolicy,
    P1R55ScientificInvalid,
    RequestLocalPDZPolicy,
    solve_pdz_floored_rate,
)
from project.run_scripts.ode_bf.p1r55_rms_pdz_rate_experiment import (
    ARMS,
    PHASE1_CELLS,
    build_phase1_binding,
    phase1_cell,
)
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_cache_sequential import (
    validate_phase3_batch_chain,
)


def _context(
    risk: torch.Tensor,
    *,
    step: int = 0,
    rho: torch.Tensor | None = None,
    semantic: torch.Tensor | None = None,
) -> P1R52AmplitudeContext:
    count = risk.numel()
    if semantic is None:
        semantic = torch.stack(
            (
                torch.linspace(0.2, 0.8, count, dtype=torch.float64),
                torch.linspace(0.9, 0.3, count, dtype=torch.float64),
            )
        )
    norm = torch.linalg.vector_norm(semantic, dim=0)
    direction = -semantic / norm.unsqueeze(0)
    if rho is None:
        rho = torch.linspace(1.0, 2.0, count, dtype=torch.float64)
    return P1R52AmplitudeContext(
        step,
        9.0,
        semantic,
        norm,
        norm + 1.0e-12,
        risk,
        norm > 1.0e-12,
        direction,
        torch.ones(count, dtype=torch.float64),
        123.0,
        rho,
    )


class P1R55RMSPDZRateTests(unittest.TestCase):
    def test_phase1_snapshot_major_mapping_and_warm_binding(self) -> None:
        self.assertEqual([item.label for item in ARMS], ["A0", "A1", "A2", "A3"])
        self.assertEqual(len(PHASE1_CELLS), 12)
        self.assertEqual(
            [(item.snapshot_batch, item.arm.label) for item in PHASE1_CELLS],
            [
                (snapshot, arm)
                for snapshot in (1, 6, 10)
                for arm in ("A0", "A1", "A2", "A3")
            ],
        )
        for cell in range(12):
            config = phase1_cell(cell)
            binding = build_phase1_binding(config)
            self.assertEqual(binding.probe_batch_index, config.snapshot_batch)
            self.assertEqual(binding.heldout_step_indices, (0, 3, 7))
            self.assertEqual(binding.writer_arm, "C3-KSTEP-CACHE")
            self.assertEqual(binding.easyedit_root, Path("/mnt/raid5/janghj/EasyEdit"))
            self.assertTrue(config.result_name.endswith("-tech-r2-v1"))
            self.assertIsNotNone(binding.canonical_prefix_amplitude_policy_factory)
            self.assertIsNotNone(binding.canonical_prefix_objective_evaluator_factory)
            metadata = binding.metadata["p1r55_rms_pdz_rate"]
            self.assertEqual(metadata["prefix_replay_scientific_denominator_count"], 0)
            self.assertEqual(metadata["accepted_z_heldout_outer_indices"], [1, 4, 8])

    def test_partial_warm_chain_validator_preserves_default_cache_algebra(self) -> None:
        rows = []
        prior = None
        for index in range(1, 7):
            entry = "w0" if prior is None else prior
            commit = f"w{index}"
            rows.append(
                {
                    "batch_index": index,
                    "cache_entry_width": (index - 1) * 100,
                    "cache_exit_width": index * 100,
                    "entry_weight_sha256": entry,
                    "commit_weight_sha256": commit,
                }
            )
            prior = commit
        receipt = validate_phase3_batch_chain(rows, expected_count=6)
        self.assertEqual(receipt["batch_count"], 6)
        self.assertEqual(receipt["commit_to_next_entry_match_count"], 5)

    def test_rms_equal_context_hard_weight_permutation_and_zero_boundary(self) -> None:
        equal = torch.full((4, 6), 2.5, dtype=torch.float64, requires_grad=True)
        mean = reduce_context_risk(equal, ObjectiveRiskPolicy.MEAN)
        rms = reduce_context_risk(equal, ObjectiveRiskPolicy.RMS)
        self.assertTrue(torch.equal(mean, rms))
        mean_gradient = torch.autograd.grad(mean.sum(), equal, retain_graph=True)[0]
        rms_gradient = torch.autograd.grad(rms.sum(), equal)[0]
        self.assertTrue(torch.equal(mean_gradient, rms_gradient))

        hard = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 9.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        rms_hard = reduce_context_risk(hard, ObjectiveRiskPolicy.RMS)
        gradient = torch.autograd.grad(rms_hard.sum(), hard)[0]
        self.assertGreater(float(gradient[0, 5]), 1.0 / 6.0)
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        self.assertTrue(torch.equal(
            reduce_context_risk(hard.detach()[:, permutation], ObjectiveRiskPolicy.RMS),
            rms_hard.detach(),
        ))

        zero = torch.zeros((3, 6), dtype=torch.float64, requires_grad=True)
        zero_risk = reduce_context_risk(zero, ObjectiveRiskPolicy.RMS)
        zero_gradient = torch.autograd.grad(zero_risk.sum(), zero)[0]
        self.assertTrue(torch.equal(zero_risk, torch.zeros(3, dtype=torch.float64)))
        self.assertTrue(torch.equal(zero_gradient, torch.zeros_like(zero)))

    def test_mean_reducer_is_exact_existing_context_mean_and_gradient(self) -> None:
        context = torch.arange(30, dtype=torch.float64).reshape(5, 6) / 17.0
        context.requires_grad_(True)
        observed = reduce_context_risk(context, ObjectiveRiskPolicy.MEAN)
        expected = context.mean(dim=1)
        self.assertTrue(torch.equal(observed, expected))
        observed_gradient = torch.autograd.grad(
            observed.sum(), context, retain_graph=True
        )[0]
        expected_gradient = torch.autograd.grad(expected.sum(), context)[0]
        self.assertTrue(torch.equal(observed_gradient, expected_gradient))

    def test_a0_pdz_is_byte_exact_to_p1r54(self) -> None:
        risk = torch.tensor([0.0, 1.0e-12, 0.2, 3.0], dtype=torch.float64)
        context = _context(risk)
        legacy = EnergyFreeLocalZPolicy(
            EnergyFreeLocalZArm.PDZ, request_count=4
        )(context)
        observed = RequestLocalPDZPolicy(
            risk_policy=ObjectiveRiskPolicy.MEAN,
            amplitude_policy=AmplitudePolicy.PDZ,
            request_count=4,
        )(context)
        self.assertTrue(torch.equal(legacy.amplitude, observed.amplitude))
        self.assertEqual(
            legacy.receipt["p1r54_kdc_direction_sha256"],
            observed.receipt["p1r55_kdc_direction_sha256"],
        )

    def test_external_amplitude_receipt_namespace_does_not_collide(self) -> None:
        decision = RequestLocalPDZPolicy(
            risk_policy=ObjectiveRiskPolicy.MEAN,
            amplitude_policy=AmplitudePolicy.PDZ,
            request_count=2,
        )(_context(torch.tensor([0.2, 0.3], dtype=torch.float64)))
        parent_keys = {
            "schema",
            "instruction_id",
            "semantic_gradient_norm_by_request",
            "semantic_gradient_sha256",
            "kdc_direction_sha256",
        }
        self.assertFalse(parent_keys.intersection(decision.receipt))
        self.assertTrue(all(key.startswith("p1r55_") for key in decision.receipt))

    def test_request_axis_microbatch_partition_keeps_one_global_scaling(self) -> None:
        context = torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
                [0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        for policy in ObjectiveRiskPolicy:
            whole = reduce_context_risk(context, policy)
            whole_gradient = torch.autograd.grad(
                whole.sum(), context, retain_graph=True
            )[0] / context.shape[0]
            partitioned_gradient = torch.zeros_like(context)
            for start, stop in ((0, 1), (1, 4)):
                partial = reduce_context_risk(context[start:stop], policy)
                partial_gradient = torch.autograd.grad(
                    partial.sum(), context, retain_graph=True
                )[0]
                partitioned_gradient.add_(partial_gradient)
            partitioned_gradient.div_(context.shape[0])
            self.assertTrue(torch.equal(partitioned_gradient, whole_gradient))

    def test_rate_closed_form_floor_cap_and_safe_branch(self) -> None:
        risk = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
        rho = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        slope = torch.tensor([4.0, 0.2, 0.01], dtype=torch.float64)
        active = torch.ones(3, dtype=torch.bool)
        amplitude, receipt = solve_pdz_floored_rate(
            risk=risk, rho=rho, slope=slope, active_mask=active
        )
        pdz = rho * (-torch.expm1(-risk))
        demand = -math.expm1(-0.125) * risk
        raw = demand / (0.125 * slope)
        expected = torch.clamp(raw, min=pdz, max=rho)
        self.assertTrue(torch.allclose(amplitude, expected, rtol=0.0, atol=0.0))
        self.assertTrue(torch.all(amplitude >= pdz))
        self.assertTrue(torch.all(amplitude <= rho))
        self.assertEqual(receipt["unsafe_predivision_count"], 0)
        self.assertEqual(receipt["rate_division_by_request"][-1], False)
        self.assertIsNone(receipt["rate_amplitude_by_request"][-1])

    def test_nonfinite_nonpositive_active_slope_fails_closed(self) -> None:
        common = dict(
            risk=torch.tensor([1.0], dtype=torch.float64),
            rho=torch.tensor([1.0], dtype=torch.float64),
            active_mask=torch.tensor([True]),
        )
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(P1R55ScientificInvalid):
                solve_pdz_floored_rate(
                    slope=torch.tensor([value], dtype=torch.float64), **common
                )
        invalid_context = _context(torch.tensor([1.0], dtype=torch.float64))
        invalid_context.semantic_gradient[0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            RequestLocalPDZPolicy(
                risk_policy=ObjectiveRiskPolicy.MEAN,
                amplitude_policy=AmplitudePolicy.PDZ,
                request_count=1,
            )(invalid_context)

    def test_request_local_invariance_b1_b10_b100_and_permutation(self) -> None:
        base_risk = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64)
        base = _context(base_risk)
        for risk_policy in ObjectiveRiskPolicy:
            for amplitude_policy in AmplitudePolicy:
                expected = RequestLocalPDZPolicy(
                    risk_policy=risk_policy,
                    amplitude_policy=amplitude_policy,
                    request_count=3,
                )(base).amplitude
                permutation = torch.tensor([2, 0, 1])
                permuted = P1R52AmplitudeContext(
                    0,
                    999.0,
                    base.semantic_gradient[:, permutation],
                    base.semantic_gradient_norm[permutation],
                    base.entry_semantic_gradient_norm[permutation],
                    base.target_new_nll[permutation],
                    base.active_mask[permutation],
                    base.kdc_direction[:, permutation],
                    base.local_parent_amplitude[permutation],
                    -1.0,
                    base.target_origin_norm[permutation],
                )
                observed = RequestLocalPDZPolicy(
                    risk_policy=risk_policy,
                    amplitude_policy=amplitude_policy,
                    request_count=3,
                )(permuted).amplitude
                self.assertTrue(torch.equal(observed[torch.argsort(permutation)], expected))
                for count in (1, 10, 100):
                    indices = torch.arange(count) % 3
                    expanded = P1R52AmplitudeContext(
                        0,
                        float(count),
                        base.semantic_gradient[:, indices],
                        base.semantic_gradient_norm[indices],
                        base.entry_semantic_gradient_norm[indices],
                        base.target_new_nll[indices],
                        base.active_mask[indices],
                        base.kdc_direction[:, indices],
                        base.local_parent_amplitude[indices],
                        0.0,
                        base.target_origin_norm[indices],
                    )
                    output = RequestLocalPDZPolicy(
                        risk_policy=risk_policy,
                        amplitude_policy=amplitude_policy,
                        request_count=count,
                    )(expanded).amplitude
                    self.assertTrue(torch.equal(output, expected[indices]))

    def test_clock_rho_and_all_zero_counters(self) -> None:
        policy = RequestLocalPDZPolicy(
            risk_policy=ObjectiveRiskPolicy.RMS,
            amplitude_policy=AmplitudePolicy.PDZ_FLOORED_RATE,
            request_count=2,
            microsteps_per_outer=2,
        )
        rho = torch.tensor([1.0, 2.0], dtype=torch.float64)
        for ordinal in range(16):
            context = _context(
                torch.tensor([0.2, 0.3], dtype=torch.float64),
                step=ordinal // 2,
                rho=rho,
            )
            decision = policy(context)
            for key in (
                "cross_request_nll_decision_access_count",
                "cross_request_gradient_reduction_decision_count",
                "batch_energy_normalization_count",
                "unused_energy_redistribution_count",
                "batch_percentile_decision_count",
                "per_request_backward_loop_count",
                "extra_model_forward_count",
                "extra_model_backward_count",
                "heldout_decision_access_count",
            ):
                self.assertEqual(decision.receipt[f"p1r55_{key}"], 0)
            policy.observe_selected(
                step_index=ordinal // 2,
                current_nll=(0.2, 0.3),
                selected_nll=(0.1, 0.2),
                field_receipt={"clamp_hit_by_request": [False, False]},
                selection=("PRIMARY", "PRIMARY"),
                target_dt=0.125,
            )
        terminal = policy.terminal_receipt()
        self.assertEqual(terminal["amplitude_call_count"], 16)
        self.assertEqual(terminal["rho_calibration_count"], 1)
        self.assertEqual(terminal["rho_refresh_count"], 0)


if __name__ == "__main__":
    unittest.main()
