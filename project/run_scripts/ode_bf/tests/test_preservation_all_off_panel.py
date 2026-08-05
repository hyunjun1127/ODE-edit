from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from project.run_scripts import (
    session05_ode_bf_preservation_alloff_fulltau_dry_plan as dry,
)
from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
    candidate_gate_decision,
    functional_p_decision_receipt,
    preservation_decision_payload,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_preservation_all_off_panel import (
    PRESERVATION_ALL_OFF_PANEL_LABELS,
    expected_preservation_all_off_result_name,
    forecast_preservation_all_off_panel,
    preservation_all_off_panel_specs,
    validate_preservation_all_off_lock,
)
from project.run_scripts.ode_bf.routing import (
    PreservationConstraintPolicy,
    QuadraticBarrier,
    RoutingProblem,
    RoutingStatus,
    project_bf_velocity,
    solve_raw_velocity,
    verify_backtracked_candidate,
)
from project.run_scripts.ode_bf.target_new_nll import RoutingObjective


ROOT = Path(__file__).resolve().parents[4]


def _functional_observation(passed: bool) -> dict[str, object]:
    return {
        "barrier": "pretrained-theta0-teacher",
        "sample_count": 10,
        "sample_order_sha256": "a" * 64,
        "budget": 1.0e-3,
        "mean_positive_damage": 0.002,
        "smooth_max_positive_damage": 0.003,
        "raw_max_positive_damage": 0.004,
        "signed_mean_damage": 0.002,
        "decision_rule": "uniform-mean-positive-part",
        "passed": passed,
    }


def _restrictive_problem() -> RoutingProblem:
    progress = np.asarray((1.0, 1.0), dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    restrictive = np.eye(2, dtype=np.float64)
    return RoutingProblem(
        progress,
        np.eye(2, dtype=np.float64),
        restrictive,
        0.05,
        np.ones(2, dtype=np.float64),
        0.5,
        1.0e-8,
        QuadraticBarrier(
            "historical", 0.0, zero, restrictive, 1.0e-4,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained", 0.0, zero, restrictive, 1.0e-4,
            "layer-local-diagonal",
        ),
    )


class PreservationAllOffPanelTests(unittest.TestCase):
    def test_alloff_removes_only_preservation_solver_constraints(self) -> None:
        problem = _restrictive_problem()
        raw = solve_raw_velocity(
            problem,
            preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
        )
        off = project_bf_velocity(
            problem,
            raw,
            preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
        )
        self.assertIs(off.status, RoutingStatus.FEASIBLE)
        self.assertIsNotNone(off.values)
        assert off.values is not None
        self.assertTrue(np.all(off.values >= 0.0))
        self.assertTrue(np.all(off.values <= problem.layer_caps))
        self.assertGreaterEqual(
            float(problem.signed_progress @ off.values),
            problem.requested_progress - 1.0e-8,
        )
        self.assertIn("preservation-observation-only", off.certificate.phase)
        locked = project_bf_velocity(problem, raw)
        self.assertIs(locked.status, RoutingStatus.PROGRESS_INFEASIBLE)

    def test_observations_remain_false_but_alloff_decision_is_true(self) -> None:
        problem = _restrictive_problem()
        raw = solve_raw_velocity(
            problem,
            preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
        )
        projected = project_bf_velocity(
            problem,
            raw,
            preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
        )
        assert projected.values is not None
        observed = verify_backtracked_candidate(
            problem,
            projected.values,
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=False,
            functional_p_pass=False,
            authoritative_bf16_pass=True,
        )
        self.assertFalse(observed.structural_h_pass)
        self.assertFalse(observed.structural_p_pass)
        self.assertFalse(observed.trust_pass)
        functional_p = functional_p_decision_receipt(
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            _functional_observation(False),
        )
        receipt = preservation_decision_payload(
            PreservationConstraintPolicy.OBSERVATION_ONLY,
            observed,
            functional_p,
        )
        self.assertFalse(any(receipt["observed_pass"].values()))
        self.assertTrue(all(receipt["decision_pass"].values()))
        self.assertEqual(
            set(receipt["decision_influence_count"].values()), {0}
        )
        self.assertEqual(candidate_gate_decision(observed, receipt), (True, None))

    def test_alloff_does_not_disable_progress_or_bf16_gate(self) -> None:
        problem = _restrictive_problem()
        velocity = np.asarray((0.25, 0.25), dtype=np.float64)
        functional_p = functional_p_decision_receipt(
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            _functional_observation(False),
        )
        no_progress = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=-1.0,
            functional_h_pass=False,
            functional_p_pass=False,
            authoritative_bf16_pass=True,
        )
        receipt = preservation_decision_payload(
            PreservationConstraintPolicy.OBSERVATION_ONLY,
            no_progress,
            functional_p,
        )
        self.assertEqual(
            candidate_gate_decision(no_progress, receipt), (False, "progress")
        )
        bad_bf16 = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=False,
            functional_p_pass=False,
            authoritative_bf16_pass=False,
        )
        receipt = preservation_decision_payload(
            PreservationConstraintPolicy.OBSERVATION_ONLY,
            bad_bf16,
            functional_p,
        )
        self.assertEqual(
            candidate_gate_decision(bad_bf16, receipt),
            (False, "authoritative_bf16"),
        )

    def test_panel_is_two_common_alloff_routing_objectives(self) -> None:
        specs = preservation_all_off_panel_specs()
        self.assertEqual(tuple(item.label for item in specs), PRESERVATION_ALL_OFF_PANEL_LABELS)
        self.assertEqual(
            tuple(item.routing_objective for item in specs),
            (RoutingObjective.MARGIN, RoutingObjective.TARGET_NEW_NLL),
        )
        self.assertTrue(
            all(
                item.functional_p_decision
                is FunctionalPDecisionPolicy.OBSERVATION_ONLY
                for item in specs
            )
        )
        self.assertTrue(
            all(
                item.preservation_constraints
                is PreservationConstraintPolicy.OBSERVATION_ONLY
                for item in specs
            )
        )

    def test_lock_and_forecast_are_fail_closed(self) -> None:
        lock_path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_preservation_alloff_fulltau.json"
        )
        value, _ = load_rooted_json(
            lock_path,
            expected_schema=(
                "ode-edit-s05-preservation-alloff-fulltau-numerical-lock/v1"
            ),
        )
        args = {
            "controller_identity_sha256": P1ControllerLock().identity(),
            "stream_root_digest": (
                "a3e2fbf27e94c3ace4e048027abf1f08f215715388dacc3cf85bb452e8e89157"
            ),
            "population_root_digest": (
                "49f3b2674d5fc649e8e17982769a8be7efde70e92b68791b550100fef627ee4b"
            ),
        }
        validate_preservation_all_off_lock(value, **args)
        changed = copy.deepcopy(value)
        changed["all_off_ablation"]["positive_progress_active"] = False
        with self.assertRaises(ODEBFContractError):
            validate_preservation_all_off_lock(changed, **args)
        forecast = forecast_preservation_all_off_panel(
            ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
            ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
            "llama3-8b-inst",
        )
        self.assertEqual(forecast.conservative_time_seconds, 81_114)
        self.assertTrue(forecast.fits_same_envelope)
        self.assertEqual(
            expected_preservation_all_off_result_name("qwen2.5-7b-inst"),
            "s05-preservation-alloff-fulltau-p1r4-qwen2.5-7b-inst-v1",
        )
        plan = dry.build_plan("c" * 40, repository_root=ROOT)
        self.assertEqual(plan["panel_labels"], list(PRESERVATION_ALL_OFF_PANEL_LABELS))
        self.assertEqual(plan["new_pair_gpu"], 2)
        self.assertEqual(plan["server1_gpu_cap"], 3)
        self.assertEqual(plan["preservation_decision_influence_count"], 0)
        self.assertFalse(plan["slurm_submit"])


if __name__ == "__main__":
    unittest.main()
