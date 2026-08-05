from __future__ import annotations

import copy
import inspect
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from project.run_scripts import (
    session05_ode_bf_functional_p_off_fulltau_dry_plan as dry,
)
from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime
from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
    functional_p_decision_receipt,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_functional_p_off_panel import (
    FUNCTIONAL_P_OFF_PANEL_LABELS,
    expected_functional_p_off_result_name,
    forecast_functional_p_off_panel,
    functional_p_off_panel_specs,
    validate_functional_p_off_lock,
    validate_r2_control_rollout,
)
from project.run_scripts.ode_bf.routing import (
    QuadraticBarrier,
    RoutingProblem,
    verify_backtracked_candidate,
)
from project.run_scripts.ode_bf.target_new_nll import RoutingObjective


ROOT = Path(__file__).resolve().parents[4]


def _observation(passed: bool) -> dict[str, object]:
    return {
        "barrier": "pretrained-theta0-teacher",
        "sample_count": 10,
        "sample_order_sha256": "a" * 64,
        "budget": 1.0e-3,
        "mean_positive_damage": 0.002,
        "smooth_max_positive_damage": 0.002,
        "raw_max_positive_damage": 0.003,
        "signed_mean_damage": 0.002,
        "decision_rule": "uniform-mean-positive-part",
        "passed": passed,
    }


class FunctionalPOffPanelTests(unittest.TestCase):
    def test_policy_changes_only_decision_bit_and_preserves_observation(self) -> None:
        observation = _observation(False)
        control = functional_p_decision_receipt(
            FunctionalPDecisionPolicy.PCTRL, observation
        )
        off = functional_p_decision_receipt(
            FunctionalPDecisionPolicy.OBSERVATION_ONLY, observation
        )
        self.assertEqual(control.observation_sha256, off.observation_sha256)
        self.assertFalse(control.decision_pass)
        self.assertEqual(control.decision_influence_count, 1)
        self.assertTrue(off.decision_pass)
        self.assertEqual(off.decision_influence_count, 0)
        self.assertEqual(off.observed_mean_positive_damage, 0.002)
        self.assertEqual(off.observed_slack_at_locked_budget, -0.001)
        self.assertTrue(off.raw_free_payload()["observation_only"])
        for bad in (None, 1, "false"):
            changed = dict(observation, passed=bad)
            with self.assertRaises(ODEBFContractError):
                functional_p_decision_receipt("FPOFF", changed)
        for bad in (True, "0.001", float("nan"), 1.1e-3):
            changed = dict(observation, budget=bad)
            with self.assertRaises(ODEBFContractError):
                functional_p_decision_receipt("FPOFF", changed)

    def test_structural_p_remains_active_when_functional_p_is_off(self) -> None:
        zero = np.zeros(2, dtype=np.float64)
        historical = QuadraticBarrier(
            "historical", 0.0, zero, np.zeros((2, 2)), 1.0,
            "layer-local-diagonal",
        )
        structural_p = QuadraticBarrier(
            "pretrained", 0.0, zero, np.eye(2), 0.1,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(2), np.eye(2), np.eye(2), 2.0, np.ones(2),
            0.5, 1.0e-8, historical, structural_p,
        )
        verdict = verify_backtracked_candidate(
            problem,
            np.asarray((0.5, 0.5)),
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        self.assertFalse(verdict.structural_p_pass)
        self.assertFalse(verdict.accepted)

    def test_specs_are_common_four_arm_causal_panel(self) -> None:
        specs = functional_p_off_panel_specs()
        self.assertEqual(tuple(item.label for item in specs), FUNCTIONAL_P_OFF_PANEL_LABELS)
        self.assertEqual(
            tuple(item.routing_objective for item in specs),
            (
                RoutingObjective.MARGIN,
                RoutingObjective.TARGET_NEW_NLL,
                RoutingObjective.MARGIN,
                RoutingObjective.TARGET_NEW_NLL,
            ),
        )
        self.assertEqual(
            tuple(item.functional_p_decision for item in specs),
            (
                FunctionalPDecisionPolicy.PCTRL,
                FunctionalPDecisionPolicy.PCTRL,
                FunctionalPDecisionPolicy.OBSERVATION_ONLY,
                FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            ),
        )
        source = inspect.getsource(runtime._run_trial)
        self.assertIn("functional_p_decision.decision_pass", source)
        self.assertIn('"functional_p": functional_p_observation', source)

    def test_lock_and_dry_forecast_fit_exact_envelope(self) -> None:
        lock_path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_functional_p_off_fulltau.json"
        )
        value, _ = load_rooted_json(
            lock_path,
            expected_schema=(
                "ode-edit-s05-functional-p-off-fulltau-numerical-lock/v1"
            ),
        )
        arguments = {
            "controller_identity_sha256": P1ControllerLock().identity(),
            "stream_root_digest": (
                "a3e2fbf27e94c3ace4e048027abf1f08f215715388dacc3cf85bb452e8e89157"
            ),
            "population_root_digest": (
                "49f3b2674d5fc649e8e17982769a8be7efde70e92b68791b550100fef627ee4b"
            ),
        }
        validate_functional_p_off_lock(value, **arguments)
        changed = copy.deepcopy(value)
        changed["functional_p_ablation"]["budget_nats"] = 1.0
        with self.assertRaises(ODEBFContractError):
            validate_functional_p_off_lock(changed, **arguments)
        plan = dry.build_plan("c" * 40, repository_root=ROOT)
        self.assertEqual(plan["panel_labels"], list(FUNCTIONAL_P_OFF_PANEL_LABELS))
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["slurm_submit"])
        for job in plan["jobs"]:
            forecast = job["forecast"]
            self.assertEqual(forecast["conservative_time_seconds"], 82_874)
            self.assertLessEqual(forecast["conservative_gpu_peak_mib"], 65_000)
            self.assertLessEqual(forecast["conservative_host_peak_mib"], 65_000)
        self.assertEqual(
            expected_functional_p_off_result_name("llama3-8b-inst"),
            "s05-functional-p-off-fulltau-p1r3-llama3-8b-inst-v1",
        )

    def test_r2_controls_fail_closed_before_fpoff(self) -> None:
        spec = functional_p_off_panel_specs()[0]
        rollout = SimpleNamespace(
            functional_p_decision="PCTRL",
            status="SAME_STATE_RETRY_EXHAUSTED",
            termination_label="SAME_STATE_RETRY_EXHAUSTED",
            accepted_t=Fraction(1, 8),
            k_acc=1,
            n_trial=6,
            n_reject=5,
            field_build_count=2,
            snapshots=[SimpleNamespace(
                snapshot_sha256=(
                    "75db974d71c37cf012c3ebf328edc19c0a424880a9731d01f6e500f2208e6a2a"
                )
            )],
        )
        validate_r2_control_rollout("llama3-8b-inst", spec, rollout)
        rollout.n_trial = 7
        with self.assertRaises(ODEBFContractError):
            validate_r2_control_rollout("llama3-8b-inst", spec, rollout)


if __name__ == "__main__":
    unittest.main()
