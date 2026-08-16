from __future__ import annotations

import ast
import copy
import inspect
import math
import unittest
from dataclasses import asdict

import numpy as np

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.layer_routing_telemetry import (
    LAYER_IDS,
    build_layer_routing_telemetry,
    distribution_summary,
    relabel_layer_routing_telemetry,
    summarize_layer_routing_trajectory,
)
from project.run_scripts.ode_bf.p1_adaptive import (
    AdaptiveTauClock,
    AdaptiveVariant,
    adaptive_lock,
)
from project.run_scripts.ode_bf.routing import (
    QuadraticBarrier,
    RoutingProblem,
    verify_backtracked_candidate,
)
from project.run_scripts.ode_edit_method.event_strength import assert_raw_free
from project.run_scripts.ode_edit_method.contracts import MethodContractError


def _row(
    *,
    step: int,
    applied=(0.8, 0.1, 0.1, 0.0, 0.0),
    realized=(8.0, 1.0, 1.0, 0.0, 0.0),
    stage="ACCEPTED_TRANSITION",
):
    signed = (1.0, 1.0, 1.0, -1.0, 0.0)
    raw = (0.7, 0.2, 0.1, 0.0, 0.0)
    bf = tuple(float(value) for value in applied)
    return build_layer_routing_telemetry(
        step_index=step,
        stage=stage,
        layer_ids=LAYER_IDS,
        signed_efficiency=signed,
        raw_velocity=raw,
        bf_velocity=bf,
        applied_coefficient=applied,
        active_direction_mask=(1, 1, 1, 0, 0),
        raw_cap_bound_mask=(0, 0, 0, 0, 0),
        bf_cap_bound_mask=(0, 0, 0, 0, 0),
        raw_zero_bound_mask=(0, 0, 0, 1, 1),
        bf_zero_bound_mask=(0, 0, 0, 1, 1),
        predicted_progress_contribution=applied,
        prequantized_update_energy=realized,
        realized_bf16_update_energy=realized,
        cumulative_bf16_capacity=realized,
        bf16_capacity_contribution=realized,
        structural_h_contribution=(0.1, 0.2, 0.3, 0.0, 0.0),
        structural_p_contribution=(0.2, 0.1, 0.3, 0.0, 0.0),
        trust_contribution=(0.3, 0.2, 0.1, 0.0, 0.0),
        field_sha256="1" * 64,
        state_sha256="2" * 64,
        target_z_sha256="3" * 64,
        factor_state_sha256="4" * 64,
        raw_solver_certificate_sha256="5" * 64,
        bf_solver_certificate_sha256="6" * 64,
        candidate_sha256="7" * 64,
    )


class LayerRoutingTelemetryTests(unittest.TestCase):
    def test_distribution_metrics_are_recomputable_and_layer_ordered(self) -> None:
        observed = distribution_summary(
            (8.0, 1.0, 1.0, 0.0, 0.0), normalization="absolute_l1"
        )
        share = np.asarray(observed["share"], dtype=np.float64)
        self.assertEqual(observed["layer_ids"], list(LAYER_IDS))
        self.assertEqual(observed["top1_layer_id"], 4)
        self.assertEqual(observed["top2_layer_id"], 5)
        self.assertAlmostEqual(observed["top1_share"], 0.8)
        self.assertAlmostEqual(observed["top2_cumulative_share"], 0.9)
        self.assertAlmostEqual(observed["hhi"], float(share @ share))
        self.assertAlmostEqual(observed["effective_layer_count"], 1.0 / float(share @ share))
        nonzero = share[share > 0.0]
        expected_entropy = float(
            -(nonzero * np.log(nonzero)).sum() / math.log(len(LAYER_IDS))
        )
        self.assertAlmostEqual(observed["normalized_entropy"], expected_entropy)
        self.assertEqual(observed["zero_count"], 2)
        self.assertEqual(observed["active_layer_count"], 3)
        self.assertTrue(observed["severe_concentration"])
        self.assertIn("TOP1_SHARE_GTE_0_80", observed["severe_metric_types"])

    def test_step_payload_is_numeric_raw_free_and_does_not_mutate_inputs(self) -> None:
        inputs = {
            "applied": [0.8, 0.1, 0.1, 0.0, 0.0],
            "realized": [8.0, 1.0, 1.0, 0.0, 0.0],
        }
        before = copy.deepcopy(inputs)
        value = _row(
            step=1,
            applied=inputs["applied"],
            realized=inputs["realized"],
            stage="REJECTED_TRANSITION",
        )
        self.assertEqual(inputs, before)
        self.assertEqual(value["stage"], "REJECTED_TRANSITION")
        self.assertEqual(value["layer_ids"], list(LAYER_IDS))
        self.assertEqual(len(value["signed_routing_efficiency"]), 5)
        self.assertEqual(value["controller_dependency_count"], 0)
        self.assertTrue(value["observation_only"])
        source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.layer_routing_telemetry",
                fromlist=["build_layer_routing_telemetry"],
            )
        )
        imported = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("torch", imported)
        for forbidden in ("request prompt", "target token", "model tensor"):
            self.assertNotIn(forbidden, source.casefold())

    def test_invalid_order_length_masks_nonfinite_and_negative_energy_fail(self) -> None:
        kwargs = dict(
            step_index=0,
            stage="FIELD",
            layer_ids=LAYER_IDS,
            signed_efficiency=(1.0,) * 5,
            raw_velocity=(1.0,) * 5,
            bf_velocity=(1.0,) * 5,
            applied_coefficient=(1.0,) * 5,
            active_direction_mask=(1,) * 5,
            raw_cap_bound_mask=(0,) * 5,
            bf_cap_bound_mask=(0,) * 5,
            raw_zero_bound_mask=(0,) * 5,
            bf_zero_bound_mask=(0,) * 5,
            predicted_progress_contribution=(1.0,) * 5,
            prequantized_update_energy=(1.0,) * 5,
            realized_bf16_update_energy=(1.0,) * 5,
            cumulative_bf16_capacity=(1.0,) * 5,
            bf16_capacity_contribution=(1.0,) * 5,
            structural_h_contribution=(1.0,) * 5,
            structural_p_contribution=(1.0,) * 5,
            trust_contribution=(1.0,) * 5,
            field_sha256="1" * 64,
            state_sha256="2" * 64,
            target_z_sha256="3" * 64,
            factor_state_sha256="4" * 64,
            raw_solver_certificate_sha256="5" * 64,
            bf_solver_certificate_sha256="6" * 64,
            candidate_sha256=None,
        )
        for key, replacement in (
            ("layer_ids", (5, 4, 6, 7, 8)),
            ("signed_efficiency", (1.0,) * 4),
            ("raw_velocity", (1.0, 1.0, math.nan, 1.0, 1.0)),
            ("active_direction_mask", (1, 1, 2, 1, 1)),
            ("active_direction_mask", (1.9, 1, 1, 1, 1)),
            ("signed_efficiency", (True, 1.0, 1.0, 1.0, 1.0)),
            ("realized_bf16_update_energy", (1.0, 1.0, -1.0, 1.0, 1.0)),
        ):
            changed = dict(kwargs)
            changed[key] = replacement
            with self.assertRaises(ODEBFContractError, msg=key):
                build_layer_routing_telemetry(**changed)

    def test_trajectory_reports_transient_collapse_switches_and_utilization(self) -> None:
        rows = (
            _row(
                step=0,
                applied=(0.9, 0.1, 0.0, 0.0, 0.0),
                realized=(9, 1, 0, 0, 0),
                stage="FIELD",
            ),
            _row(step=1, applied=(0.85, 0.15, 0.0, 0.0, 0.0), realized=(8.5, 1.5, 0, 0, 0)),
            _row(
                step=2,
                applied=(0.1, 0.4, 0.3, 0.1, 0.1),
                realized=(1, 4, 3, 1, 1),
            ),
        )
        observed = summarize_layer_routing_trajectory(
            rows, first_hit_step_index=1
        )
        coefficient = observed["distribution_trajectory"]["coefficient_share"]
        self.assertEqual(coefficient["longest_consecutive_severe_run"], 2)
        self.assertEqual(
            coefficient["severe_run_by_metric_type"]["TOP1_SHARE_GTE_0_80"],
            {"step_count": 2, "longest_consecutive_run": 2},
        )
        self.assertEqual(coefficient["top_layer_switch_count"], 1)
        self.assertEqual(len(coefficient["successive_step_jensen_shannon"]), 2)
        self.assertTrue(all(value >= 0.0 for value in coefficient["successive_step_jensen_shannon"]))
        self.assertEqual(observed["first_hit_step_index"], 1)
        self.assertEqual(
            observed["cumulative_per_layer_utilization"],
            [0.95, 0.55, 0.3, 0.1, 0.1],
        )
        self.assertEqual(
            observed["cumulative_per_layer_omega"],
            [9.5, 5.5, 3.0, 1.0, 1.0],
        )
        self.assertEqual(observed["table"][0]["trajectory_roles"], ["INITIAL"])
        self.assertEqual(
            observed["table"][1]["trajectory_roles"],
            ["ACCEPTED", "FIRST_HIT"],
        )
        self.assertEqual(
            observed["table"][2]["trajectory_roles"],
            ["ACCEPTED", "FINAL"],
        )

    def test_relabel_is_numeric_copy_and_rejected_rows_cannot_enter_summary(self) -> None:
        source = _row(step=1, stage="TRIAL")
        before = copy.deepcopy(source)
        rejected = relabel_layer_routing_telemetry(
            source, stage="REJECTED_TRANSITION"
        )
        self.assertEqual(source, before)
        self.assertEqual(rejected["stage"], "REJECTED_TRANSITION")
        self.assertEqual(
            rejected["source_telemetry_sha256"], source["identity_sha256"]
        )
        with self.assertRaises(ODEBFContractError):
            summarize_layer_routing_trajectory(
                (_row(step=0, stage="FIELD"), rejected),
                first_hit_step_index=None,
            )

    def test_emitted_payload_and_summary_pass_raw_firewall(self) -> None:
        initial = _row(step=0, stage="FIELD")
        accepted = relabel_layer_routing_telemetry(
            _row(step=1, stage="TRIAL"),
            stage="ACCEPTED_TRANSITION",
        )
        summary = summarize_layer_routing_trajectory(
            (initial, accepted), first_hit_step_index=1
        )
        assert_raw_free(initial, "layer routing field")
        assert_raw_free(accepted, "layer routing accepted")
        assert_raw_free(summary, "layer routing trajectory")
        contaminated = copy.deepcopy(accepted)
        contaminated["prompt"] = "forbidden"
        with self.assertRaises(MethodContractError):
            assert_raw_free(contaminated, "layer routing contaminated")

    def test_observation_does_not_change_decision_or_tau_transition(self) -> None:
        zero = np.zeros(5, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical",
            0.0,
            zero,
            np.zeros((5, 5), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(5, dtype=np.float64),
            np.eye(5, dtype=np.float64),
            np.eye(5, dtype=np.float64),
            2.0,
            np.ones(5, dtype=np.float64),
            0.5,
            1.0e-8,
            barrier,
            barrier,
        )
        velocity = np.full(5, 0.1, dtype=np.float64)
        before_problem = problem.identity()
        before_velocity = velocity.tobytes()
        expected = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        _row(step=0)
        observed = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        self.assertEqual(asdict(expected), asdict(observed))
        self.assertEqual(problem.identity(), before_problem)
        self.assertEqual(velocity.tobytes(), before_velocity)

        left = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        right = AdaptiveTauClock(adaptive_lock(AdaptiveVariant.FR_A8))
        left_transition = left.accept(trial=left.begin_trial(), rho=0.5)
        _row(step=1)
        right_transition = right.accept(trial=right.begin_trial(), rho=0.5)
        self.assertEqual(
            left_transition.raw_free_payload(), right_transition.raw_free_payload()
        )
        self.assertEqual(left.raw_free_payload(), right.raw_free_payload())


if __name__ == "__main__":
    unittest.main()
