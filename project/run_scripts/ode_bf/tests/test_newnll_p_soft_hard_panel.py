from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts import (
    session05_ode_bf_newnll_p_soft_hard_dry_plan as dry,
)
from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime
from project.run_scripts.ode_bf.artifacts import load_rooted_json
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional_p_secant import FunctionalPFieldPolicy
from project.run_scripts.ode_bf.p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_newnll_p_soft_hard_panel import (
    NEWNLL_P_SOFT_HARD_PANEL_LABELS,
    _field_common_projection,
    _local_p_model_diagnostic,
    _matched_policy_prefix,
    expected_newnll_p_soft_hard_result_name,
    forecast_newnll_p_soft_hard_panel,
    newnll_p_soft_hard_panel_specs,
    validate_newnll_p_soft_hard_lock,
    validate_pctrl_probe_control,
)
from project.run_scripts.ode_bf.routing import (
    PreservationConstraintPolicy,
    QuadraticBarrier,
    RoutingProblem,
    verify_backtracked_candidate,
)
from project.run_scripts.ode_bf.target_new_nll import RoutingObjective


ROOT = Path(__file__).resolve().parents[4]


class NewNLLPSoftHardPanelTests(unittest.TestCase):
    def test_three_arm_panel_changes_only_field_or_p_decision_policy(self) -> None:
        specs = newnll_p_soft_hard_panel_specs()
        self.assertEqual(tuple(item.label for item in specs), NEWNLL_P_SOFT_HARD_PANEL_LABELS)
        self.assertEqual(
            tuple(item.routing_objective for item in specs),
            (RoutingObjective.TARGET_NEW_NLL,) * 3,
        )
        self.assertEqual(
            tuple(item.functional_p_decision for item in specs),
            (
                FunctionalPDecisionPolicy.PCTRL,
                FunctionalPDecisionPolicy.PCTRL,
                FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            ),
        )
        self.assertEqual(
            tuple(item.functional_p_field_policy for item in specs),
            (
                FunctionalPFieldPolicy.PROBE_ONLY,
                FunctionalPFieldPolicy.SOFT_HARD,
                FunctionalPFieldPolicy.PROBE_ONLY,
            ),
        )
        self.assertEqual(
            tuple(item.preservation_constraints for item in specs),
            (PreservationConstraintPolicy.LOCKED,) * 3,
        )

    def test_lock_and_forecast_fit_the_existing_pair_envelope(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/"
            "numerical_lock_s05_newnll_p_soft_hard.json"
        )
        value, _ = load_rooted_json(
            path,
            expected_schema=(
                "ode-edit-s05-newnll-p-soft-hard-p1r5-numerical-lock/v1"
            ),
        )
        validate_newnll_p_soft_hard_lock(
            value,
            controller_identity_sha256=P1ControllerLock().identity(),
            stream_root_digest=value["stream_root_digest"],
            population_root_digest=value["p_population_root_digest"],
        )
        changed = copy.deepcopy(value)
        changed["functional_replay_soft_hard"]["h_ref"] = 0.25
        with self.assertRaisesRegex(ODEBFContractError, "numerical lock"):
            validate_newnll_p_soft_hard_lock(
                changed,
                controller_identity_sha256=P1ControllerLock().identity(),
                stream_root_digest=value["stream_root_digest"],
                population_root_digest=value["p_population_root_digest"],
            )
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            forecast = forecast_newnll_p_soft_hard_panel(
                ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
                ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
                alias,
            )
            self.assertEqual(forecast.conservative_time_seconds, 83_601)
            self.assertLessEqual(forecast.conservative_gpu_peak_mib, 65_000)
            self.assertLessEqual(forecast.conservative_host_peak_mib, 65_000)
        self.assertEqual(
            expected_newnll_p_soft_hard_result_name("llama3-8b-inst"),
            "s05-newnll-p-soft-hard-p1r5-llama3-8b-inst-v1-r2",
        )
        plan = dry.build_plan("c" * 40, repository_root=ROOT)
        self.assertEqual(plan["panel_labels"], list(NEWNLL_P_SOFT_HARD_PANEL_LABELS))
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["slurm_submit"])
        self.assertEqual(plan["new_pair_gpu"], 2)
        self.assertEqual(plan["server1_project_gpu_cap"], 3)

    def test_pctrl_probe_reproduces_the_locked_semantic_control(self) -> None:
        spec = newnll_p_soft_hard_panel_specs()[0]
        rollout = SimpleNamespace(
            routing_objective="TARGET_NEW_NLL",
            functional_p_decision="PCTRL",
            functional_p_field_policy="PROBE_ONLY",
            status="SAME_STATE_RETRY_EXHAUSTED",
            termination_label="SAME_STATE_RETRY_EXHAUSTED",
            accepted_t=Fraction(1, 8),
            k_acc=1,
            n_trial=6,
            n_reject=5,
            field_build_count=2,
            snapshots=[
                SimpleNamespace(
                    snapshot_sha256=(
                        "d3d601af7dcb2f8835766f72f7a11e8112ee8a26b3205c99a2b0fae382ca1ce4"
                    )
                )
            ],
        )
        validate_pctrl_probe_control("llama3-8b-inst", spec, rollout)
        changed = copy.copy(rollout)
        changed.accepted_t = Fraction(1, 4)
        with self.assertRaisesRegex(ODEBFContractError, "HOLD_INVALID_CONTROL"):
            validate_pctrl_probe_control("llama3-8b-inst", spec, changed)

    def test_actual_hard_h_and_p_gates_remain_in_the_trial_path(self) -> None:
        source = inspect.getsource(runtime._run_trial)
        self.assertIn("functional.historical.passed", source)
        self.assertIn("functional_p_decision.decision_pass", source)
        self.assertIn("predicted_candidate_controller_p", source)
        self.assertNotIn("predicted_pass_at_locked_budget,\n        authoritative", source)
        zero = np.zeros(2, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical", 0.0, zero, np.zeros((2, 2)), 1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(2), np.eye(2), np.eye(2), 2.0, np.ones(2),
            0.5, 1.0e-8, barrier,
            QuadraticBarrier(
                "pretrained", 0.0, zero, np.zeros((2, 2)), 1.0,
                "layer-local-diagonal",
            ),
        )
        rejected_h = verify_backtracked_candidate(
            problem, np.asarray((0.25, 0.25)), beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=False,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        rejected_p = verify_backtracked_candidate(
            problem, np.asarray((0.25, 0.25)), beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=True,
            functional_p_pass=False,
            authoritative_bf16_pass=True,
        )
        self.assertFalse(rejected_h.accepted)
        self.assertFalse(rejected_p.accepted)

    def test_probe_is_field_cached_and_retry_does_not_recompute_it(self) -> None:
        helper = inspect.getsource(runtime._functional_p_probe_and_transform)
        variant = inspect.getsource(runtime._run_variant)
        self.assertIn("for ordinal in range(dimension)", helper)
        self.assertIn("same_state_retry_cache", helper)
        self.assertIn("if not outcome.gate_accepted", variant)
        adaptive = variant.split("clock = AdaptiveTauClock", 1)[1]
        reject_branch = adaptive.split("if not outcome.gate_accepted:", 1)[1].split(
            "accepted_t = clock.tau", 1
        )[0]
        self.assertNotIn("_build_active_field", reject_branch)
        self.assertNotIn("_functional_p_probe_and_transform", reject_branch)

    def test_no_raw_or_heldout_payload_enters_the_soft_module(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/functional_p_secant.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "target_true", "paraphrase", "neighborhood", "generation",
            "model.generate", "prompt", "subject",
        ):
            self.assertNotIn(forbidden, source)

    def test_probe_only_overlay_is_six_call_cached_and_rng_neutral(self) -> None:
        layer_ids = (4, 5, 6, 7, 8)
        layers = tuple(
            SimpleNamespace(
                layer=layer,
                weight_name=f"layers.{ordinal}.weight",
                residual=torch.ones((2, 10), dtype=torch.float32),
                q=torch.ones((2, 10), dtype=torch.float32),
            )
            for ordinal, layer in enumerate(layer_ids)
        )
        field = SimpleNamespace(identity_sha256="f" * 64, layers=layers)
        zero = np.zeros(5, dtype=np.float64)
        problem = RoutingProblem(
            np.ones(5), np.eye(5), np.eye(5), 3.0, np.ones(5),
            0.5, 1.0e-8,
            QuadraticBarrier(
                "historical", 0.0, zero, np.zeros((5, 5)), 10.0,
                "layer-local-diagonal",
            ),
            QuadraticBarrier(
                "pretrained", 0.0, zero, np.zeros((5, 5)), 10.0,
                "layer-local-diagonal",
            ),
        )
        sample = "a" * 64
        empty = runtime.canonical_hash([])

        def risk(barrier: str, damage: float, count: int, order: str):
            return SimpleNamespace(
                barrier=barrier,
                item_count=count,
                sample_order_sha256=order,
                budget=1.0e-3,
                mean_positive_damage=damage,
                smooth_max_positive_damage=damage,
                raw_max_positive_damage=damage,
                signed_mean_damage=damage,
                decision_rule=(
                    "uniform-mean-positive-part"
                    if barrier == "pretrained-theta0-teacher"
                    else "mean-and-smooth-max"
                ),
                passed=damage <= 1.0e-3,
            )

        values = [5.0e-4, 6.0e-4, 7.0e-4, 4.0e-4, 5.0e-4, 8.0e-4]
        pairs = [
            SimpleNamespace(
                pretrained=risk("pretrained-theta0-teacher", value, 10, sample),
                historical=risk("historical-current-teacher", 0.0, 0, empty),
                pretrained_entry_receipt=SimpleNamespace(
                    request_order_sha256=sample,
                    value_sha256="b" * 64,
                ),
                historical_entry_receipt=SimpleNamespace(
                    value_sha256="c" * 64,
                ),
                pretrained_trial_receipt=SimpleNamespace(
                    value_sha256=f"{index + 1:064x}"
                ),
            )
            for index, value in enumerate(values)
        ]
        target = torch.zeros((2, 10), dtype=torch.float32)
        capture = SimpleNamespace(
            entry_sha256={
                f"layers.{index}.weight": "d" * 64 for index in range(5)
            }
        )
        factor_state = runtime._factor_state(capture.entry_sha256, {}, target)
        replay = SimpleNamespace(
            pretrained_baseline=SimpleNamespace(
                sample_order_sha256=sample,
                entry_kl_identity_sha256="b" * 64,
            ),
            history_entry_sha256="c" * 64,
            schedule_sha256="e" * 64,
        )
        history = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(digest="h" * 64)
        )
        schedule = SimpleNamespace(state_digest="s" * 64)
        ledger = runtime.ComputeLedger()
        model = torch.nn.Linear(2, 2, bias=False)
        rng_before = torch.get_rng_state().clone()
        with mock.patch.object(
            runtime, "_functional_trial", side_effect=pairs
        ) as observed:
            final, payload = runtime._functional_p_probe_and_transform(
                model,
                object(),
                alias="llama3-8b-inst",
                field=field,
                accepted_index=0,
                factors={},
                target_state=target,
                capture=capture,
                problem=problem,
                pre_soft_velocity=(0.1,) * 5,
                replay_entry=replay,
                theta0_cache=SimpleNamespace(),
                lock=P1ControllerLock(),
                ledger=ledger,
                touched={"weight": model.weight},
                history=history,
                schedule=schedule,
                policy=FunctionalPFieldPolicy.PROBE_ONLY,
                factor_state_sha256=factor_state,
            )
        self.assertEqual(observed.call_count, 6)
        np.testing.assert_array_equal(final, np.asarray((0.1,) * 5))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertFalse(payload["historical_soft_active"])
        self.assertEqual(payload["historical_soft_reason"], "EMPTY_HISTORY")
        self.assertEqual(payload["field_decision_influence_count"], 0)
        self.assertEqual(payload["probe_cost"]["actuator_probe_count"], 5)

    def test_common_field_projection_excludes_policy_specific_field_hash(self) -> None:
        common = {
            "signed_slopes": [1.0, 2.0],
            "raw_velocity": [0.25, 0.5],
            "functional_p_field": {
                "pre_soft_velocity": [0.25, 0.5],
                "controller_p_sample_order_sha256": "a" * 64,
                "controller_p_baseline_identity_sha256": "b" * 64,
                "factor_state_sha256": "c" * 64,
                "secant": {"cache_identity_sha256": "d" * 64},
                "generic_hp_soft_capable": True,
                "historical_soft_active": False,
                "historical_soft_reason": "EMPTY_HISTORY",
            },
        }
        control = {**copy.deepcopy(common), "field_sha256": "e" * 64}
        soft = {**copy.deepcopy(common), "field_sha256": "f" * 64}
        self.assertEqual(
            _field_common_projection(control),
            _field_common_projection(soft),
        )
        self.assertNotEqual(control["field_sha256"], soft["field_sha256"])
        changed = copy.deepcopy(soft)
        changed["raw_velocity"][0] = 0.125
        self.assertNotEqual(
            _field_common_projection(control),
            _field_common_projection(changed),
        )

    def test_matched_prefix_compares_common_field_not_policy_hash(self) -> None:
        def field(identity: str, raw_velocity: list[float]) -> dict[str, object]:
            return {
                "category": "field",
                "field_sha256": identity,
                "signed_slopes": [1.0, 2.0],
                "raw_velocity": raw_velocity,
                "functional_p_field": {
                    "pre_soft_velocity": [0.25, 0.5],
                    "controller_p_sample_order_sha256": "a" * 64,
                    "controller_p_baseline_identity_sha256": "b" * 64,
                    "factor_state_sha256": "c" * 64,
                    "secant": {"cache_identity_sha256": "d" * 64},
                    "generic_hp_soft_capable": True,
                    "historical_soft_active": False,
                    "historical_soft_reason": "EMPTY_HISTORY",
                },
            }

        def trial(identity: str, accepted: bool) -> dict[str, object]:
            return {
                "category": "trial",
                "snapshot_sha256": "1" * 64,
                "routing": {
                    "field_sha256": identity,
                    "coefficient": [0.125, 0.25],
                },
                "functional_p_decision": {"observation_sha256": "2" * 64},
                "gate_accepted": accepted,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_root, right_root = root / "left", root / "right"
            left_root.mkdir()
            right_root.mkdir()
            for target, value in (
                (left_root / "field-0000.json", field("e" * 64, [0.25, 0.5])),
                (right_root / "field-0000.json", field("f" * 64, [0.25, 0.5])),
                (
                    left_root / "field-0001.json",
                    {"category": "field", "field_sha256": "3" * 64},
                ),
                (
                    right_root / "field-0001.json",
                    {"category": "field", "field_sha256": "4" * 64},
                ),
                (left_root / "trial-0000.json", trial("e" * 64, False)),
                (right_root / "trial-0000.json", trial("f" * 64, True)),
            ):
                target.write_text(json.dumps(value), encoding="utf-8")
            left = SimpleNamespace(recorder=SimpleNamespace(root=left_root))
            right = SimpleNamespace(recorder=SimpleNamespace(root=right_root))
            result = _matched_policy_prefix(left, right)
            self.assertEqual(result["first_policy_caused_divergence_trial"], 0)
            self.assertEqual(
                result["common_candidate_count_through_policy_divergence"], 1
            )
            (right_root / "field-0000.json").write_text(
                json.dumps(field("f" * 64, [0.125, 0.5])), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ODEBFContractError, "initial field or probe differs"
            ):
                _matched_policy_prefix(left, right)

    def test_local_p_model_hold_uses_only_accepted_trial_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = (
                (True, 5.0e-4, 7.0e-4, 8.0e-4),
                (False, 5.0e-4, 7.0e-4, 9.0e-3),
                (True, 5.0e-4, 4.0e-4, 2.0e-3),
            )
            for ordinal, (accepted, baseline, predicted, actual) in enumerate(rows):
                (root / f"trial-{ordinal:04d}.json").write_text(
                    json.dumps(
                        {
                            "category": "trial",
                            "ordinal": ordinal,
                            "gate_accepted": accepted,
                            "functional_p_probe_prediction": {
                                "baseline_damage": baseline,
                                "predicted_candidate_controller_p": predicted,
                                "actual_candidate_controller_p": actual,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            rollout = SimpleNamespace(recorder=SimpleNamespace(root=root))
            diagnostic = _local_p_model_diagnostic(rollout)
        self.assertEqual(diagnostic["status"], "HOLD_LOCAL_P_MODEL")
        self.assertEqual(diagnostic["accepted_trial_count"], 2)
        self.assertEqual(diagnostic["violating_trial_ordinals"], [2])
        self.assertFalse(
            diagnostic["accepted_trial_receipts"][0]["sign_disagreement"]
        )
        self.assertTrue(
            diagnostic["accepted_trial_receipts"][1]["sign_disagreement"]
        )


if __name__ == "__main__":
    unittest.main()
