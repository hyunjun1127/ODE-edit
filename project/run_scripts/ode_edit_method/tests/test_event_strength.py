from __future__ import annotations

import inspect
import math
import unittest
from types import SimpleNamespace

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    EventReading,
    MethodContractError,
)
from project.run_scripts.ode_edit_method.easyedit_backend import EasyEditMemitBackend
from project.run_scripts.ode_edit_method.event_strength import (
    absolute_event_metrics,
    assert_raw_free,
    build_event_strength_trace,
)
from project.run_scripts.ode_edit_method.events import event_from_log_likelihoods
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation


TAU = 0.1


def _reading(new: tuple[float, ...], old: tuple[float, ...]) -> EventReading:
    return event_from_log_likelihoods(new, old, tau=TAU, nfe=2)


def _history(source: str, reading: EventReading) -> dict[str, object]:
    holder = SimpleNamespace(_event_history=[])
    EasyEditMemitBackend._record_event(holder, source, reading)
    return holder._event_history[0]


class AbsoluteEventReadingTests(unittest.TestCase):
    def test_absolute_panels_are_retained_and_margin_aligned(self) -> None:
        reading = _reading((-3.0, -1.5), (-2.0, -2.0))
        self.assertEqual(reading.target_new_log_likelihoods, (-3.0, -1.5))
        self.assertEqual(reading.target_old_log_likelihoods, (-2.0, -2.0))
        self.assertEqual(reading.context_margins, (-1.0, 0.5))
        self.assertTrue(reading.has_absolute_likelihoods)
        with self.assertRaisesRegex(MethodContractError, "misaligned"):
            EventReading(
                hard_phi=1.0,
                smooth_phi=1.0,
                context_margins=(-1.0, 0.5),
                target_new_log_likelihoods=(-3.0,),
                target_old_log_likelihoods=(-2.0,),
            )
        with self.assertRaisesRegex(MethodContractError, "differs"):
            EventReading(
                hard_phi=1.0,
                smooth_phi=1.0,
                context_margins=(-0.9, 0.5),
                target_new_log_likelihoods=(-3.0, -1.5),
                target_old_log_likelihoods=(-2.0, -2.0),
            )

    def test_entry_anchored_metrics_are_uniform_and_recomputable(self) -> None:
        entry = _reading((-3.0, -3.0), (-2.0, -2.0))
        current = _reading((-1.0, -3.0), (-2.0, -2.0))
        metrics = absolute_event_metrics(
            current,
            entry,
            tau=TAU,
            denominator_epsilon=1e-10,
        )
        self.assertEqual(metrics["uniform_context_weights"], [0.5, 0.5])
        self.assertEqual(metrics["mean_new_log_likelihood"], -2.0)
        self.assertEqual(metrics["mean_old_log_likelihood"], -2.0)
        self.assertEqual(metrics["mean_margin"], 0.0)
        self.assertEqual(metrics["min_margin"], -1.0)
        self.assertEqual(metrics["entry_anchor_denominator_raw"], 1.0)
        self.assertEqual(metrics["entry_anchor_q"], 1.0)
        self.assertFalse(metrics["entry_anchor_denominator_degenerate"])

    def test_degenerate_entry_denominator_is_fail_safe_and_recorded(self) -> None:
        entry = _reading((-1.0, -1.0), (-2.0, -2.0))
        metrics = absolute_event_metrics(
            entry,
            entry,
            tau=TAU,
            denominator_epsilon=1e-6,
        )
        self.assertTrue(metrics["entry_anchor_denominator_degenerate"])
        self.assertEqual(metrics["entry_anchor_denominator"], 1e-6)
        self.assertEqual(metrics["entry_anchor_q"], 0.0)


class AbsoluteEventTraceTests(unittest.TestCase):
    def test_trace_maps_trials_commits_and_rollback_without_new_measurement(self) -> None:
        entry = _reading((-3.0, -3.0), (-2.0, -2.0))
        accepted_trial = _reading((-2.75, -2.25), (-2.0, -2.0))
        accepted_commit = _reading((-2.5, -2.5), (-2.0, -2.0))
        mean_positive_near_miss = _reading((-1.0, -3.0), (-2.0, -2.0))
        field = _reading((-2.9, -2.8), (-2.0, -2.0))
        result = {
            "status": "trust_rejection_limit",
            "omega_appended": False,
            "steps": [
                {"accepted": True},
                {"accepted": False},
            ],
        }
        trace = build_event_strength_trace(
            Arm.FULL_ODE_EDIT,
            result,
            (
                _history("event", entry),
                _history("field", field),
                _history("event", accepted_trial),
                _history("event", accepted_commit),
                _history("event", mean_positive_near_miss),
            ),
            tau=TAU,
            denominator_epsilon=1e-10,
            event_tolerance=1e-6,
        )
        roles = [row["role"] for row in trace["decision_observations"]]
        self.assertEqual(
            roles,
            ["entry", "trial", "accepted_committed", "trial"],
        )
        self.assertEqual(len(trace["field_observations"]), 1)
        self.assertEqual(trace["anchors"]["last_trial"]["mean_margin"], 0.0)
        self.assertEqual(trace["anchors"]["last_trial"]["min_margin"], -1.0)
        self.assertTrue(
            trace["diagnostics"][
                "failed_last_trial_mean_positive_worst_context_negative"
            ]
        )
        self.assertTrue(trace["anchors"]["rollback"]["measurement_reused_no_forward"])
        self.assertEqual(trace["anchors"]["rollback"]["entry_anchor_q"], 0.0)
        self.assertEqual(trace["diagnostics"]["additional_model_forwards"], 0)
        self.assertEqual(trace["diagnostics"]["additional_backwards"], 0)

    def test_native_endpoint_is_diagnostic_reference_only(self) -> None:
        entry = _reading((-3.0, -3.0), (-2.0, -2.0))
        endpoint = _reading((-1.0, -1.0), (-2.5, -2.5))
        trace = build_event_strength_trace(
            Arm.NATIVE_MEMIT,
            {"status": "event_hit", "omega_appended": True, "steps": [{"accepted": True}]},
            (_history("event", entry), _history("event", endpoint)),
            tau=TAU,
            denominator_epsilon=1e-10,
            event_tolerance=1e-6,
        )
        self.assertEqual(
            trace["diagnostics"]["native_absolute_nll_role"],
            "diagnostic-reference-only",
        )
        self.assertTrue(
            trace["diagnostics"][
                "relative_pass_with_mean_old_degradation_possible"
            ]
        )
        self.assertFalse(trace["diagnostics"]["under_write_claim_authorized"])

    def test_raw_firewall_and_source_have_no_forward_or_controller_dependency(self) -> None:
        with self.assertRaisesRegex(MethodContractError, "forbidden raw"):
            assert_raw_free({"prompt": "secret"})
        source = inspect.getsource(absolute_event_metrics) + inspect.getsource(
            build_event_strength_trace
        )
        for forbidden in (
            "model(",
            "torch.autograd",
            "controller_config",
            "model_alias",
            "llama3",
            "qwen2",
        ):
            self.assertNotIn(forbidden, source)


class PathAwareRawFirewallTests(unittest.TestCase):
    def test_actual_compute_snapshot_zero_accounting_is_allowed(self) -> None:
        snapshot = EditInstrumentation("firewall-snapshot").finalize().to_dict()
        self.assertEqual(snapshot["component_wall_seconds"]["evaluation"], 0.0)
        self.assertEqual(snapshot["component_gpu_seconds"]["evaluation"], 0.0)
        assert_raw_free({"schema_version": "p0-compute", **snapshot})
        assert_raw_free({"p1_compute_row": snapshot})

    def test_exact_zero_accounting_leaf_paths_are_allowed(self) -> None:
        for parent in ("component_wall_seconds", "component_gpu_seconds"):
            for value in (0, 0.0):
                with self.subTest(parent=parent, value=value):
                    assert_raw_free({parent: {"evaluation": value}})
                    assert_raw_free({"row": {parent: {"evaluation": value}}})

    def test_zero_accounting_rejects_invalid_values(self) -> None:
        invalid = (True, 1, -1.0, "0", [], {}, math.nan, math.inf, -math.inf)
        for parent in ("component_wall_seconds", "component_gpu_seconds"):
            for value in invalid:
                with self.subTest(parent=parent, value=repr(value)):
                    with self.assertRaises(MethodContractError):
                        assert_raw_free({parent: {"evaluation": value}})

    def test_evaluation_outside_exact_accounting_paths_fails_closed(self) -> None:
        forbidden = (
            {"evaluation": 0.0},
            {"other": {"evaluation": 0.0}},
            {"component_wall_seconds": {"payload": {"evaluation": 0.0}}},
            {"component_gpu_seconds": {"evaluation": {"value": 0.0}}},
            {"component_wall_seconds": [{"evaluation": 0.0}]},
        )
        for payload in forbidden:
            with self.subTest(payload=payload):
                with self.assertRaises(MethodContractError):
                    assert_raw_free(payload)

    def test_generation_and_other_raw_fields_remain_forbidden_everywhere(self) -> None:
        for key in (
            "generation",
            "prompt",
            "subject",
            "target_new",
            "target_old",
            "raw_context",
            "context_templates",
            "templates",
        ):
            for payload in (
                {key: 0.0},
                {"component_wall_seconds": {key: 0.0}},
                {"row": [{key: "secret"}]},
            ):
                with self.subTest(key=key, payload=payload):
                    with self.assertRaises(MethodContractError):
                        assert_raw_free(payload)


if __name__ == "__main__":
    unittest.main()
