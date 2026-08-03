from __future__ import annotations

import time
import unittest

from project.run_scripts.ode_edit_method.contracts import (
    EventReading,
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.retry import RejectedRetryCache


def _field(state_id: str) -> tuple[EventReading, ProposalBatch]:
    return (
        EventReading(hard_phi=1.0, smooth_phi=1.0, context_margins=(-1.0,)),
        ProposalBatch(
            snapshot_id=state_id,
            proposals=(
                LayerProposal(
                    layer=0,
                    state_id=state_id,
                    direction_id="direction-0",
                ),
            ),
            slopes=(1.0,),
            semantics=ProposalSemantics.CURRENT_COORDINATE,
        ),
    )


class InstrumentationTests(unittest.TestCase):
    def test_schema_timers_and_first_hit_freeze(self) -> None:
        metrics = EditInstrumentation("edit-1")
        with metrics.component("direct_z"):
            time.sleep(0.001)
        metrics.increment("N_z")
        metrics.increment("N_field")
        metrics.increment("N_bw")
        metrics.increment("N_trial")
        metrics.increment("K_acc")
        metrics.increment("N_write")
        metrics.mark_first_hit()

        for counter in (
            "N_z",
            "N_state_fwd",
            "N_field",
            "N_bw",
            "N_trial",
            "N_write",
            "K_acc",
        ):
            with self.subTest(counter=counter):
                with self.assertRaises(MethodContractError):
                    metrics.increment(counter)
        with self.assertRaises(MethodContractError):
            with metrics.component("field"):
                pass

        # Held-out evaluation is a separate post-action phase.
        with metrics.component("evaluation"):
            metrics.increment("N_eval")
        snapshot = metrics.finalize()
        payload = snapshot.to_dict()
        self.assertEqual(payload["schema_version"], "ode-edit-compute-accounting/v1")
        self.assertEqual(payload["counters"]["N_z"], 1)
        self.assertEqual(payload["counters"]["N_eval"], 1)
        self.assertGreater(payload["component_cpu_seconds"]["direct_z"], 0.0)
        self.assertEqual(payload["gpu_seconds_per_edit"], 0.0)
        self.assertEqual(payload["controller_gpu_seconds"], 0.0)
        self.assertEqual(payload["evaluation_gpu_seconds"], 0.0)
        self.assertEqual(payload["peak_memory_allocated_bytes"], 0)
        with self.assertRaises(MethodContractError):
            metrics.finalize()

    def test_unknown_or_nested_component_fails_closed(self) -> None:
        metrics = EditInstrumentation("edit-2")
        with self.assertRaises(MethodContractError):
            metrics.increment("unknown")
        with self.assertRaises(MethodContractError):
            with metrics.component("unknown"):
                pass
        with metrics.component("field"):
            with self.assertRaises(MethodContractError):
                with metrics.component("field"):
                    pass


class RetryReuseTests(unittest.TestCase):
    def test_exact_retry_reuses_field_and_identical_qp(self) -> None:
        metrics = EditInstrumentation("edit-retry")
        cache = RejectedRetryCache()
        calls = {"field": 0, "qp": 0}

        def build() -> tuple[EventReading, ProposalBatch]:
            calls["field"] += 1
            return _field("state-a")

        event1, batch1, reused1 = cache.get_or_build_field(
            current_state_id="state-a",
            field_payload={"request": "r", "arm": "full"},
            build=build,
            instrumentation=metrics,
        )
        event2, batch2, reused2 = cache.get_or_build_field(
            current_state_id="state-a",
            field_payload={"request": "r", "arm": "full"},
            build=build,
            instrumentation=metrics,
        )
        self.assertFalse(reused1)
        self.assertTrue(reused2)
        self.assertIs(event1, event2)
        self.assertIs(batch1, batch2)
        self.assertEqual(calls["field"], 1)

        def solve() -> object:
            calls["qp"] += 1
            return object()

        qp1, qp_reused1 = cache.get_or_solve_qp(
            current_state_id="state-a",
            qp_payload={"radius": 1.0, "progress": 0.5},
            solve=solve,
            instrumentation=metrics,
        )
        qp2, qp_reused2 = cache.get_or_solve_qp(
            current_state_id="state-a",
            qp_payload={"progress": 0.5, "radius": 1.0},
            solve=solve,
            instrumentation=metrics,
        )
        self.assertFalse(qp_reused1)
        self.assertTrue(qp_reused2)
        self.assertIs(qp1, qp2)

        # Contracting trust changes only the cheap QP solve, not the field.
        _qp3, qp_reused3 = cache.get_or_solve_qp(
            current_state_id="state-a",
            qp_payload={"radius": 0.5, "progress": 0.5},
            solve=solve,
            instrumentation=metrics,
        )
        self.assertFalse(qp_reused3)
        self.assertEqual(calls, {"field": 1, "qp": 2})
        cache.assert_exact_rollback("state-a", metrics)

        snapshot = metrics.finalize().to_dict()
        self.assertEqual(snapshot["counters"]["N_field"], 1)
        self.assertEqual(snapshot["counters"]["N_reject"], 1)

    def test_rollback_state_mismatch_invalidates_cache(self) -> None:
        metrics = EditInstrumentation("edit-mismatch")
        cache = RejectedRetryCache()
        cache.get_or_build_field(
            current_state_id="state-a",
            field_payload={"request": "r"},
            build=lambda: _field("state-a"),
            instrumentation=metrics,
        )
        with self.assertRaises(MethodContractError):
            cache.assert_exact_rollback("state-b", metrics)
        self.assertIsNone(cache.state_id)


if __name__ == "__main__":
    unittest.main()
