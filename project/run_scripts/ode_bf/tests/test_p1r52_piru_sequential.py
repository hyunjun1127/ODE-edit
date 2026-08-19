from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p1_backend import (
    FULL_CURRENT_RESIDUAL_VELOCITY_DEFINITION,
    P1LayerField,
)
from project.run_scripts.ode_bf.p1_controller import (
    AcceptedLayerContribution,
    _cumulative_structural_terms,
)
from project.run_scripts.ode_bf.p1r52_pir_writer import (
    PIRWriterResult,
    P1R52PIRPolicy,
)
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX,
    P1R52_PIRU_SEQUENTIAL_METHOD_ID,
    P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
    P1R52_PIRU_SEQUENTIAL_ROLE,
    bind_piru_sequential_history,
    is_piru_structural_h_role,
    pir_policy_for_role,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    R52_H_ROLE,
    R52_ROLES,
    R52_STRUCTURAL_H_ROLES,
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_sequential_contract import (
    scoped_atomic_sequential_adapter,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r52_piru_sequential_panel import (
    LOCK_FILE,
    load_and_validate_lock,
    verify_reference_results,
)
from project.run_scripts import session05_ode_bf_p1r52_piru_sequential_b100x10_dry_plan as dry


class P1R52PIRUSequentialTests(unittest.TestCase):
    @staticmethod
    def _field(layer: int, history_width: int, *, offset: float = 0.0) -> P1LayerField:
        residual = torch.tensor(
            [[1.0 + offset, 2.0], [3.0, 4.0 + offset]], dtype=torch.float32
        )
        q = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]], dtype=torch.float32
        )
        factor = WaypointFactor(
            f"layer.{layer}.weight",
            layer,
            0,
            0,
            layer,
            1.0,
            residual.clone(),
            q.clone(),
            global_batch_size=2,
        )
        history = torch.arange(
            residual.shape[0] * history_width,
            dtype=torch.float64,
        ).reshape(residual.shape[0], history_width)
        return P1LayerField(
            layer,
            factor.weight_name,
            q.clone(),
            q.clone(),
            residual,
            FULL_CURRENT_RESIDUAL_VELOCITY_DEFINITION,
            1,
            q,
            factor,
            1.0,
            q.clone(),
            q.T.double() @ q.double(),
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            history,
        )

    @staticmethod
    def _result(fields: tuple[P1LayerField, ...]) -> PIRWriterResult:
        return PIRWriterResult(
            P1R52PIRPolicy.PIR_U,
            {},
            (1.0,) * 5,
            fields,
            0.0,
            None,  # type: ignore[arg-type]
            {},
            torch.empty((2, 2), dtype=torch.float32),
            torch.empty((2, 2), dtype=torch.float32),
            1.0,
            {"layers": [{} for _ in fields], "identity_sha256": "old"},
        )

    def test_nonzero_b2_history_is_rebound_through_k1_k2(self) -> None:
        entry_fields = tuple(self._field(layer, 100) for layer in range(4, 9))
        current_fields = tuple(
            self._field(layer, 0, offset=0.1) for layer in range(4, 9)
        )
        current_fields = (entry_fields[0], *current_fields[1:])
        result = bind_piru_sequential_history(
            self._result(current_fields),
            SimpleNamespace(layers=entry_fields),
        )
        rebound_fields = result.layer_fields
        for rebound, entry in zip(rebound_fields, entry_fields, strict=True):
            self.assertEqual(rebound.history_action.shape[1], 100)
            self.assertEqual(
                rebound.history_action.data_ptr(), entry.history_action.data_ptr()
            )
            self.assertTrue(torch.equal(rebound.history_action, entry.history_action))

        accepted = {
            field.layer: [AcceptedLayerContribution.from_field(field, field.factor)]
            for field in rebound_fields
        }
        dynamic = SimpleNamespace(layers=entry_fields)
        _cumulative_structural_terms(dynamic, accepted)
        for field in rebound_fields:
            accepted[field.layer].append(
                AcceptedLayerContribution.from_field(field, field.factor)
            )
        _cumulative_structural_terms(dynamic, accepted)

    def test_empty_b1_history_rebind_is_exact(self) -> None:
        entries = tuple(self._field(layer, 0) for layer in range(4, 9))
        currents = (entries[0],) + tuple(
            self._field(layer, 0, offset=0.1) for layer in range(5, 9)
        )
        rebound = bind_piru_sequential_history(
            self._result(currents),
            SimpleNamespace(layers=entries),
        )
        for current, entry in zip(rebound.layer_fields, entries, strict=True):
            self.assertEqual(current.history_action.shape, (2, 0))
            self.assertEqual(
                current.history_action.data_ptr(), entry.history_action.data_ptr()
            )
            self.assertTrue(torch.equal(current.history_action, entry.history_action))

    def test_piru_role_is_a_structural_h_r52_role(self) -> None:
        self.assertIn(P1R52_PIRU_SEQUENTIAL_ROLE, R52_ROLES)
        self.assertIn(P1R52_PIRU_SEQUENTIAL_ROLE, R52_STRUCTURAL_H_ROLES)
        self.assertIn(R52_H_ROLE, R52_STRUCTURAL_H_ROLES)
        self.assertTrue(is_piru_structural_h_role(P1R52_PIRU_SEQUENTIAL_ROLE))
        self.assertIs(
            pir_policy_for_role(P1R52_PIRU_SEQUENTIAL_ROLE),
            P1R52PIRPolicy.PIR_U,
        )
        self.assertIsNone(pir_policy_for_role(R52_H_ROLE))

    def test_piru_result_identity_is_create_once_and_b100_only(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                P1R52_PIRU_SEQUENTIAL_ROLE,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix=P1R52_PIRU_SEQUENTIAL_ATTEMPT_SUFFIX,
            ),
            P1R52_PIRU_SEQUENTIAL_RESULT_NAME,
        )

    def test_piru_dry_plan_has_one_cell_and_no_entry_evaluator(self) -> None:
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["job_count"], 1)
        self.assertEqual(plan["request_count"], 1000)
        self.assertEqual(plan["writer"], "PIR-U")
        self.assertEqual(plan["structural_h"], "ON")
        self.assertEqual(plan["batch_entry_evaluator_count"], 0)
        self.assertEqual(plan["retry_backtracking_line_search"], [0, 0, 0])

    def test_piru_runtime_uses_narrow_policy_switch(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        self.assertIn("pir_policy = pir_policy_for_role(role)", source)
        self.assertIn("p1r52_pir_policy=pir_policy", source)
        self.assertIn("structural_h_decision_enabled=structural_h_enabled", source)
        self.assertIn("P1R52_PIRU_SEQUENTIAL_METHOD_ID", source)
        adapter_source = inspect.getsource(scoped_atomic_sequential_adapter)
        self.assertIn("pir_history_rebind_enabled", adapter_source)
        self.assertIn("experiment_module.plan_pir_writer", adapter_source)
        self.assertIn("bind_piru_sequential_history", adapter_source)

    def test_numerical_lock_and_sealed_references(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        lock, lock_sha = load_and_validate_lock(
            repo / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(len(lock_sha), 64)
        self.assertEqual(lock["root_digest"], "95d2e909d0e59bbf8ce65f181778dfb23b38ef12735e84649674b5e2cfccea67")
        references = verify_reference_results()
        self.assertEqual(len(references["references"]), 3)
        self.assertTrue(all(item["W0_restored"] for item in references["references"].values()))


if __name__ == "__main__":
    unittest.main()
