from __future__ import annotations

import ast
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import torch

from project.run_scripts import session05_ode_bf_p1r52_sequential_b100x10_fourcell_dry_plan as dry
from project.run_scripts.ode_bf.p1_backend import _validate_history_keys
from project.run_scripts.ode_bf.p1_evaluator import (
    PrefixNLLPair,
    _ordered_request_digest_for_batch,
    _metric_receipt,
    _prompt_metric_receipt,
)
from project.run_scripts.ode_bf.analysis import PrimaryMetric
from project.run_scripts.ode_bf.p1_state import P1HistoryLedger, P1HistoryRecord
from project.run_scripts.ode_bf.p1r29_sequential_preparation import SequentialArmState
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    verify_b10_prefix,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    RESULT_NAMES,
    RESULT_NAMES_B100X10,
    RESULT_NAMES_B100X10_TECH_R1,
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import (
    P1R52_B10X10_SCALE,
    P1R52_B100X10_SCALE,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import P1R23_LAYER_ORDER
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_scalable_v1


REPO_ROOT = Path(__file__).resolve().parents[4]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records(version: int, count: int) -> tuple[P1HistoryRecord, ...]:
    return tuple(
        P1HistoryRecord(
            _sha(f"request-{version}-{index}"),
            version * 1000 + index,
            _sha(f"collision-{version}-{index}"),
            _sha(f"target-{version}-{index}"),
            _sha(f"event-{version}"),
            version,
        )
        for index in range(count)
    )


class P1R52SequentialB100x10Tests(unittest.TestCase):
    def test_scale_is_explicit_and_b10_default_is_unchanged(self) -> None:
        self.assertEqual(P1R52_B10X10_SCALE.batch_size, 10)
        self.assertEqual(P1R52_B10X10_SCALE.request_count, 100)
        self.assertEqual(P1R52_B100X10_SCALE.batch_size, 100)
        self.assertEqual(P1R52_B100X10_SCALE.request_count, 1000)
        self.assertEqual(P1R52_B100X10_SCALE.history_counts, tuple(range(0, 1000, 100)))
        for role, name in RESULT_NAMES.items():
            self.assertEqual(
                expected_p1r52_sequential_result_name("llama3-8b-inst", role),
                name,
            )
        self.assertEqual(len(RESULT_NAMES_B100X10), 4)
        self.assertTrue(all("10xb100" in name for name in RESULT_NAMES_B100X10.values()))
        self.assertTrue(all("tech-r1" in name for name in RESULT_NAMES_B100X10_TECH_R1.values()))

    def test_stream_is_outcome_free_unique_and_preserves_b10_prefix(self) -> None:
        locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
        stream = verify_p1r52_b100x10_stream(
            json.loads((locks / "p1r52_sequential_b100x10_stream_seal.json").read_text())
        )
        prefix = json.loads(
            (locks / "p1r24_independent_b10x10_stream_seal.json").read_text()
        )
        verify_b10_prefix(prefix, stream)
        self.assertEqual(len(stream["requests"]), 1000)
        self.assertEqual(len({row["request_sha256"] for row in stream["requests"]}), 1000)
        self.assertEqual(stream["model_import_or_access_count"], 0)
        self.assertEqual(stream["evaluator_import_or_access_count"], 0)

    def test_history_geometry_accepts_only_b100_multiples_to_nine_hundred(self) -> None:
        for width in range(0, 901, 100):
            values = {
                layer: torch.zeros((3, width), dtype=torch.float32)
                for layer in P1R23_LAYER_ORDER
            }
            observed = _validate_history_keys(
                values,
                P1R23_LAYER_ORDER,
                maximum_history_columns=900,
                batch_size=100,
            )
            self.assertEqual({value.shape[1] for value in observed.values()}, {width})
        with self.assertRaises(Exception):
            _validate_history_keys(
                {
                    layer: torch.zeros((3, 50), dtype=torch.float32)
                    for layer in P1R23_LAYER_ORDER
                },
                P1R23_LAYER_ORDER,
                maximum_history_columns=900,
                batch_size=100,
            )

    def test_history_ledger_b100_finalize_and_idempotent_replay(self) -> None:
        ledger = P1HistoryLedger(
            layer_order=P1R23_LAYER_ORDER,
            maximum_records=1000,
            batch_size=100,
        )
        values = {
            layer: torch.full((3, 100), float(layer), dtype=torch.float32)
            for layer in P1R23_LAYER_ORDER
        }
        prospective = ledger.prospective(
            transaction_id="b100-1",
            expected_version=0,
            records=_records(1, 100),
            solve_keys_by_layer=values,
            risk_keys_by_layer=values,
        )
        receipt = ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={layer: 1.0 for layer in P1R23_LAYER_ORDER},
        )
        replay = ledger.finalize(
            prospective,
            post_commit_verified=True,
            load_increment_by_layer={layer: 1.0 for layer in P1R23_LAYER_ORDER},
        )
        self.assertEqual(receipt.appended_count, 100)
        self.assertEqual(ledger.snapshot().version, 1)
        self.assertEqual(len(ledger.snapshot().active_records), 100)
        self.assertEqual(replay.appended_count, 0)
        self.assertTrue(replay.idempotent_replay)

    def test_sequential_state_uses_explicit_b100_ledger_without_changing_default(self) -> None:
        default = SequentialArmState("default", P1R23_LAYER_ORDER)
        scaled = SequentialArmState(
            "scaled",
            P1R23_LAYER_ORDER,
            maximum_history_records=1000,
            history_batch_size=100,
        )
        self.assertEqual(default.ledger.batch_size, 10)
        self.assertEqual(default.ledger.maximum_records, 100)
        self.assertEqual(scaled.ledger.batch_size, 100)
        self.assertEqual(scaled.ledger.maximum_records, 1000)

    def test_runtime_uses_scale_for_batch_history_and_final_rows(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        tree = ast.parse(source)
        attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "scale"
        }
        self.assertTrue(
            {"batch_size", "round_count", "request_count", "history_counts"}.issubset(attributes)
        )
        self.assertIn("expected_batch_size=scale.batch_size", source)
        self.assertIn("batch_size=scale.batch_size", source)

    def test_evaluator_receipts_accept_b100_without_changing_metric_arithmetic(self) -> None:
        scores = tuple((PrefixNLLPair(0.5, 1.0),) for _ in range(100))
        official = _metric_receipt(PrimaryMetric.EFFICACY, scores)
        prompt = _prompt_metric_receipt("rewrite_success", ((1,) for _ in range(100)), scores)
        self.assertEqual(official.numerator, 100)
        self.assertEqual(official.denominator, 100)
        self.assertEqual(prompt.prompt_numerator, 100)
        self.assertEqual(prompt.strict_request_denominator, 100)

    def test_tech_r1_uses_existing_scalable_digest_only_for_b100(self) -> None:
        values = tuple(_sha(f"digest-{index}") for index in range(100))
        self.assertEqual(
            _ordered_request_digest_for_batch(values, expected_batch_size=100),
            ordered_request_digest_scalable_v1(values),
        )

    def test_four_cell_plan_is_antialiased_and_single_wave(self) -> None:
        plan = dry.build_plan("1" * 40)
        self.assertEqual(plan["array"], "0-3%4")
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual([job["array_index"] for job in plan["jobs"]], list(range(4)))
        self.assertEqual(len({job["role"] for job in plan["jobs"]}), 4)
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertEqual(plan["request_count_per_job"], 1000)
        self.assertEqual(plan["monitor_cadence_after_initial_gate_seconds"], 3600)


if __name__ == "__main__":
    unittest.main()
