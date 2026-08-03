from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.oracle_absolute_lock import (
    load_oracle_absolute_lock,
)
from project.run_scripts.ode_edit_method.oracle_lock import load_oracle_lock
from project.run_scripts.session02_oracle_absolute_mean_margin_v3_p0 import (
    dry_plan as p0_dry_plan,
)
from project.run_scripts.session02_oracle_absolute_mean_margin_v3_p1 import (
    dry_plan as p1_dry_plan,
)
import project.run_scripts.session02_oracle_absolute_mean_margin_v3_p0 as p0_runner
import project.run_scripts.session02_oracle_absolute_mean_margin_v3_p1 as p1_runner


REPO = Path(__file__).resolve().parents[4]
LOCK_PATH = REPO / "project/run_scripts/ode_edit_method/oracle_absolute_mean_margin_v3.json"


class OracleAbsoluteV3LockTests(unittest.TestCase):
    def test_lock_and_pair_dry_plans_are_common_and_non_authorizing(self) -> None:
        lock = load_oracle_absolute_lock()
        self.assertEqual(
            lock["instruction_id"],
            "ODEEDIT-S02-ORACLE-ABSOLUTE-MEAN-MARGIN-V3-P0P1",
        )
        event = lock["event"]
        self.assertEqual(event["rho"], 0.5)
        self.assertEqual(event["required_mean_margin"], 0.0)
        self.assertFalse(event["q_margin_decision"])
        self.assertEqual(event["calibration_total_model_forwards"], 4)
        self.assertEqual(event["oracle_extra_model_forwards"], 2)
        self.assertEqual(event["oracle_backward_count"], 0)
        self.assertFalse(
            lock["legacy_diagnostic_provenance"]["legacy_replay_executed"]
        )
        for plan in (p0_dry_plan(lock), p1_dry_plan(lock)):
            self.assertFalse(plan["submission_authorized"])
            self.assertEqual(plan["pair_gpu"], 2)
            self.assertEqual(plan["server1_project_gpu_cap"], 3)
            self.assertEqual(
                [job["model_alias"] for job in plan["jobs"]],
                ["llama3-8b-inst", "qwen2.5-7b-inst"],
            )
        self.assertNotEqual(
            p0_dry_plan(lock)["jobs"][0]["output_root"],
            p1_dry_plan(lock)["jobs"][0]["output_root"],
        )

    def test_v2_lock_is_unchanged_and_not_a_v3_fallback(self) -> None:
        v2 = load_oracle_lock()
        v3 = load_oracle_absolute_lock()
        self.assertEqual(v2["proposal_id"], v3["v2_provenance"]["proposal_id"])
        self.assertEqual(v2["lock_sha256"], v3["v2_provenance"]["lock_sha256"])
        self.assertFalse(v3["v2_provenance"]["reused_as_runtime_fallback"])
        for section in ("selection", "runtime", "resources"):
            self.assertEqual(v3[section], v2[section])

    def test_mutated_mean_margin_or_accounting_fails_closed(self) -> None:
        raw = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("required_mean_margin", 0.5),
            ("oracle_extra_model_forwards", 4),
            ("q_margin_decision", True),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                candidate_payload = json.loads(json.dumps(raw))
                candidate_payload["event"][key] = value
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "lock.json"
                    candidate.write_text(
                        json.dumps(candidate_payload), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(MethodContractError, "differs"):
                        load_oracle_absolute_lock(candidate)


class OracleAbsoluteV3RunnerSourceTests(unittest.TestCase):
    def test_common_backend_has_no_alias_policy_or_qmargin_decision(self) -> None:
        for module in (p0_runner, p1_runner):
            source = inspect.getsource(module)
            self.assertIn("OracleAbsoluteMeanMarginEasyEditBackend", source)
            self.assertNotIn("if args.model_alias ==", source)
            self.assertNotIn('terminal["q_margin"] + config.event_tolerance', source)
            self.assertNotIn("evaluation_payload", source)
            self.assertNotIn("open_evaluation", source)
            self.assertIn("oracle_extra_model_forwards", source)
            self.assertIn("calibration_total_model_forwards", source)

    def test_p1_has_no_legacy_replay_import_or_call(self) -> None:
        source = inspect.getsource(p1_runner)
        for forbidden in (
            "PinnedDirectZEasyEditBackend",
            "legacy_r2_trajectories",
            "validate_r2_sources",
            "_run_legacy_rca",
            "expected_legacy_root",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"legacy_replay_executed": False', source)
        self.assertIn('"existing_legacy_diagnostic_only": True', source)

    def test_new_roots_are_distinct_and_absent(self) -> None:
        lock = load_oracle_absolute_lock()
        proposal = lock["proposal_id"]
        roots = []
        for alias in lock["runtime"]["models"]:
            roots.extend(
                (
                    REPO / p0_runner.expected_output_root(alias, proposal),
                    REPO / p1_runner.expected_output_root(alias, proposal),
                )
            )
        self.assertEqual(len(roots), len(set(roots)))
        self.assertTrue(all(not root.exists() for root in roots))


if __name__ == "__main__":
    unittest.main()
