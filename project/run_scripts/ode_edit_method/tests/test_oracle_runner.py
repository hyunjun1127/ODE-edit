from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.legacy_rca import legacy_r2_trajectories
from project.run_scripts.ode_edit_method.oracle_lock import load_oracle_lock
from project.run_scripts.session02_oracle_mean_event_p0 import dry_plan as p0_dry_plan
from project.run_scripts.session02_oracle_mean_event_p1 import dry_plan as p1_dry_plan
import project.run_scripts.session02_oracle_mean_event_p0 as p0_runner
import project.run_scripts.session02_oracle_mean_event_p1 as p1_runner


REPO = Path(__file__).resolve().parents[4]


class OracleDevelopmentLockTests(unittest.TestCase):
    def test_lock_and_pair_dry_plans_are_common_and_non_authorizing(self) -> None:
        lock = load_oracle_lock()
        self.assertEqual(
            lock["instruction_id"],
            "ODEEDIT-S02-ORACLE-MEAN-EVENT-V2-R1-P0P1",
        )
        self.assertEqual(lock["revision_id"], "R1_EXACT_ZERO_ACCOUNTING_FIREWALL")
        self.assertEqual(lock["event"]["rho"], 0.5)
        self.assertEqual(lock["event"]["context_weights"], "uniform")
        self.assertFalse(lock["event"]["per_context_decision"])
        self.assertEqual(lock["resources"]["server1_project_gpu_cap"], 3)
        p0 = p0_dry_plan(lock)
        p1 = p1_dry_plan(lock)
        for plan in (p0, p1):
            self.assertFalse(plan["submission_authorized"])
            self.assertEqual(plan["pair_gpu"], 2)
            self.assertEqual(plan["server1_project_gpu_cap"], 3)
            self.assertEqual(len(plan["jobs"]), 2)
            self.assertEqual(
                [row["model_alias"] for row in plan["jobs"]],
                ["llama3-8b-inst", "qwen2.5-7b-inst"],
            )
        self.assertNotEqual(
            p0["jobs"][0]["output_root"], p1["jobs"][0]["output_root"]
        )
        self.assertTrue(
            all("oracle-mean-event-p0-r1" in row["output_root"] for row in p0["jobs"])
        )
        self.assertTrue(
            all("oracle-mean-event-p1-r1" in row["output_root"] for row in p1["jobs"])
        )

    def test_mutated_policy_fails_closed(self) -> None:
        source = json.loads(
            Path(
                "project/run_scripts/ode_edit_method/oracle_mean_event_v2.json"
            ).read_text(encoding="utf-8")
        )
        source["event"]["rho"] = 0.6
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "lock.json"
            candidate.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(MethodContractError, "differs"):
                load_oracle_lock(candidate)


class LegacyR2PinTests(unittest.TestCase):
    def test_selected_native_full_panel_is_fully_pinned(self) -> None:
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            with self.subTest(alias=alias):
                rows = legacy_r2_trajectories(REPO, alias)
                self.assertEqual(len(rows), 8)
                self.assertEqual(
                    [row.case_id for row in rows[:4]],
                    ["2022", "12498", "20964", "768"],
                )
                self.assertTrue(all(row.direct_z_path.is_file() for row in rows))
                self.assertTrue(
                    all(
                        row.controller_record["pre_edit_state_id"]
                        == row.direct_z_source_state_id
                        for row in rows
                    )
                )


class OracleRunnerSourceTests(unittest.TestCase):
    def test_both_runners_use_one_common_backend_without_alias_policy_branch(self) -> None:
        for module in (p0_runner, p1_runner):
            source = inspect.getsource(module)
            self.assertIn("OracleMeanEasyEditBackend", source)
            self.assertNotIn("if args.model_alias ==", source)
            self.assertNotIn("evaluation_payload", source)
            self.assertNotIn("open_evaluation", source)
        p1_source = inspect.getsource(p1_runner)
        self.assertIn("PinnedDirectZEasyEditBackend", p1_source)
        self.assertIn("excluded_from_v2_compute", p1_source)
        self.assertIn("legacy_r2_trajectories", p1_source)

    def test_stage_roots_are_distinct_and_absent_at_cpu_gate(self) -> None:
        lock = load_oracle_lock()
        proposal = lock["proposal_id"]
        paths = []
        for alias in lock["runtime"]["models"]:
            paths.extend(
                (
                    REPO / p0_runner.expected_output_root(alias, proposal),
                    REPO / p1_runner.expected_output_root(alias, proposal),
                    REPO / p1_runner.expected_legacy_root(alias, proposal),
                )
            )
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(not path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
