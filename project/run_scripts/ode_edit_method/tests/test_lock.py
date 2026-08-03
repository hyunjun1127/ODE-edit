from __future__ import annotations

import copy
from contextlib import redirect_stderr
import io
import unittest

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.lock import (
    controller_config,
    dry_plan,
    load_lock,
    validate_lock,
)
from project.run_scripts.session02_compute_aware_p01 import build_parser


class NumericalLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        loaded = load_lock()
        loaded.pop("proposal_id")
        cls.lock = loaded

    def test_common_controller_and_exact_case_order_are_locked(self) -> None:
        config = controller_config(self.lock)
        self.assertEqual(config.s_max, 6)
        self.assertEqual(
            self.lock["selection"]["canonical_case_ids"],
            ["2022", "12498", "20964", "768"],
        )
        self.assertFalse(self.lock["controller"]["model_specific_policy"])
        self.assertEqual(
            self.lock["trial_backend"]["cached_trial_graph"],
            "UNSUPPORTED_FAIL_CLOSED",
        )

    def test_p01_plan_is_paired_fail_closed_and_within_cap(self) -> None:
        for stage, expected_cases in (("p0", 1), ("p1", 4)):
            plan = dry_plan(self.lock, stage)
            self.assertEqual(plan["status"], "DRY_RUN_ONLY; NO_GPU; NO_SLURM")
            self.assertTrue(plan["same_submission_batch_required"])
            self.assertEqual(plan["aggregate_gpus"], 2)
            self.assertEqual(len(plan["jobs"]), 2)
            self.assertEqual(
                {job["model"] for job in plan["jobs"]},
                {"llama3-8b-inst", "qwen2.5-7b-inst"},
            )
            self.assertEqual({len(job["case_ids"]) for job in plan["jobs"]}, {expected_cases})
            self.assertEqual(len({tuple(job["arms"]) for job in plan["jobs"]}), 1)
            self.assertEqual(len({job["backend"] for job in plan["jobs"]}), 1)

    def test_launcher_requires_explicit_dry_run(self) -> None:
        parser = build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--stage", "p0"])
        parsed = parser.parse_args(["--stage", "p0", "--dry-run"])
        self.assertTrue(parsed.dry_run)

    def test_lock_rejects_outcomes_cached_mode_and_per_model_policy(self) -> None:
        for mutate in (
            lambda value: value.__setitem__("outcome_count_at_proposal", 1),
            lambda value: value["trial_backend"].__setitem__(
                "cached_trial_graph", "SUPPORTED"
            ),
            lambda value: value["controller"].__setitem__(
                "model_specific_policy", True
            ),
        ):
            candidate = copy.deepcopy(self.lock)
            mutate(candidate)
            with self.assertRaises(MethodContractError):
                validate_lock(candidate)

    def test_lock_rejects_ordered_in_p3_or_submission_authority(self) -> None:
        candidate = copy.deepcopy(self.lock)
        candidate["stages"]["p3"]["arms"].append("ordered-adaptive")
        with self.assertRaises(MethodContractError):
            validate_lock(candidate)

        candidate = copy.deepcopy(self.lock)
        candidate["resource_forecast"]["submission_authorized"] = True
        with self.assertRaises(MethodContractError):
            validate_lock(candidate)

    def test_compute_schema_and_structural_trial_bounds_are_complete(self) -> None:
        required = set(self.lock["compute_accounting"]["required_counters"])
        artifact = set(self.lock["artifact_schema"]["compute_required_fields"])
        self.assertTrue(required <= artifact)
        forecast = self.lock["resource_forecast"]
        self.assertEqual(forecast["p0_n_field_upper_bound_per_model"], 9)
        self.assertEqual(forecast["p1_n_field_upper_bound_per_model"], 36)
        self.assertEqual(
            forecast["p0_n_trial_upper_bound_per_model_per_repetition"], 71
        )
        self.assertEqual(forecast["p0_profile_n_trial_upper_bound_per_model"], 284)
        self.assertEqual(forecast["p1_n_trial_upper_bound_per_model"], 284)


if __name__ == "__main__":
    unittest.main()
