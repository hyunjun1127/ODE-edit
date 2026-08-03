from __future__ import annotations

import ast
import copy
from contextlib import redirect_stderr
import io
import inspect
import unittest

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.lock import (
    controller_config,
    dry_plan,
    load_lock,
    validate_lock,
)
from project.run_scripts.ode_edit_method.memit_adapter import (
    functional_trial_for_batch,
)
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner
from project.run_scripts import session02_compute_aware_p0 as executable_module
from project.run_scripts.session02_compute_aware_p01 import build_parser
from project.run_scripts.session02_compute_aware_p0 import (
    build_parser as build_executable_parser,
)


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
        self.assertEqual(
            self.lock["trial_backend"]["selected_common_backend"],
            "quantized-rowblock-commit-emulator",
        )
        self.assertEqual(self.lock["trial_backend"]["row_block"], 64)
        self.assertFalse(self.lock["trial_backend"]["two_tier_pretrial"])
        self.assertEqual(
            self.lock["dtype_contract"]["method_dtype_policy"],
            "checkpoint-original",
        )
        self.assertEqual(
            self.lock["dtype_contract"]["checkpoint_original_dtype"],
            "torch.bfloat16",
        )
        self.assertEqual(config.h0_fraction, 0.25)
        self.assertEqual(config.h_max_fraction, 0.5)

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
            if stage == "p0":
                self.assertTrue(
                    all("--execute" in job["executable_command"] for job in plan["jobs"])
                )
                self.assertTrue(
                    all(job["submission_authorized"] is False for job in plan["jobs"])
                )
                self.assertEqual(
                    {job["dtype_policy"] for job in plan["jobs"]},
                    {"checkpoint-original"},
                )
                self.assertEqual(
                    {job["checkpoint_original_dtype"] for job in plan["jobs"]},
                    {"torch.bfloat16"},
                )
                self.assertEqual(
                    {job["trial_backend"] for job in plan["jobs"]},
                    {"quantized-rowblock-commit-emulator"},
                )
                for job in plan["jobs"]:
                    output_root = job["executable_command"][-2]
                    self.assertIn(
                        "session02-p0-original-dtype-simple-t-v1-",
                        output_root,
                    )
                    self.assertNotIn("session02-p0-tech-v1", output_root)

    def test_launcher_requires_explicit_dry_run(self) -> None:
        parser = build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--stage", "p0"])
        parsed = parser.parse_args(["--stage", "p0", "--dry-run"])
        self.assertTrue(parsed.dry_run)

        executable = build_executable_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                executable.parse_args(
                    [
                        "--model-alias",
                        "llama3-8b-inst",
                        "--output-root",
                        "local/results/session02-test",
                    ]
                )
        parsed_execute = executable.parse_args(
            [
                "--model-alias",
                "llama3-8b-inst",
                "--output-root",
                "local/results/session02-test",
                "--execute",
            ]
        )
        self.assertTrue(parsed_execute.execute)

    def test_lock_rejects_outcomes_cached_mode_and_per_model_policy(self) -> None:
        for mutate in (
            lambda value: value.__setitem__("outcome_count_at_proposal", 1),
            lambda value: value["trial_backend"].__setitem__(
                "cached_trial_graph", "SUPPORTED"
            ),
            lambda value: value["controller"].__setitem__(
                "model_specific_policy", True
            ),
            lambda value: value["dtype_contract"].__setitem__(
                "method_dtype_policy", "legacy-motivation-float32"
            ),
            lambda value: value["trial_backend"].__setitem__(
                "two_tier_pretrial", True
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
            forecast["p0_n_trial_upper_bound_per_model_per_repetition"], 56
        )
        self.assertEqual(forecast["p0_profile_n_trial_upper_bound_per_model"], 224)
        self.assertEqual(forecast["p1_n_trial_upper_bound_per_model"], 224)
        self.assertEqual(
            forecast["p0_profile_n_field_state_fwd_upper_bound_per_model"],
            257,
        )
        self.assertEqual(
            forecast["p1_n_field_state_fwd_upper_bound_per_model"],
            252,
        )
        self.assertNotIn("N_state_fwd", required)
        self.assertIn("N_model_fwd", required)
        controller_fields = set(
            self.lock["artifact_schema"]["controller_step_required_fields"]
        )
        self.assertTrue(
            {
                "model",
                "case_id",
                "arm",
                "order_sha256",
                "seed",
                "commit",
                "hashes",
                "status",
                "result",
                "event",
                "Omega",
            }
            <= controller_fields
        )
        self.assertEqual(
            self.lock["trial_backend"]["unchanged_trial_state_guard"],
            "storage-pointer-version-shape-dtype-device-without-dense-rehash",
        )
        self.assertEqual(
            self.lock["resource_forecast"]["p0_forecast_gpu_hours_per_model"],
            "UNMEASURED_CHECKPOINT_ORIGINAL_BF16_P0",
        )
        self.assertFalse(
            self.lock["resource_forecast"][
                "simple_t_microfixture_is_model_scale_forecast"
            ]
        )

    def test_runner_and_adaptive_trial_source_guards(self) -> None:
        runner_source = inspect.getsource(executable_module)
        calls = {
            node.func.id
            for node in ast.walk(ast.parse(runner_source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("load_fixed_model_checkpoint_original", calls)
        self.assertNotIn("load_fixed_model", calls)

        adapter_source = inspect.getsource(functional_trial_for_batch)
        self.assertIn("QuantizedRowBlockFunctionalTrial", adapter_source)
        self.assertIn("SIMPLE_T_ROW_BLOCK", adapter_source)
        self.assertNotIn("LowRankFunctionalTrial", adapter_source)

        adaptive_source = inspect.getsource(FiveArmRunner._run_adaptive)
        self.assertIn('instrumentation.component("trial")', adaptive_source)
        self.assertIn(
            "functional trial event differs from committed write",
            adaptive_source,
        )

    def test_lock_rejects_ambiguous_event_tolerance_type(self) -> None:
        candidate = copy.deepcopy(self.lock)
        candidate["event_backend"]["p0_two_forward_atol"] = None
        with self.assertRaises(MethodContractError):
            validate_lock(candidate)


if __name__ == "__main__":
    unittest.main()
