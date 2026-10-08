"""Fake-exec CPU accounting/privacy tests; no actual Slurm query or model."""
import copy
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from official.runners.server1.costs import AccountingError, collect_costs


def manifest():
    return dict(schema="official-server1-job-manifest-v1",
        source=dict(main_commit="a" * 40, official_tree="b" * 40), plan_sha256="c" * 64,
        jobs={"cf-memit": "80001", "collector": "80002"},
        profiles=[dict(key="cf-memit"), dict(key="collector")])


class CostsTests(unittest.TestCase):
    def test_one_query_exact_ids_parent_only_cost_with_step_peak_RSS(self):
        calls = []
        def runner(argv, *, timeout):
            calls.append((argv, timeout))
            return ("80001|COMPLETED|12|8|cpu=8,mem=65536M,node=1,billing=8,gres/gpu=2,gres/gpu:a100=2|00:03.250|\n"
                    "80001.batch|COMPLETED|999|8|cpu=8,gres/gpu=2|9-00:00:00|100K\n"
                    "80001.0|COMPLETED|999|8|cpu=8,gres/gpu=2|9-00:00:00|200K\n"
                    "80002|RUNNING|4|8|cpu=8,mem=24576M,node=1|00:05.000|30K\n")
        result = collect_costs(manifest(), runner=runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 10)
        self.assertEqual(calls[0][0][0], "sacct")
        self.assertIn("--jobs=80001,80002", calls[0][0])
        self.assertIn("--units=K", calls[0][0])
        row = result["rows"][0]
        self.assertEqual(row["allocation_GPU_seconds"], 24)
        self.assertEqual(row["allocation_CPU_seconds"], 96)
        self.assertEqual(row["total_CPU_seconds"], 3.25)
        self.assertEqual(row["max_RSS_bytes"], 200 * 1024)
        self.assertEqual(row["max_RSS_source"], "MAX_REPORTED_STEP_TASK_PEAK_NOT_SUM")
        self.assertEqual(result["totals"]["known_total_CPU_seconds"], 8.25)
        self.assertEqual(result["totals"]["known_allocation_GPU_seconds"], 24)
        self.assertEqual(result["requested_parent_jobs"], 2)
        self.assertFalse(result["all_costs_final"])
        self.assertTrue(result["scientific_success_not_inferred"])

    def test_missing_parent_and_fields_are_unknown_not_zero(self):
        stdout = ("80001|COMPLETED|Unknown|8|cpu=8,gres/gpu=1||\n"
                  "80002.batch|COMPLETED|2|8|cpu=8|00:01|100K\n")
        result = collect_costs(manifest(), runner=lambda argv, timeout: stdout)
        self.assertEqual(result["status"], "ACCOUNTING_PARTIAL_ID_COVERAGE")
        self.assertIsNone(result["rows"][0]["allocation_GPU_seconds"])
        self.assertIsNone(result["rows"][0]["total_CPU_seconds"])
        self.assertIsNone(result["rows"][0]["max_RSS_bytes"])
        self.assertEqual(result["rows"][1]["accounting_status"], "NOT_OBSERVED")
        self.assertIsNone(result["rows"][1]["elapsed_seconds"])
        self.assertIsNone(result["totals"]["known_allocation_GPU_seconds"])
        self.assertEqual(result["totals"]["allocation_GPU_seconds_unknown_jobs"], 2)

    def test_absent_TRES_not_interpreted_as_CPU_only_allocation(self):
        stdout = "80001|FAILED|20|8||00:02|\n"
        result = collect_costs(manifest(), runner=lambda argv, timeout: stdout)
        self.assertIsNone(result["rows"][0]["allocated_GPU_count"])
        self.assertIsNone(result["rows"][0]["allocation_GPU_seconds"])
        self.assertIn("MISSING_ALLOCATED_TRES", result["rows"][0]["uncertainty"])

    def test_duplicate_parent_and_step_rows_fail_closed_without_double_count(self):
        rows = ("80001|COMPLETED|20|8|cpu=8,gres/gpu=1|00:02|200K\n",
                "80001.batch|COMPLETED|20|8|cpu=8|00:02|200K\n")
        for row, code in ((rows[0], "ACCOUNTING_DUPLICATE_PARENT"),
                          (rows[1], "ACCOUNTING_DUPLICATE_STEP")):
            with self.subTest(code=code):
                result = collect_costs(manifest(), runner=lambda argv, timeout: row + row)
                self.assertEqual(result["status"], "ACCOUNTING_UNAVAILABLE")
                self.assertEqual(result["error_code"], code)
                self.assertIsNone(result["totals"]["known_total_CPU_seconds"])

    def test_unrequested_job_ID_or_invalid_TRES_or_conflicting_GPU_counts_rejected(self):
        for row, code in (
            ("80003|COMPLETED|20|8|cpu=8,gres/gpu=1|00:02|200K\n", "ACCOUNTING_UNREQUESTED_OR_INVALID_JOB"),
            ("80001|COMPLETED|20|8|cpu=8,secret=PRIVATE_VALUE|00:02|200K\n", "ACCOUNTING_ALLOC_TRES_WHITELIST"),
            ("80001|COMPLETED|20|8|cpu=8,gres/gpu=1,gres/gpu:a100=2|00:02|200K\n",
                "ACCOUNTING_GPU_GENERIC_TYPED_CONFLICT")):
            with self.subTest(code=code):
                result = collect_costs(manifest(), runner=lambda argv, timeout: row)
                self.assertEqual(result["error_code"], code)
                self.assertNotIn("PRIVATE_VALUE", json.dumps(result))

    def test_manifest_duplicate_ids_profiles_or_source_do_not_query(self):
        calls = []
        def runner(argv, timeout):
            calls.append(argv)
            return ""
        for change in ("duplicate", "profiles", "source"):
            value = copy.deepcopy(manifest())
            if change == "duplicate":
                value["jobs"]["collector"] = "80001"
            elif change == "profiles":
                value["profiles"][0]["key"] = "wrong"
            else:
                value["source"]["secret"] = "PRIVATE_VALUE"
            with self.subTest(change=change), self.assertRaises(AccountingError):
                collect_costs(value, runner=runner)
        self.assertEqual(calls, [])

    def test_timeout_is_bounded_and_no_secret_stdout_or_stderr_is_published(self):
        failure = subprocess.TimeoutExpired("sacct", 10, output="PRIVATE_STDOUT", stderr="PRIVATE_TOKEN")
        with patch("official.runners.server1.costs.subprocess.run", side_effect=failure) as call:
            result = collect_costs(manifest())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["timeout"], 10)
        self.assertEqual(result["error_code"], "ACCOUNTING_QUERY_TIMEOUT")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertFalse(result["automatic_retry"])

    def test_error_return_code_is_typed_without_raw_error_or_blind_retry(self):
        completed = SimpleNamespace(returncode=1, stdout="PRIVATE_STDOUT", stderr="PRIVATE_TOKEN")
        with patch("official.runners.server1.costs.subprocess.run", return_value=completed) as call:
            result = collect_costs(manifest())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["error_code"], "ACCOUNTING_QUERY_FAILED")
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_requested_timeout_cannot_expand_beyond_10_seconds(self):
        for timeout in (0, -1, 11, True, "10"):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(AccountingError, "TIMEOUT_BOUND"):
                collect_costs(manifest(), runner=lambda argv, timeout: "", timeout=timeout)

    def test_raw_state_and_task_memory_units_are_whitelisted_not_private_reason(self):
        stdout = ("80001|CANCELLED by 1234|1|8|cpu=8,gres/gpu=1|00:00.125|1.5M|\n"
                  "80002|COMPLETED|1|8|cpu=8|1-02:03:04.5|2G|\n")
        result = collect_costs(manifest(), runner=lambda argv, timeout: stdout)
        self.assertEqual(result["rows"][0]["state"], "CANCELLED")
        self.assertEqual(result["rows"][0]["max_RSS_bytes"], int(1.5 * 1024**2))
        self.assertEqual(result["rows"][1]["max_RSS_bytes"], 2 * 1024**3)
        self.assertEqual(result["rows"][1]["total_CPU_seconds"], 93784.5)
        self.assertNotIn("by 1234", json.dumps(result))
        self.assertTrue(result["all_costs_final"])


if __name__ == "__main__":
    unittest.main()
