"""Synthetic admission tests; scheduler/model/GPU actions are zero."""
import dataclasses
import unittest

from .resources import (Allocation, BudgetAuthority, ResourceBoundary, ResourcePolicy,
                        SchedulerSnapshot, UsageInterval, UsageLedger,
                        assess_admission, validate_sbatch_memory)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.policy = ResourcePolicy()
        self.budget = BudgetAuthority(2.0, "USER-TEST-AUTHORITY", "a" * 64)
        self.snapshot = SchedulerSnapshot(True, True, 100, "b" * 64)

    def decide(self, **overrides):
        args = dict(policy=self.policy, budget=self.budget, snapshot=self.snapshot,
                    ledger=UsageLedger(), new_processes=2,
                    reserved_wall_seconds_per_process=1800, now_seconds=101)
        args.update(overrides)
        return assess_admission(**args)

    def test_unassigned_budget_holds_only_admission(self):
        result = self.decide(budget=BudgetAuthority())
        self.assertEqual(result["blocking_reasons"], ["GPU_HOUR_BUDGET_UNASSIGNED"])
        self.assertIsNone(result["remaining_gpu_seconds"])
        self.assertFalse(result["is_reservation"])

    def test_user_authority_required(self):
        with self.assertRaisesRegex(ResourceBoundary, "USER_BUDGET_AUTHORITY"):
            BudgetAuthority(8)
        with self.assertRaises(ResourceBoundary):
            BudgetAuthority(None, "old-pilot", "a" * 64)

    def test_cap2_and_single_gpu_per_process(self):
        self.assertEqual(self.decide()["status"], "ALLOW_NEW_SUBMISSION")
        self.assertIn("PROJECT_GPU_CAP_EXCEEDED", self.decide(new_processes=3)["blocking_reasons"])
        with self.assertRaises(ResourceBoundary):
            ResourcePolicy(project_gpu_cap=4)
        with self.assertRaises(ResourceBoundary):
            ResourcePolicy(gpus_per_process=2)

    def test_grandfathered_completing_and_configuring_count(self):
        for state in ("RUNNING", "COMPLETING", "CONFIGURING"):
            with self.subTest(state=state):
                snap = dataclasses.replace(self.snapshot, allocations=(Allocation("job1", state, 4, grandfathered=True),))
                result = self.decide(snapshot=snap, new_processes=1)
                self.assertEqual(result["active_project_gpus"], 4)
                self.assertEqual(result["existing_job_mutation_count"], 0)
                self.assertIn("PROJECT_GPU_CAP_EXCEEDED", result["blocking_reasons"])

    def test_cancelled_but_uncleared_allocation_counts(self):
        snap = dataclasses.replace(self.snapshot, allocations=(Allocation("job1", "CANCELLED", 2),))
        self.assertEqual(self.decide(snapshot=snap)["active_project_gpus"], 2)
        snap = dataclasses.replace(snap, allocations=(Allocation("job1", "CANCELLED", 2, process_exit_confirmed=True),))
        self.assertEqual(self.decide(snapshot=snap)["active_project_gpus"], 0)

    def test_failed_or_incomplete_query_is_not_zero_usage(self):
        for field in ("query_ok", "allocation_query_complete"):
            result = self.decide(snapshot=dataclasses.replace(self.snapshot, **{field: False}))
            self.assertIsNone(result["active_project_gpus"])
            self.assertIn("SCHEDULER_QUERY_UNRESOLVED", result["blocking_reasons"])

    def test_stale_or_future_snapshot_fails(self):
        for timestamp in (0, 102, float("nan")):
            result = self.decide(snapshot=dataclasses.replace(self.snapshot, captured_at_seconds=timestamp))
            self.assertIn("SCHEDULER_EVIDENCE_STALE_OR_MISSING", result["blocking_reasons"])

    def test_unknown_state_gpu_and_duplicate_fail_closed(self):
        for allocations in ((Allocation("x", "COMPLETING", None),), (Allocation("x", "UNKNOWN", 1),),
                            (Allocation("x", "RUNNING", 1), Allocation("x", "RUNNING", 1))):
            self.assertNotEqual(self.decide(snapshot=dataclasses.replace(self.snapshot, allocations=allocations))["status"], "ALLOW_NEW_SUBMISSION")

    def test_ledger_is_immutable_and_charges_technical_attempts(self):
        original = UsageLedger()
        first = UsageInterval("job1", "TECH-R1", "S", "TECHNICAL_ATTEMPT", 10, 70, 1, "c" * 64)
        second = UsageInterval("job1", "TECH-R1", "S", "ALLOCATED_RESIDENCY", 70, 130, 1, "d" * 64)
        ledger = original.append(first).append(second)
        self.assertEqual(original.gpu_seconds, 0)
        self.assertEqual(ledger.gpu_seconds, 120)
        self.assertEqual(ledger.receipt()["technical_attempts_excluded_from_budget_count"], 0)
        self.assertEqual(ledger.receipt(), ledger.receipt())

    def test_ledger_double_charge_and_invalid_values_fail(self):
        a = UsageInterval("job1", "TECH-R1", "D", "PRIMARY", 0, 100, 1, "c" * 64)
        with self.assertRaises(ResourceBoundary):
            UsageLedger().append(a).append(dataclasses.replace(a, start_seconds=99))
        with self.assertRaises(ResourceBoundary):
            _ = UsageLedger((a, a)).gpu_seconds
        with self.assertRaises(ResourceBoundary):
            _ = dataclasses.replace(a, end_seconds=float("inf")).gpu_seconds

    def test_charged_plus_reserved_plus_new_must_fit(self):
        item = UsageInterval("job1", "TECH-R1", "D", "TECHNICAL_ATTEMPT", 0, 3600, 1, "c" * 64)
        result = self.decide(ledger=UsageLedger((item,)), other_reserved_gpu_seconds=1)
        self.assertIn("GPU_HOUR_BUDGET_EXCEEDED", result["blocking_reasons"])
        self.assertEqual(self.decide(ledger=UsageLedger((item,)))["status"], "ALLOW_NEW_SUBMISSION")

    def test_exact_one_mem_directive(self):
        text = "#!/bin/bash\n#SBATCH --mem=182272M\n#SBATCH --gres=gpu:1\n"
        self.assertEqual(validate_sbatch_memory(text)["explicit_mem_count"], 1)
        for bad in ("", text + "#SBATCH --mem=182272M\n", "#SBATCH --mem=183296M", "#SBATCH --mem-per-gpu=182272M",
                    "#SBATCH --mem=182272M --mem-per-cpu=100M"):
            with self.subTest(text=bad), self.assertRaises(ResourceBoundary):
                validate_sbatch_memory(bad)

    def test_policy_and_requested_count_are_finite_non_boolean(self):
        for value in (True, 2.5, 0, -1):
            with self.subTest(value=value), self.assertRaises(ResourceBoundary):
                self.decide(new_processes=value)
        with self.assertRaises(ResourceBoundary):
            self.decide(reserved_wall_seconds_per_process=float("nan"))


if __name__ == "__main__":
    unittest.main()
