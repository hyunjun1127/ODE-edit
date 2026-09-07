"""Small CPU-only S grid, prefix, inventory and outcome-blind sampling gates."""
from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from .sampling import (ExclusionInventory, REQUIRED_INVENTORIES, SampleBoundary,
                       canonical_hash, evaluation_inventory, load_git_inventories, select_cohorts)
from .sweep import (ACTUAL_LAMBDAS, MODELS, SweepBoundary, dry_plan,
                    endpoint_binding, endpoints, paths)


def _records(count=460):
    return [dict(case_id=i,
                 requested_rewrite=dict(prompt="{} is", subject=f"entity{i}",
                                        target_new={"str": "new"}, target_true={"str": "old"}),
                 paraphrase_prompts=["p0", "p1"] if i % 7 else ["p0", "p1", "p2"],
                 neighborhood_prompts=["n"] * (10 + int(i % 11 == 0))) for i in range(count)]


def _inventories():
    return tuple(ExclusionInventory(label, (i,), {"sha256": canonical_hash(label),
                                                  "scope": "CPU_FIXTURE"})
                 for i, label in enumerate(sorted(REQUIRED_INVENTORIES)))


class SweepTests(unittest.TestCase):
    def test_all_five_lambdas_actual_both_models(self):
        plan = dry_plan()
        self.assertEqual(tuple(plan["models"]), MODELS)
        candidates = [row for row in endpoints() if row.config.T == 2 and row.config.N == 4]
        self.assertEqual(tuple(sorted(row.config.lambda_response for row in candidates)), ACTUAL_LAMBDAS)
        for model in MODELS:
            self.assertEqual(len(candidates), 5, model)
        self.assertTrue(all(row["actual_gpu_endpoint_required"] for row in plan["endpoints"]))

    def test_nine_exact_labels_seven_paths_thirtyfour_nodes(self):
        self.assertEqual({row.candidate_id for row in endpoints()}, {
            "JV-BASE", "JV-LAM-001", "JV-LAM-00316", "JV-LAM-0316", "JV-LAM-1",
            "JV-RES-N2", "JV-RES-N8", "JV-HOR-T1", "JV-HOR-T4"})
        self.assertEqual(len(paths()), 7)
        self.assertEqual(sum(row.config.N for row in paths()), 34)
        self.assertEqual(dry_plan()["JV_endpoint_count"], 18)
        self.assertEqual(dry_plan()["all_endpoint_count_including_entry_and_Official"], 22)

    def test_prefix_effective_parent_clock_separated(self):
        parent = paths()[0]
        prefix = next(row for row in endpoints() if row.candidate_id == "JV-BASE")
        row = endpoint_binding(prefix, parent)
        self.assertEqual((row["effective_T"], row["effective_N"], row["h"]), (2, 4, .5))
        self.assertEqual((row["parent_T"], row["parent_N"]), (4, 8))
        self.assertEqual(row["history_append_count"], 0)
        self.assertEqual(row["persistent_endpoint_capture_count"], 0)
        self.assertFalse(row["prefix_is_sequential_resume_checkpoint"])

    def test_wrong_prefix_binding_fails(self):
        prefix = endpoints()[0]
        for bad in (replace(prefix, completed_nodes=3),
                    replace(prefix, derived_observation_only=False),
                    replace(prefix, config=replace(prefix.config, lambda_response=.01))):
            with self.assertRaisesRegex(SweepBoundary, "PREFIX_CONFIG_BINDING"):
                endpoint_binding(bad, paths()[0])

    def test_S_not_blocked_on_D_and_no_invented_budget(self):
        plan = dry_plan()
        self.assertFalse(plan["D_exact_replay_dependency"])
        self.assertIsNone(plan["GPU_hour_cap"])
        self.assertEqual(plan["GPU_submission_status"], "GPU_HOUR_BUDGET_UNASSIGNED")
        self.assertEqual(plan["fixed_z_capture_per_model_cohort"], 1)

    def test_intermediate_lambdas_remain_required_lower_priority(self):
        late = [row for row in paths() if row.priority_group == "INTERMEDIATE_LAMBDA_ACTUAL"]
        self.assertEqual({row.config.lambda_response for row in late},
                         {.03162277660168379, .31622776601683794})
        self.assertEqual(len(late), 2)


class SamplingTests(unittest.TestCase):
    def seal(self, rows=None, inventory=None):
        return select_cohorts(_records() if rows is None else rows,
                              _inventories() if inventory is None else inventory,
                              dataset_identity={"sha256": "CPU_FIXTURE"})

    def test_disjoint_and_no_model_duplication(self):
        seal = self.seal()
        dev = {r["case_id"] for r in seal["cohorts"]["S_DEV"]["records"]}
        audit = {r["case_id"] for r in seal["cohorts"]["S_AUDIT_RESERVED"]["records"]}
        self.assertEqual((len(dev), len(audit)), (100, 300))
        self.assertFalse(dev & audit)
        self.assertFalse((dev | audit) & {0, 1, 2, 3})
        self.assertEqual(seal["model_payload_duplication"], 0)
        self.assertFalse(seal["D_restore_dependency"])

    def test_pool_order_and_outcomes_cannot_change_selected_IDs(self):
        original = self.seal()
        rows = list(reversed(_records()))
        for row in rows:
            row["nll"] = -1000 * row["case_id"]
            row["previous_success"] = row["case_id"] % 2 == 0
            row["residual"] = 1e-30
        changed = self.seal(rows)
        for group in ("S_DEV", "S_AUDIT_RESERVED"):
            self.assertEqual([r["case_id"] for r in original["cohorts"][group]["records"]],
                             [r["case_id"] for r in changed["cohorts"][group]["records"]])
        self.assertEqual(changed["outcome_selection_count"], 0)

    def test_actual_prompt_counts_not_forced_or_selected(self):
        seal = self.seal()
        chosen = seal["cohorts"]["S_DEV"]
        raw = {r["case_id"]: r for r in _records()}
        selected = [raw[r["case_id"]] for r in chosen["records"]]
        actual = chosen["evaluation"]["denominators"]
        self.assertEqual(actual["RS"], 100)
        self.assertEqual(actual["PS"], sum(len(r["paraphrase_prompts"]) for r in selected))
        self.assertEqual(actual["NS"], sum(len(r["neighborhood_prompts"]) for r in selected))
        self.assertGreater(actual["PS"], 200)
        self.assertGreater(actual["NS"], 1000)
        altered = _records()
        for row in altered:
            row["paraphrase_prompts"] = ["single"]
            row["neighborhood_prompts"] = ["single"]
        self.assertEqual([r["case_id"] for r in chosen["records"]],
                         [r["case_id"] for r in self.seal(altered)["cohorts"]["S_DEV"]["records"]])

    def test_missing_reserved_inventory_is_typed_HOLD(self):
        with self.assertRaisesRegex(SampleBoundary, "EXCLUSION_INVENTORY_UNAVAILABLE"):
            self.seal(inventory=_inventories()[:-1])

    def test_unavailable_git_object_is_typed_HOLD_without_remote_poll(self):
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(128, "git")) as git:
            with self.assertRaisesRegex(SampleBoundary, "EXCLUSION_INVENTORY_UNAVAILABLE"):
                load_git_inventories(Path("/cpu-fixture-no-remote"))
            self.assertEqual(git.call_count, 1)

    def test_insufficient_pool_no_relaxation(self):
        with self.assertRaisesRegex(SampleBoundary, "SAMPLE_POOL_INSUFFICIENT"):
            self.seal(_records(403))

    def test_duplicate_case_and_invalid_evaluator_schema_fail(self):
        with self.assertRaisesRegex(SampleBoundary, "DATASET_DUPLICATE_CASE_ID"):
            self.seal(_records() + [_records()[0]])
        rows = _records()
        rows[10]["neighborhood_prompts"] = []
        with self.assertRaisesRegex(SampleBoundary, "EVALUATOR_INPUT_SCHEMA_BOUNDARY"):
            self.seal(rows)

    def test_manifest_identity_deterministic_and_raw_free(self):
        seal = self.seal()
        self.assertEqual(seal, self.seal())
        body = dict(seal)
        identity = body.pop("manifest_identity")
        self.assertEqual(canonical_hash(body), identity)
        for row in seal["cohorts"]["S_DEV"]["records"]:
            self.assertFalse(set(row) & {"prompt", "subject", "target_new", "target_true"})

    def test_prompt_identity_changes_without_changing_selection(self):
        rows = _records(1)
        before = evaluation_inventory(rows)
        rows[0]["neighborhood_prompts"][0] = "changed observation input"
        after = evaluation_inventory(rows)
        self.assertNotEqual(before["input_order_sha256"], after["input_order_sha256"])
        self.assertEqual(before["denominators"], after["denominators"])


if __name__ == "__main__":
    unittest.main()
