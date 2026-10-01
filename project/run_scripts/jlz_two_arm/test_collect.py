"""Synthetic CPU-only reducer checks; no model, scheduler, or experiment I/O."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .collect import (accounting_snapshot, collect, expected_identity, extract_rows,
                      paired, reduce_rows, validate_ledger, validate_rows)


def record(cid):
    return dict(case_id=cid, requested_rewrite=dict(subject="fixture " + str(cid), prompt="{} is",
                target_new={"str": "new"}, target_true={"str": "true"}),
                paraphrase_prompts=["p0 " + str(cid), "p1 " + str(cid)],
                neighborhood_prompts=["n" + str(j) + ":" + str(cid) for j in range(10)])


def rows(records, convention="JLZ_UTF8_KIND"):
    output = []
    for kind, count in (("R", 1), ("P", 2), ("N", 10)):
        for rec in records:
            for index in range(count):
                output.append(dict(kind=kind, case_id=rec["case_id"], prompt_index=index,
                                   identity=expected_identity(rec, kind, index, convention),
                                   identity_convention=convention, true_nll=2., new_nll=1.,
                                   true_token_correct=1, true_token_count=1, true_strict=True,
                                   new_token_correct=1, new_token_count=2, new_strict=False,
                                   active_at_endpoint=True))
    return output


def state(char):
    return {field: {str(i): char * 64 for i in range(4, 9)} for field in ("W", "H")}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.records = [record(i) for i in range(2)]
        self.by_id = {r["case_id"]: r for r in self.records}
        self.ids = {k: [0, 1] for k in ("R", "P", "N")}
        self.rows = rows(self.records)

    def test_raw_preference_ties_and_tf_denominators(self):
        validate_rows(self.rows, self.by_id, self.ids)
        self.rows[0].update(new_nll=2.)
        reduced = reduce_rows(self.rows)
        self.assertEqual((reduced["R"]["numerator"], reduced["R"]["denominator"], reduced["R"]["ties"]), (1, 2, 1))
        self.assertEqual(reduced["R"]["token_micro"], .5)
        self.assertEqual(reduced["R"]["strict_rate"], 0.)
        self.assertEqual(reduced["N"]["numerator"], 0)
        self.assertEqual(reduced["N"]["token_micro"], 1.)

    def test_prompt_target_identity_and_order_are_independent(self):
        bad = copy.deepcopy(self.rows)
        bad[0]["identity"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PROMPT_TARGET_IDENTITY"):
            validate_rows(bad, self.by_id, self.ids)
        bad = [self.rows[1], self.rows[0], *self.rows[2:]]
        with self.assertRaisesRegex(ValueError, "CASE_PROMPT_ORDER"):
            validate_rows(bad, self.by_id, self.ids)
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            validate_rows(self.rows + [self.rows[0]], self.by_id, self.ids)

    def test_nonfinite_and_strict_corruption_block(self):
        bad = copy.deepcopy(self.rows)
        bad[0]["new_nll"] = float("nan")
        with self.assertRaisesRegex(ValueError, "NONFINITE_NLL"):
            validate_rows(bad, self.by_id, self.ids)
        bad = copy.deepcopy(self.rows)
        bad[0]["new_strict"] = True
        with self.assertRaisesRegex(ValueError, "STRICT_TOKEN"):
            validate_rows(bad, self.by_id, self.ids)

    def test_missing_tf_is_not_fabricated(self):
        missing = [{k: v for k, v in r.items() if "token_" not in k and not k.endswith("_strict")} for r in self.rows]
        validate_rows(missing, self.by_id, self.ids)
        reduced = reduce_rows(missing)
        self.assertEqual(reduced["R"]["tf_status"], "NOT_RECORDED")
        self.assertIsNone(reduced["R"]["token_micro"])

    def test_native_hash_convention_normalizes_without_hash_equivalence_claim(self):
        native = rows(self.records, "NATIVE_ASCII")
        document = {"metrics": {tag: {"rows": [{k: v for k, v in r.items() if k not in ("kind", "identity_convention")}
                                                for r in native if r["kind"] == kind]}
                                for tag, kind in (("RS", "R"), ("PS", "P"), ("NS", "N"))}}
        normalized = extract_rows(document)
        validate_rows(normalized, self.by_id, self.ids)
        self.assertNotEqual(normalized[0]["identity"], self.rows[0]["identity"])
        summary, _ = paired(normalized, self.rows, "native_to_jlz")
        self.assertTrue(all(r["lost"] == r["gained"] == 0 for r in summary))

    def test_pair_transition_ids_and_missing_not_silently_intersected(self):
        after = copy.deepcopy(self.rows)
        after[0]["new_nll"] = 3.
        table, identities = paired(self.rows, after, "fixture")
        summary = next(r for r in table if r["kind"] == "R" and r["split"] == "all")
        self.assertEqual((summary["lost"], summary["gained"], summary["denominator"]), (1, 0, 2))
        self.assertEqual(identities[0]["lost_ids"], [["R", 0, 0]])
        with self.assertRaisesRegex(ValueError, "PAIRED_MISSING"):
            paired(self.rows, after[1:], "bad")

    def test_ledger_budget_chain_and_final_reserve(self):
        ledger = [dict(batch=1, case_ids=list(range(100)), commit=True, history_appends=5,
                       entry=state("a"), post=state("b"), solver=dict(calls=2, status="BUDGET_STOP",
                       final_recomputed=True, commit_eligible=True,
                       budget=dict(cap=120, actual_calls=2, events=["initial", "final"],
                                   final_reserved=True, final_started=True)))]
        result = validate_ledger(ledger, list(range(2000)), stage="main")
        self.assertEqual((result["commits"], result["calls"], result["history_appends"]), (1, 2, 5))
        self.assertEqual(result["warnings"][0]["action"], "RECORD_ONLY_USER_DIRECTED")
        bad = copy.deepcopy(ledger)
        bad[0]["solver"]["budget"]["events"][-1] = "trial"
        with self.assertRaisesRegex(ValueError, "FINAL_RESERVE"):
            validate_ledger(bad, list(range(2000)), stage="main")
        bad = copy.deepcopy(ledger)
        bad[0]["history_appends"] = 4
        with self.assertRaisesRegex(ValueError, "HISTORY"):
            validate_ledger(bad, list(range(2000)), stage="main")

    def test_all40_commits200_appends4800_calls_reference_schedule(self):
        ledger, schedule = [], []
        for batch in range(1, 21):
            current = list(range((batch - 1) * 100, batch * 100))
            general = list(range(10000, 10016))
            replay = list(range(max(0, (batch - 1) * 100 - 16), (batch - 1) * 100))
            before = {field: {str(layer): f"{batch - 1:064x}" for layer in range(4, 9)} for field in ("W", "H")}
            after = {field: {str(layer): f"{batch:064x}" for layer in range(4, 9)} for field in ("W", "H")}
            ledger.append(dict(batch=batch, case_ids=current, commit=True, history_appends=5,
                               entry=before, post=after,
                               references=dict(current_ids=current, general_ids=general, replay_ids=replay),
                               solver=dict(calls=120, status="BUDGET_STOP", final_recomputed=True,
                                           commit_eligible=True, budget=dict(cap=120, actual_calls=120,
                                           events=["initial"] + ["trial"] * 118 + ["final"],
                                           final_reserved=True, final_started=True))))
            schedule.append(dict(ids=current, general=general, replay=replay))
        result = validate_ledger(ledger, list(range(2000)), stage="main", schedule=schedule)
        self.assertEqual((2 * result["commits"], 2 * result["history_appends"], 2 * result["calls"]), (40, 200, 4800))
        self.assertEqual(result["reference_membership"], "VERIFIED")
        self.assertEqual(result["chain_links"], 19)
        bad = copy.deepcopy(ledger)
        bad[9]["entry"] = state("a")
        with self.assertRaisesRegex(ValueError, "CHAIN_CONTINUITY"):
            validate_ledger(bad, list(range(2000)), stage="main", schedule=schedule)
        bad = copy.deepcopy(ledger)
        bad[0]["references"]["replay_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "REFERENCE_MEMBERSHIP"):
            validate_ledger(bad, list(range(2000)), stage="main", schedule=schedule)
        bad = copy.deepcopy(ledger)
        bad[0]["solver"]["status"] = "NONFINITE"
        with self.assertRaisesRegex(ValueError, "INVALID_COMMITTED_SOLVER_STATUS"):
            validate_ledger(bad, list(range(2000)), stage="main", schedule=schedule)

    def test_success_aggregate_is_never_used_instead_of_raw(self):
        bad = copy.deepcopy(self.rows)
        bad[0]["success"] = False
        with self.assertRaisesRegex(ValueError, "SUCCESS_STRICT_TIE_FAILURE"):
            validate_rows(bad, self.by_id, self.ids)
        normalized = extract_rows(dict(rows=self.rows, summary={"R": {"numerator": -999}}))
        self.assertEqual(reduce_rows(normalized)["R"]["numerator"], 2)

    def test_single_parent_snapshot_no_steps_double_count(self):
        calls = []
        def command(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stderr="", stdout=
                "10|fixture|owner|COMPLETED|0:0|100|cpu=8,gres/gpu=1|start|end|server4|\n"
                "10.batch|fixture|owner|COMPLETED|0:0|100|cpu=8,gres/gpu=1|start|end|server4|\n"
                "11|fixture|owner|FAILED|1:0|20|cpu=8,gres/gpu=1|start|end|server4|\n")
        report = accounting_snapshot(["10", "11"], command)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(report["parents"]), 2)
        self.assertEqual(report["allocated_gpu_seconds"], 120)
        self.assertTrue(report["parent_only"])

    def test_incomplete_run_still_publishes_not_complete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "stream.json"
            dataset.write_text(json.dumps([record(i) for i in range(2000)]))
            config = root / "config.json"
            config.write_text(json.dumps({"stream": str(dataset), "baselines": []}))
            result = collect(root, config, {"parents": [], "allocated_gpu_seconds": None})
            self.assertEqual(result["status"], "NOT_COMPLETE")
            self.assertEqual(result["main_totals"]["commits"], 0)
            self.assertTrue((root / "report/report-ko.md").exists())
            self.assertTrue((root / "report/package-manifest.json").exists())
            self.assertFalse(result["baseline_scope_complete"])


if __name__ == "__main__":
    unittest.main()
