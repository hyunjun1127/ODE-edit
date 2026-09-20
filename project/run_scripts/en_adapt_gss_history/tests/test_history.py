"""Policy and persistence-boundary regression tests with small semantic streams."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from project.run_scripts.en_adapt_gss_history.history import (
    HistoryLedger, gss_prune, history_weights, priority,
)


ROOT = Path(__file__).resolve().parents[4]
REFERENCE_PATH = ROOT / "audits/global/2026-09-20-en-adapt-gss-history-design-v1/history_reference.py"
spec = importlib.util.spec_from_file_location("canonical_history_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def record(case, subject=None, target="new", prompt="{} is"):
    return dict(case_id=case, requested_rewrite=dict(
        subject=str(case) if subject is None else subject, relation_id="P1", prompt=prompt,
        target_new=dict(id="Q-" + target, str=target), target_true=dict(id="Q-old", str="old")),
        paraphrase_prompts=["Observer only"], neighborhood_prompts=["Observer only"])


def teachers(prep):
    return {vid: dict(path=f"cold/{vid}.npy", sha256="0" * 64,
                      input_identity="canonical-prefix", created_batch=prep["batch"])
            for vid in prep["teacher_required_ids"]}


def apply(ledger, rows, selected=None, terminal=False):
    batch = ledger.last_batch + 1
    prep = ledger.prepare(batch, rows, terminal=terminal)
    if selected is None:
        selected = prep["pool_ids"][:ledger.cap]
    result = ledger.commit(batch, rows, selected, teachers(prep), preparation=prep, terminal=terminal)
    return prep, result


class VersionLedgerTests(unittest.TestCase):
    def test_latest_override_rejects_uniform_gss_arm(self):
        with self.assertRaises(ValueError):
            HistoryLedger("EN_ADAPT_H_GSS")

    def test_prepare_is_pure_and_same_target_keeps_original_teacher_birth(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC", cap=2, pending_capacity=2)
        first = record(1, "subject")
        prep, _ = apply(ledger, [first])
        vid = ledger.active_ids()[0]
        original = copy.deepcopy(ledger.versions[vid])
        before = ledger.snapshot()
        repeat = record(2, "subject", prompt="About {}, the value is")
        prep2 = ledger.prepare(2, [repeat])
        self.assertEqual(before, ledger.snapshot())
        self.assertEqual(prep2["pool_ids"], [])
        self.assertEqual(prep2["teacher_required_ids"], [])
        self.assertEqual(prep2["new_versions"], [])
        ledger.commit(2, [repeat], [], {}, preparation=prep2)
        self.assertEqual(ledger.active_ids(), [vid])
        updated = ledger.versions[vid]
        self.assertEqual(updated["record"], original["record"])
        self.assertEqual(updated["teacher_binding"], original["teacher_binding"])
        self.assertEqual(updated["created_batch"], 1)
        self.assertEqual(updated["occurrence_case_ids"], [1, 2])
        self.assertEqual(updated["latest_record"], repeat)
        self.assertEqual(ledger.latest_records(), [repeat])
        self.assertEqual(ledger.pending_ids, [vid])
        self.assertEqual(ledger.active_case_ids(), [1, 2])
        self.assertEqual(ledger.prepare(3, [record(3)])["pool_ids"], [vid])

    def test_changed_target_retires_old_version_only_on_commit(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC", cap=2, pending_capacity=2)
        apply(ledger, [record(1, "x"), record(2, "y")])
        old = ledger.active_ids()[0]
        old_teacher = copy.deepcopy(ledger.versions[old]["teacher_binding"])
        replacement = record(3, "x", "changed")
        preparation = ledger.prepare(2, [replacement])
        self.assertTrue(ledger.versions[old]["active"])
        self.assertNotIn(old, preparation["pool_ids"])
        new = preparation["new_versions"][0]["version_id"]
        ledger.commit(2, [replacement], preparation["pool_ids"], teachers(preparation),
                      preparation=preparation)
        self.assertFalse(ledger.versions[old]["active"])
        self.assertEqual(ledger.versions[old]["teacher_binding"], old_teacher)
        self.assertEqual(ledger.versions[old]["retired_batch"], 2)
        self.assertTrue(ledger.versions[new]["active"])
        self.assertEqual(ledger.active_case_ids(), [2, 3])
        self.assertEqual(len(ledger.occurrence_rows()), 3)
        self.assertTrue(ledger.occurrence_rows()[0]["superseded"])

    def test_within_batch_latest_order_no_false_simultaneous_target_success(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC", cap=3, pending_capacity=3)
        apply(ledger, [record(1, "x", "A")])
        old = ledger.active_ids()[0]
        rows = [record(2, "x", "B"), record(3, "x", "A"), record(4, "x", "A")]
        prep, result = apply(ledger, rows)
        self.assertEqual(len(prep["created_versions"]), 2)
        self.assertEqual(len(prep["new_versions"]), 1)
        latest = ledger.active_ids()[0]
        self.assertNotEqual(latest, old)  # A -> B -> A is a genuinely new version.
        self.assertEqual(ledger.versions[latest]["created_batch"], 2)
        self.assertEqual(ledger.versions[latest]["record"]["case_id"], 3)
        self.assertEqual(ledger.versions[latest]["latest_record"]["case_id"], 4)
        self.assertEqual(ledger.versions[latest]["occurrence_case_ids"], [3, 4])
        self.assertEqual(ledger.active_case_ids(), [3, 4])
        transient = [v for v in ledger.versions.values() if v["target"]["str"] == "B"][0]
        self.assertFalse(transient["active"])
        self.assertIsNone(transient["teacher_binding"])
        self.assertEqual(transient["teacher_not_required"], "superseded_same_batch_before_commit")
        self.assertEqual(result["teacher_created_ids"], [latest])

    def test_target_identity_binds_entity_id_and_supplied_text(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        apply(ledger, [record(1, "x")])
        changed = record(2, "x")
        changed["requested_rewrite"]["target_new"]["id"] = "Q-different"
        prep = ledger.prepare(2, [changed])
        self.assertEqual(len(prep["new_versions"]), 1)

    def test_unsealed_fact_identity_is_error(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        for key in ("subject", "relation_id"):
            malformed = record(1)
            del malformed["requested_rewrite"][key]
            with self.assertRaises(ValueError):
                ledger.prepare(1, [malformed])
        self.assertEqual(ledger.last_batch, 0)

    def test_terminal_skips_only_unused_new_teacher_and_pending(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC")
        apply(ledger, [record(1)])
        previous = ledger.active_ids()[0]
        old_binding = copy.deepcopy(ledger.versions[previous]["teacher_binding"])
        prep, _ = apply(ledger, [record(2)], terminal=True)
        self.assertEqual(prep["teacher_required_ids"], [])
        self.assertEqual(ledger.pending_ids, [])
        self.assertEqual(ledger.versions[previous]["teacher_binding"], old_binding)
        latest = ledger.active_ids()[-1]
        self.assertEqual(ledger.versions[latest]["teacher_not_required"], "terminal_no_future_replay")
        self.assertFalse(ledger.snapshot()["save_checkpoints"])
        self.assertEqual(ledger.snapshot()["exact_crash_resume"], "NOT_AVAILABLE")


class BankPolicyTests(unittest.TestCase):
    def test_gss_eviction_does_not_resurrect_and_repeat_reproposes_original(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC", cap=2, pending_capacity=2)
        apply(ledger, [record(1), record(2)])
        first_ids = ledger.active_ids()
        apply(ledger, [record(3), record(4)])
        third_records = [record(5), record(6)]
        preparation = ledger.prepare(3, third_records)
        self.assertEqual(len(preparation["pool_ids"]), 4)
        keep = preparation["pool_ids"][2:]
        evicted = first_ids[0]
        original = copy.deepcopy(ledger.versions[evicted])
        ledger.commit(3, third_records, keep, teachers(preparation), preparation=preparation)
        preparation4 = ledger.prepare(4, [record(7), record(8)])
        self.assertNotIn(evicted, preparation4["pool_ids"])
        self.assertTrue(ledger.versions[evicted]["active"])
        self.assertEqual(ledger.versions[evicted]["teacher_binding"], original["teacher_binding"])
        # Reappearance is excluded from this batch loss and admitted for the next.
        rows = [record(7, "1"), record(8)]
        preparation4 = ledger.prepare(4, rows)
        self.assertNotIn(evicted, preparation4["pool_ids"])
        ledger.commit(4, rows, preparation4["pool_ids"][:2], teachers(preparation4),
                      preparation=preparation4)
        self.assertIn(evicted, ledger.pending_ids)
        self.assertIn(evicted, ledger.prepare(5, [record(9)])["pool_ids"])
        self.assertEqual(ledger.versions[evicted]["teacher_binding"], original["teacher_binding"])
        self.assertEqual(ledger.versions[evicted]["created_batch"], 1)

    def test_res_replenishes_whole_active_ledger_after_current_exclusion(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES", cap=2, pending_capacity=2)
        apply(ledger, [record(1), record(2)])
        apply(ledger, [record(3), record(4)])
        rows = [record(5), record(6)]
        prep, _ = apply(ledger, rows)
        all_active = set(ledger.active_ids())
        current_bank = set(ledger.bank_ids)
        outside = all_active - current_bank - set(ledger.pending_ids)
        self.assertEqual(len(outside), 2)
        # Exclude both prior bank facts: RES must consult all active metadata.
        repeated = [record(7 + i, ledger.versions[vid]["fact"][0])
                    for i, vid in enumerate(ledger.bank_ids)]
        prep4 = ledger.prepare(4, repeated)
        eligible = all_active - current_bank
        expected = sorted(eligible, key=priority)[:2]
        self.assertEqual(prep4["pool_ids"], expected)
        self.assertNotEqual(set(expected), current_bank)
        ledger.commit(4, repeated, expected, teachers(prep4), preparation=prep4)
        self.assertEqual(set(ledger.pending_ids), current_bank)
        self.assertEqual(set(ledger.bank_ids), set(expected))

    def test_actual_stream_capacity_and_first_overflow_not_hardcoded(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC")
        pools = []
        for batch in range(1, 9):
            rows = [record((batch - 1) * 100 + i) for i in range(100)]
            prep, _ = apply(ledger, rows)
            pools.append(len(prep["pool_ids"]))
            self.assertLessEqual(len(ledger.bank_ids), 512)
            self.assertLessEqual(len(ledger.pending_ids), 100)
        self.assertEqual(pools, [0, 100, 200, 300, 400, 500, 600, 612])
        repeated = HistoryLedger("EN_ADAPT_H_GSS_REC")
        for batch in range(1, 9):
            rows = [record((batch - 1) * 100 + i, str(i)) for i in range(100)]
            prep, _ = apply(repeated, rows)
            self.assertEqual(prep["pool_ids"], [])  # all current facts excluded
        self.assertEqual(len(repeated.versions), 100)
        self.assertEqual(len(repeated.eventledger), 800)

    def test_selection_requires_exact_bank_capacity_and_pool_identity(self):
        ledger = HistoryLedger("EN_ADAPT_H_GSS_REC", cap=2, pending_capacity=2)
        apply(ledger, [record(1), record(2)])
        prep = ledger.prepare(2, [record(3)])
        for bad in ([prep["pool_ids"][0]], ["unknown", prep["pool_ids"][0]],
                    [prep["pool_ids"][0]] * 2):
            with self.assertRaises(ValueError):
                ledger.select(prep, bad)
        self.assertEqual(ledger.select(prep, prep["pool_ids"])["selected_ids"], prep["pool_ids"])


class AtomicCommitTests(unittest.TestCase):
    def test_no_partial_mutation_and_retry_then_duplicate_rejected(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        rows = [record(1), record(2)]
        prep = ledger.prepare(1, rows)
        before = ledger.snapshot()
        wrong_binding = teachers(prep)
        del wrong_binding[prep["teacher_required_ids"][0]]
        with self.assertRaises(ValueError):
            ledger.commit(1, rows, [], wrong_binding, preparation=prep)
        self.assertEqual(before, ledger.snapshot())
        good = teachers(prep)
        ledger.commit(1, rows, [], good, preparation=prep)
        after = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.commit(1, rows, [], good, preparation=prep)
        self.assertEqual(after, ledger.snapshot())

    def test_no_silent_teacher_overwrite_or_tensor_metadata(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        prep, _ = apply(ledger, [record(1)])
        old = ledger.active_ids()[0]
        rows = [record(2)]
        prep = ledger.prepare(2, rows)
        before = ledger.snapshot()
        bindings = teachers(prep)
        bindings[old] = dict(path="overwrite-not-allowed")
        with self.assertRaises(ValueError):
            ledger.commit(2, rows, prep["pool_ids"], bindings, preparation=prep)
        for invalid in (np.zeros((1, 2)), float("nan"), float("inf")):
            bindings = teachers(prep)
            bindings[prep["teacher_required_ids"][0]]["bad"] = invalid
            with self.assertRaises((TypeError, ValueError)):
                ledger.commit(2, rows, prep["pool_ids"], bindings, preparation=prep)
            self.assertEqual(before, ledger.snapshot())

    def test_tampered_request_and_preparation_are_rejected(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        rows = [record(1)]
        prep = ledger.prepare(1, rows)
        changed = copy.deepcopy(rows)
        changed[0]["requested_rewrite"]["prompt"] = "{} has changed"
        with self.assertRaises(ValueError):
            ledger.commit(1, changed, [], teachers(prep), preparation=prep)
        changed_prep = copy.deepcopy(prep)
        changed_prep["new_versions"][0]["created_batch"] = 0
        with self.assertRaises(ValueError):
            ledger.select(changed_prep, [])
        self.assertEqual(ledger.last_batch, 0)

    def test_prepare_is_stale_after_another_commit(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        stale = ledger.prepare(1, [record(1)])
        apply(ledger, [record(2)])
        with self.assertRaises(ValueError):
            ledger.select(stale, [])

    def test_snapshot_is_json_only_and_defensive_copy(self):
        ledger = HistoryLedger("EN_ADAPT_H_RES")
        apply(ledger, [record(1)])
        snapshot = ledger.snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        snapshot["versions"][0]["teacher_binding"]["path"] = "mutated"
        self.assertNotEqual(ledger.versions[ledger.active_ids()[0]]["teacher_binding"]["path"], "mutated")


class MathematicalReferenceTests(unittest.TestCase):
    def test_signed_pruning_matches_canonical_including_ties_and_zero(self):
        fixtures = [
            np.array([[1., 0.], [-1., 0.], [1., 0.], [0., 1.]]),
            np.array([[1., 0.], [0., 0.], [0., 0.], [0., 0.]]),
            np.random.default_rng(42).normal(size=(30, 12)),
            np.ones((12, 3)),
        ]
        for z in fixtures:
            ids = [f"version-{i}" for i in range(len(z))]
            for cap in (2, len(z), len(z) + 1):
                actual = gss_prune(z, ids, cap)
                expected = reference.gss_prune(z, ids, cap)
                self.assertEqual(actual, expected)
                json.dumps(actual, allow_nan=False)  # np scalar regression boundary
        signed = gss_prune(fixtures[0][:3], ["a", "b", "c"], 2)
        self.assertIn(1, signed["selected"])  # opposite direction is not redundant

    def test_recency_matches_reference_without_extra_microbatch_average(self):
        for created in ([1, 1], [1, 2, 4], []):
            for recency in (True, False):
                actual = history_weights(created, 5, recency=recency)
                expected = reference.history_weights(created, 5, recency=recency)
                np.testing.assert_array_equal(actual, expected)
        weights = history_weights([1, 99], 100)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertLessEqual(weights.max() / weights.min(), 2)
        np.testing.assert_array_equal(history_weights([99] * 3, 100), np.full(3, 1 / 3))
        with self.assertRaises(ValueError):
            history_weights([5], 5)
        with self.assertRaises(ValueError):
            history_weights([float("nan")], 5)

    def test_nonfinite_features_are_technical_errors_even_below_capacity(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                gss_prune([[value, 0]], ["v"], 512)
        with self.assertRaises(ValueError):
            gss_prune([[1, 2], [3, 4]], ["duplicate", "duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
