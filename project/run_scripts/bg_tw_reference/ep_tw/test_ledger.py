"""CPU ledger chronology/acceptance fixtures, not actual history/GPU tests."""
from copy import deepcopy
import json
import unittest

from .ledger import AcceptedLedger, LedgerError, request_identity


def record(case, target="A", subject=" exact subject ", relation="P1"):
    return {"case_id": case, "requested_rewrite": {"subject": subject,
            "relation_id": relation, "target_new": {"str": target}}}


class LedgerFixtures(unittest.TestCase):
    def add(self, ledger, records, successes):
        return ledger.commit_batch(records, successes, {r["case_id"]: .2 for r in records},
                                   batch_index=ledger.processed_batches + 1,
                                   endpoint_identity={"weight_sha256": "fixture-selected"},
                                   expected_count=len(records))

    def test_empty_cold_ledger(self):
        ledger = AcceptedLedger()
        self.assertEqual(ledger.next_ordinal, 0)
        self.assertEqual(ledger.accepted_case_ids, [])
        self.assertEqual(ledger.processed_batches, 0)

    def test_only_final_strict_successes_accepted_all_requests_retained(self):
        ledger = AcceptedLedger()
        receipt = self.add(ledger, [record(0), record(1), record(2)], {0, 2})
        self.assertEqual(ledger.accepted_case_ids, [0, 2])
        self.assertEqual(len(ledger.requested_events), 3)
        self.assertEqual(receipt["unaccepted_count"], 1)
        self.assertEqual(ledger.accepted_events[0]["acceptance_loss"], .2)

    def test_failed_intent_never_supersedes_accepted_target(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0, "A")], {0})
        self.add(ledger, [record(1, "B")], set())
        self.assertEqual(ledger.annotated_accepted()[0]["status"], "ACTIVE_TARGET")
        self.assertEqual(ledger.latest_accepted_targets()[(" exact subject ", "P1")], "A")
        self.assertFalse(ledger.annotate_records([record(1, "B")])[0]["previously_accepted"])

    def test_same_target_reissue_and_later_accepted_conflict(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0, "A"), record(1, "A")], {0, 1})
        self.assertEqual([x["status"] for x in ledger.annotated_accepted()], ["ACTIVE_TARGET"] * 2)
        self.add(ledger, [record(2, "B")], {2})
        self.assertEqual([x["status"] for x in ledger.annotated_accepted()],
                         ["SUPERSEDED", "SUPERSEDED", "ACTIVE_TARGET"])
        self.add(ledger, [record(3, "A")], set())
        self.assertEqual(ledger.annotated_accepted()[-1]["target"], "B")

    def test_exact_case_whitespace_relation_no_normalization(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0, "A", "Alice"), record(1, "B", "alice"),
                          record(2, "C", "Alice "), record(3, "D", "Alice", "P2")], {0, 1, 2, 3})
        self.assertEqual(len(ledger.latest_accepted_targets()), 4)
        self.assertTrue(all(x["status"] == "ACTIVE_TARGET" for x in ledger.annotated_accepted()))

    def test_same_batch_accepted_order_and_failed_later_intent(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0, "A"), record(1, "B"), record(2, "C")], {0, 1})
        self.assertEqual([x["status"] for x in ledger.annotated_accepted()], ["SUPERSEDED", "ACTIVE_TARGET"])
        self.assertEqual(ledger.next_ordinal, 3)

    def test_requested_unknown_is_not_accepted(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0)], set())
        self.assertEqual(ledger.annotate_requested()[0]["status"], "UNKNOWN")
        self.assertEqual(ledger.accepted_case_ids, [])

    def test_duplicate_case_and_batch_fail_without_mutation(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0)], {0})
        before = ledger.state_hash()
        with self.assertRaisesRegex(LedgerError, "DUPLICATE_CASE"):
            self.add(ledger, [record(0)], set())
        with self.assertRaisesRegex(LedgerError, "DUPLICATE_FINALIZATION"):
            ledger.commit_batch([record(1)], {1}, {1: .3}, batch_index=1,
                                endpoint_identity={"sha": "test"}, expected_count=1)
        self.assertEqual(before, ledger.state_hash())

    def test_strict_membership_missing_nll_or_nan_fail_before_append(self):
        ledger = AcceptedLedger()
        for strict, nll in [({1}, {0: .2}), ({0}, {}), ({0}, {0: float("nan")})]:
            with self.assertRaises(LedgerError):
                ledger.commit_batch([record(0)], strict, nll, batch_index=1,
                                    endpoint_identity={"sha": "test"}, expected_count=1)
            self.assertEqual(ledger.next_ordinal, 0)
            self.assertEqual(ledger.processed_batches, 0)

    def test_snapshot_restore_exact_and_detached_copy(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0), record(1, "B")], {0})
        self.add(ledger, [record(2, "C")], {2})
        snapshot = json.loads(json.dumps(ledger.snapshot(), allow_nan=False))
        restored = AcceptedLedger.restore(snapshot)
        self.assertEqual(restored.state_hash(), ledger.state_hash())
        self.assertEqual(restored.annotated_accepted(), ledger.annotated_accepted())
        snapshot["requested_events"][0]["target"] = "CHANGED"
        self.assertEqual(ledger.requested_events[0]["target"], "A")
        with self.assertRaisesRegex(LedgerError, "RESTORED_CONTENT"):
            AcceptedLedger.from_dict(snapshot)

    def test_corrupted_chronology_or_acceptance_does_not_load(self):
        ledger = AcceptedLedger()
        self.add(ledger, [record(0)], {0})
        for field, value in [("next_ordinal", 100), ("accepted_events", [])]:
            payload = ledger.snapshot()
            payload[field] = value
            with self.assertRaises(LedgerError):
                AcceptedLedger.restore(payload)
        payload = ledger.snapshot()
        payload["requested_events"][0]["ordinal"] = 9
        with self.assertRaisesRegex(LedgerError, "ORDINAL"):
            AcceptedLedger.restore(payload)

    def test_default_b100_denominator_and_request_identity(self):
        ledger = AcceptedLedger()
        with self.assertRaisesRegex(LedgerError, "DENOMINATOR"):
            ledger.commit_batch([record(0)], {0}, {0: .2}, batch_index=1,
                                endpoint_identity={"sha": "test"})
        self.assertEqual(request_identity(record(1))["subject"], " exact subject ")
        with self.assertRaises(LedgerError):
            request_identity({"case_id": 1, "subject": "A", "target_new": "B"})


if __name__ == "__main__":
    unittest.main()
