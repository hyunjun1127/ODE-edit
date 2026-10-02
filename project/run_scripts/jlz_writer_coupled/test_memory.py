"""CPU-only synthetic memory-contract checks; no model or GPU qualification."""
import copy
import json
import pickle
import random
import unittest
from unittest import mock

import torch

from .memory import NativeMemory


def event(fact, version="v1", identity=None, quality=1.0, alias=None):
    identity = identity or f"kl-{fact}"
    return {
        "fact_id": str(fact), "version": str(version),
        "record": {"case_id": str(fact), "requested_rewrite": {
            "prompt": "{} has native target", "subject": alias or f"subject-{fact}",
            "target_new": {"str": f"target-{version}", "id": str(version)},
            "relation_id": "R1"}},
        "context_nll": [float(quality), float(quality + .5)],
        "kl_identity": identity,
        "kl_input": {"input_ids": [1, sum(identity.encode()) % 100 + 2],
                     "attention_mask": [1, 1], "positions": [1],
                     "readout": "native_last_position"},
        "teacher": torch.log_softmax(torch.arange(11, dtype=torch.float32)
                                     * float(quality), dim=0),
    }


def by_id(records):
    return {r["fact_id"]: r for r in records}


class MemoryTests(unittest.TestCase):
    def test_defaults_and_exact_first_unique_algorithm_r(self):
        memory = NativeMemory()
        self.assertEqual((memory.capacity, memory.reference_cap, memory.seed),
                         (128, 16, 20261002))
        expected = []
        rng = random.Random(20261002)
        events = []
        for i in range(400):
            fact = str(i)
            events.append(event(fact))
            # Repeats must neither advance n nor consume an Algorithm R draw.
            if i % 7 == 0:
                events.append(event(fact, "latest", quality=100.))
            if i < 128:
                expected.append(fact)
            else:
                draw = rng.randint(1, i + 1)
                if draw <= 128:
                    expected[draw - 1] = fact
        memory.admit(events)
        self.assertEqual(memory.fact_ids, tuple(expected))
        self.assertEqual(len(memory), 128)
        self.assertEqual(memory.summary()["seen_count"], 400)
        self.assertEqual(memory.summary()["owner_count"], 128)

    def test_uniform_subset_current_exclusion_and_once_per_batch(self):
        memory = NativeMemory(8, 3, 17)
        memory.admit([event(str(i)) for i in range(8)])
        before = memory.summary()
        records = memory.sample(["2", "6", "2"], actual_B=2)
        expected = random.Random(17).sample(["0", "1", "3", "4", "5", "7"], 2)
        self.assertEqual([r["fact_id"] for r in records], expected)
        after = memory.summary()
        self.assertNotEqual(before["rng_hash"], after["rng_hash"])
        repeated = memory.sample(["6", "2"], actual_B=2)
        self.assertEqual([r["fact_id"] for r in repeated], expected)
        self.assertEqual(memory.summary(), after)
        with self.assertRaisesRegex(RuntimeError, "batch changed"):
            memory.sample(["3"], actual_B=2)
        self.assertEqual(memory.summary(), after)
        # A successful empty commit still closes the sampling boundary.
        memory.admit([])
        self.assertEqual(len(memory.sample(["1"], actual_B=100)), 3)

    def test_reference_count_minimum_and_empty_first_batch(self):
        for cap, actual, excluded, count in (
                (16, 2, [], 2), (1, 100, [], 1), (16, 100, ["a"], 2),
                (16, 0, [], 0), (0, 10, [], 0),
                (16, 10, ["a", "b", "c"], 0)):
            with self.subTest(cap=cap, actual=actual, excluded=excluded):
                memory = NativeMemory(3, cap, 11)
                self.assertEqual(memory.sample(["a"], actual), [])
                memory.admit([event("a"), event("b"), event("c")])
                before_rng = memory.summary()["rng_hash"]
                self.assertEqual(len(memory.sample(excluded, actual)), count)
                if not count:
                    self.assertEqual(memory.summary()["rng_hash"], before_rng)

    def test_seen_nonresident_repeat_never_reenters(self):
        memory = NativeMemory(1, 1, 2)
        memory.admit([event("a"), event("b")])
        self.assertEqual(memory.fact_ids, ("b",))
        before = memory.summary()
        receipt = memory.admit([event("a", "retry", quality=1000.)])
        self.assertEqual(memory.fact_ids, ("b",))
        self.assertEqual(receipt["nonresident_repeats"], 1)
        self.assertEqual(memory.summary()["seen_count"], 2)
        self.assertEqual(memory.summary()["rng_hash"], before["rng_hash"])
        # A previously rejected first observation has the same rule.
        rejected = NativeMemory(1, 1, 0)  # n=2 draw is 2, so b never resides.
        rejected.admit([event("a"), event("b"), event("b", "retry")])
        self.assertEqual(rejected.fact_ids, ("a",))
        self.assertEqual(rejected.summary()["nonresident_repeats"], 1)

    def test_resident_repeat_replaces_rewrite_but_freezes_kl_pair(self):
        memory = NativeMemory(2, 2, 4)
        original = event("a", identity="first-alias")
        repeat = event("a", "v2", identity="new-alias", quality=3., alias="changed")
        memory.admit([original])
        before_rng = memory.summary()["rng_hash"]
        self.assertEqual(memory.sample(["a"], 2), [])
        memory.admit([repeat])
        resident = memory.sample([], 2)[0]
        self.assertEqual(resident["version"], "v2")
        self.assertEqual(resident["record"], repeat["record"])
        self.assertEqual(resident["context_nll"], repeat["context_nll"])
        self.assertEqual(resident["kl_identity"], original["kl_identity"])
        self.assertEqual(resident["kl_input"], original["kl_input"])
        self.assertTrue(torch.equal(resident["teacher"], original["teacher"]))
        # Sampling happened after update; repeat itself has consumed no RNG.
        self.assertEqual(memory.summary()["seen_count"], 1)
        self.assertEqual(memory.summary()["kl_identity_count"], 1)
        self.assertEqual(before_rng, NativeMemory(2, 2, 4).summary()["rng_hash"])

    def test_shared_teacher_earliest_resident_owner_and_fact_exclusion(self):
        memory = NativeMemory(3, 3, 3)
        first = event("a", identity="shared", quality=1.)
        second = event("b", identity="shared", quality=9.)
        memory.admit([first, second])
        all_records = memory.sample([], 3)
        records = by_id(all_records)
        self.assertIs(records["a"]["teacher"], records["b"]["teacher"])
        self.assertIs(records["a"]["kl_input"], records["b"]["kl_input"])
        self.assertTrue(torch.equal(records["b"]["teacher"], first["teacher"]))
        self.assertEqual(memory.summary()["owner_count"], 2)
        self.assertEqual(memory.summary()["kl_identity_count"], 1)
        self.assertEqual(memory.summary()["teacher_bytes"], first["teacher"].numel() * 4)
        memory.admit([])
        # Current a excludes a, not another owner of its KL identity.
        self.assertEqual([r["fact_id"] for r in memory.sample(["a"], 3)], ["b"])

    def test_refcounts_keep_teacher_until_last_owner_eviction(self):
        memory = NativeMemory(2, 2, 1)
        first = event("a", identity="shared", quality=1.)
        memory.admit([first, event("b", identity="shared", quality=2.),
                      event("c", identity="other", quality=3.),
                      event("d", identity="shared", quality=4.)])
        # Seed 1 replaces slot zero twice: a -> c -> d; b keeps old teacher alive.
        self.assertEqual(memory.fact_ids, ("d", "b"))
        records = by_id(memory.sample([], 2))
        self.assertTrue(torch.equal(records["d"]["teacher"], first["teacher"]))
        self.assertEqual(memory.summary()["kl_identity_count"], 1)
        self.assertEqual(memory.summary()["owner_count"], 2)

    def test_same_batch_last_owner_eviction_then_new_admission_renews_teacher(self):
        memory = NativeMemory(1, 1, 2)
        first = event("a", identity="shared", quality=1.)
        renewed = event("c", identity="shared", quality=3.)
        memory.admit([first, event("b", identity="different", quality=2.), renewed])
        self.assertEqual(memory.fact_ids, ("c",))
        resident = memory.sample([], 1)[0]
        self.assertTrue(torch.equal(resident["teacher"], renewed["teacher"]))
        self.assertFalse(torch.equal(resident["teacher"], first["teacher"]))
        self.assertEqual(memory.summary()["kl_identity_count"], 1)
        direct = NativeMemory(1, 1, 2)
        replacement = event("b", identity="shared", quality=2.)
        direct.admit([first, replacement])
        self.assertTrue(torch.equal(direct.sample([], 1)[0]["teacher"],
                                    replacement["teacher"]))

    def test_batch_conflicts_follow_stream_order_without_dropping_occurrences(self):
        memory = NativeMemory(2, 2, 2)
        first = event("a", "v1", identity="first")
        latest = event("a", "v3", identity="third", quality=10.)
        receipt = memory.admit([first, event("a", "v2"), event("b"), latest])
        self.assertEqual(receipt["events"], 4)
        self.assertEqual(receipt["resident_repeats"], 2)
        resident = by_id(memory.sample([], 2))["a"]
        self.assertEqual(resident["version"], "v3")
        self.assertEqual(resident["context_nll"], latest["context_nll"])
        self.assertEqual(resident["kl_identity"], "first")
        evicted = NativeMemory(1, 1, 2)
        evicted.admit([first, event("b"), latest])
        self.assertEqual(evicted.fact_ids, ("b",))
        self.assertEqual(evicted.summary()["nonresident_repeats"], 1)

    def test_same_metadata_selects_same_ids_independently_of_arm_values(self):
        left, right = NativeMemory(4, 3, 9), NativeMemory(4, 3, 9)
        for start in range(0, 32, 4):
            current = [str(i) for i in range(start, start + 4)]
            self.assertEqual([r["fact_id"] for r in left.sample(current, 4)],
                             [r["fact_id"] for r in right.sample(current, 4)])
            left.admit([event(f, quality=1.) for f in current])
            right.admit([event(f, quality=1000.) for f in current])
            self.assertEqual(left.fact_ids, right.fact_ids)
            self.assertEqual(left.summary()["rng_hash"], right.summary()["rng_hash"])
        self.assertNotEqual(left.summary()["anchor_hash"], right.summary()["anchor_hash"])

    def test_full_transaction_rollback_exact_rng_seen_anchors_and_pending(self):
        memory = NativeMemory(3, 2, 12)
        memory.admit([event("a"), event("b"), event("c")])
        before = memory.summary()
        snapshot = memory.snapshot()
        selection = [r["fact_id"] for r in memory.sample(["a", "d"], 2)]
        sampled = memory.summary()
        sampled_snapshot = memory.snapshot()
        events = [event("a", "v2", identity="alias"), event("d"), event("e")]
        memory.admit(events)
        committed = memory.summary()
        memory.restore(sampled_snapshot)
        self.assertEqual(memory.summary(), sampled)
        self.assertEqual([r["fact_id"] for r in memory.sample(["d", "a"], 2)], selection)
        self.assertEqual(memory.summary(), sampled)
        memory.admit(events)
        self.assertEqual(memory.summary(), committed)
        for _ in range(2):
            memory.restore(snapshot)
            self.assertEqual(memory.summary(), before)
            self.assertEqual([r["fact_id"] for r in memory.sample(["a", "d"], 2)], selection)
            memory.admit(events)
            self.assertEqual(memory.summary(), committed)

    def test_invalid_admission_rolls_back_partial_eviction_and_rng(self):
        memory = NativeMemory(2, 2, 0)
        memory.admit([event("a", identity="shared"), event("b")])
        bad = event("c", identity="shared")
        bad["kl_input"]["positions"] = [0]
        before = memory.summary()
        # Seed 0 evicts b, then discovers mismatched tuple of still-resident a.
        with self.assertRaisesRegex(ValueError, "different native input tuples"):
            memory.admit([bad])
        self.assertEqual(memory.summary(), before)
        malformed = event("d")
        malformed["context_nll"] = [float("nan")]
        with self.assertRaises(ValueError):
            memory.admit([event("a", "v2"), malformed])
        self.assertEqual(memory.summary(), before)

    def test_zero_capacity_tracks_seen_without_teacher_payload(self):
        memory = NativeMemory(0, 16, 7)
        memory.admit([event("a"), event("b"), event("a", "v2")])
        self.assertEqual(memory.fact_ids, ())
        self.assertEqual(memory.sample([], 100), [])
        self.assertEqual(memory.summary()["seen_count"], 2)
        self.assertEqual(memory.summary()["teacher_bytes"], 0)
        self.assertEqual(memory.summary()["kl_identity_count"], 0)

    def test_input_and_sample_mutation_cannot_change_frozen_bank(self):
        memory = NativeMemory(2, 2)
        original = event("a")
        expected = copy.deepcopy(original)
        memory.admit([original])
        original["teacher"].zero_()
        original["kl_input"]["input_ids"][0] = -10
        original["record"]["requested_rewrite"]["subject"] = "mutated"
        original["context_nll"][0] = 500.
        sample = memory.sample([], 2)
        before = memory.summary()
        self.assertEqual(sample[0]["record"], expected["record"])
        self.assertTrue(torch.equal(sample[0]["teacher"], expected["teacher"]))
        sample[0]["teacher"].zero_()
        sample[0]["kl_input"]["input_ids"].clear()
        sample[0]["context_nll"].clear()
        again = memory.sample([], 2)[0]
        self.assertEqual(again["kl_input"], expected["kl_input"])
        self.assertEqual(again["context_nll"], expected["context_nll"])
        self.assertTrue(torch.equal(again["teacher"], expected["teacher"]))
        self.assertEqual(memory.summary(), before)

    def test_ram_only_snapshot_and_payload_free_summary(self):
        memory = NativeMemory()
        with mock.patch("torch.save", side_effect=AssertionError("no teacher disk writes")):
            memory.admit([event("sensitive-fact", alias="sensitive-subject")])
            snapshot = memory.snapshot()
            memory.sample([], 1)
            memory.restore(snapshot)
            summary = memory.summary()
        self.assertTrue(all(isinstance(v, (str, int, bool)) for v in summary.values()))
        encoded = json.dumps(summary)
        self.assertNotIn("sensitive", encoded)
        self.assertNotIn("native target", encoded)
        self.assertEqual(repr(snapshot), "<NativeMemory snapshot: RAM only>")
        for obj in (memory, snapshot):
            with self.assertRaisesRegex(TypeError, "RAM-only"):
                pickle.dumps(obj)

    def test_invalid_types_and_non_native_records_are_rejected(self):
        for args in ((-1, 2), (2, -1), (True, 1), (2, 1.5)):
            with self.assertRaises(ValueError):
                NativeMemory(*args)
        memory = NativeMemory()
        with self.assertRaises(ValueError):
            memory.sample("not-an-id-list", 1)
        bad = event("a")
        bad["record"]["neighborhood_prompts"] = ["not native"]
        with self.assertRaisesRegex(ValueError, "only requested_rewrite"):
            memory.admit([bad])
        bad = event("a")
        bad["teacher"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            memory.admit([bad])
        with self.assertRaises(ValueError):
            NativeMemory(1).restore(memory.snapshot())
        with self.assertRaises(TypeError):
            memory.restore({})


if __name__ == "__main__":
    unittest.main()
