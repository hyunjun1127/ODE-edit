from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.contracts import BATCH_SIZE, ODEBFContractError
from project.run_scripts.ode_bf.p1_selection import (
    EXPECTED_BASE,
    P_POPULATION_COUNT,
    P_POPULATION_SALT,
    STREAM_REQUEST_COUNT,
    STREAM_SALT,
    build_p1_seals,
    load_p1_population_requests,
    load_p1_stream_batches,
    scan_prior_tracked_seals,
    verify_p1_population_seal,
    verify_p1_stream_seal,
)
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.sampling import (
    SampleLineage,
    StatelessReplaySchedule,
    assert_cross_arm_common_random_numbers,
    load_p1_sampling_seal,
)


REPO = Path(__file__).resolve().parents[4]
LOCKS = Path(__file__).resolve().parents[1] / "locks"
DATASET = Path("/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json")


def _load(name: str) -> dict[str, object]:
    return json.loads((LOCKS / name).read_text(encoding="utf-8"))


class P1SealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET.is_file():
            raise unittest.SkipTest("pinned CounterFact dataset is unavailable")
        cls.stream = _load("p1_seqb10_stream_seal.json")
        cls.population = _load("p1_p_population_seal.json")

    def test_exact_seals_rebuild_from_locked_base_without_local_results(self) -> None:
        rebuilt_stream, rebuilt_population = build_p1_seals(
            DATASET,
            REPO,
            base_commit=EXPECTED_BASE,
        )
        self.assertEqual(rebuilt_stream, self.stream)
        self.assertEqual(rebuilt_population, self.population)
        self.assertEqual(self.stream["salt"], STREAM_SALT)
        self.assertEqual(self.population["salt"], P_POPULATION_SALT)
        self.assertEqual(
            self.stream["prior_selection_scan"]["local_result_paths_read"], 0
        )

    def test_prior_scan_is_base_commit_scoped_and_raw_free(self) -> None:
        first = scan_prior_tracked_seals(REPO)
        second = scan_prior_tracked_seals(REPO)
        self.assertEqual(first, second)
        self.assertEqual(first.base_commit, EXPECTED_BASE)
        self.assertGreater(first.source_count, 0)
        self.assertEqual(len(first.sources_digest), 64)

    def test_four_ordered_joint_batches_are_distinct_and_digest_locked(self) -> None:
        value = verify_p1_stream_seal(self.stream)
        requests = value["requests"]
        self.assertEqual(len(requests), STREAM_REQUEST_COUNT)
        self.assertEqual(len({item["case_id"] for item in requests}), STREAM_REQUEST_COUNT)
        self.assertEqual(
            len({item["collision_sha256"] for item in requests}),
            STREAM_REQUEST_COUNT,
        )
        for index in range(4):
            batch = requests[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
            self.assertEqual(len(batch), BATCH_SIZE)
            self.assertEqual(
                ordered_request_digest_v1([item["request_sha256"] for item in batch]),
                value["batch_ordered_request_digest_v1"][index],
            )

    def test_population_is_160_disjoint_and_lineages_are_common_random_numbers(self) -> None:
        value = verify_p1_population_seal(self.population, stream=self.stream)
        self.assertEqual(len(value["items"]), P_POPULATION_COUNT)
        stream_ids = {item["request_sha256"] for item in self.stream["requests"]}
        population_ids = {item["request_sha256"] for item in value["items"]}
        self.assertTrue(stream_ids.isdisjoint(population_ids))
        schedule = StatelessReplaySchedule(
            load_p1_sampling_seal(
                LOCKS / "p1_p_population_seal.json",
                stream_path=LOCKS / "p1_seqb10_stream_seal.json",
            )
        )
        before = schedule.state_digest
        batches = {
            arm: schedule.batch(
                SampleLineage.CONTROLLER,
                outer_batch_index=3,
                correction_cycle=0,
                waypoint=7,
                replay_batch_id=0,
            )
            for arm in ("F_G", "F_BF", "R_BF")
        }
        assert_cross_arm_common_random_numbers(batches)
        self.assertEqual(schedule.state_digest, before)

    def test_canonical_load_opens_only_rewrite_surface_and_preserves_order(self) -> None:
        batches = load_p1_stream_batches(DATASET, self.stream)
        population = load_p1_population_requests(
            DATASET,
            self.population,
            stream=self.stream,
        )
        self.assertEqual(tuple(map(len, batches)), (10, 10, 10, 10))
        self.assertEqual(len(population), 160)
        allowed = {
            "case_id",
            "prompt",
            "relation_id",
            "subject",
            "target_new",
            "target_true",
            "request_sha256",
        }
        self.assertTrue(all(set(item) == allowed for batch in batches for item in batch))
        self.assertTrue(all(set(item) == allowed for item in population))

    def test_reorder_duplicate_missing_and_stream_collision_fail_closed(self) -> None:
        reordered = json.loads(json.dumps(self.stream))
        reordered["requests"][0], reordered["requests"][1] = (
            reordered["requests"][1],
            reordered["requests"][0],
        )
        reordered.pop("root_digest")
        from project.run_scripts.ode_bf.contracts import canonical_hash

        reordered["root_digest"] = canonical_hash(reordered)
        with self.assertRaisesRegex(ODEBFContractError, "ordinal"):
            verify_p1_stream_seal(reordered)

        duplicate = json.loads(json.dumps(self.population))
        duplicate["items"][1]["request_sha256"] = duplicate["items"][0][
            "request_sha256"
        ]
        duplicate.pop("root_digest")
        duplicate["root_digest"] = canonical_hash(duplicate)
        with self.assertRaisesRegex(ODEBFContractError, "not distinct"):
            verify_p1_population_seal(duplicate, stream=self.stream)

        collision = json.loads(json.dumps(self.population))
        collision["items"][0]["request_sha256"] = self.stream["requests"][0][
            "request_sha256"
        ]
        collision.pop("root_digest")
        collision["root_digest"] = canonical_hash(collision)
        with self.assertRaisesRegex(ODEBFContractError, "collides"):
            verify_p1_population_seal(collision, stream=self.stream)


if __name__ == "__main__":
    unittest.main()
