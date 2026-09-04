from __future__ import annotations

import unittest

from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import (
    counterfact_locality_target_new_pairs,
)
from project.run_scripts.ordered_response_barrier_ode.preedit_ns_backfill import (
    PROMPT_PAIR_COUNT,
    PreEditNSBoundary,
    build_locality_pairs,
    build_plan,
    reduce_locality_rows,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "case_id": index,
            "requested_rewrite": {
                "target_new": {"str": f"new-{index}"},
                "target_true": {"str": f"true-{index}"},
            },
            "neighborhood_prompts": [f"prompt-{index}-{prompt}" for prompt in range(10)],
        }
        for index in range(100)
    ]


def _sealed() -> list[dict[str, object]]:
    return [
        {
            "case_id": index,
            "request_sha256": f"{index:064x}",
        }
        for index in range(100)
    ]


def _evaluated(kind: str, *, true: bool = False, tie_last: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(100):
        for prompt in range(10):
            nll = 1.0 if true else 2.0
            if tie_last and index == 99 and prompt == 9:
                nll = 2.0
            rows.append(
                {
                    "case_id": index,
                    "kind": kind,
                    "prompt_index": prompt,
                    "prompt": f"prompt-{index}-{prompt}",
                    "target": f"{'true' if true else 'new'}-{index}",
                    "target_token_ids": [index + 1],
                    "nll": nll,
                    "all_tokens_correct": true,
                }
            )
    return rows


class PreEditNSBackfillTests(unittest.TestCase):
    def test_endpoint_target_new_pairs_preserve_request_prompt_order(self) -> None:
        pairs = counterfact_locality_target_new_pairs(
            [
                {
                    "case_id": 7,
                    "requested_rewrite": {"target_new": {"str": "new-value"}},
                    "neighborhood_prompts": ["p0", "p1"],
                },
                {
                    "case_id": 9,
                    "requested_rewrite": {"target_new": {"str": "other-new"}},
                    "neighborhood_prompts": ["p2"],
                },
            ]
        )
        self.assertEqual(
            [(value.case_id, value.kind, value.prompt_index, value.prompt, value.target) for value in pairs],
            [
                (7, "locality_target_new", 0, "p0", "new-value"),
                (7, "locality_target_new", 1, "p1", "new-value"),
                (9, "locality_target_new", 0, "p2", "other-new"),
            ],
        )

    def test_pair_builder_keeps_exact_request_prompt_order(self) -> None:
        new, true = build_locality_pairs(_records())
        self.assertEqual(len(new), PROMPT_PAIR_COUNT)
        self.assertEqual(len(true), PROMPT_PAIR_COUNT)
        self.assertEqual((new[0].case_id, new[0].prompt_index), (0, 0))
        self.assertEqual((new[-1].case_id, new[-1].prompt_index), (99, 9))
        self.assertEqual(new[317].prompt, true[317].prompt)

    def test_canonical_ns_uses_strict_true_less_new_and_tie_fails(self) -> None:
        rows, summary = reduce_locality_rows(
            _evaluated("locality_target_new"),
            _evaluated("locality_target_true", true=True, tie_last=True),
            _sealed(),
        )
        self.assertEqual(len(rows), PROMPT_PAIR_COUNT)
        self.assertEqual(summary["ns_numerator"], PROMPT_PAIR_COUNT - 1)
        self.assertEqual(summary["ns_denominator"], PROMPT_PAIR_COUNT)
        self.assertEqual(summary["nll_tie_count"], 1)
        self.assertNotIn("prompt", rows[0])
        self.assertNotIn("target", rows[0])

    def test_alignment_mismatch_fails_closed(self) -> None:
        true = _evaluated("locality_target_true", true=True)
        true[0]["prompt"] = "different"
        with self.assertRaises(PreEditNSBoundary):
            reduce_locality_rows(_evaluated("locality_target_new"), true, _sealed())

    def test_plan_is_evaluation_only_cap_two(self) -> None:
        plan = build_plan()
        self.assertEqual(plan["array"], "0-1%2")
        self.assertEqual(plan["project_gpu_cap"], 2)
        for name in (
            "editor_run_count",
            "compute_z_count",
            "writer_count",
            "key_capture_count",
            "solve_count",
            "cache_history_mutation_count",
            "model_update_count",
        ):
            self.assertEqual(plan[name], 0)


if __name__ == "__main__":
    unittest.main()
