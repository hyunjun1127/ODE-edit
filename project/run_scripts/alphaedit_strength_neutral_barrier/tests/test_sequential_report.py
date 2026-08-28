from __future__ import annotations

import unittest

from project.run_scripts.alphaedit_strength_neutral_barrier.build_sequential_report import (
    paired_nll,
    subset_endpoint,
)


def _row(case_id: int, prompt_index: int, nll: float) -> dict:
    return {
        "case_id": case_id,
        "prompt_index": prompt_index,
        "prompt": f"prompt-{case_id}-{prompt_index}",
        "nll": nll,
    }


class SequentialReportTests(unittest.TestCase):
    def test_subset_endpoint_preserves_metric_rows_and_order(self):
        endpoint = {
            metric: [_row(0, 0, 1.0), _row(1, 0, 2.0), _row(2, 0, 3.0)]
            for metric in (
                "rewrite_target_new",
                "rewrite_target_true",
                "rephrase_target_new",
                "rephrase_target_true",
                "locality_target_true",
            )
        }
        observed = subset_endpoint(endpoint, [2, 0])
        self.assertEqual(
            [row["case_id"] for row in observed["rewrite_target_new"]], [0, 2]
        )

    def test_paired_nll_is_pointwise_not_aggregate_difference(self):
        left = {"rewrite_target_new": [_row(0, 0, 3.0), _row(1, 0, 5.0)]}
        right = {"rewrite_target_new": [_row(0, 0, 1.0), _row(1, 0, 4.0)]}
        observed = paired_nll(left, right, "rewrite_target_new")
        self.assertEqual(observed["count"], 2)
        self.assertEqual(observed["mean"], 1.5)
        self.assertEqual(observed["median"], 1.5)
        self.assertEqual(observed["max"], 2.0)

    def test_paired_nll_rejects_row_mismatch(self):
        left = {"rewrite_target_new": [_row(0, 0, 1.0)]}
        right = {"rewrite_target_new": [_row(1, 0, 1.0)]}
        with self.assertRaises(RuntimeError):
            paired_nll(left, right, "rewrite_target_new")


if __name__ == "__main__":
    unittest.main()
