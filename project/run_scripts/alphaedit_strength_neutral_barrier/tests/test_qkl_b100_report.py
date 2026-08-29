from __future__ import annotations

import unittest

from project.run_scripts.alphaedit_strength_neutral_barrier.build_qkl_b100_report import (
    summarize_endpoint,
    summarize_geometry,
)


def _row(case: int, prompt: int, nll: float, correct: bool) -> dict:
    return {
        "case_id": case,
        "prompt_index": prompt,
        "prompt": f"p-{case}-{prompt}",
        "nll": nll,
        "all_tokens_correct": correct,
        "token_correct": [correct],
    }


class QKLB100ReportTests(unittest.TestCase):
    def test_endpoint_uses_pointwise_nll_and_strict_rephrase(self):
        endpoint = {
            "rewrite_target_new": [_row(0, 0, 1.0, True)],
            "rewrite_target_true": [_row(0, 0, 3.0, False)],
            "rephrase_target_new": [_row(0, 0, 1.0, True), _row(0, 1, 2.0, False)],
            "rephrase_target_true": [_row(0, 0, 3.0, False), _row(0, 1, 4.0, False)],
        }
        observed = summarize_endpoint(endpoint)
        self.assertEqual(observed["rephrase_target_new"]["nll"]["median"], 1.5)
        self.assertEqual(observed["rewrite_preference"]["rate"], 1.0)
        self.assertEqual(observed["rephrase_strict_exact"]["rate"], 0.0)

    def test_missing_official_geometry_is_explicit(self):
        observed = summarize_geometry({})
        self.assertEqual(observed["availability"], "NOT_RECORDED")
        self.assertIsNone(observed["terminal_qkl_last_layer"])

    def test_projected_geometry_aggregates_nodes(self):
        node = {
            "correction_active": True,
            "native_lookahead": {"aggregate_q_kl": 2.0},
            "post_projected": {"aggregate_q_kl": 1.0},
            "removed_energy_fraction": 0.25,
            "positive_projected_rate_violation": 1e-7,
        }
        observed = summarize_geometry(
            {"layers": [{"terminal_barrier_q_kl": 0.5, "nodes": [node, node]}]}
        )
        self.assertEqual(observed["active_node_fraction"], 1.0)
        self.assertEqual(observed["native_predictive_qkl"]["mean"], 2.0)
        self.assertEqual(observed["post_projected_qkl"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
