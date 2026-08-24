from __future__ import annotations

import unittest

from project.run_scripts.session05_ode_bf_p1r54_fz_c3_terminal_report import (
    INDEPENDENT_BASELINE_SHA,
    SEQUENTIAL_BASELINE_SHA,
    STREAM_ORDER,
    STREAM_ROOT,
    _analysis_endpoint,
)


class P1R54FZTerminalReportTest(unittest.TestCase):
    def test_common_seal_and_native_package_identities_are_fixed(self) -> None:
        self.assertEqual(len(STREAM_ROOT), 64)
        self.assertEqual(len(STREAM_ORDER), 64)
        self.assertEqual(
            INDEPENDENT_BASELINE_SHA["identity"],
            "ea5261b1f4b08732fa7ae08895dbe655d53a630d3c4f5b7bb86a5a74b644ff60",
        )
        self.assertEqual(
            SEQUENTIAL_BASELINE_SHA["identity"],
            "fb9b1e9709ec8c86319e8f88a36a9d99726fd4a22177f5cf8c2cb2607347812e",
        )

    def test_analysis_endpoint_keeps_z_and_w_denominators_explicit(self) -> None:
        performance = {
            label: {"numerator": 1, "denominator": 2, "rate": 0.5}
            for label in (
                "Eff",
                "Rewrite_accuracy",
                "Gen",
                "Gen_strict",
                "Rephrase_accuracy",
                "Rephrase_strict_accuracy",
                "Loc",
            )
        }
        distribution = {
            "count": 2,
            "sum": 3.0,
            "mean": 1.5,
            "median": 1.5,
            "p90_nearest_rank": 2.0,
            "max": 2.0,
            "nonfinite_count": 0,
        }
        surface = {
            "performance": performance,
            **{
                f"{prompt}_target_{target}_nll": distribution
                for prompt in ("rewrite", "rephrase")
                for target in ("new", "true")
            },
        }
        endpoint = _analysis_endpoint(surface)
        self.assertEqual(endpoint["rewrite"]["target_new"]["count"], 2)
        self.assertEqual(endpoint["rephrase"]["strict_success"]["denominator"], 2)
        self.assertIs(endpoint["performance"], performance)


if __name__ == "__main__":
    unittest.main()
