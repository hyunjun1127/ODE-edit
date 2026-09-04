from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from project.run_scripts.ordered_response_barrier_ode.round0_baseline_report import (
    AnalysisBoundary,
    _legacy_detail,
    build_metric_correction,
    build_performance,
)


PACKAGE = Path(
    "/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-orbode-round0-analysis-v1/"
    "experiment-reports/servers/server1/"
    "ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1"
)


class Round0BaselineReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PACKAGE.is_dir():
            raise unittest.SkipTest("immutable v1 round0 package unavailable")
        cls.core = pd.read_csv(PACKAGE / "core-performance-summary.csv")
        cls.requests = pd.read_csv(PACKAGE / "request-endpoint-metrics.csv.gz")
        cls.prompt = pd.read_csv(PACKAGE / "prompt-nll.csv.gz", low_memory=False)
        cls.backfill = {
            "llama3-8b-inst": {"summary": {"ns_numerator": 893, "ns_denominator": 1000, "ns_rate": 0.893}},
            "qwen2.5-7b-inst": {"summary": {"ns_numerator": 849, "ns_denominator": 1000, "ns_rate": 0.849}},
        }

    def test_explicit_w0_and_official_baseline_rows(self) -> None:
        frame = build_performance(self.core, self.requests, self.backfill)
        self.assertEqual(len(frame), 22)
        self.assertEqual((frame.method_group == "W0_BASELINE").sum(), 2)
        self.assertEqual((frame.method_group == "OFFICIAL_BASELINE").sum(), 4)
        self.assertEqual((frame.method_group == "OURS").sum(), 16)
        self.assertEqual(set(frame[frame.method_group == "OFFICIAL_BASELINE"].method_label), {"Official MEMIT", "Official AlphaEdit"})

    def test_ns_and_prediction_preservation_are_not_conflated(self) -> None:
        frame = build_performance(self.core, self.requests, self.backfill)
        entry = frame[frame.arm == "PRE_EDIT"]
        endpoint = frame[frame.arm != "PRE_EDIT"]
        self.assertTrue((entry.canonical_ns_denominator == 1000).all())
        self.assertTrue(endpoint.canonical_ns_denominator.isna().all())
        self.assertTrue((endpoint.pp_prompt_denominator == 1000).all())
        self.assertTrue((endpoint.pp_token_denominator == 1010).all())

    def test_1010_is_exact_token_decomposition_not_prompt_ns(self) -> None:
        correction = build_metric_correction(self.prompt)
        rows = correction.set_index("metric")
        self.assertEqual(rows.loc["canonical_ns", "denominator_per_model_or_endpoint"], 1000)
        self.assertEqual(rows.loc["pp_prompt", "denominator_per_model_or_endpoint"], 1000)
        self.assertEqual(rows.loc["pp_token", "denominator_per_model_or_endpoint"], 1010)
        selected = self.prompt[
            (self.prompt.cell_id == 0)
            & (self.prompt.stage == "PRE_EDIT")
            & (self.prompt.kind == "locality_target_true")
        ]
        self.assertEqual(selected.target_token_count.value_counts().sort_index().to_dict(), {1: 990, 2: 10})

    def test_preedit_cross_writer_drift_fails_closed(self) -> None:
        broken = self.core.copy()
        index = broken[(broken.model == "llama3-8b-inst") & (broken.writer_family == "MEMIT") & (broken.stage == "PRE_EDIT")].index[0]
        broken.loc[index, "rewrite_success_count"] += 1
        with self.assertRaises(AnalysisBoundary):
            build_performance(broken, self.requests, self.backfill)

    def test_legacy_detail_renames_uppercase_ns_to_pp(self) -> None:
        text = (PACKAGE / "ordered-response-barrier-ode-round0-b100-exhaustive-factual-ko.md").read_text(encoding="utf-8")
        detail = _legacy_detail(text)
        self.assertNotIn("primary RS/PS/NS", detail)
        self.assertIn("target-new margin과 PP", detail)
        self.assertIn("RS/PS/token-PP", detail)
        self.assertIn("### 6.1 Rewrite NLL 분포", detail)


if __name__ == "__main__":
    unittest.main()
