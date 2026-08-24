from __future__ import annotations

import csv
import io
import unittest

from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll_analysis import (
    METHOD_ALPHA,
    METHOD_FZ,
    METHOD_MEMIT,
    summarize,
    update_headline_csv,
    update_report_markdown,
)


def _summary(method: str, offset: float) -> dict[str, object]:
    return {
        "method": method,
        "metrics": {
            prompt: {
                target: {
                    "denominator": 1000,
                    "mean": offset + 1.0,
                    "median": offset + 2.0,
                    "p90_nearest_rank": offset + 3.0,
                    "max": offset + 4.0,
                }
                for target in ("target_new", "target_true")
            }
            for prompt in ("rewrite", "rephrase")
        },
    }


class NativeSequentialWNLLAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = {
            method: _summary(method, float(index))
            for index, method in enumerate((METHOD_ALPHA, METHOD_MEMIT, METHOD_FZ))
        }

    def test_nearest_rank_summary(self) -> None:
        value = summarize([float(index) for index in range(1, 11)])
        self.assertEqual(value["denominator"], 10)
        self.assertEqual(value["median"], 5.5)
        self.assertEqual(value["p90_nearest_rank"], 9.0)
        self.assertEqual(value["max"], 10.0)

    def test_headline_backfill_changes_only_three_W_columns_groups(self) -> None:
        header = [
            "method", "W_rewrite_mean", "W_rewrite_median",
            "W_rewrite_p90_nearest_rank", "W_rewrite_max",
            "W_rephrase_mean", "W_rephrase_median",
            "W_rephrase_p90_nearest_rank", "W_rephrase_max",
        ]
        source = io.StringIO()
        writer = csv.DictWriter(source, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for method in (*self.summaries, "FZ-RESET"):
            writer.writerow({"method": method})
        rows = list(csv.DictReader(io.StringIO(update_headline_csv(source.getvalue(), self.summaries))))
        by_method = {row["method"]: row for row in rows}
        self.assertEqual(by_method[METHOD_ALPHA]["W_rewrite_mean"], "1.0")
        self.assertEqual(by_method[METHOD_FZ]["W_rephrase_max"], "6.0")
        self.assertEqual(by_method["FZ-RESET"]["W_rewrite_mean"], "")

    def test_markdown_replaces_all_NR_rows_and_adds_true_NLL_detail(self) -> None:
        rows = "\n".join(
            f"| {method} | 1 | 1 | 1 | 1 | z | NR/NR/NR/NR | z | NR/NR/NR/NR |"
            for method in self.summaries
        )
        source = f"# R\n\n{rows}\n\n## Rewrite/Rephrase 세부\n"
        output = update_report_markdown(source, self.summaries)
        self.assertNotIn("NR/NR/NR/NR", output)
        self.assertIn("Baseline final-W10 NLL backfill", output)
        self.assertIn("target-true mean/median/p90/max", output)
        self.assertIn("requests |", output)

    def test_integrated_update_does_not_replace_independent_memit_row(self) -> None:
        source = (
            "## Independent 최종 표\n"
            f"| {METHOD_MEMIT} | 1 | 1 | 1 | 1 | z | KEEP | z | KEEP |\n\n"
            "## Sequential 최종 표\n"
            + "\n".join(
                f"| {method} | 1 | 1 | 1 | 1 | z | NR/NR/NR/NR | z | NR/NR/NR/NR |"
                for method in self.summaries
            )
            + "\n\n## Sequential−Independent descriptive delta\n"
        )
        output = update_report_markdown(
            source, self.summaries, start_heading="## Sequential 최종 표"
        )
        self.assertIn(f"| {METHOD_MEMIT} | 1 | 1 | 1 | 1 | z | KEEP | z | KEEP |", output)
        self.assertEqual(output.count("NR/NR/NR/NR"), 0)


if __name__ == "__main__":
    unittest.main()
