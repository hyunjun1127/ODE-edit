from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import unittest

from project.run_scripts.official_layer_realization_debt.lifelong_v5_analysis import (
    AGE_ORDER,
    ARM_ORDER,
    CATEGORIES,
    _average_ranks,
    _category_transition_rows,
    _core_transition_rows,
    _recent_minus_early,
    _v3_section_body,
)
from project.run_scripts.official_layer_realization_debt.lifelong_v5_figures import (
    CATEGORY_ORDER,
    generate,
    sha256_file,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class LifelongV5AnalysisTests(unittest.TestCase):
    def test_transition_denominators_and_total_span(self) -> None:
        core = []
        categories = []
        for arm_index, arm in enumerate(ARM_ORDER):
            for count in (1000, 2000):
                core.append(
                    {
                        "arm": arm,
                        "accepted_edit_count": count,
                        "request_denominator": count,
                        "eff_numerator": count // 2,
                        "eff": 0.5,
                        "gen_prompt_numerator": count,
                        "gen_prompt_denominator": 2 * count,
                        "gen_prompt": 0.5,
                        "gen_strict_numerator": count // 4,
                        "gen_strict_denominator": count,
                        "gen_strict": 0.25,
                        "loc_numerator": count,
                        "loc_denominator": 10 * count,
                        "loc": 0.1,
                        "rewrite_new_preferred_numerator": count // 2,
                        "rewrite_new_preferred_rate": 0.5,
                        "rephrase_new_preferred_numerator": count // 2,
                        "rephrase_new_preferred_rate": 0.5,
                    }
                )
                for category in CATEGORIES:
                    row = {
                        "arm": arm,
                        "accepted_edit_count": count,
                        "category": category,
                        "prompt_denominator": count,
                        "strict_rate": 0.5,
                    }
                    for family in (
                        "request_cluster_nll",
                        "request_cluster_min_margin",
                        "prompt_nll",
                        "prompt_margin",
                    ):
                        for statistic in ("mean", "median", "p90", "max"):
                            row[f"{family}_{statistic}"] = arm_index + count / 1000
                    categories.append(row)
        core_delta = _core_transition_rows(core)
        category_delta = _category_transition_rows(categories)
        self.assertEqual(len(core_delta), 4 * 2 * 6)
        self.assertEqual(len(category_delta), 4 * 5 * 2)
        self.assertEqual(
            {row["span"] for row in core_delta}, {"ADJACENT", "TOTAL_1K_TO_10K"}
        )

    def test_relative_age_delta_keeps_checkpoint_denominators(self) -> None:
        rows = []
        for arm in ARM_ORDER:
            for count in (1000,):
                for index, age in enumerate(AGE_ORDER):
                    row = {
                        "arm": arm,
                        "accepted_edit_count": count,
                        "age_stratum": age,
                        "request_denominator": (200, 600, 200)[index],
                        "eff": 0.1 * index,
                        "gen_prompt": 0.1 * index,
                        "gen_strict": 0.1 * index,
                        "loc": 0.1 * index,
                    }
                    for category in CATEGORIES:
                        for quantity in ("nll", "margin"):
                            for statistic in ("mean", "median", "p90", "max"):
                                row[f"{category}_{quantity}_{statistic}"] = float(index)
                    rows.append(row)
        delta = _recent_minus_early(rows)
        self.assertEqual(len(delta), 4)
        self.assertEqual(delta[0]["early_request_denominator"], 200)
        self.assertEqual(delta[0]["recent_request_denominator"], 200)
        self.assertAlmostEqual(delta[0]["delta_recent_minus_early_eff"], 0.2)

    def test_average_ranks_are_tie_aware(self) -> None:
        self.assertEqual(_average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_v3_sections_are_main_body_ready_and_repoint_figures(self) -> None:
        repository = Path(__file__).resolve().parents[4]
        v3 = repository / (
            "experiment-reports/servers/server4/"
            "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3"
        )
        text = _v3_section_body(v3, 3)
        self.assertIn("### 3.1 10k sentinel q 분포", text)
        self.assertNotIn("immutable appendix", text.lower())
        self.assertIn("../official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3/", text)

    def test_all_v5_plots_are_byte_stable_and_policy_clean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, first, second = root / "tables", root / "first", root / "second"
            tables.mkdir()
            first.mkdir()
            second.mkdir()
            final_rows = []
            core_rows = []
            category_rows = []
            age_rows = []
            association_rows = []
            for arm_index, arm in enumerate(ARM_ORDER):
                final_rows.append(
                    {
                        "arm": arm,
                        "eff": 0.1 + arm_index / 10,
                        "gen_prompt": 0.2 + arm_index / 10,
                        "gen_strict": 0.15 + arm_index / 10,
                        "loc": 0.05 + arm_index / 100,
                    }
                )
                for count in (1000, 2000):
                    core_rows.append(
                        {
                            "arm": arm,
                            "accepted_edit_count": count,
                            "eff": 0.2,
                            "gen_prompt": 0.15,
                            "gen_strict": 0.1,
                            "loc": 0.05,
                            "rewrite_new_preferred_rate": 0.7,
                            "rephrase_new_preferred_rate": 0.6,
                        }
                    )
                    for category in CATEGORY_ORDER:
                        category_rows.append(
                            {
                                "arm": arm,
                                "accepted_edit_count": count,
                                "category": category,
                                "request_cluster_nll_median": 1.0 + arm_index,
                                "request_cluster_nll_p90": 2.0 + arm_index,
                                "request_cluster_min_margin_median": -1.0 + arm_index,
                                "request_cluster_min_margin_p90": 1.0 + arm_index,
                            }
                        )
                    for age in AGE_ORDER:
                        age_rows.append(
                            {
                                "arm": arm,
                                "accepted_edit_count": count,
                                "age_stratum": age,
                                "eff": 0.2,
                                "gen_prompt": 0.15,
                                "gen_strict": 0.1,
                                "loc": 0.05,
                            }
                        )
                for mechanism in (
                    "q_pre_L8_median", "q_post_L8_median", "mean_rho_median",
                    "mean_tau_median", "d_parallel_median", "d_perp_median",
                    "D_TV_median", "update_magnitude_total", "layer8_update_share",
                ):
                    for outcome in ("eff", "gen_prompt", "gen_strict", "loc", "rewrite_new_nll_median"):
                        association_rows.append(
                            {
                                "arm": arm,
                                "mechanism_metric": mechanism,
                                "outcome_metric": outcome,
                                "spearman": 0.1 * (arm_index - 1),
                            }
                        )
            _write(tables / "finalw-full10k-arm-summary.csv", final_rows)
            _write(tables / "cumulative-core-rates.csv", core_rows)
            _write(tables / "cumulative-category-distributions.csv", category_rows)
            _write(tables / "cumulative-age-strata-full.csv", age_rows)
            _write(tables / "mechanism-cumulative-associations.csv", association_rows)
            first_paths = generate(tables, first)
            second_paths = generate(tables, second)
            self.assertEqual(len(first_paths), 7)
            self.assertEqual(
                [sha256_file(path) for path in first_paths],
                [sha256_file(path) for path in second_paths],
            )
            source = Path(generate.__code__.co_filename).read_text(encoding="utf-8")
            self.assertNotIn("bars:", source.lower())
            self.assertNotIn("equal-share", source.lower())
            self.assertIn("Final W₁₀₀₀₀ on All 10,000 Seen Requests", source)


if __name__ == "__main__":
    unittest.main()
