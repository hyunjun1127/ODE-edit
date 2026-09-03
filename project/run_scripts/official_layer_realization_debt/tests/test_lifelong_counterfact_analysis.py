from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.official_layer_realization_debt.lifelong_counterfact_analysis import (
    _store_summary,
    _transition_rows,
)
from project.run_scripts.official_layer_realization_debt.lifelong_counterfact_figures import (
    generate,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LifelongCounterFactAnalysisTests(unittest.TestCase):
    def test_primary_store_keeps_prompt_denominator_and_ties_fail(self) -> None:
        store = {
            "rs": {
                "bits": [1, 0], "new": [0.1, 0.2], "true": [0.2, 0.2],
                "advantage": [0.1, 0.0], "order": [["a", 0], ["b", 0]],
                "pairs": [[0.1, 0.2], [0.2, 0.2]],
            },
            "ps": {
                "bits": [1, 1], "new": [0.1, 0.2], "true": [0.3, 0.4],
                "advantage": [0.2, 0.2], "order": [["a", 0], ["a", 1]],
                "pairs": [[0.1, 0.3], [0.2, 0.4]],
            },
            "ns": {
                "bits": [1, 0], "new": [0.3, 0.1], "true": [0.2, 0.2],
                "advantage": [0.1, -0.1], "order": [["a", 0], ["a", 1]],
                "pairs": [[0.3, 0.2], [0.1, 0.2]],
            },
        }
        core, rows = _store_summary(store)
        self.assertEqual(core["rs_numerator"], 1)
        self.assertEqual(core["rs_denominator"], 2)
        self.assertEqual(core["ps"], 1.0)
        self.assertEqual(core["ns"], 0.5)
        self.assertEqual(rows[0]["tie_count"], 1)

    def test_transition_rows_include_all_primary_and_secondary_fields(self) -> None:
        rows = []
        for arm in ("LM", "LA", "QM", "QA"):
            for index, count in enumerate((1000, 1500, 2000, 3000, 5000, 7500, 10000)):
                rows.append(
                    {
                        "arm": arm,
                        "accepted_edit_count": count,
                        "rs": 0.9 - index * 0.1,
                        "ps": 0.8 - index * 0.1,
                        "ns": 0.7 - index * 0.1,
                        "rewrite_acc": 0.6,
                        "rephrase_acc": 0.5,
                        "rephrase_acc_strict_all_2": 0.4,
                        "neighborhood_target_true_teacher_forced_acc": 0.3,
                    }
                )
        self.assertEqual(len(_transition_rows(rows)), 196)

    def test_deterministic_plot_contract_and_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            final = []
            age = []
            distributions = []
            associations = []
            for arm_index, arm in enumerate(("LM", "LA", "QM", "QA")):
                for count in (1000, 1500, 2000, 3000, 5000, 7500, 10000):
                    row = {
                        "arm": arm,
                        "accepted_edit_count": count,
                        "rs": 0.8 - arm_index * 0.05,
                        "ps": 0.7 - arm_index * 0.05,
                        "ns": 0.9 - arm_index * 0.05,
                        "rewrite_acc": 0.6,
                        "rephrase_acc": 0.5,
                        "rephrase_acc_strict_all_2": 0.4,
                        "neighborhood_target_true_teacher_forced_acc": 0.3,
                    }
                    checkpoints.append(row)
                    if count == 10000:
                        final.append(row)
                    for stratum in ("EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT"):
                        age.append({**row, "age_stratum": stratum})
                    for metric in ("RS", "PS", "NS"):
                        distributions.append(
                            {
                                "arm": arm,
                                "accepted_edit_count": count,
                                "metric": metric,
                                "nll_advantage_median": 0.2,
                                "nll_advantage_p90": 0.4,
                            }
                        )
                for mechanism in ("q_pre_L8_median", "D_TV_median"):
                    for outcome in ("rs", "ps", "ns"):
                        associations.append(
                            {"arm": arm, "mechanism_metric": mechanism, "outcome_metric": outcome, "spearman": 0.1}
                        )
            _write(root / "counterfact-primary-checkpoints.csv", checkpoints)
            _write(root / "counterfact-finalw-full10k.csv", final)
            _write(root / "counterfact-primary-age-strata.csv", age)
            _write(root / "counterfact-primary-distributions.csv", distributions)
            _write(root / "counterfact-mechanism-associations.csv", associations)
            outputs = generate(root, root)
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))
            source = Path(__file__).resolve().parents[1] / "lifelong_counterfact_figures.py"
            text = source.read_text(encoding="utf-8").lower()
            self.assertNotIn("equal-share", text)
            self.assertNotIn("bars:", text)
            self.assertNotIn("imagegen", text)


if __name__ == "__main__":
    unittest.main()
