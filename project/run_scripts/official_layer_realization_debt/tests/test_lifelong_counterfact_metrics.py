from __future__ import annotations

from pathlib import Path
import unittest

from project.run_scripts.official_layer_realization_debt.lifelong_counterfact_contracts import (
    CounterFactMetricLock,
)
from project.run_scripts.official_layer_realization_debt.lifelong_counterfact_metrics import (
    paired_prompt_metrics,
    strict_nll_pair_success,
)


class LifelongCounterFactMetricTests(unittest.TestCase):
    def test_strict_inequality_direction_and_tie(self) -> None:
        self.assertEqual(strict_nll_pair_success(0.2, 0.7, locality=False), 1)
        self.assertEqual(strict_nll_pair_success(0.7, 0.2, locality=False), 0)
        self.assertEqual(strict_nll_pair_success(0.7, 0.2, locality=True), 1)
        self.assertEqual(strict_nll_pair_success(0.2, 0.7, locality=True), 0)
        self.assertEqual(strict_nll_pair_success(0.2, 0.2, locality=False), 0)
        self.assertEqual(strict_nll_pair_success(0.2, 0.2, locality=True), 0)

    def test_prompt_pair_aggregation_is_prompt_level(self) -> None:
        new = [{"nll": 0.1}, {"nll": 3.0}]
        true = [{"nll": 0.2}, {"nll": 2.0}]
        value = paired_prompt_metrics(new, true, locality=False)
        self.assertEqual(value["numerator"], 1)
        self.assertEqual(value["denominator"], 2)
        self.assertEqual(value["rate"], 0.5)
        self.assertEqual(value["tie_count"], 0)

    def test_locality_pair_aggregation_reverses_preference(self) -> None:
        new = [{"nll": 3.0}, {"nll": 0.1}]
        true = [{"nll": 2.0}, {"nll": 0.2}]
        value = paired_prompt_metrics(new, true, locality=True)
        self.assertEqual(value["numerator"], 1)
        self.assertEqual(value["denominator"], 2)
        self.assertEqual(value["rate"], 0.5)

    def test_metric_lock_requires_only_missing_locality_new(self) -> None:
        lock = CounterFactMetricLock()
        self.assertEqual(lock.backfill_categories, ("locality_target_new",))
        self.assertEqual(lock.rewrite_prompts_per_request, 1)
        self.assertEqual(lock.rephrase_prompts_per_request, 2)
        self.assertEqual(lock.locality_prompts_per_request, 10)
        self.assertEqual(lock.edit_replay_count, 0)

    def test_launcher_has_cap_two_and_one_safe_memory_request(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[2]
            / "session06_official_layer_debt_lifelong_counterfact_ns_backfill_server4.sbatch"
        )
        text = launcher.read_text(encoding="utf-8")
        self.assertEqual(text.count("#SBATCH --mem="), 1)
        self.assertIn("#SBATCH --mem=60416M", text)
        self.assertIn("#SBATCH --array=0-3%2", text)
        self.assertIn("#SBATCH --nodelist=server4", text)


if __name__ == "__main__":
    unittest.main()
