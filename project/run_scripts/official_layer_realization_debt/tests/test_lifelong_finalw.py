from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from project.run_scripts.official_layer_realization_debt.lifelong_finalw_contracts import (
    ABSENT_EXACT_CHECKPOINTS,
    AMENDED_CHECKPOINTS,
    EvaluationLock,
    ORIGINAL_REQUESTED_CHECKPOINTS,
)
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_evaluation import age_stratum
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_runtime import (
    SummaryAccumulator,
    _gzip_jsonl_create,
)


def _metric(prompt_count: int, *, strict: int, nll: float) -> dict:
    prompts = [
        {
            "nll": nll + index,
            "margin": 1.0 - index,
            "strict": index < strict,
            "token_accuracy": 1.0 if index < strict else 0.0,
            "token_correct_count": 1 if index < strict else 0,
            "target_token_count": 1,
        }
        for index in range(prompt_count)
    ]
    return {
        "prompt_count": prompt_count,
        "strict_count": strict,
        "token_correct_count": strict,
        "target_token_count": prompt_count,
        "prompts": prompts,
    }


def _record(ordinal: int) -> dict:
    return {
        "ordinal": ordinal,
        "metrics": {
            "rewrite_target_new": _metric(1, strict=1, nll=0.1),
            "rewrite_target_true": _metric(1, strict=0, nll=1.1),
            "rephrase_target_new": _metric(2, strict=2, nll=0.2),
            "rephrase_target_true": _metric(2, strict=0, nll=1.2),
            "locality_target_true": _metric(10, strict=9, nll=0.3),
        },
    }


class FinalWeightBackfillTests(unittest.TestCase):
    def test_outcome_blind_schedule_is_exact(self) -> None:
        self.assertEqual(AMENDED_CHECKPOINTS, (1000, 1500, 2000, 3000, 5000, 7500, 10000))
        self.assertEqual(ABSENT_EXACT_CHECKPOINTS, (100, 500, 4000, 6000, 8000))
        self.assertEqual(set(ORIGINAL_REQUESTED_CHECKPOINTS) - set(AMENDED_CHECKPOINTS), set(ABSENT_EXACT_CHECKPOINTS))
        lock = EvaluationLock()
        self.assertEqual(lock.edit_replay_count, 0)
        self.assertEqual(lock.imputation_count, 0)

    def test_age_strata_are_fixed_twenty_sixty_twenty(self) -> None:
        self.assertEqual(age_stratum(0, 10), "EARLY_FIRST_20PCT")
        self.assertEqual(age_stratum(1, 10), "EARLY_FIRST_20PCT")
        self.assertEqual(age_stratum(2, 10), "MIDDLE_60PCT")
        self.assertEqual(age_stratum(7, 10), "MIDDLE_60PCT")
        self.assertEqual(age_stratum(8, 10), "RECENT_LAST_20PCT")
        self.assertEqual(age_stratum(9, 10), "RECENT_LAST_20PCT")

    def test_summary_uses_request_prompt_and_token_denominators(self) -> None:
        value = SummaryAccumulator(10)
        for ordinal in range(10):
            value.add(_record(ordinal))
        payload = value.payload()
        self.assertEqual(payload["request_denominator"], 10)
        self.assertEqual(payload["rephrase_prompt_denominator"], 20)
        self.assertEqual(payload["locality_prompt_denominator"], 100)
        self.assertEqual(payload["eff"], 1.0)
        self.assertEqual(payload["gen_prompt"], 1.0)
        self.assertEqual(payload["gen_strict"], 1.0)
        self.assertEqual(payload["loc"], 0.9)
        self.assertEqual(
            sorted(group["request_denominator"] for group in payload["age_strata"].values()),
            [2, 2, 6],
        )

    def test_gzip_artifact_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl.gz"
            second = Path(directory) / "second.jsonl.gz"
            records = [{"b": 2, "a": 1}, {"a": 3}]
            self.assertEqual(_gzip_jsonl_create(first, records), _gzip_jsonl_create(second, records))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_launcher_has_one_safe_explicit_memory_request(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[2]
            / "session06_official_layer_debt_lifelong_finalw_full10k_server4.sbatch"
        )
        text = launcher.read_text(encoding="utf-8")
        self.assertEqual(text.count("#SBATCH --mem="), 1)
        self.assertIn("#SBATCH --mem=60416M", text)
        self.assertIn("#SBATCH --array=0-3%2", text)


if __name__ == "__main__":
    unittest.main()
