from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_analysis import (
    _recompute,
)
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_figures import (
    ARM_ORDER,
    generate,
    sha256_file,
)
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_contracts import (
    INSTRUCTION_ID,
)
from project.run_scripts.official_layer_realization_debt.lifelong_finalw_package_verify import (
    verify,
)


def _category(prompt_count: int, ordinal: int) -> dict:
    prompts = [
        {
            "nll": 0.1 + 0.01 * ordinal + 0.02 * prompt,
            "margin": 1.0 - 0.01 * ordinal - 0.02 * prompt,
            "strict": True,
            "target_token_count": 1,
            "token_accuracy": "NOT_RECORDED_EVALUATOR_SCHEMA",
            "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
        }
        for prompt in range(prompt_count)
    ]
    return {
        "prompt_count": prompt_count,
        "strict_count": prompt_count,
        "target_token_count": prompt_count,
        "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
        "prompts": prompts,
    }


def _record(ordinal: int) -> dict:
    value = {
        "ordinal": ordinal,
        "batch_index": ordinal // 100 + 1,
        "case_identity_sha256": f"case-{ordinal}",
        "request_sha256": f"request-{ordinal}",
        "age_stratum": (
            "EARLY_FIRST_20PCT" if ordinal < 2
            else "RECENT_LAST_20PCT" if ordinal >= 8
            else "MIDDLE_60PCT"
        ),
        "metrics": {
            "rewrite_target_new": _category(1, ordinal),
            "rewrite_target_true": _category(1, ordinal + 100),
            "rephrase_target_new": _category(2, ordinal),
            "rephrase_target_true": _category(2, ordinal + 100),
            "locality_target_true": _category(10, ordinal),
        },
        "raw_prompt_logit_generation_publish_count": 0,
    }
    value["identity_sha256"] = canonical_hash(value)
    return value


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class FinalWeightAnalysisTests(unittest.TestCase):
    def test_recompute_keeps_exact_aggregation_units_and_age_metrics(self) -> None:
        value = _recompute([_record(index) for index in range(10)])
        self.assertEqual(value["request_denominator"], 10)
        self.assertEqual(value["gen_prompt_denominator"], 20)
        self.assertEqual(value["loc_denominator"], 100)
        self.assertEqual(value["gen_strict_numerator"], 10)
        self.assertEqual(
            [value["age"][key]["request_denominator"] for key in (
                "EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT"
            )],
            [2, 6, 2],
        )
        self.assertIn(
            "rephrase_target_true_margin_p90",
            value["age"]["MIDDLE_60PCT"],
        )

    def test_plots_are_complete_ordered_and_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = root / "tables"
            first = root / "first"
            second = root / "second"
            tables.mkdir()
            first.mkdir()
            second.mkdir()
            final_rows = []
            cumulative_rows = []
            age_rows = []
            for arm_index, arm in enumerate(ARM_ORDER):
                final_rows.append(
                    {
                        "arm": arm,
                        "eff": 0.1 + 0.1 * arm_index,
                        "gen_prompt": 0.2 + 0.1 * arm_index,
                        "gen_strict": 0.15 + 0.1 * arm_index,
                        "loc": 0.8 - 0.1 * arm_index,
                        "rewrite_target_new_nll_median": 1.0 + arm_index,
                        "rewrite_target_true_nll_median": 2.0 + arm_index,
                        "rephrase_target_new_nll_median": 1.5 + arm_index,
                        "rephrase_target_true_nll_median": 2.5 + arm_index,
                    }
                )
                for checkpoint in (1000, 1500, 2000, 3000, 5000, 7500, 10000):
                    cumulative_rows.append(
                        {
                            "arm": arm,
                            "accepted_edit_count": checkpoint,
                            "eff": 0.1 + 0.1 * arm_index,
                            "gen_prompt": 0.2 + 0.1 * arm_index,
                            "gen_strict": 0.15 + 0.1 * arm_index,
                            "loc": 0.8 - 0.1 * arm_index,
                        }
                    )
                for stratum in (
                    "EARLY_FIRST_20PCT", "MIDDLE_60PCT", "RECENT_LAST_20PCT"
                ):
                    age_rows.append(
                        {
                            "arm": arm,
                            "accepted_edit_count": 10000,
                            "age_stratum": stratum,
                            "eff": 0.1 + 0.1 * arm_index,
                            "gen_prompt": 0.2 + 0.1 * arm_index,
                            "gen_strict": 0.15 + 0.1 * arm_index,
                            "loc": 0.8 - 0.1 * arm_index,
                        }
                    )
            _write(tables / "finalw-full10k-arm-summary.csv", final_rows)
            _write(tables / "cumulative-seen-prefix-arm-summary.csv", cumulative_rows)
            _write(tables / "edit-age-strata-summary.csv", age_rows)
            first_paths = generate(tables, first)
            second_paths = generate(tables, second)
            self.assertEqual(len(first_paths), 4)
            self.assertEqual([path.name for path in first_paths], [path.name for path in second_paths])
            self.assertTrue(all(path.stat().st_size > 0 for path in first_paths))
            self.assertEqual(
                [sha256_file(path) for path in first_paths],
                [sha256_file(path) for path in second_paths],
            )
            source = Path(generate.__code__.co_filename).read_text(encoding="utf-8")
            self.assertIn("Final W₁₀₀₀₀ on All 10,000 Seen Requests", source)
            self.assertNotIn("bars:", source.lower())
            self.assertNotIn("interpol", source.lower())

    def test_package_verifier_rehashes_package_and_external_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            root.mkdir()
            external = Path(directory) / "external.json"
            external.write_text("{}\n", encoding="utf-8")
            external_sha = hashlib.sha256(external.read_bytes()).hexdigest()
            raw_members = [
                {
                    "kind": "FIXTURE",
                    "arm": "GLOBAL",
                    "accepted_edit_count": "LOCK",
                    "path": str(external),
                    "bytes": external.stat().st_size,
                    "sha256": external_sha,
                }
            ]
            inventory = root / "evaluation-raw-member-inventory.json"
            inventory.write_text(
                json.dumps(raw_members, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            report = root / "report.md"
            report.write_text("fixture\n", encoding="utf-8")
            members = [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted((inventory, report), key=lambda value: value.name)
            ]
            manifest = {
                "instruction_id": INSTRUCTION_ID,
                "members": members,
                "member_root": canonical_hash(members),
                "raw_member_root": canonical_hash(raw_members),
            }
            manifest["identity_sha256"] = canonical_hash(manifest)
            manifest_path = root / "analysis-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            receipt = {
                "instruction_id": INSTRUCTION_ID,
                "manifest": {
                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "identity_sha256": manifest["identity_sha256"],
                },
                "report": {
                    "path": str(report),
                    "bytes": report.stat().st_size,
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                },
                "member_root": manifest["member_root"],
                "external_raw_member_root": manifest["raw_member_root"],
            }
            receipt["identity_sha256"] = canonical_hash(receipt)
            (root / "rooted-analysis-receipt.json").write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            result = verify(root)
            self.assertEqual(result["status"], "FULL_REHASH_PASS")
            self.assertEqual(result["package_member_count"], 2)
            self.assertEqual(result["external_raw_member_count"], 1)


if __name__ == "__main__":
    unittest.main()
