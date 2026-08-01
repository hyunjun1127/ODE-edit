from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "direct_z_possibility_analysis.py"
_SPEC = importlib.util.spec_from_file_location("direct_z_possibility_analysis", _ANALYZER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load direct-z possibility analyzer")
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


class DirectZPossibilityAnalysisTests(unittest.TestCase):
    def _arm_metrics(self, arm: str, case_index: int) -> dict[str, object]:
        offset = case_index * 0.01
        values: dict[str, dict[str, object]] = {
            analysis.NO_OP: {
                "z_residual_ratio": 1.0 + offset,
                "generated_delta_error_mean": 1.0 + offset,
                "generated_delta_error_worst": 1.2 + offset,
                "delta_gain": 0.0,
                "delta_cosine": 0.0,
                "off_token_spill_ratio": 0.0,
                "output_progress": 0.0,
                "output_nll_reduction": 0.0,
                "exact_margin_min": -1.0,
                "exact_satisfied": False,
                "paraphrase_nll_reduction": 0.0,
                "heldout_kl": 0.0,
                "endpoint_c_energy": 0.0,
                "endpoint_frobenius_norm": 0.0,
                "nfe": 1,
            },
            analysis.ORACLE_DO_Z: {
                "z_residual_ratio": 0.0,
                "generated_delta_error_mean": 0.0,
                "generated_delta_error_worst": 0.0,
                "delta_gain": 1.0,
                "delta_cosine": 1.0,
                "off_token_spill_ratio": 0.0,
                "output_progress": 2.0 + offset,
                "output_nll_reduction": 2.0 + offset,
                "exact_margin_min": 1.0 + offset,
                "exact_satisfied": True,
                "paraphrase_nll_reduction": 1.8 + offset,
                "heldout_kl": 0.0,
                "endpoint_c_energy": 0.0,
                "endpoint_frobenius_norm": 0.0,
                "nfe": 1,
            },
            analysis.NATIVE_FULL: {
                "z_residual_ratio": 0.20 + offset,
                "generated_delta_error_mean": 0.22 + offset,
                "generated_delta_error_worst": 0.32 + offset,
                "delta_gain": 0.80 + offset,
                "delta_cosine": 0.84,
                "off_token_spill_ratio": 0.20,
                "output_progress": 1.30 + offset,
                "output_nll_reduction": 1.30 + offset,
                "exact_margin_min": 0.30 + offset,
                "exact_satisfied": True,
                "paraphrase_nll_reduction": 1.00 + offset,
                "heldout_kl": 0.40 + offset,
                "endpoint_c_energy": 4.0 + offset,
                "endpoint_frobenius_norm": 2.0 + offset,
                "nfe": 1,
            },
            analysis.NATIVE_ALPHA: {
                "z_residual_ratio": 0.45 + offset,
                "generated_delta_error_mean": 0.50 + offset,
                "generated_delta_error_worst": 0.65 + offset,
                "delta_gain": 0.55 + offset,
                "delta_cosine": 0.70,
                "off_token_spill_ratio": 0.12,
                "output_progress": 0.90 + offset,
                "output_nll_reduction": 0.90 + offset,
                "exact_margin_min": 0.10 + offset,
                "exact_satisfied": True,
                "paraphrase_nll_reduction": 0.70 + offset,
                "heldout_kl": 0.15 + offset,
                "endpoint_c_energy": 1.0,
                "endpoint_frobenius_norm": 1.0 + offset,
                "nfe": 1,
            },
            analysis.BF_CURRENT: {
                "z_residual_ratio": 0.30 + offset,
                "generated_delta_error_mean": 0.35 + offset,
                "generated_delta_error_worst": 0.48 + offset,
                "delta_gain": 0.68 + offset,
                "delta_cosine": 0.78,
                "off_token_spill_ratio": 0.10,
                "output_progress": 1.10 + offset,
                "output_nll_reduction": 1.10 + offset,
                "exact_margin_min": 0.20 + offset,
                "exact_satisfied": True,
                "paraphrase_nll_reduction": 0.90 + offset,
                "heldout_kl": 0.10 + offset,
                "endpoint_c_energy": 1.0,
                "endpoint_frobenius_norm": 0.90 + offset,
                "nfe": 12,
            },
            analysis.SYNC_CONE: {
                "z_residual_ratio": 0.10 + offset,
                "generated_delta_error_mean": 0.12 + offset,
                "generated_delta_error_worst": 0.20 + offset,
                "delta_gain": 0.90 + offset,
                "delta_cosine": 0.92,
                "off_token_spill_ratio": 0.06,
                "output_progress": 1.50 + offset,
                "output_nll_reduction": 1.50 + offset,
                "exact_margin_min": 0.50 + offset,
                "exact_satisfied": True,
                "paraphrase_nll_reduction": 1.20 + offset,
                "heldout_kl": 0.20 + offset,
                "endpoint_c_energy": 0.80 + offset,
                "endpoint_frobenius_norm": 0.80 + offset,
                "nfe": 20,
            },
        }
        result = dict(values[arm])
        result["preservation_score"] = -float(result["heldout_kl"])
        return result

    def _write_run(
        self,
        root: Path,
        *,
        failed_cases: set[int] | None = None,
        mutate_payload=None,
        rollback_failure: tuple[int, str] | None = None,
    ) -> Path:
        failed_cases = failed_cases or set()
        run_id = "direct-z-possibility-test-v1"
        manifest = {
            "schema_version": analysis.MANIFEST_SCHEMA,
            "run_id": run_id,
            "model_alias": "llama3-8b-inst",
            "case_count": analysis.EXPECTED_CASE_COUNT,
            "arm_order": list(analysis.ARM_ORDER),
            "selection_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED,
            "bootstrap_resamples": analysis.BOOTSTRAP_RESAMPLES,
            "claim_boundary": analysis.CLAIM_BOUNDARY,
        }
        pass_count = analysis.EXPECTED_CASE_COUNT - len(failed_cases)
        summary = {
            "schema_version": analysis.SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": "llama3-8b-inst",
            "planned_case_count": analysis.EXPECTED_CASE_COUNT,
            "attempted_case_count": analysis.EXPECTED_CASE_COUNT,
            "pass_case_count": pass_count,
            "failed_case_count": len(failed_cases),
            "outcome_count": analysis.EXPECTED_OUTCOME_COUNT,
            "arm_order": list(analysis.ARM_ORDER),
            "selection_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "all_rollbacks_exact": rollback_failure is None,
            "firewall_pass": True,
            "receipt_before_outcome": True,
            "direct_z_once_per_case": True,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows = []
        sequence = 0
        for case_index in range(analysis.EXPECTED_CASE_COUNT):
            case_pass = case_index not in failed_cases
            for arm in analysis.ARM_ORDER:
                payload = {
                    "case_id": case_index,
                    "request_id": f"{case_index + 1:064x}",
                    "arm_id": arm,
                    "case_pass": case_pass,
                    "arm_success": case_pass,
                    "technical_pass": True,
                    "rollback_exact": not (
                        rollback_failure == (case_index, arm)
                    ),
                    "firewall_pass": True,
                    "receipt_before_outcome": True,
                    **self._arm_metrics(arm, case_index),
                }
                if mutate_payload is not None:
                    mutate_payload(payload, case_index, arm)
                rows.append(
                    {
                        "schema_version": analysis.STREAM_SCHEMA,
                        "run_id": run_id,
                        "sequence": sequence,
                        "recorded_at": "2026-08-01T00:00:00Z",
                        "event": analysis.STREAM_EVENT,
                        "payload": payload,
                    }
                )
                sequence += 1
        with (root / "outcomes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
        return root

    def test_valid_run_classifies_ceiling_cone_and_bf_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory))
            result = analysis.analyze_run(run)
        self.assertTrue(result["artifact_validation"]["valid"])
        self.assertEqual(
            result["classification"]["verdict"],
            "CONE_FEASIBLE_CURRENT_BF_GAP",
        )
        oracle = result["effects"]["oracle_ceiling"][
            "output_nll_gain_oracle_minus_noop"
        ]
        self.assertEqual(oracle["n_itd"], 8)
        self.assertEqual(oracle["positive_count"], 8)
        self.assertTrue(oracle["lenient_positive"])
        self.assertEqual(oracle["bootstrap_resamples"], 4000)
        self.assertTrue(result["compute"]["endpoint_c_native_alpha_bf_matched"])

    def test_failed_case_remains_in_itd_with_zero_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory), failed_cases={7})
            result = analysis.analyze_run(run)
        self.assertTrue(result["artifact_validation"]["valid"])
        self.assertEqual(result["artifact_validation"]["failed_case_count"], 1)
        effect = result["effects"]["cone_feasibility"][
            "output_nll_gain_cone_minus_noop"
        ]
        self.assertEqual(effect["n_itd"], 8)
        self.assertEqual(effect["positive_count"], 7)
        self.assertEqual(effect["zero_count"], 1)
        self.assertEqual(result["case_diagnostics"][7]["cone_output_gain"], 0.0)

    def test_technical_failure_blocks_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(
                Path(directory), rollback_failure=(0, analysis.BF_CURRENT)
            )
            result = analysis.analyze_run(run)
        self.assertFalse(result["artifact_validation"]["valid"])
        self.assertEqual(
            result["classification"]["verdict"], "BLOCK_TECHNICAL_INVALID"
        )

    def test_extra_unsafe_payload_field_is_rejected(self) -> None:
        def mutate(payload, case_index, arm):
            if case_index == 0 and arm == analysis.NO_OP:
                payload["raw_prompt"] = "secret"

        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory), mutate_payload=mutate)
            with self.assertRaises(analysis.DirectZPossibilityAnalysisError):
                analysis.analyze_run(run)

    def test_nonfinite_scalar_is_rejected(self) -> None:
        def mutate(payload, case_index, arm):
            if case_index == 0 and arm == analysis.BF_CURRENT:
                payload["output_nll_reduction"] = math.nan

        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory), mutate_payload=mutate)
            with self.assertRaises(analysis.DirectZPossibilityAnalysisError):
                analysis.analyze_run(run)

    def test_preservation_score_must_be_negative_heldout_kl(self) -> None:
        def mutate(payload, case_index, arm):
            if case_index == 0 and arm == analysis.BF_CURRENT:
                payload["preservation_score"] = 0.5

        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory), mutate_payload=mutate)
            with self.assertRaises(analysis.DirectZPossibilityAnalysisError):
                analysis.analyze_run(run)

    def test_analysis_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory))
            first = analysis.analyze_run(run)
            second = analysis.analyze_run(run)
        self.assertEqual(first, second)
        self.assertEqual(first["analysis_sha256"], second["analysis_sha256"])

    def test_cli_outputs_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            self._write_run(run)
            output_json = root / "analysis.json"
            output_md = root / "analysis.md"
            argv = [
                "--run-directory",
                str(run),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            self.assertEqual(analysis.main(argv), 0)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_md.is_file())
            self.assertEqual(analysis.main(argv), 2)


if __name__ == "__main__":
    unittest.main()
