"""Synthetic contract tests for the paired Alpha direct-z analyzer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_ANALYZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "direct_z_alpha_analysis.py"
)
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "direct_z_alpha_analysis_test_module", _ANALYZER_PATH
)
if _ANALYZER_SPEC is None or _ANALYZER_SPEC.loader is None:
    raise RuntimeError("unable to load direct-z Alpha analyzer for unit tests")
analyzer = importlib.util.module_from_spec(_ANALYZER_SPEC)
sys.modules[_ANALYZER_SPEC.name] = analyzer
_ANALYZER_SPEC.loader.exec_module(analyzer)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest(run_id: str) -> dict[str, object]:
    return {
        "schema_version": analyzer.MANIFEST_SCHEMA,
        "run_id": run_id,
        "model_alias": "llama3-8b-inst",
        "case_count": analyzer.EXPECTED_CASE_COUNT,
        "arm_order": list(analyzer.ARM_ORDER),
        "selection_sha256": _digest("selection"),
        "config_sha256": _digest("config"),
        "bootstrap_seed": analyzer.BOOTSTRAP_SEED,
        "bootstrap_resamples": analyzer.BOOTSTRAP_RESAMPLES,
        "paired_memit_run_id": analyzer.EXPECTED_MEMIT_RUN_BY_MODEL[
            "llama3-8b-inst"
        ],
        "paired_memit_manifest_sha256": analyzer.EXPECTED_MEMIT_MANIFEST_BY_MODEL[
            "llama3-8b-inst"
        ],
        "replay_lock_id": analyzer.EXPECTED_REPLAY_LOCK_ID,
        "claim_boundary": analyzer.CLAIM_BOUNDARY,
        "target_lineage_sha256": _digest("frozen-target-lineage"),
    }


def _arm_values(arm_id: str, case_index: int) -> dict[str, float | bool | int]:
    # The construction intentionally creates clean, per-axis positive effects
    # for the reference complete artifact.  Tiny case variation prevents tests
    # from relying on a degenerate bootstrap interval.
    jitter = case_index * 0.001
    values: dict[str, dict[str, float | bool | int]] = {
        analyzer.NO_OP: {
            "z": 1.0,
            "generated": 1.0,
            "gain": 0.0,
            "cosine": 0.0,
            "spill": 0.60,
            "progress": 0.0,
            "nll": 0.0,
            "margin": 0.0,
            "exact": False,
            "paraphrase": 0.0,
            "kl": 0.0,
            "c": 0.0,
            "fro": 0.0,
            "nfe": 1,
            "leak": 0.0,
        },
        analyzer.ORACLE_DO_Z: {
            "z": 0.0,
            "generated": 0.0,
            "gain": 1.0,
            "cosine": 1.0,
            "spill": 0.0,
            "progress": 10.0,
            "nll": 10.0,
            "margin": 10.0,
            "exact": True,
            "paraphrase": 2.0,
            "kl": 0.20,
            "c": 0.0,
            "fro": 0.0,
            "nfe": 1,
            "leak": 0.0,
        },
        analyzer.GENUINE_FULL: {
            "z": 0.25,
            "generated": 0.25,
            "gain": 0.75,
            "cosine": 0.75,
            "spill": 0.16,
            "progress": 8.0,
            "nll": 8.0,
            "margin": 8.0,
            "exact": True,
            "paraphrase": 1.4,
            "kl": 0.24,
            "c": 2.0,
            "fro": 3.0,
            "nfe": 5,
            "leak": 1e-7,
        },
        analyzer.GENUINE_C: {
            "z": 0.40,
            "generated": 0.40,
            "gain": 0.60,
            "cosine": 0.60,
            "spill": 0.22,
            "progress": 6.0,
            "nll": 6.0,
            "margin": 6.0,
            "exact": True,
            "paraphrase": 1.1,
            "kl": 0.18,
            "c": 1.0,
            "fro": 2.0,
            "nfe": 4,
            "leak": 1e-7,
        },
        analyzer.POSTHOC_FULL: {
            "z": 0.35,
            "generated": 0.35,
            "gain": 0.65,
            "cosine": 0.65,
            "spill": 0.20,
            "progress": 7.0,
            "nll": 7.0,
            "margin": 7.0,
            "exact": True,
            "paraphrase": 1.2,
            "kl": 0.28,
            "c": 2.3,
            "fro": 3.5,
            "nfe": 5,
            "leak": 1e-7,
        },
        analyzer.POSTHOC_C: {
            "z": 0.60,
            "generated": 0.60,
            "gain": 0.40,
            "cosine": 0.40,
            "spill": 0.30,
            "progress": 5.0,
            "nll": 5.0,
            "margin": 5.0,
            "exact": True,
            "paraphrase": 0.8,
            "kl": 0.28,
            "c": 1.0,
            "fro": 2.4,
            "nfe": 4,
            "leak": 1e-7,
        },
        analyzer.BF_GENUINE: {
            "z": 0.30,
            "generated": 0.30,
            "gain": 0.70,
            "cosine": 0.70,
            "spill": 0.18,
            "progress": 7.0,
            "nll": 7.0,
            "margin": 7.0,
            "exact": True,
            "paraphrase": 1.3,
            "kl": 0.16,
            "c": 1.0,
            "fro": 1.8,
            "nfe": 8,
            "leak": 1e-7,
        },
        analyzer.SYNC_CONE_GENUINE: {
            "z": 0.50,
            "generated": 0.50,
            "gain": 0.50,
            "cosine": 0.50,
            "spill": 0.25,
            "progress": 5.5,
            "nll": 5.5,
            "margin": 5.5,
            "exact": True,
            "paraphrase": 0.9,
            "kl": 0.22,
            "c": 1.0,
            "fro": 1.7,
            "nfe": 4,
            "leak": 1e-7,
        },
    }
    value = values[arm_id]
    return {
        "z_residual_ratio": float(value["z"]) + jitter,
        "generated_delta_error_mean": float(value["generated"]) + jitter,
        "generated_delta_error_worst": float(value["generated"]) + jitter,
        "delta_gain": float(value["gain"]) - jitter,
        "delta_cosine": float(value["cosine"]) - jitter,
        "off_token_spill_ratio": float(value["spill"]) + jitter,
        "output_progress": float(value["progress"]) - jitter,
        "output_nll_reduction": float(value["nll"]) - jitter,
        "exact_margin_min": float(value["margin"]) - jitter,
        "exact_satisfied": bool(value["exact"]),
        "paraphrase_nll_reduction": float(value["paraphrase"]) - jitter,
        "heldout_kl": float(value["kl"]) + jitter,
        "endpoint_c_energy": float(value["c"]),
        "endpoint_frobenius_norm": float(value["fro"])
        + (0.0 if arm_id in (analyzer.NO_OP, analyzer.ORACLE_DO_Z) else jitter),
        "nfe": int(value["nfe"]),
        "endpoint_right_projector_violation_ratio": float(value["leak"]),
    }


def _payload(
    arm_id: str,
    case_index: int,
    *,
    passed: bool,
) -> dict[str, object]:
    values = _arm_values(arm_id, case_index)
    heldout_kl = float(values.pop("heldout_kl"))
    return {
        "case_id": case_index,
        "request_id": _digest(f"request-{case_index}"),
        "target_anchor_id": _digest(f"target-anchor-{case_index}"),
        "arm_id": arm_id,
        "case_pass": passed,
        "arm_success": passed,
        "technical_pass": True,
        "rollback_exact": True,
        "firewall_pass": True,
        "receipt_before_outcome": True,
        **values,
        "heldout_kl": heldout_kl,
        "preservation_score": -heldout_kl,
        "arm_construction": analyzer.ARM_CONSTRUCTION[arm_id],
        "frozen_target_replay_exact": True,
        "projector_precomputed_only": True,
        "projector_integrity_exact": True,
        "genuine_alpha_solver_used": arm_id in analyzer.GENUINE_ALPHA_ARMS,
        "posthoc_b_at_p_only": arm_id in analyzer.POSTHOC_BP_ARMS,
        "reference_c_energy": 1.0,
    }


def _write_run(directory: Path, *, failed_case: int | None = None) -> None:
    run_id = analyzer.EXPECTED_ALPHA_RUN_BY_MODEL["llama3-8b-inst"]
    manifest = _manifest(run_id)
    passed_count = analyzer.EXPECTED_CASE_COUNT - int(failed_case is not None)
    summary = {
        "schema_version": analyzer.SUMMARY_SCHEMA,
        "run_id": run_id,
        "model_alias": manifest["model_alias"],
        "planned_case_count": analyzer.EXPECTED_CASE_COUNT,
        "attempted_case_count": analyzer.EXPECTED_CASE_COUNT,
        "pass_case_count": passed_count,
        "failed_case_count": analyzer.EXPECTED_CASE_COUNT - passed_count,
        "outcome_count": analyzer.EXPECTED_OUTCOME_COUNT,
        "arm_order": list(analyzer.ARM_ORDER),
        "selection_sha256": manifest["selection_sha256"],
        "config_sha256": manifest["config_sha256"],
        "frozen_target_replay_exact": True,
        "frozen_target_load_once_per_case": True,
        "direct_z_recompute_count_total": 0,
        "projector_precomputed_only": True,
        "projector_integrity_exact": True,
        "all_rollbacks_exact": True,
        "firewall_pass": True,
        "receipt_before_outcome": True,
        "paired_memit_run_id": manifest["paired_memit_run_id"],
        "paired_memit_manifest_sha256": manifest[
            "paired_memit_manifest_sha256"
        ],
        "replay_lock_id": manifest["replay_lock_id"],
        "genuine_alpha_ordered_solve_once_per_case": True,
        "genuine_alpha_bf_all_hops_genuine": True,
        "posthoc_arms_projection_only": True,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    records: list[str] = []
    sequence = 0
    for case_index in range(analyzer.EXPECTED_CASE_COUNT):
        case_passed = case_index != failed_case
        for arm_id in analyzer.ARM_ORDER:
            records.append(
                json.dumps(
                    {
                        "schema_version": analyzer.STREAM_SCHEMA,
                        "run_id": run_id,
                        "sequence": sequence,
                        "recorded_at": "2026-08-01T00:00:00Z",
                        "event": analyzer.STREAM_EVENT,
                        "payload": _payload(
                            arm_id, case_index, passed=case_passed
                        ),
                    },
                    sort_keys=True,
                )
            )
            sequence += 1
    (directory / "outcomes.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


class DirectZAlphaAnalysisTest(unittest.TestCase):
    def test_complete_contract_cli_and_sign_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory)
            output_json = directory / "analysis.json"
            output_markdown = directory / "analysis.md"
            self.assertEqual(
                analyzer.main(
                    [
                        "--run-directory",
                        str(directory),
                        "--output-json",
                        str(output_json),
                        "--output-markdown",
                        str(output_markdown),
                    ]
                ),
                0,
            )
            analysis = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(analysis["artifact_validation"]["valid"])
            self.assertEqual(
                analysis["classification"]["verdict"],
                "PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION",
            )
            residual = analysis["contrasts"]["genuine_c_vs_posthoc_c"]["metrics"][
                "z_residual_ratio"
            ]
            self.assertEqual(residual["direction"], "lower_is_better")
            self.assertTrue(residual["sign_normalized"])
            self.assertAlmostEqual(residual["mean"], 0.2)
            self.assertTrue(residual["lenient_positive"])
            self.assertIn("method-superiority", output_markdown.read_text(encoding="utf-8"))

    def test_failed_case_blocks_validity_but_remains_in_itd_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory, failed_case=7)
            analysis = analyzer.analyze_run(directory)
            self.assertFalse(analysis["artifact_validation"]["valid"])
            self.assertEqual(
                analysis["classification"]["verdict"], "BLOCK_TECHNICAL_INVALID"
            )
            residual = analysis["contrasts"]["genuine_c_vs_posthoc_c"]["metrics"][
                "z_residual_ratio"
            ]
            self.assertEqual(residual["n_itd"], analyzer.EXPECTED_CASE_COUNT)
            self.assertEqual(residual["positive_count"], 7)
            self.assertEqual(residual["zero_count"], 1)
            self.assertAlmostEqual(residual["mean"], 0.2 * 7 / 8)
            self.assertIn(
                "summary: failed_case_count must be zero",
                analysis["artifact_validation"]["failures"],
            )

    def test_schema_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["arm_order"] = list(reversed(manifest["arm_order"]))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                analyzer.analyze_run(directory)

    def test_outcome_scalar_schema_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory)
            outcomes_path = directory / "outcomes.jsonl"
            records = outcomes_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(records[0])
            first["payload"]["unexpected_scalar"] = 1.0
            records[0] = json.dumps(first, sort_keys=True)
            outcomes_path.write_text("\n".join(records) + "\n", encoding="utf-8")
            with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                analyzer.analyze_run(directory)

    def test_arm_solver_semantics_are_rejected_when_mislabeled(self) -> None:
        mutations = (
            (2, analyzer.GENUINE_FULL, "genuine_alpha_solver_used", False),
            (4, analyzer.POSTHOC_FULL, "posthoc_b_at_p_only", False),
            (0, analyzer.NO_OP, "genuine_alpha_solver_used", True),
            (0, analyzer.NO_OP, "posthoc_b_at_p_only", True),
        )
        for index, arm_id, field, value in mutations:
            with self.subTest(arm_id=arm_id, field=field, value=value):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    _write_run(directory)
                    outcomes_path = directory / "outcomes.jsonl"
                    records = outcomes_path.read_text(encoding="utf-8").splitlines()
                    record = json.loads(records[index])
                    self.assertEqual(record["payload"]["arm_id"], arm_id)
                    record["payload"][field] = value
                    records[index] = json.dumps(record, sort_keys=True)
                    outcomes_path.write_text(
                        "\n".join(records) + "\n", encoding="utf-8"
                    )
                    with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                        analyzer.analyze_run(directory)

    def test_replay_and_projector_exactness_are_required_per_outcome(self) -> None:
        for field in (
            "frozen_target_replay_exact",
            "projector_precomputed_only",
            "projector_integrity_exact",
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    _write_run(directory)
                    outcomes_path = directory / "outcomes.jsonl"
                    records = outcomes_path.read_text(encoding="utf-8").splitlines()
                    first = json.loads(records[0])
                    first["payload"][field] = False
                    records[0] = json.dumps(first, sort_keys=True)
                    outcomes_path.write_text(
                        "\n".join(records) + "\n", encoding="utf-8"
                    )
                    with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                        analyzer.analyze_run(directory)

    def test_manifest_and_summary_replay_identities_must_match_the_fixed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory)
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["paired_memit_manifest_sha256"] = _digest("wrong-source")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                analyzer.analyze_run(directory)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_run(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["replay_lock_id"] = _digest("wrong-lock")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(analyzer.DirectZAlphaAnalysisError):
                analyzer.analyze_run(directory)

    def test_reference_c_match_cone_cap_and_zero_cone_contract_are_required(self) -> None:
        mutations = (
            (3, "endpoint_c_energy", 1.001),
            (7, "endpoint_c_energy", 1.001),
            (7, "endpoint_c_energy", 0.0),
        )
        for index, field, value in mutations:
            with self.subTest(index=index, field=field, value=value):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    _write_run(directory)
                    outcomes_path = directory / "outcomes.jsonl"
                    records = outcomes_path.read_text(encoding="utf-8").splitlines()
                    record = json.loads(records[index])
                    record["payload"][field] = value
                    if index == 7 and value == 0.0:
                        record["payload"]["endpoint_frobenius_norm"] = 1.0
                    records[index] = json.dumps(record, sort_keys=True)
                    outcomes_path.write_text(
                        "\n".join(records) + "\n", encoding="utf-8"
                    )
                    analysis = analyzer.analyze_run(directory)
                    self.assertFalse(analysis["artifact_validation"]["valid"])
                    self.assertEqual(
                        analysis["classification"]["verdict"],
                        "BLOCK_TECHNICAL_INVALID",
                    )


if __name__ == "__main__":
    unittest.main()
