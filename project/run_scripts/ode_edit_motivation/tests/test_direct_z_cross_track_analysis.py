"""Synthetic contract tests for the MEMIT/Alpha direct-z cross-track join."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "direct_z_cross_track_analysis.py"
_SPEC = importlib.util.spec_from_file_location(
    "direct_z_cross_track_analysis_test_module", _ANALYZER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load direct-z cross-track analyzer")
analyzer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _envelope(
    *, schema: str, run_id: str, sequence: int, event: str, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "run_id": run_id,
        "sequence": sequence,
        "recorded_at": "2026-08-01T00:00:00Z",
        "event": event,
        "payload": payload,
    }


def _metrics(track: str, arm: str, case_index: int) -> dict[str, Any]:
    jitter = case_index * 0.001
    if track == "memit" and arm == analyzer.MEMIT_NATIVE:
        z, nll, fro, nfe = 0.40 + jitter, 0.50 + jitter, 1.10 + jitter, 4
    elif track == "memit" and arm == analyzer.MEMIT_BF:
        z, nll, fro, nfe = 0.50 + jitter, 0.40 + jitter, 1.20 + jitter, 8
    elif track == "alpha" and arm == analyzer.ALPHA_NATIVE:
        z, nll, fro, nfe = 0.60 + jitter, 0.40 + jitter, 1.30 + jitter, 4
    elif track == "alpha" and arm == analyzer.ALPHA_BF:
        z, nll, fro, nfe = 0.30 + jitter, 0.80 + jitter, 0.90 + jitter, 8
    else:
        z, nll, fro, nfe = 0.70 + jitter, 0.20 + jitter, 1.50 + jitter, 2
    heldout = max(0.0, 0.1 + z * 0.1)
    selected = arm in {
        analyzer.MEMIT_NATIVE,
        analyzer.MEMIT_BF,
        analyzer.ALPHA_NATIVE,
        analyzer.ALPHA_BF,
    }
    return {
        "z_residual_ratio": z,
        "generated_delta_error_mean": z + 0.01,
        "generated_delta_error_worst": z + 0.02,
        "delta_gain": 1.0 - z,
        "delta_cosine": 0.8 - z * 0.1,
        "off_token_spill_ratio": z * 0.2,
        "output_progress": nll,
        "output_nll_reduction": nll,
        "exact_margin_min": nll - 0.1,
        "exact_satisfied": nll > 0.3,
        "paraphrase_nll_reduction": nll * 0.8,
        "heldout_kl": heldout,
        "preservation_score": -heldout,
        "endpoint_c_energy": 1.0 if selected else 0.5,
        "endpoint_frobenius_norm": fro,
        "nfe": nfe,
    }


class SyntheticPanel:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.selection = _digest("selection")
        self.directories: dict[str, dict[str, Path]] = {}
        self.artifacts: dict[str, list[dict[str, Any]]] = {}
        for model_alias in analyzer.MODEL_ALIASES:
            short = "llama" if model_alias.startswith("llama") else "qwen"
            self.directories[model_alias] = {}
            for track in ("memit", "alpha"):
                directory = root / f"{track}-{short}"
                directory.mkdir()
                self.directories[model_alias][track] = directory
            self.artifacts[model_alias] = [
                {
                    "case_id": case_id,
                    "request_id": _digest(f"request-{case_index}"),
                    "artifact_sha256": _digest(f"{model_alias}-artifact-{case_index}"),
                    "artifact_size": 1000 + case_index,
                    "tensor_sha256": _digest(f"{model_alias}-tensor-{case_index}"),
                    "origin_lineage_id": _digest(f"{model_alias}-lineage-{case_index}"),
                    "target_token_sha256": _digest(f"{model_alias}-target-{case_index}"),
                    "reference_c_energy": 1.0,
                }
                for case_index, case_id in enumerate(analyzer.EXPECTED_CASE_ORDER)
            ]
        self._write_memit_metadata()
        self.lock = self._build_lock()
        self.lock_id = analyzer._canonical_sha256(self.lock)
        analyzer.EXPECTED_REPLAY_LOCK_ID = self.lock_id
        self.lock_path = root / "replay-lock.json"
        _write_json(self.lock_path, self.lock)
        self._write_alpha_metadata()
        self._write_streams()

    def _manifest(self, model_alias: str, track: str) -> dict[str, Any]:
        arms = analyzer.MEMIT_ARM_ORDER if track == "memit" else analyzer.ALPHA_ARM_ORDER
        schema = (
            analyzer.MEMIT_MANIFEST_SCHEMA
            if track == "memit"
            else analyzer.ALPHA_MANIFEST_SCHEMA
        )
        manifest: dict[str, Any] = {
            "schema_version": schema,
            "run_id": analyzer.RUN_IDS[model_alias][track],
            "model_alias": model_alias,
            "case_count": analyzer.EXPECTED_CASE_COUNT,
            "arm_order": list(arms),
            "selection_sha256": self.selection,
            "config_sha256": _digest(f"{model_alias}-{track}-config"),
            "bootstrap_seed": analyzer.BOOTSTRAP_SEED,
            "bootstrap_resamples": analyzer.BOOTSTRAP_RESAMPLES,
            "claim_boundary": (
                analyzer.MEMIT_CLAIM_BOUNDARY
                if track == "memit"
                else analyzer.ALPHA_CLAIM_BOUNDARY
            ),
        }
        if track == "alpha":
            source = self.lock["models"][model_alias]
            manifest.update(
                {
                    "paired_memit_run_id": source["source_run_id"],
                    "paired_memit_manifest_sha256": source["source_manifest"]["sha256"],
                    "replay_lock_id": self.lock_id,
                }
            )
        return manifest

    def _summary(self, model_alias: str, track: str, manifest: dict[str, Any]) -> dict[str, Any]:
        arms = analyzer.MEMIT_ARM_ORDER if track == "memit" else analyzer.ALPHA_ARM_ORDER
        summary: dict[str, Any] = {
            "schema_version": (
                analyzer.MEMIT_SUMMARY_SCHEMA
                if track == "memit"
                else analyzer.ALPHA_SUMMARY_SCHEMA
            ),
            "run_id": analyzer.RUN_IDS[model_alias][track],
            "model_alias": model_alias,
            "planned_case_count": analyzer.EXPECTED_CASE_COUNT,
            "attempted_case_count": analyzer.EXPECTED_CASE_COUNT,
            "pass_case_count": analyzer.EXPECTED_CASE_COUNT,
            "failed_case_count": 0,
            "outcome_count": analyzer.EXPECTED_CASE_COUNT * len(arms),
            "arm_order": list(arms),
            "selection_sha256": manifest["selection_sha256"],
            "config_sha256": manifest["config_sha256"],
            "all_rollbacks_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
        }
        if track == "memit":
            summary["direct_z_once_per_case"] = True
        else:
            summary.update(
                {
                    "frozen_target_replay_exact": True,
                    "frozen_target_load_once_per_case": True,
                    "projector_precomputed_only": True,
                    "projector_integrity_exact": True,
                    "genuine_alpha_ordered_solve_once_per_case": True,
                    "genuine_alpha_bf_all_hops_genuine": True,
                    "posthoc_arms_projection_only": True,
                    "direct_z_recompute_count_total": 0,
                    "paired_memit_run_id": manifest["paired_memit_run_id"],
                    "paired_memit_manifest_sha256": manifest[
                        "paired_memit_manifest_sha256"
                    ],
                    "replay_lock_id": manifest["replay_lock_id"],
                }
            )
        return summary

    def _write_memit_metadata(self) -> None:
        for model_alias in analyzer.MODEL_ALIASES:
            directory = self.directories[model_alias]["memit"]
            manifest = self._manifest(model_alias, "memit")
            summary = self._summary(model_alias, "memit", manifest)
            _write_json(directory / "manifest.json", manifest)
            _write_json(directory / "summary.json", summary)

    def _build_lock(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for model_alias in analyzer.MODEL_ALIASES:
            directory = self.directories[model_alias]["memit"]
            models[model_alias] = {
                "source_run_id": analyzer.RUN_IDS[model_alias]["memit"],
                "source_manifest": _identity(directory / "manifest.json"),
                "source_summary": _identity(directory / "summary.json"),
                "artifacts": self.artifacts[model_alias],
            }
        return {
            "schema_version": analyzer.LOCK_SCHEMA,
            "selection_sha256": self.selection,
            "rank_slice": list(analyzer.EXPECTED_RANK_SLICE),
            "case_order": list(analyzer.EXPECTED_CASE_ORDER),
            "models": models,
        }

    def _write_alpha_metadata(self) -> None:
        for model_alias in analyzer.MODEL_ALIASES:
            directory = self.directories[model_alias]["alpha"]
            manifest = self._manifest(model_alias, "alpha")
            summary = self._summary(model_alias, "alpha", manifest)
            _write_json(directory / "manifest.json", manifest)
            _write_json(directory / "summary.json", summary)

    def _write_streams(self) -> None:
        for model_alias in analyzer.MODEL_ALIASES:
            for track in ("memit", "alpha"):
                self._write_features(model_alias, track)
                self._write_events(model_alias, track)
                self._write_outcomes(model_alias, track)

    def _write_features(self, model_alias: str, track: str) -> None:
        rows: list[dict[str, Any]] = []
        run_id = analyzer.RUN_IDS[model_alias][track]
        schema = analyzer.MEMIT_STREAM_SCHEMA if track == "memit" else analyzer.ALPHA_STREAM_SCHEMA
        event = "direct_z_possibility_feature" if track == "memit" else "direct_z_alpha_feature"
        for case_index, artifact in enumerate(self.artifacts[model_alias]):
            if track == "memit":
                payload: dict[str, Any] = {
                    "case_id": artifact["case_id"],
                    "request_id": artifact["request_id"],
                    "target_identity": {
                        "direct_z_artifact_sha256": artifact["artifact_sha256"],
                        "direct_z_compute_count": 1,
                        "direct_z_tensor_sha256": artifact["tensor_sha256"],
                        "origin_lineage_id": artifact["origin_lineage_id"],
                        "target_token_sha256": artifact["target_token_sha256"],
                    },
                }
            else:
                source = self.lock["models"][model_alias]
                anchor: dict[str, Any] = {
                    "alpha_solver_config_id": _digest(f"{model_alias}-alpha-config"),
                    "alpha_solver_yaml_sha256": _digest(f"{model_alias}-alpha-yaml"),
                    "context_id": _digest(f"{model_alias}-context"),
                    "cross_solver_frozen_target_anchor": True,
                    "direct_z_artifact_sha256": artifact["artifact_sha256"],
                    "direct_z_artifact_size": artifact["artifact_size"],
                    "direct_z_compute_count": 0,
                    "direct_z_load_count": 1,
                    "direct_z_tensor_sha256": artifact["tensor_sha256"],
                    "memit_target_hparams_sha256": _digest(f"{model_alias}-memit-hparams"),
                    "model_id": f"synthetic-{model_alias}",
                    "origin_lineage_id": artifact["origin_lineage_id"],
                    "origin_parameter_hashes": {
                        "layer.0": _digest(f"{model_alias}-W0-layer-0"),
                        "layer.1": _digest(f"{model_alias}-W0-layer-1"),
                    },
                    "origin_snapshot_id": _digest(f"{model_alias}-snapshot-{case_index}"),
                    "origin_state_id": _digest(f"{model_alias}-state-{case_index}"),
                    "request_ids": [artifact["request_id"]],
                    "source_manifest_sha256": source["source_manifest"]["sha256"],
                    "source_run_id": source["source_run_id"],
                    "source_summary_sha256": source["source_summary"]["sha256"],
                    "target_token_sha256": artifact["target_token_sha256"],
                }
                anchor["target_anchor_id"] = analyzer._canonical_sha256(anchor)
                payload = {
                    "case_id": artifact["case_id"],
                    "request_id": artifact["request_id"],
                    "reference_c_energy": artifact["reference_c_energy"],
                    "target_anchor": anchor,
                    "target_identity": {
                        "direct_z_tensor_sha256": artifact["tensor_sha256"],
                        "origin_lineage_id": artifact["origin_lineage_id"],
                        "target_anchor_id": anchor["target_anchor_id"],
                        "target_token_sha256": artifact["target_token_sha256"],
                    },
                }
            payload["feature_hash"] = analyzer._canonical_sha256(payload)
            rows.append(
                _envelope(
                    schema=schema,
                    run_id=run_id,
                    sequence=case_index,
                    event=event,
                    payload=payload,
                )
            )
        _write_jsonl(self.directories[model_alias][track] / "features.jsonl", rows)

    def _write_events(self, model_alias: str, track: str) -> None:
        rows: list[dict[str, Any]] = []
        run_id = analyzer.RUN_IDS[model_alias][track]
        schema = analyzer.MEMIT_STREAM_SCHEMA if track == "memit" else analyzer.ALPHA_STREAM_SCHEMA
        event = "direct_z_possibility_case" if track == "memit" else "direct_z_alpha_case"
        arms = analyzer.MEMIT_ARM_ORDER if track == "memit" else analyzer.ALPHA_ARM_ORDER
        payload_schema = (
            "ode-edit-direct-z-possibility-event/v2"
            if track == "memit"
            else "ode-edit-direct-z-alpha-paired-event/v1"
        )
        for case_index, artifact in enumerate(self.artifacts[model_alias]):
            payload: dict[str, Any] = {
                "schema_version": payload_schema,
                "case_id": artifact["case_id"],
                "request_id": artifact["request_id"],
                "evaluation_payload_sha256": _digest(f"heldout-{case_index}"),
                "feature_count": 1,
                "outcome_count": len(arms),
                "pass": True,
                "technical": {"firewall_pass": True, "rollback_exact": True},
            }
            if track == "memit":
                payload["direct_z_compute_count"] = 1
            else:
                payload["direct_z_compute_count"] = 0
                payload["direct_z_load_count"] = 1
            rows.append(
                _envelope(
                    schema=schema,
                    run_id=run_id,
                    sequence=case_index,
                    event=event,
                    payload=payload,
                )
            )
        _write_jsonl(self.directories[model_alias][track] / "events.jsonl", rows)

    def _write_outcomes(self, model_alias: str, track: str) -> None:
        rows: list[dict[str, Any]] = []
        run_id = analyzer.RUN_IDS[model_alias][track]
        schema = analyzer.MEMIT_STREAM_SCHEMA if track == "memit" else analyzer.ALPHA_STREAM_SCHEMA
        event = "direct_z_possibility_outcome" if track == "memit" else "direct_z_alpha_outcome"
        arms = analyzer.MEMIT_ARM_ORDER if track == "memit" else analyzer.ALPHA_ARM_ORDER
        alpha_feature_rows = None
        if track == "alpha":
            alpha_feature_rows = [
                json.loads(line)
                for line in (self.directories[model_alias][track] / "features.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        sequence = 0
        for case_index, artifact in enumerate(self.artifacts[model_alias]):
            for arm in arms:
                payload: dict[str, Any] = {
                    "case_id": artifact["case_id"],
                    "request_id": artifact["request_id"],
                    "arm_id": arm,
                    "case_pass": True,
                    "arm_success": True,
                    "technical_pass": True,
                    "rollback_exact": True,
                    "firewall_pass": True,
                    "receipt_before_outcome": True,
                    **_metrics(track, arm, case_index),
                }
                if track == "alpha":
                    assert alpha_feature_rows is not None
                    payload["target_anchor_id"] = alpha_feature_rows[case_index]["payload"]["target_anchor"]["target_anchor_id"]
                    payload["reference_c_energy"] = artifact["reference_c_energy"]
                    payload["arm_construction"] = analyzer.ALPHA_ARM_CONSTRUCTION[arm]
                    payload["frozen_target_replay_exact"] = True
                    payload["projector_precomputed_only"] = True
                    payload["projector_integrity_exact"] = True
                    payload["genuine_alpha_solver_used"] = (
                        arm in analyzer.ALPHA_GENUINE_ARMS
                    )
                    payload["posthoc_b_at_p_only"] = (
                        arm in analyzer.ALPHA_POSTHOC_ARMS
                    )
                rows.append(
                    _envelope(
                        schema=schema,
                        run_id=run_id,
                        sequence=sequence,
                        event=event,
                        payload=payload,
                    )
                )
                sequence += 1
        _write_jsonl(self.directories[model_alias][track] / "outcomes.jsonl", rows)

    def kwargs(self) -> dict[str, Any]:
        return {
            "replay_lock": self.lock_path,
            "memit_llama_directory": self.directories["llama3-8b-inst"]["memit"],
            "alpha_llama_directory": self.directories["llama3-8b-inst"]["alpha"],
            "memit_qwen_directory": self.directories["qwen2.5-7b-inst"]["memit"],
            "alpha_qwen_directory": self.directories["qwen2.5-7b-inst"]["alpha"],
        }


def _mutate_jsonl(path: Path, row_index: int, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[row_index])
    _write_jsonl(path, rows)


class DirectZCrossTrackAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_lock_id = analyzer.EXPECTED_REPLAY_LOCK_ID

    def tearDown(self) -> None:
        analyzer.EXPECTED_REPLAY_LOCK_ID = self.original_lock_id

    def test_valid_join_sign_normalization_and_distinct_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPanel(Path(temporary))
            result = analyzer.analyze_cross_track(**fixture.kwargs())
        self.assertTrue(result["artifact_validation"]["valid"])
        self.assertFalse(result["fixed_contract"]["model_pooling"])
        self.assertEqual(set(result["models"]), set(analyzer.MODEL_ALIASES))
        for model_alias in analyzer.MODEL_ALIASES:
            panel = result["models"][model_alias]
            self.assertEqual(
                len(panel["canonical_cross_bound_join_certificates"]), 8
            )
            self.assertEqual(
                len(panel["alpha_self_bound_provenance_certificates"]), 8
            )
            residual = panel["effects"]["z_residual_ratio"]
            self.assertEqual(residual["direction"], "lower_is_better")
            self.assertTrue(residual["sign_normalized"])
            self.assertAlmostEqual(
                residual["memit_bf_minus_native_lift"]["mean"], -0.1
            )
            self.assertAlmostEqual(
                residual["alpha_bf_minus_native_lift"]["mean"], 0.3
            )
            self.assertAlmostEqual(
                residual["alpha_minus_memit_bf_native_2x2_interaction"]["mean"],
                0.4,
            )
            self.assertAlmostEqual(
                residual["alpha_bf_minus_memit_bf_endpoint_contrast"]["mean"],
                0.2,
            )
            self.assertTrue(
                residual["alpha_minus_memit_bf_native_2x2_interaction"][
                    "lenient_positive"
                ]
            )
            self.assertEqual(
                residual["alpha_minus_memit_bf_native_2x2_interaction"][
                    "bootstrap_resamples"
                ],
                4000,
            )

    def test_unrehashed_identity_tampering_blocks(self) -> None:
        mutations = {
            "W0_lineage": lambda row: row["payload"]["target_anchor"].__setitem__(
                "origin_lineage_id", _digest("wrong-lineage")
            ),
            "context": lambda row: row["payload"]["target_anchor"].__setitem__(
                "context_id", _digest("wrong-context")
            ),
            "source_state": lambda row: row["payload"]["target_anchor"].__setitem__(
                "origin_state_id", _digest("wrong-state")
            ),
            "source_snapshot": lambda row: row["payload"]["target_anchor"].__setitem__(
                "origin_snapshot_id", _digest("wrong-snapshot")
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticPanel(Path(temporary))
                feature_path = fixture.directories["llama3-8b-inst"]["alpha"] / "features.jsonl"
                _mutate_jsonl(feature_path, 0, mutation)
                result = analyzer.analyze_cross_track(**fixture.kwargs())
                self.assertFalse(result["artifact_validation"]["valid"])
                self.assertEqual(
                    result["classification"]["verdict"],
                    "BLOCK_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_OR_C_BUDGET_INVALID",
                )

    def test_coherent_alpha_internal_rebinding_is_self_bound_not_cross_pinned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPanel(Path(temporary))
            before = analyzer.analyze_cross_track(**fixture.kwargs())
            model_alias = "llama3-8b-inst"
            alpha_root = fixture.directories[model_alias]["alpha"]
            feature_path = alpha_root / "features.jsonl"
            feature_rows = [
                json.loads(line)
                for line in feature_path.read_text(encoding="utf-8").splitlines()
            ]
            rebound_anchor_ids: list[str] = []
            for case_index, row in enumerate(feature_rows):
                payload = row["payload"]
                anchor = payload["target_anchor"]
                anchor.update(
                    {
                        "alpha_solver_config_id": _digest("rebound-alpha-config"),
                        "alpha_solver_yaml_sha256": _digest("rebound-alpha-yaml"),
                        "context_id": _digest("rebound-context"),
                        "memit_target_hparams_sha256": _digest(
                            "rebound-memit-hparams"
                        ),
                        "model_id": "coherently-rebound-alpha-model",
                        "origin_parameter_hashes": {
                            "rebound.layer.0": _digest("rebound-W0-layer-0"),
                            "rebound.layer.1": _digest("rebound-W0-layer-1"),
                        },
                        "origin_snapshot_id": _digest(
                            f"rebound-snapshot-{case_index}"
                        ),
                        "origin_state_id": _digest(f"rebound-state-{case_index}"),
                    }
                )
                anchor_body = dict(anchor)
                anchor_body.pop("target_anchor_id")
                anchor["target_anchor_id"] = analyzer._canonical_sha256(anchor_body)
                payload["target_identity"]["target_anchor_id"] = anchor[
                    "target_anchor_id"
                ]
                feature_body = dict(payload)
                feature_body.pop("feature_hash")
                payload["feature_hash"] = analyzer._canonical_sha256(feature_body)
                rebound_anchor_ids.append(anchor["target_anchor_id"])
            _write_jsonl(feature_path, feature_rows)

            outcome_path = alpha_root / "outcomes.jsonl"
            outcome_rows = [
                json.loads(line)
                for line in outcome_path.read_text(encoding="utf-8").splitlines()
            ]
            for sequence, row in enumerate(outcome_rows):
                case_index = sequence // len(analyzer.ALPHA_ARM_ORDER)
                row["payload"]["target_anchor_id"] = rebound_anchor_ids[case_index]
            _write_jsonl(outcome_path, outcome_rows)

            after = analyzer.analyze_cross_track(**fixture.kwargs())

        self.assertTrue(after["artifact_validation"]["valid"])
        before_panel = before["models"][model_alias]
        after_panel = after["models"][model_alias]
        self.assertEqual(
            [
                item["certificate_sha256"]
                for item in before_panel[
                    "canonical_cross_bound_join_certificates"
                ]
            ],
            [
                item["certificate_sha256"]
                for item in after_panel[
                    "canonical_cross_bound_join_certificates"
                ]
            ],
        )
        self.assertNotEqual(
            [
                item["certificate_sha256"]
                for item in before_panel[
                    "alpha_self_bound_provenance_certificates"
                ]
            ],
            [
                item["certificate_sha256"]
                for item in after_panel[
                    "alpha_self_bound_provenance_certificates"
                ]
            ],
        )
        self.assertEqual(before_panel["effects"], after_panel["effects"])

    def test_heldout_and_source_manifest_identity_mismatch_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPanel(Path(temporary))
            event_path = fixture.directories["qwen2.5-7b-inst"]["alpha"] / "events.jsonl"
            _mutate_jsonl(
                event_path,
                0,
                lambda row: row["payload"].__setitem__(
                    "evaluation_payload_sha256", _digest("wrong-heldout")
                ),
            )
            result = analyzer.analyze_cross_track(**fixture.kwargs())
            self.assertFalse(result["artifact_validation"]["valid"])
            self.assertTrue(
                any(
                    "heldout_identity" in failure
                    for failure in result["models"]["qwen2.5-7b-inst"][
                        "artifact_validation"
                    ]["failures"]
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPanel(Path(temporary))
            manifest_path = fixture.directories["llama3-8b-inst"]["memit"] / "manifest.json"
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            result = analyzer.analyze_cross_track(**fixture.kwargs())
            self.assertFalse(result["artifact_validation"]["valid"])
            self.assertTrue(
                any(
                    "source_manifest_identity" in failure
                    for failure in result["models"]["llama3-8b-inst"][
                        "artifact_validation"
                    ]["failures"]
                )
            )

    def test_C_energy_cap_violation_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPanel(Path(temporary))
            outcome_path = fixture.directories["llama3-8b-inst"]["alpha"] / "outcomes.jsonl"
            bf_index = analyzer.ALPHA_ARM_ORDER.index(analyzer.ALPHA_BF)
            _mutate_jsonl(
                outcome_path,
                bf_index,
                lambda row: row["payload"].__setitem__("endpoint_c_energy", 1.01),
            )
            result = analyzer.analyze_cross_track(**fixture.kwargs())
        self.assertFalse(result["artifact_validation"]["valid"])
        failures = result["models"]["llama3-8b-inst"]["artifact_validation"]["failures"]
        self.assertTrue(any("C_energy_cap" in failure for failure in failures))

    def test_cli_writes_exclusive_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticPanel(root)
            output_json = root / "analysis.json"
            output_markdown = root / "analysis.md"
            kwargs = fixture.kwargs()
            argv = [
                "--replay-lock",
                str(kwargs["replay_lock"]),
                "--memit-llama-directory",
                str(kwargs["memit_llama_directory"]),
                "--alpha-llama-directory",
                str(kwargs["alpha_llama_directory"]),
                "--memit-qwen-directory",
                str(kwargs["memit_qwen_directory"]),
                "--alpha-qwen-directory",
                str(kwargs["alpha_qwen_directory"]),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
            self.assertEqual(analyzer.main(argv), 0)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_markdown.is_file())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["artifact_validation"]["valid"])
            self.assertNotIn("synthetic-llama3-8b-inst", output_json.read_text(encoding="utf-8"))
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIsInstance(markdown, str)
            self.assertIn("2×2 interaction", markdown)
            self.assertIn("endpoint contrast", markdown)
            self.assertIn("self-bound provenance", markdown)
            self.assertEqual(analyzer.main(argv), 2)


if __name__ == "__main__":
    unittest.main()
