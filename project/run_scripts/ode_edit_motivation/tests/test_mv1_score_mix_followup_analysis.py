from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from project.run_scripts.ode_edit_motivation.contracts import ContractError
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
)
from project.run_scripts.ode_edit_motivation.mv1_analysis import (
    build_confirmatory_fold_manifest,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_analysis import (
    NUMERIC_COMPARISON_TOLERANCE,
    _canonical_json,
    _selection_manifest_id,
    _sha256_bytes,
    _sha256_file,
    ScoreMixAnalysisError,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory_analysis import (
    CONFIRMATORY_ANALYSIS_SCHEMA,
    CONFIRMATORY_FOLD_COUNT,
    CONFIRMATORY_FOLD01_INPUT_SCHEMA,
    CONFIRMATORY_FOLD_SEED,
    CONFIRMATORY_OUTCOME_ACTIONS,
    CONFIRMATORY_ANALYSIS_WAVE_LOCK,
    ScoreMixAnalysisWaveLock,
    _effect_summary,
    _pearson,
    analyze_confirmatory_run,
    analyze_confirmatory_run_for_fold01,
    analyze_score_mix_wave,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_followup import (
    FOLLOWUP_JOB_NAMES,
    FOLLOWUP_RUN_IDS,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_followup_analysis import (
    FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
    FOLD01_AGGREGATE_BOOTSTRAP_SEED,
    FOLD01_AGGREGATE_SCHEMA,
    FOLLOWUP_ANALYSIS_SCHEMA,
    FOLLOWUP_BOOTSTRAP_REPLICATES,
    FOLLOWUP_BOOTSTRAP_SEED,
    PRIMARY_ESTIMAND,
    FOLLOWUP_ANALYSIS_WAVE_LOCKS,
    aggregate_fold01_analyses,
    analyze_followup_run,
    followup_analysis_wave_lock,
    main as followup_analysis_main,
)
from project.run_scripts.ode_edit_motivation.tests.test_mv1_score_mix_confirmatory_analysis import (
    MODEL,
    build_run,
    write_json,
)


def digest(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _rewrite_stream_run_id(path: Path, run_id: str) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["run_id"] = run_id
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def convert_c1_fixture_to_fold1(root: Path) -> Path:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    selection = manifest["selection"]
    selected = tuple(manifest["selected_case_ids"])
    confirmatory = list(selection["case_ids"]["confirmatory"])

    def rank_key(case_id: str):
        return (
            hashlib.sha256(
                CONFIRMATORY_FOLD_SEED.encode("utf-8")
                + b"\0"
                + case_id.encode("utf-8")
            ).digest(),
            case_id,
        )

    ranked = sorted(confirmatory, key=rank_key)
    removed = ranked[-1]
    if removed in selected:
        raise AssertionError("rank-59 fixture case unexpectedly belongs to fold0")
    smallest_key = rank_key(ranked[0])
    all_case_ids = {
        case_id
        for split in selection["case_ids"].values()
        for case_id in split
    }
    replacement = None
    for index in range(10000):
        candidate = f"fold1-minimum-pad-{index:04d}"
        if candidate not in all_case_ids and rank_key(candidate) < smallest_key:
            replacement = candidate
            break
    if replacement is None:
        raise AssertionError("could not construct deterministic fold1 fixture")
    confirmatory[confirmatory.index(removed)] = replacement
    selection["seed"] = DEFAULT_SELECTION_SEED
    selection["case_ids"]["confirmatory"] = confirmatory
    calibration = selection["case_ids"]["calibration"]
    untouched = selection["case_ids"]["untouched"]
    selection["order_hash"] = digest(calibration + confirmatory + untouched)
    selection["split_hash"] = digest(
        {
            "calibration": calibration,
            "confirmatory": confirmatory,
            "untouched": untouched,
        }
    )
    selection["manifest_id"] = _selection_manifest_id(selection)
    folds = build_confirmatory_fold_manifest(
        confirmatory,
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    expected = tuple(
        case_id for case_id in confirmatory if folds.fold_for(case_id) == 1
    )
    if expected != selected:
        raise AssertionError("fixture transformation did not preserve exact fold1 IDs")

    run_id = FOLLOWUP_RUN_IDS["fold1"][MODEL]
    destination = root.parent / run_id
    root.rename(destination)
    root = destination
    slurm = {
        "under_slurm": True,
        "job_id": "50001",
        "job_name": FOLLOWUP_JOB_NAMES["fold1"],
        "node": "devbox",
        "slice_label": "fold1",
        "fold": 1,
    }
    manifest["run_id"] = run_id
    manifest["selection"] = selection
    manifest["confirmatory_folds"] = folds.to_dict()
    manifest["selected_fold"] = {
        "label": "fold1",
        "fold": 1,
        "count": 12,
        "case_ids": list(selected),
        "run_seed": 17,
    }
    manifest["slurm"] = slurm
    summary["run_id"] = run_id
    summary["slice"] = {
        "label": "fold1",
        "split": "confirmatory",
        "fold": 1,
        "fold_count": 5,
        "count": 12,
        "fold_manifest_id": folds.manifest_id,
        "run_seed": 17,
    }
    summary["slurm"] = slurm
    summary["selection_manifest_id"] = selection["manifest_id"]
    write_json(root / "manifest.json", manifest)
    for name in (
        "features.jsonl",
        "actions.jsonl",
        "outcomes.jsonl",
        "events.jsonl",
    ):
        _rewrite_stream_run_id(root / name, run_id)
    summary["artifacts"].update(
        {
            "manifest_sha256": _sha256_file(root / "manifest.json"),
            "features_sha256": _sha256_file(root / "features.jsonl"),
            "actions_sha256": _sha256_file(root / "actions.jsonl"),
            "outcomes_sha256": _sha256_file(root / "outcomes.jsonl"),
            "events_sha256": _sha256_file(root / "events.jsonl"),
        }
    )
    write_json(root / "summary.json", summary)
    return root


def canonicalize_c1_fixture_selection(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    selection = manifest["selection"]
    selection["seed"] = DEFAULT_SELECTION_SEED
    selection["manifest_id"] = _selection_manifest_id(selection)
    manifest["selection"] = selection
    write_json(root / "manifest.json", manifest)
    summary["selection_manifest_id"] = selection["manifest_id"]
    summary["artifacts"]["manifest_sha256"] = _sha256_file(
        root / "manifest.json"
    )
    write_json(root / "summary.json", summary)


def synthetic_fold1_analysis(
    c1_input: dict,
    *,
    replay_envelope: float | None = None,
) -> dict:
    c1 = c1_input["c1_analysis"]
    identity = c1_input["aggregate_identity"]
    report = copy.deepcopy(c1)
    report["schema_version"] = FOLLOWUP_ANALYSIS_SCHEMA
    report["claim_status"] = "single_model_followup_gate_inputs_only"
    report["analysis_status"] = "followup_single_model_complete"
    report["run_id"] = FOLLOWUP_RUN_IDS["fold1"][report["model_alias"]]
    report["mode"] = "fold1"
    artifact = report["artifact_validation"]
    artifact.update(
        {
            "exact_fold": "confirmatory_hash_fold_1",
            "selection_manifest_id": identity["selection_manifest_id"],
            "fold_manifest_id": identity["fold_manifest_id"],
            "context_id": identity["context_id"],
            "provenance_id": identity["provenance_id"],
            "q": 1.0 / 256.0,
            "outcome_action_order": list(CONFIRMATORY_OUTCOME_ACTIONS),
            "calibration_hash": identity["calibration_hash"],
            "forecast_beta": identity["forecast_beta"],
            "run_seed": 17,
            "primary_estimand": PRIMARY_ESTIMAND,
            "bootstrap_seed": FOLLOWUP_BOOTSTRAP_SEED,
            "bootstrap_replicates": FOLLOWUP_BOOTSTRAP_REPLICATES,
        }
    )
    report["wave_replay_envelope"] = report.pop("c1_replay_envelope")
    if replay_envelope is not None:
        report["wave_replay_envelope"] = replay_envelope
        report["replay_envelope"] = max(
            replay_envelope,
            report["calibration_replay_envelope"],
        )
    folds = build_confirmatory_fold_manifest(
        identity["confirmatory_case_ids"],
        seed=CONFIRMATORY_FOLD_SEED,
        fold_count=CONFIRMATORY_FOLD_COUNT,
    )
    fold1_case_ids = [
        case_id
        for case_id in identity["confirmatory_case_ids"]
        if folds.fold_for(case_id) == 1
    ]
    for row, case_id in zip(report["case_diagnostics"], fold1_case_ids):
        row["case_id"] = case_id
        row["effect_above_replay_envelope"] = (
            row["adaptive_static_effect"] > report["replay_envelope"]
        )
    artifact["selected_case_ids_sha256"] = _sha256_bytes(
        _canonical_json(
            [row["case_id"] for row in report["case_diagnostics"]]
        ).encode("utf-8")
    )
    effects = [
        row["adaptive_static_effect"] for row in report["case_diagnostics"]
    ]
    oracles = [
        row["finite_panel_oracle_opportunity"]
        for row in report["case_diagnostics"]
    ]
    report["primary_adaptive_minus_frozen_static"] = _effect_summary(
        effects,
        expected_case_count=12,
        replay_envelope=report["replay_envelope"],
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    report["finite_panel_oracle_opportunity"] = _effect_summary(
        oracles,
        expected_case_count=12,
        replay_envelope=report["replay_envelope"],
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED + 1,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    report["single_model_gate_inputs"] = {
        "clear_continue_input": False,
        "scientific_kill_input": False,
        "controller_pivot_input": False,
        "gray_input": False,
        "architecture_conditional_tolerance_input": report[
            "calibration_residual_envelope"
        ],
        "pair_level_decision_computed": False,
    }
    report["execution_lock"] = {
        "selected_split": "confirmatory",
        "fold": 1,
        "case_count": 12,
        "run_seed": 17,
        "q": 1.0 / 256.0,
        "outcome_action_order": list(CONFIRMATORY_OUTCOME_ACTIONS),
        "primary_estimand": PRIMARY_ESTIMAND,
        "bootstrap_seed": FOLLOWUP_BOOTSTRAP_SEED,
        "bootstrap_replicates": FOLLOWUP_BOOTSTRAP_REPLICATES,
        "pair_level_decision_computed": False,
    }
    report["model_single_run_gate_inputs"] = {
        "replay_envelope": report["replay_envelope"],
        "primary": report["primary_adaptive_minus_frozen_static"],
        "oracle": report["finite_panel_oracle_opportunity"],
        "forecast_calibration": {
            "predicted_vs_realized": report["predicted_vs_realized"],
            "calibration_residual_envelope": report[
                "calibration_residual_envelope"
            ],
            "calibration_replay_envelope": report[
                "calibration_replay_envelope"
            ],
        },
        "pair_level_decision_computed": False,
    }
    report["claim_boundary"] = (
        "One fixed-model 12-case fold1 run; no pair decision."
    )
    rewrite_analysis_effects(
        report,
        [
            row["adaptive_static_effect"]
            for row in report["case_diagnostics"]
        ],
    )
    return report


def rehash_c1_input(c1_input: dict) -> None:
    c1_input["analysis_hash"] = _sha256_bytes(
        _canonical_json(
            {
                key: value
                for key, value in c1_input.items()
                if key != "analysis_hash"
            }
        ).encode("utf-8")
    )


def rewrite_analysis_effects(
    report: dict,
    effects: list[float],
    *,
    oracle_values: list[float] | None = None,
) -> None:
    if len(effects) != 12:
        raise AssertionError("synthetic fold must contain 12 effects")
    if oracle_values is None:
        oracle_values = [
            row["finite_panel_oracle_opportunity"]
            for row in report["case_diagnostics"]
        ]
    if len(oracle_values) != 12:
        raise AssertionError("synthetic fold must contain 12 oracle values")
    replay = report["replay_envelope"]
    predicted_values = []
    for row, effect, oracle in zip(
        report["case_diagnostics"],
        effects,
        oracle_values,
    ):
        predicted = row["predicted_adaptive_static_gain"]
        row["adaptive_static_effect"] = effect
        row["finite_panel_oracle_opportunity"] = oracle
        row["prediction_error"] = effect - predicted
        row["effect_above_replay_envelope"] = effect > replay
        predicted_values.append(predicted)
    report["primary_adaptive_minus_frozen_static"] = _effect_summary(
        effects,
        expected_case_count=12,
        replay_envelope=replay,
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    report["finite_panel_oracle_opportunity"] = _effect_summary(
        oracle_values,
        expected_case_count=12,
        replay_envelope=replay,
        bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED + 1,
        bootstrap_replicates=FOLLOWUP_BOOTSTRAP_REPLICATES,
    )
    denominator = sum(value * value for value in predicted_values)
    prediction = {
        "predicted_mean": statistics.fmean(predicted_values),
        "realized_mean": statistics.fmean(effects),
        "mean_error": statistics.fmean(
            realized - predicted
            for predicted, realized in zip(predicted_values, effects)
        ),
        "mean_absolute_error": statistics.fmean(
            abs(realized - predicted)
            for predicted, realized in zip(predicted_values, effects)
        ),
        "median_absolute_error": statistics.median(
            abs(realized - predicted)
            for predicted, realized in zip(predicted_values, effects)
        ),
        "zero_intercept_realized_on_predicted_slope": (
            sum(
                predicted * realized
                for predicted, realized in zip(predicted_values, effects)
            )
            / denominator
            if denominator > NUMERIC_COMPARISON_TOLERANCE
            else None
        ),
        "pearson": _pearson(predicted_values, effects),
        "positive_direction_concordance": sum(
            (predicted > 0.0) == (realized > replay)
            for predicted, realized in zip(predicted_values, effects)
        )
        / len(effects),
    }
    report["predicted_vs_realized"] = prediction
    if report.get("mode") == "fold1":
        inputs = report["model_single_run_gate_inputs"]
        inputs["primary"] = report[
            "primary_adaptive_minus_frozen_static"
        ]
        inputs["oracle"] = report["finite_panel_oracle_opportunity"]
        inputs["forecast_calibration"][
            "predicted_vs_realized"
        ] = prediction
    else:
        primary = report["primary_adaptive_minus_frozen_static"]
        oracle = report["finite_panel_oracle_opportunity"]
        clear = bool(
            primary["mean"] > replay
            and (
                primary["trimmed_mean_20pct"] > replay
                or primary["median"] > replay
                or primary[
                    "positive_sign_count_above_replay_envelope"
                ]
                >= 7
            )
        )
        kill = bool(
            primary["mean"] <= replay
            and primary["trimmed_mean_20pct"] <= replay
            and primary[
                "positive_sign_fraction_above_replay_envelope"
            ]
            <= 0.50
            and oracle["mean"] <= replay
        )
        pivot = bool(
            primary["mean"] <= replay
            and primary["trimmed_mean_20pct"] <= replay
            and primary[
                "positive_sign_fraction_above_replay_envelope"
            ]
            <= 0.50
            and oracle["mean"] > replay
            and (
                oracle["trimmed_mean_20pct"] > replay
                or oracle["median"] > replay
                or oracle[
                    "positive_sign_count_above_replay_envelope"
                ]
                >= 7
            )
        )
        gate = report["single_model_gate_inputs"]
        gate.update(
            {
                "clear_continue_input": clear,
                "scientific_kill_input": kill,
                "controller_pivot_input": pivot,
                "gray_input": bool(not clear and not kill and not pivot),
            }
        )


class FollowupAnalysisTests(unittest.TestCase):
    def test_full_fold1_run_uses_parameterized_c1_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            root = convert_c1_fixture_to_fold1(root)
            report = analyze_followup_run(
                root,
                mode="fold1",
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
            )
        self.assertEqual(report["schema_version"], FOLLOWUP_ANALYSIS_SCHEMA)
        self.assertEqual(report["mode"], "fold1")
        self.assertTrue(report["artifact_validation"]["valid"])
        self.assertEqual(
            report["artifact_validation"]["exact_fold"],
            "confirmatory_hash_fold_1",
        )
        self.assertEqual(report["artifact_validation"]["case_count"], 12)
        self.assertEqual(
            report["artifact_validation"]["selected_case_ids_sha256"],
            _sha256_bytes(
                _canonical_json(
                    [
                        row["case_id"]
                        for row in report["case_diagnostics"]
                    ]
                ).encode("utf-8")
            ),
        )
        self.assertEqual(
            report["execution_lock"]["outcome_action_order"],
            list(CONFIRMATORY_OUTCOME_ACTIONS),
        )
        inputs = report["model_single_run_gate_inputs"]
        self.assertEqual(inputs["primary"]["case_count"], 12)
        self.assertIn("trimmed_mean_20pct", inputs["primary"])
        self.assertIn("median", inputs["primary"])
        self.assertIn("paired_case_bootstrap_mean_ci_95", inputs["primary"])
        self.assertIn("forecast_calibration", inputs)
        self.assertFalse(inputs["pair_level_decision_computed"])

    def test_untouched_lock_is_exact_20_and_bootstrap_is_not_tunable(self):
        lock = followup_analysis_wave_lock("untouched")
        self.assertEqual(
            (lock.selected_split, lock.fold, lock.case_count),
            ("untouched", None, 20),
        )
        self.assertEqual(FOLLOWUP_BOOTSTRAP_SEED, 20260731)
        self.assertEqual(FOLLOWUP_BOOTSTRAP_REPLICATES, 4000)
        with self.assertRaises(ContractError):
            ScoreMixAnalysisWaveLock(
                label="fold1",
                selected_split="confirmatory",
                fold=True,
                case_count=12,
                job_name=FOLLOWUP_JOB_NAMES["fold1"],
                run_ids=FOLLOWUP_RUN_IDS["fold1"],
                run_seed=17,
                require_canonical_selection_seed=True,
            )
        fake = SimpleNamespace(
            label="fold1",
            selected_split="confirmatory",
            fold=1,
            case_count=12,
            job_name=FOLLOWUP_JOB_NAMES["fold1"],
            run_ids=FOLLOWUP_RUN_IDS["fold1"],
            run_seed=17,
            require_canonical_selection_seed=True,
        )
        with self.assertRaisesRegex(ContractError, "runtime type"):
            analyze_score_mix_wave(
                "/not-opened",
                wave=fake,
                static_policy={},
                forecast_policy={},
                forecast_analysis={},
            )
        with self.assertRaisesRegex(
            ScoreMixAnalysisError,
            "bootstrap configuration",
        ):
            analyze_score_mix_wave(
                "/not-opened",
                wave=FOLLOWUP_ANALYSIS_WAVE_LOCKS["untouched"],
                static_policy={},
                forecast_policy={},
                forecast_analysis={},
                bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED + 1,
            )
        values = [float(index) for index in range(20)]
        first = _effect_summary(
            values,
            expected_case_count=20,
            replay_envelope=1e-12,
            bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
            bootstrap_replicates=100,
        )
        second = _effect_summary(
            values,
            expected_case_count=20,
            replay_envelope=1e-12,
            bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
            bootstrap_replicates=100,
        )
        self.assertEqual(first, second)
        with self.assertRaises(ScoreMixAnalysisError):
            followup_analysis_wave_lock("c1")

    def _analysis_pair(self):
        temporary = tempfile.TemporaryDirectory()
        root, static, forecast, forecast_analysis = build_run(temporary.name)
        canonicalize_c1_fixture_selection(root)
        c1_input = analyze_confirmatory_run_for_fold01(
            root,
            static_policy=static,
            forecast_policy=forecast,
            forecast_analysis=forecast_analysis,
        )
        return (
            temporary,
            c1_input,
            synthetic_fold1_analysis(c1_input),
        )

    def test_fold01_aggregate_recomputes_24_event_inputs_deterministically(self):
        temporary, c1, fold1 = self._analysis_pair()
        try:
            self.assertEqual(
                set(c1["c1_analysis"]["artifact_validation"]),
                {
                    "valid",
                    "panel_complete",
                    "error_codes",
                    "exact_fold",
                    "case_count",
                    "arm_count_per_case",
                    "static_policy_hash",
                    "forecast_policy_hash",
                    "outcome_firewall_and_receipts_valid",
                    "equal_c_and_rollback_valid",
                },
            )
            first = aggregate_fold01_analyses(c1, fold1)
            second = aggregate_fold01_analyses(c1, fold1)
        finally:
            temporary.cleanup()
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], FOLD01_AGGREGATE_SCHEMA)
        self.assertEqual(
            first["bootstrap_lock"],
            {
                "seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED,
                "oracle_seed": FOLD01_AGGREGATE_BOOTSTRAP_SEED + 1,
                "replicates": FOLD01_AGGREGATE_BOOTSTRAP_REPLICATES,
                "statistical_unit": "24 unique events",
            },
        )
        primary = first["primary_adaptive_minus_frozen_static"]
        self.assertEqual(primary["case_count"], 24)
        self.assertAlmostEqual(primary["mean"], 0.35)
        self.assertAlmostEqual(primary["trimmed_mean_20pct"], 0.35)
        self.assertAlmostEqual(primary["median"], 0.35)
        self.assertEqual(
            first["model_gate_inputs"]["clear_sign_count_threshold"],
            14,
        )
        self.assertEqual(
            first["model_gate_inputs"]["kill_sign_count_upper_bound"],
            12,
        )
        self.assertFalse(
            first["model_gate_inputs"]["pair_level_decision_computed"]
        )
        self.assertFalse(
            first["model_gate_inputs"]["cross_model_decision_computed"]
        )
        self.assertEqual(
            first["locked_identity"]["runtime_seed_evidence"],
            {"fold0": None, "fold1": 17},
        )
        self.assertRegex(first["analysis_hash"], r"^[0-9a-f]{64}$")

    def test_default_c1_shape_is_legacy_and_enriched_input_is_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            canonicalize_c1_fixture_selection(root)
            legacy = analyze_confirmatory_run(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
            )
            enriched = analyze_confirmatory_run_for_fold01(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
            )
        self.assertEqual(legacy["schema_version"], CONFIRMATORY_ANALYSIS_SCHEMA)
        self.assertNotIn("aggregate_identity", legacy)
        self.assertEqual(
            enriched["schema_version"],
            CONFIRMATORY_FOLD01_INPUT_SCHEMA,
        )
        self.assertEqual(enriched["c1_analysis"], legacy)
        self.assertEqual(
            enriched["analysis_hash"],
            _sha256_bytes(
                _canonical_json(
                    {
                        key: value
                        for key, value in enriched.items()
                        if key != "analysis_hash"
                    }
                ).encode("utf-8")
            ),
        )
        blocked = aggregate_fold01_analyses(
            legacy,
            synthetic_fold1_analysis(enriched),
        )
        self.assertEqual(
            blocked["analysis_status"],
            "technical_block_invalid_analysis_json",
        )

    def test_enriched_c1_generation_requires_canonical_fixed_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            with self.assertRaisesRegex(
                ScoreMixAnalysisError,
                "selection seed",
            ):
                analyze_confirmatory_run_for_fold01(
                    root,
                    static_policy=static,
                    forecast_policy=forecast,
                    forecast_analysis=forecast_analysis,
                )
        for kwargs in (
            {"bootstrap_seed": float(FOLLOWUP_BOOTSTRAP_SEED)},
            {
                "bootstrap_replicates": (
                    FOLLOWUP_BOOTSTRAP_REPLICATES + 1
                )
            },
            {"emit_fold01_aggregate_input": 1},
        ):
            arguments = {
                "wave": CONFIRMATORY_ANALYSIS_WAVE_LOCK,
                "static_policy": {},
                "forecast_policy": {},
                "forecast_analysis": {},
                "emit_fold01_aggregate_input": True,
                **kwargs,
            }
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ScoreMixAnalysisError):
                    analyze_score_mix_wave("/not-opened", **arguments)

    def test_invalid_c1_envelope_redacts_untrusted_selection_ids(self):
        secret = "SECRET PROMPT RAW CONTENT"
        with tempfile.TemporaryDirectory() as temporary:
            root, static, forecast, forecast_analysis = build_run(temporary)
            canonicalize_c1_fixture_selection(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["selection"]["case_ids"]["confirmatory"][0] = secret
            write_json(root / "manifest.json", manifest)
            envelope = analyze_confirmatory_run_for_fold01(
                root,
                static_policy=static,
                forecast_policy=forecast,
                forecast_analysis=forecast_analysis,
            )
        self.assertEqual(
            envelope["analysis_status"],
            "technical_block_invalid_artifacts",
        )
        self.assertIsNone(envelope["aggregate_identity"])
        self.assertNotIn(secret, json.dumps(envelope, sort_keys=True))

    def test_aggregate_cli_writes_block_for_malformed_analysis_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c1_path = root / "c1.json"
            fold1_path = root / "fold1.json"
            output_path = root / "aggregate.json"
            c1_path.write_text("{", encoding="utf-8")
            fold1_path.write_text("{}", encoding="utf-8")
            with mock.patch("builtins.print"):
                return_code = followup_analysis_main(
                    [
                        "--c1-analysis",
                        str(c1_path),
                        "--fold1-analysis",
                        str(fold1_path),
                        "--analysis-output",
                        str(output_path),
                    ]
                )
            report = json.loads(output_path.read_text())
        self.assertEqual(return_code, 2)
        self.assertEqual(
            report["analysis_status"],
            "technical_block_invalid_analysis_json",
        )
        self.assertIsNone(report["primary_adaptive_minus_frozen_static"])

    def test_aggregate_replay_expands_to_fold1_envelope(self):
        temporary, c1, _fold1 = self._analysis_pair()
        try:
            fold1 = synthetic_fold1_analysis(c1, replay_envelope=0.5)
            aggregate = aggregate_fold01_analyses(c1, fold1)
        finally:
            temporary.cleanup()
        self.assertEqual(aggregate["replay_envelope"], 0.5)
        self.assertEqual(
            aggregate["primary_adaptive_minus_frozen_static"][
                "positive_sign_count_above_replay_envelope"
            ],
            8,
        )
        self.assertTrue(
            all(
                row["effect_above_replay_envelope"]
                is (row["adaptive_static_effect"] > 0.5)
                for row in aggregate["case_diagnostics"]
            )
        )

    def test_aggregate_14_clear_13_gray_and_12_kill_boundaries(self):
        scenarios = {
            "clear": (
                [1.0] * 14 + [0.0] * 10,
                {
                    "clear_continue_input": True,
                    "gray_input": False,
                    "scientific_kill_input": False,
                },
            ),
            "gray": (
                [0.1] * 13 + [-0.2] * 11,
                {
                    "clear_continue_input": False,
                    "gray_input": True,
                    "scientific_kill_input": False,
                },
            ),
            "kill": (
                [0.03] * 12 + [-0.03] * 12,
                {
                    "clear_continue_input": False,
                    "gray_input": False,
                    "scientific_kill_input": True,
                },
            ),
        }
        for label, (effects, expected) in scenarios.items():
            with self.subTest(label=label):
                temporary, c1, fold1 = self._analysis_pair()
                try:
                    rewrite_analysis_effects(
                        c1["c1_analysis"],
                        effects[:12],
                        oracle_values=[0.0] * 12,
                    )
                    rehash_c1_input(c1)
                    rewrite_analysis_effects(
                        fold1,
                        effects[12:],
                        oracle_values=[0.0] * 12,
                    )
                    report = aggregate_fold01_analyses(c1, fold1)
                finally:
                    temporary.cleanup()
                self.assertEqual(
                    report["analysis_status"],
                    "fold01_aggregate_single_model_complete",
                )
                for field, value in expected.items():
                    self.assertIs(report["model_gate_inputs"][field], value)

    def test_cross_run_identity_disjoint_bootstrap_and_estimand_fail_closed(self):
        temporary, c1, fold1 = self._analysis_pair()
        try:
            mutations = []
            wrong_hash = copy.deepcopy(fold1)
            wrong_hash["artifact_validation"]["static_policy_hash"] = "f" * 64
            mutations.append(wrong_hash)
            overlap = copy.deepcopy(fold1)
            for left, right in zip(
                overlap["case_diagnostics"],
                c1["c1_analysis"]["case_diagnostics"],
            ):
                left["case_id"] = right["case_id"]
            mutations.append(overlap)
            arbitrary_case_id = copy.deepcopy(fold1)
            arbitrary_case_id["case_diagnostics"][0][
                "case_id"
            ] = "arbitrary-not-in-fold1"
            mutations.append(arbitrary_case_id)
            rehashed_arbitrary_case_id = copy.deepcopy(arbitrary_case_id)
            rehashed_arbitrary_case_id["artifact_validation"][
                "selected_case_ids_sha256"
            ] = _sha256_bytes(
                _canonical_json(
                    [
                        row["case_id"]
                        for row in rehashed_arbitrary_case_id[
                            "case_diagnostics"
                        ]
                    ]
                ).encode("utf-8")
            )
            mutations.append(rehashed_arbitrary_case_id)
            wrong_seed = copy.deepcopy(fold1)
            wrong_seed["artifact_validation"]["bootstrap_seed"] += 1
            mutations.append(wrong_seed)
            wrong_estimand = copy.deepcopy(fold1)
            wrong_estimand["artifact_validation"]["primary_estimand"] = "other"
            mutations.append(wrong_estimand)
            wrong_nested_pair = copy.deepcopy(fold1)
            wrong_nested_pair["model_single_run_gate_inputs"][
                "pair_level_decision_computed"
            ] = True
            mutations.append(wrong_nested_pair)
            extra_pair_verdict = copy.deepcopy(fold1)
            extra_pair_verdict["single_model_gate_inputs"][
                "pair_verdict"
            ] = "continue"
            mutations.append(extra_pair_verdict)
            near_residual_mismatch = copy.deepcopy(fold1)
            near_residual_mismatch["calibration_residual_envelope"] += 1e-8
            near_residual_mismatch["single_model_gate_inputs"][
                "architecture_conditional_tolerance_input"
            ] += 1e-8
            near_residual_mismatch["model_single_run_gate_inputs"][
                "forecast_calibration"
            ]["calibration_residual_envelope"] += 1e-8
            mutations.append(near_residual_mismatch)
            near_replay = copy.deepcopy(fold1)
            near_replay["replay_envelope"] += 1e-8
            mutations.append(near_replay)
            near_prediction_error = copy.deepcopy(fold1)
            near_prediction_error["case_diagnostics"][0][
                "prediction_error"
            ] += 1e-8
            mutations.append(near_prediction_error)
            for mutation in mutations:
                with self.subTest(
                    mutation=mutation["artifact_validation"].get(
                        "primary_estimand"
                    )
                ):
                    blocked = aggregate_fold01_analyses(c1, mutation)
                    self.assertEqual(
                        blocked["analysis_status"],
                        "technical_block_invalid_analysis_json",
                    )
                    self.assertIsNone(
                        blocked["primary_adaptive_minus_frozen_static"]
                    )
                    self.assertIsNone(
                        blocked["finite_panel_oracle_opportunity"]
                    )
                    self.assertFalse(
                        blocked["model_gate_inputs"][
                            "pair_level_decision_computed"
                        ]
                    )
        finally:
            temporary.cleanup()

    def test_cross_run_firewall_and_nonfinite_inputs_fail_closed(self):
        temporary, c1, fold1 = self._analysis_pair()
        try:
            complete = aggregate_fold01_analyses(c1, fold1)
            raw = copy.deepcopy(fold1)
            raw["prompt"] = "forbidden"
            blocked_raw = aggregate_fold01_analyses(c1, raw)
            self.assertEqual(
                blocked_raw["analysis_status"],
                "technical_block_invalid_analysis_json",
            )
            self.assertEqual(set(complete), set(blocked_raw))
            self.assertEqual(
                set(complete["model_gate_inputs"]),
                set(blocked_raw["model_gate_inputs"]),
            )

            nonfinite = copy.deepcopy(fold1)
            nonfinite["case_diagnostics"][0][
                "adaptive_static_effect"
            ] = math.nan
            blocked_nonfinite = aggregate_fold01_analyses(c1, nonfinite)
            self.assertEqual(blocked_raw, blocked_nonfinite)

            invalid_residual = copy.deepcopy(fold1)
            invalid_residual["calibration_residual_envelope"] = "not-a-number"
            self.assertEqual(
                aggregate_fold01_analyses(c1, invalid_residual),
                blocked_raw,
            )

            invalid_beta = copy.deepcopy(fold1)
            invalid_beta["artifact_validation"]["forecast_beta"] = "invalid"
            self.assertEqual(
                aggregate_fold01_analyses(c1, invalid_beta),
                blocked_raw,
            )

            malformed_order = copy.deepcopy(fold1)
            malformed_order["artifact_validation"]["outcome_action_order"] = None
            self.assertEqual(
                aggregate_fold01_analyses(c1, malformed_order),
                blocked_raw,
            )
        finally:
            temporary.cleanup()

    def test_all_cross_wave_identities_and_envelope_hash_fail_closed(self):
        temporary, c1, fold1 = self._analysis_pair()
        try:
            invalid_hash = copy.deepcopy(c1)
            invalid_hash["analysis_hash"] = "f" * 64
            expected_block = aggregate_fold01_analyses(
                invalid_hash,
                fold1,
            )
            self.assertEqual(
                expected_block["analysis_status"],
                "technical_block_invalid_analysis_json",
            )

            mutations = []
            replacements = {
                "selection_manifest_id": "1" * 64,
                "fold_manifest_id": "2" * 64,
                "context_id": "3" * 64,
                "provenance_id": "4" * 64,
                "calibration_hash": "5" * 64,
                "forecast_beta": 0.125,
                "static_policy_hash": "6" * 64,
                "forecast_policy_hash": "7" * 64,
                "q": 0.5,
                "outcome_action_order": list(
                    reversed(CONFIRMATORY_OUTCOME_ACTIONS)
                ),
                "primary_estimand": "other",
                "bootstrap_seed": FOLLOWUP_BOOTSTRAP_SEED + 1,
                "bootstrap_replicates": (
                    FOLLOWUP_BOOTSTRAP_REPLICATES + 1
                ),
            }
            for field, value in replacements.items():
                mutation = copy.deepcopy(c1)
                mutation["aggregate_identity"][field] = value
                rehash_c1_input(mutation)
                mutations.append((field, mutation))

            wrong_case_hash = copy.deepcopy(c1)
            wrong_case_hash["aggregate_identity"][
                "selected_case_ids_sha256"
            ] = "8" * 64
            rehash_c1_input(wrong_case_hash)
            mutations.append(("selected_case_ids_sha256", wrong_case_hash))

            wrong_confirmatory_panel = copy.deepcopy(c1)
            wrong_confirmatory_panel["aggregate_identity"][
                "confirmatory_case_ids"
            ][0] = "arbitrary-confirmatory-case"
            rehash_c1_input(wrong_confirmatory_panel)
            mutations.append(
                ("confirmatory_case_ids", wrong_confirmatory_panel)
            )

            nested_spoof = copy.deepcopy(c1)
            nested_spoof["c1_analysis"]["single_model_gate_inputs"][
                "pair_level_decision_computed"
            ] = 0
            rehash_c1_input(nested_spoof)
            mutations.append(("nested_pair_bool", nested_spoof))

            near_replay = copy.deepcopy(c1)
            near_replay["c1_analysis"]["replay_envelope"] += 1e-8
            rehash_c1_input(near_replay)
            mutations.append(("nested_c1_replay", near_replay))

            for field, mutation in mutations:
                with self.subTest(field=field):
                    self.assertEqual(
                        aggregate_fold01_analyses(mutation, fold1),
                        expected_block,
                    )
        finally:
            temporary.cleanup()

    def test_exact_json_types_and_finite_huge_values_fail_closed(self):
        temporary, c1, fold1 = self._analysis_pair()
        try:
            raw = copy.deepcopy(fold1)
            raw["prompt"] = "forbidden"
            expected_block = aggregate_fold01_analyses(c1, raw)
            fold1_mutations = []
            for location, field, value in (
                ("artifact_validation", "case_count", 12.0),
                ("artifact_validation", "arm_count_per_case", 6.0),
                ("artifact_validation", "run_seed", 17.0),
                (
                    "artifact_validation",
                    "bootstrap_seed",
                    float(FOLLOWUP_BOOTSTRAP_SEED),
                ),
                ("execution_lock", "case_count", 12.0),
                ("execution_lock", "run_seed", 17.0),
                (
                    "single_model_gate_inputs",
                    "pair_level_decision_computed",
                    0,
                ),
                ("single_model_gate_inputs", "clear_continue_input", 0),
            ):
                mutation = copy.deepcopy(fold1)
                mutation[location][field] = value
                fold1_mutations.append((f"{location}.{field}", mutation))

            nested_primary = copy.deepcopy(fold1)
            nested_primary["model_single_run_gate_inputs"]["primary"][
                "case_count"
            ] = 12.0
            fold1_mutations.append(("nested_primary.case_count", nested_primary))
            nested_oracle = copy.deepcopy(fold1)
            nested_oracle["model_single_run_gate_inputs"]["oracle"][
                "bootstrap_seed"
            ] = float(FOLLOWUP_BOOTSTRAP_SEED + 1)
            fold1_mutations.append(
                ("nested_oracle.bootstrap_seed", nested_oracle)
            )
            nested_pair = copy.deepcopy(fold1)
            nested_pair["model_single_run_gate_inputs"][
                "pair_level_decision_computed"
            ] = 0
            fold1_mutations.append(("nested_pair_flag", nested_pair))

            huge = copy.deepcopy(fold1)
            for row in huge["case_diagnostics"]:
                row["adaptive_static_effect"] = 1e308
                row["prediction_error"] = 1e308
                row["effect_above_replay_envelope"] = True
            fold1_mutations.append(("finite_huge_effects", huge))

            for label, mutation in fold1_mutations:
                with self.subTest(label=label):
                    self.assertEqual(
                        aggregate_fold01_analyses(c1, mutation),
                        expected_block,
                    )

            for field, value in (
                ("fold", 0.0),
                ("case_count", 12.0),
                ("bootstrap_seed", float(FOLLOWUP_BOOTSTRAP_SEED)),
            ):
                mutation = copy.deepcopy(c1)
                mutation["aggregate_identity"][field] = value
                rehash_c1_input(mutation)
                with self.subTest(c1_identity_field=field):
                    self.assertEqual(
                        aggregate_fold01_analyses(mutation, fold1),
                        expected_block,
                    )
            with self.assertRaises(ScoreMixAnalysisError):
                _effect_summary(
                    [1e308] * 12,
                    expected_case_count=12,
                    replay_envelope=1e-12,
                    bootstrap_seed=FOLLOWUP_BOOTSTRAP_SEED,
                    bootstrap_replicates=100,
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
