import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    canonical_json,
    sha256_bytes,
)
from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    CounterFactSelectionManifest,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    SanitizedJsonlWriter,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import (
    ACTION_UNIFORM,
    _feature_hash,
    build_unit_c_actions,
    proposal_c_energy,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory import (
    BUNDLE_ID,
    CALIBRATION_METHOD,
    CONFIRMATORY_CASE_COUNT,
    CONFIRMATORY_EVENT_FIELDS,
    CONFIRMATORY_FOLD,
    CONFIRMATORY_FOLD_COUNT,
    CONFIRMATORY_FOLD_SEED,
    CONFIRMATORY_JOB_NAME,
    CONFIRMATORY_MANIFEST_FIELDS,
    CONFIRMATORY_MANIFEST_SCHEMA,
    CONFIRMATORY_OUTCOME_ACTIONS,
    CONFIRMATORY_POLICY_PATHS,
    CONFIRMATORY_RECEIPT_SCHEMA,
    CONFIRMATORY_RUN_IDS,
    CONFIRMATORY_STREAM_SCHEMA,
    CONFIRMATORY_SUMMARY_FIELDS,
    CONFIRMATORY_SUMMARY_SCHEMA,
    FORECAST_POLICY_SCHEMA,
    FROZEN_STATIC_ACTION,
    LAYER_ACTIONS,
    SCORE_MIX_ACTION,
    SCORE_MIX_Q,
    SOURCE_RUN_IDS,
    STATIC_POLICY_SCHEMA,
    ForecastPolicyLock,
    StaticPolicyLock,
    _confirmatory_slurm_state,
    _execution_envelope,
    _run_confirmatory_event,
    _validate_execution_mode,
    build_frozen_static_direction,
    build_parser,
    commit_confirmatory_action,
    compute_forecast_terms,
    load_forecast_policy,
    load_fixed_model,
    load_static_policy,
    run_confirmatory_event_loop,
    select_confirmatory_fold_cases,
    validate_forecast_policy,
    validate_static_policy,
)


def with_policy_hash(payload):
    result = dict(payload)
    result["policy_hash"] = sha256_bytes(
        canonical_json(result).encode("utf-8")
    )
    return result


def static_policy_payload(model_alias="llama3-8b-inst"):
    weights = {
        "layer_4": 0.6,
        "layer_5": 0.8,
        "layer_6": 0.0,
        "layer_7": 0.0,
        "layer_8": 0.0,
    }
    return with_policy_hash(
        {
            "schema_version": STATIC_POLICY_SCHEMA,
            "model_alias": model_alias,
            "source_runs": SOURCE_RUN_IDS[model_alias],
            "source_slices": {
                "d0": {"start": 3, "count": 5},
                "d1": {"start": 8, "count": 12},
            },
            "feature_case_count": 17,
            "feature_panel_sha256": "a" * 64,
            "fit_numeric_inputs": [
                f"action_scores.{action}" for action in LAYER_ACTIONS
            ],
            "outcome_fields_used": [],
            "sbar_single_layer_slopes": {
                "layer_4": 3.0,
                "layer_5": 4.0,
                "layer_6": -2.0,
                "layer_7": 0.0,
                "layer_8": -1.0,
            },
            "layer_weights": weights,
            "predicted_mean_score": 5.0,
            "controller": "relu(sbar)/L2-else-max-onehot",
            "controller_branch": "positive-relu-l2",
            "tie_policy": "exact-tie-lower-layer",
            "weight_l2_norm": 1.0,
            "claim_boundary": "calibration-only frozen comparator",
        }
    )


def forecast_policy_payload(static_hash, model_alias="llama3-8b-inst"):
    return with_policy_hash(
        {
            "schema_version": FORECAST_POLICY_SCHEMA,
            "model_alias": model_alias,
            "static_policy_hash": static_hash,
            "source_runs": SOURCE_RUN_IDS[model_alias],
            "beta": 0.75,
            "calibration_method": CALIBRATION_METHOD,
            "calibration_hash": "b" * 64,
            "confirmatory_outcomes_used": [],
        }
    )


class _FiveLayerToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for layer in range(4, 9):
            setattr(self, f"layer{layer}", torch.nn.Linear(2, 2, bias=False))


class _MemoryWriter:
    def __init__(self):
        self.rows = []

    def write(self, event, payload):
        self.rows.append((event, dict(payload)))


class ConfirmatoryPolicyTests(unittest.TestCase):
    def test_static_policy_exact_contract_and_fail_closed_mutations(self):
        payload = static_policy_payload()
        lock = validate_static_policy(
            payload,
            model_alias="llama3-8b-inst",
        )
        self.assertEqual(lock.weights, (0.6, 0.8, 0.0, 0.0, 0.0))
        self.assertEqual(lock.policy_hash, payload["policy_hash"])
        self.assertEqual(lock.source_runs, SOURCE_RUN_IDS["llama3-8b-inst"])

        mutations = []
        wrong_schema = dict(payload)
        wrong_schema["schema_version"] = "wrong"
        mutations.append(wrong_schema)
        wrong_model = dict(payload)
        wrong_model["model_alias"] = "qwen2.5-7b-inst"
        mutations.append(wrong_model)
        wrong_runs = dict(payload)
        wrong_runs["source_runs"] = {
            "d0": "other",
            "d1": SOURCE_RUN_IDS["llama3-8b-inst"]["d1"],
        }
        mutations.append(wrong_runs)
        outcomes_used = dict(payload)
        outcomes_used["outcome_fields_used"] = ["progress"]
        mutations.append(outcomes_used)
        wrong_weights = json.loads(json.dumps(payload))
        wrong_weights["layer_weights"]["layer_4"] = 0.7
        mutations.append(wrong_weights)
        forged_weights = json.loads(json.dumps(payload))
        forged_weights["layer_weights"] = {
            "layer_4": 1.0,
            "layer_5": 0.0,
            "layer_6": 0.0,
            "layer_7": 0.0,
            "layer_8": 0.0,
        }
        forged_weights["predicted_mean_score"] = 3.0
        forged_weights = with_policy_hash(
            {
                key: value
                for key, value in forged_weights.items()
                if key != "policy_hash"
            }
        )
        mutations.append(forged_weights)
        wrong_norm = dict(payload)
        wrong_norm["weight_l2_norm"] = 0.9
        mutations.append(wrong_norm)
        wrong_hash = dict(payload)
        wrong_hash["policy_hash"] = "0" * 64
        mutations.append(wrong_hash)
        extra_field = dict(payload)
        extra_field["extra"] = True
        mutations.append(extra_field)

        for mutation in mutations:
            with self.subTest(keys=set(mutation)):
                with self.assertRaises(ContractError):
                    validate_static_policy(
                        mutation,
                        model_alias="llama3-8b-inst",
                    )

    def test_forecast_policy_linkage_beta_and_outcome_firewall(self):
        static = static_policy_payload()
        payload = forecast_policy_payload(static["policy_hash"])
        lock = validate_forecast_policy(
            payload,
            model_alias="llama3-8b-inst",
            static_policy_hash=static["policy_hash"],
        )
        self.assertEqual(lock.beta, 0.75)
        self.assertEqual(lock.static_policy_hash, static["policy_hash"])
        self.assertEqual(lock.policy_hash, payload["policy_hash"])

        negative = dict(payload)
        negative["beta"] = -0.1
        negative = with_policy_hash(
            {key: value for key, value in negative.items() if key != "policy_hash"}
        )
        with self.assertRaises(ContractError):
            validate_forecast_policy(
                negative,
                model_alias="llama3-8b-inst",
                static_policy_hash=static["policy_hash"],
            )

        wrong_method = dict(payload)
        wrong_method["calibration_method"] = "origin-through-zero-least-squares"
        wrong_method = with_policy_hash(
            {
                key: value
                for key, value in wrong_method.items()
                if key != "policy_hash"
            }
        )
        with self.assertRaises(ContractError):
            validate_forecast_policy(
                wrong_method,
                model_alias="llama3-8b-inst",
                static_policy_hash=static["policy_hash"],
            )

        used = dict(payload)
        used["confirmatory_outcomes_used"] = ["case-1"]
        used = with_policy_hash(
            {key: value for key, value in used.items() if key != "policy_hash"}
        )
        with self.assertRaises(ContractError):
            validate_forecast_policy(
                used,
                model_alias="llama3-8b-inst",
                static_policy_hash=static["policy_hash"],
            )

        with self.assertRaises(ContractError):
            validate_forecast_policy(
                payload,
                model_alias="llama3-8b-inst",
                static_policy_hash="c" * 64,
            )

        poisoned = dict(payload)
        poisoned["policy_hash"] = "0" * 64
        with self.assertRaises(ContractError):
            validate_forecast_policy(
                poisoned,
                model_alias="llama3-8b-inst",
                static_policy_hash=static["policy_hash"],
            )

    def test_strict_policy_file_loading_rejects_duplicates_and_nonfinite(self):
        static = static_policy_payload()
        forecast = forecast_policy_payload(static["policy_hash"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            static_path = root / "static.json"
            forecast_path = root / "forecast.json"
            static_path.write_text(
                json.dumps(static, sort_keys=True),
                encoding="utf-8",
            )
            forecast_path.write_text(
                json.dumps(forecast, sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(
                load_static_policy(
                    static_path,
                    model_alias="llama3-8b-inst",
                ).policy_hash,
                static["policy_hash"],
            )
            self.assertEqual(
                load_forecast_policy(
                    forecast_path,
                    model_alias="llama3-8b-inst",
                    static_policy_hash=static["policy_hash"],
                ).policy_hash,
                forecast["policy_hash"],
            )

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":"x","schema_version":"y"}')
            with self.assertRaises(ContractError):
                load_static_policy(
                    duplicate,
                    model_alias="llama3-8b-inst",
                )

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}')
            with self.assertRaises(ContractError):
                load_static_policy(
                    nonfinite,
                    model_alias="llama3-8b-inst",
                )


class ConfirmatoryFoldEnvelopeTests(unittest.TestCase):
    @staticmethod
    def selection(confirmatory=None, *, seed=DEFAULT_SELECTION_SEED):
        return CounterFactSelectionManifest(
            seed=seed,
            source_sha256=ExpectedFileIdentity(
                sha256="d" * 64,
                size=1,
            ).sha256,
            source_size=1,
            source_row_count=100,
            calibration=tuple(f"c{index}" for index in range(20)),
            confirmatory=(
                tuple(f"x{index}" for index in range(60))
                if confirmatory is None
                else tuple(confirmatory)
            ),
            untouched=tuple(f"u{index}" for index in range(20)),
        )

    def test_fixed_confirmatory_fold_zero_is_balanced_and_deterministic(self):
        selection = self.selection()
        selected, folds = select_confirmatory_fold_cases(selection)
        again, repeated = select_confirmatory_fold_cases(selection)

        self.assertEqual(selected, again)
        self.assertEqual(folds.manifest_id, repeated.manifest_id)
        self.assertEqual(len(selected), CONFIRMATORY_CASE_COUNT)
        self.assertEqual(len(set(selected)), CONFIRMATORY_CASE_COUNT)
        self.assertEqual(folds.seed, CONFIRMATORY_FOLD_SEED)
        self.assertEqual(folds.fold_count, CONFIRMATORY_FOLD_COUNT)
        self.assertTrue(
            all(folds.fold_for(case_id) == CONFIRMATORY_FOLD for case_id in selected)
        )
        self.assertEqual(
            selected,
            tuple(
                case_id
                for case_id in selection.confirmatory
                if folds.fold_for(case_id) == CONFIRMATORY_FOLD
            ),
        )

        reversed_selection = self.selection(reversed(selection.confirmatory))
        reversed_selected, reversed_folds = select_confirmatory_fold_cases(
            reversed_selection
        )
        self.assertEqual(folds.manifest_id, reversed_folds.manifest_id)
        self.assertEqual(set(selected), set(reversed_selected))

    def test_fold_rejects_nonfixed_split(self):
        selection = self.selection(confirmatory=tuple(f"x{i}" for i in range(59)))
        with self.assertRaises(ContractError):
            select_confirmatory_fold_cases(selection)

    def test_fold_rejects_noncanonical_selection_seed(self):
        selection = self.selection(seed="different-outcome-blind-seed")
        with self.assertRaises(ContractError):
            select_confirmatory_fold_cases(selection)

    def test_production_rejects_injected_seams_and_test_mode_requires_both(self):
        def injected_loader(_model_alias):
            return object()

        def injected_event(**_kwargs):
            return {}

        self.assertTrue(
            _validate_execution_mode(
                slurm_state={"under_slurm": True},
                model_loader=load_fixed_model,
                event_runner=_run_confirmatory_event,
            )
        )
        for loader, runner in (
            (injected_loader, _run_confirmatory_event),
            (load_fixed_model, injected_event),
            (injected_loader, injected_event),
        ):
            with self.subTest(loader=loader, runner=runner):
                with self.assertRaises(Exception):
                    _validate_execution_mode(
                        slurm_state={"under_slurm": True},
                        model_loader=loader,
                        event_runner=runner,
                    )
        self.assertFalse(
            _validate_execution_mode(
                slurm_state={"under_slurm": False},
                model_loader=injected_loader,
                event_runner=injected_event,
            )
        )
        for loader, runner in (
            (load_fixed_model, injected_event),
            (injected_loader, _run_confirmatory_event),
        ):
            with self.subTest(test_loader=loader, test_runner=runner):
                with self.assertRaises(Exception):
                    _validate_execution_mode(
                        slurm_state={"under_slurm": False},
                        model_loader=loader,
                        event_runner=runner,
                    )

    def test_exact_run_slurm_cli_and_schema_envelopes(self):
        for model_alias, run_id in CONFIRMATORY_RUN_IDS.items():
            self.assertEqual(
                _execution_envelope(model_alias, run_id),
                "c1",
            )
        with self.assertRaises(Exception):
            _execution_envelope(
                "llama3-8b-inst",
                CONFIRMATORY_RUN_IDS["qwen2.5-7b-inst"],
            )

        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "40001",
                "SLURM_JOB_NAME": CONFIRMATORY_JOB_NAME,
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            state = _confirmatory_slurm_state(
                "llama3-8b-inst",
                CONFIRMATORY_RUN_IDS["llama3-8b-inst"],
            )
            self.assertTrue(state["under_slurm"])
            self.assertEqual(state["fold"], 0)

        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "40001",
                "SLURM_JOB_NAME": "wrong",
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            with self.assertRaises(Exception):
                _confirmatory_slurm_state(
                    "llama3-8b-inst",
                    CONFIRMATORY_RUN_IDS["llama3-8b-inst"],
                )

        parsed = build_parser().parse_args(
            [
                "--easyedit-root",
                "/tmp/easyedit",
                "--model",
                "llama3-8b-inst",
                "--run-id",
                CONFIRMATORY_RUN_IDS["llama3-8b-inst"],
                "--static-policy",
                "/tmp/static.json",
                "--forecast-policy",
                "/tmp/forecast.json",
            ]
        )
        self.assertEqual(parsed.static_policy, Path("/tmp/static.json"))
        self.assertEqual(parsed.forecast_policy, Path("/tmp/forecast.json"))
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "--easyedit-root",
                        "/tmp/easyedit",
                        "--model",
                        "llama3-8b-inst",
                        "--run-id",
                        CONFIRMATORY_RUN_IDS["llama3-8b-inst"],
                        "--static-policy",
                        "/tmp/static.json",
                        "--forecast-policy",
                        "/tmp/forecast.json",
                        "--selection-seed",
                        "forbidden",
                    ]
                )
        self.assertEqual(
            CONFIRMATORY_POLICY_PATHS,
            {
                "llama3-8b-inst": {
                    "static": (
                        "experiment-reports/global/"
                        "2026-07-31-mv1mix-llama-d1-v1-static-policy.json"
                    ),
                    "forecast": (
                        "experiment-reports/global/"
                        "2026-07-31-mv1mix-llama-d0-d1-forecast-v1.policy.json"
                    ),
                },
                "qwen2.5-7b-inst": {
                    "static": (
                        "experiment-reports/global/"
                        "2026-07-31-mv1mix-qwen-d1-v1-static-policy.json"
                    ),
                    "forecast": (
                        "experiment-reports/global/"
                        "2026-07-31-mv1mix-qwen-d0-d1-forecast-v1.policy.json"
                    ),
                },
            },
        )
        self.assertEqual(
            CONFIRMATORY_OUTCOME_ACTIONS,
            (
                "score_mix",
                "frozen_static_mix",
                "uniform",
                "ordered_global_alpha",
                "native_memit_full",
                "no_op_replay",
            ),
        )
        self.assertEqual(
            (
                CONFIRMATORY_MANIFEST_SCHEMA,
                CONFIRMATORY_STREAM_SCHEMA,
                CONFIRMATORY_RECEIPT_SCHEMA,
                CONFIRMATORY_SUMMARY_SCHEMA,
            ),
            (
                "ode-edit-mv1-score-mix-confirmatory-manifest/v1",
                "ode-edit-mv1-score-mix-confirmatory/v1",
                "ode-edit-mv1-score-mix-confirmatory-receipt/v1",
                "ode-edit-mv1-score-mix-confirmatory-summary/v1",
            ),
        )
        self.assertEqual(
            CONFIRMATORY_MANIFEST_FIELDS,
            frozenset(
                {
                    "schema_version",
                    "run_id",
                    "ode_edit_git",
                    "slurm",
                    "model",
                    "hparams_relative_path",
                    "selection",
                    "selected_split",
                    "confirmatory_folds",
                    "selected_fold",
                    "selected_case_ids",
                    "selected_request_ids",
                    "contexts",
                    "provenance_id",
                    "fixed_files",
                    "q",
                    "q_label",
                    "probe_ratio",
                    "probe_action_set",
                    "controller_input_set",
                    "outcome_action_set",
                    "static_policy",
                    "forecast_policy",
                    "decision_policy",
                    "forecast_policy_formula",
                    "action_receipt_policy",
                    "utility_policy",
                    "teacher_suffix_policy",
                    "projector_policy",
                    "covariance_policy",
                    "direct_z_policy",
                    "artifact_firewall",
                    "expected_counts",
                }
            ),
        )
        self.assertEqual(len(CONFIRMATORY_SUMMARY_FIELDS), 35)
        self.assertIn("projector_files_loaded", CONFIRMATORY_SUMMARY_FIELDS)
        self.assertIn("expected_counts_exact", CONFIRMATORY_SUMMARY_FIELDS)
        self.assertIn("stream_sequences", CONFIRMATORY_SUMMARY_FIELDS)
        self.assertEqual(
            CONFIRMATORY_EVENT_FIELDS,
            frozenset(
                {
                    "case_id",
                    "request_id",
                    "event_seed",
                    "target_token_count",
                    "native_c_energy",
                    "q",
                    "probe_direction_count",
                    "feature_count",
                    "commitment_count",
                    "outcome_count",
                    "direct_z_artifact_sha256",
                    "direct_z_artifact_size",
                    "static_policy_hash",
                    "forecast_policy_hash",
                    "action_receipt",
                    "rollback_exact",
                    "pass",
                }
            ),
        )


class ConfirmatoryActionTests(unittest.TestCase):
    def setUp(self):
        self.layers = tuple(range(4, 9))
        self.model = _FiveLayerToy().eval()
        self.request = EditRequest.from_mapping(
            {
                "case_id": "confirmatory-case",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        contexts = ContextManifest.freeze([["{}"]], source="confirmatory-test")
        weight_names = tuple(f"layer{layer}.weight" for layer in self.layers)
        snapshot = capture_snapshot(
            self.model,
            model_id="toy",
            requests=(self.request,),
            context_id=contexts.manifest_id,
            hparams={"layers": list(self.layers)},
            weight_names=weight_names,
        )
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.tensor([[1.0], [0.0]]),
                right=torch.tensor([[1.0], [0.0]]),
                expected_weight_sha256=snapshot.parameter(name).sha256,
            )
            for name in weight_names
        )
        self.proposal = MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="confirmatory-test",
            residual_denominator=5,
        )
        self.covariances = {layer: torch.eye(2) for layer in self.layers}
        self.layer_by_weight = {
            name: layer for name, layer in zip(weight_names, self.layers)
        }
        self.static = validate_static_policy(
            static_policy_payload(),
            model_alias="llama3-8b-inst",
        )
        self.forecast = validate_forecast_policy(
            forecast_policy_payload(self.static.policy_hash),
            model_alias="llama3-8b-inst",
            static_policy_hash=self.static.policy_hash,
        )

    def test_frozen_static_direction_reuses_unit_c_actions(self):
        unit_actions = build_unit_c_actions(
            self.proposal,
            self.covariances,
            self.layer_by_weight,
        )
        static = build_frozen_static_direction(
            synchronous=self.proposal,
            unit_actions=unit_actions,
            policy=self.static,
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
        )
        self.assertEqual(static.action_id, FROZEN_STATIC_ACTION)
        self.assertAlmostEqual(static.c_squared_norm, 1.0, places=6)
        self.assertEqual(
            tuple(factor.weight_name for factor in static.proposal.factors),
            ("layer4.weight", "layer5.weight"),
        )
        self.assertAlmostEqual(
            proposal_c_energy(
                static.proposal,
                self.covariances,
                self.layer_by_weight,
            ),
            1.0,
            places=6,
        )

    def test_forecast_known_answer_and_nonnegative_beta_can_predict_negative(self):
        terms = compute_forecast_terms(
            single_layer_scores={
                "layer_4": 3.0,
                "layer_5": 4.0,
                "layer_6": -2.0,
                "layer_7": 0.0,
                "layer_8": -1.0,
            },
            adaptive_weights={
                "layer_4": 1.0,
                "layer_5": 0.0,
                "layer_6": 0.0,
                "layer_7": 0.0,
                "layer_8": 0.0,
            },
            static_weights=self.static.layer_weights,
            operational_c_distance=0.5,
            beta=2.0,
        )
        self.assertAlmostEqual(terms.adaptive_score, 3.0)
        self.assertAlmostEqual(terms.static_score, 5.0)
        self.assertAlmostEqual(terms.predicted_score_gap, -2.0)
        self.assertAlmostEqual(terms.x_as, -1.0)
        self.assertAlmostEqual(terms.predicted_gain, -2.0)

        with self.assertRaises(ContractError):
            compute_forecast_terms(
                single_layer_scores={
                    action: 1.0 for action in LAYER_ACTIONS
                },
                adaptive_weights=self.static.layer_weights,
                static_weights=self.static.layer_weights,
                operational_c_distance=1.0,
                beta=-1.0,
            )

    def valid_feature_action(self):
        adaptive_weights = {
            "layer_4": 1.0,
            "layer_5": 0.0,
            "layer_6": 0.0,
            "layer_7": 0.0,
            "layer_8": 0.0,
        }
        scores = {
            **{action: float(index) for index, action in enumerate(LAYER_ACTIONS)},
            ACTION_UNIFORM: 0.25,
        }
        feature = {
            "case_id": self.request.case_id,
            "request_id": self.request.request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "native_c_energy": 64.0,
            "operational_c_distance": 0.5,
            "probe_c_distance": 0.125,
            "action_scores": scores,
            "context_action_scores": {
                key: [value] for key, value in scores.items()
            },
            "feature_policy": "synthetic-central-fd",
            "static_policy_hash": self.static.policy_hash,
            "forecast_policy_hash": self.forecast.policy_hash,
            "adaptive_layer_weights": adaptive_weights,
            "static_layer_weights": self.static.layer_weights,
            "adaptive_predicted_score": 0.0,
            "static_predicted_score": 0.8,
            "predicted_adaptive_static_score_gap": -0.8,
            "x_as": -0.4,
            "forecast_beta": self.forecast.beta,
            "predicted_adaptive_static_gain": -0.3,
        }
        feature["feature_hash"] = _feature_hash(feature)
        action = {
            "case_id": self.request.case_id,
            "request_id": self.request.request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "feature_hash": feature["feature_hash"],
            "bundle_id": BUNDLE_ID,
            "adaptive_action_id": SCORE_MIX_ACTION,
            "adaptive_layer_weights": adaptive_weights,
            "adaptive_predicted_score": 0.0,
            "adaptive_actual_unit_c_energy": 1.0,
            "adaptive_controller": "five-single-slopes/relu-l2-else-max-onehot",
            "adaptive_controller_branch": "positive-relu-l2",
            "static_action_id": FROZEN_STATIC_ACTION,
            "static_layer_weights": self.static.layer_weights,
            "static_predicted_score": 0.8,
            "static_actual_unit_c_energy": 1.0,
            "static_policy_hash": self.static.policy_hash,
            "forecast_policy_hash": self.forecast.policy_hash,
            "predicted_adaptive_static_score_gap": -0.8,
            "x_as": -0.4,
            "forecast_beta": self.forecast.beta,
            "predicted_adaptive_static_gain": -0.3,
            "tie_policy": "exact-tie-lower-layer",
        }
        action["commitment_hash"] = _feature_hash(action)
        return feature, action

    def test_adaptive_static_forecast_commitment_is_durable_and_outcome_free(self):
        feature, action = self.valid_feature_action()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "confirmatory-test",
                    schema_version=CONFIRMATORY_STREAM_SCHEMA,
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "confirmatory-test",
                    schema_version=CONFIRMATORY_STREAM_SCHEMA,
                ) as action_writer,
            ):
                receipt_name, receipt_hash = commit_confirmatory_action(
                    feature_writer=feature_writer,
                    action_writer=action_writer,
                    receipt_root=receipts,
                    feature=feature,
                    action=action,
                )

            self.assertRegex(receipt_hash, r"^[0-9a-f]{64}$")
            receipt = json.loads((receipts / receipt_name).read_text())
            self.assertEqual(
                receipt["durability"],
                "feature+adaptive-static-forecast-action-write+flush+fsync-"
                "before-exclusive-receipt",
            )
            self.assertFalse(receipt["outcomes_observed_before_commitment"])
            self.assertEqual(
                receipt["adaptive_layer_weights"],
                feature["adaptive_layer_weights"],
            )
            self.assertEqual(
                receipt["static_layer_weights"],
                feature["static_layer_weights"],
            )
            self.assertEqual(
                receipt["static_policy_hash"],
                self.static.policy_hash,
            )
            self.assertEqual(
                receipt["forecast_policy_hash"],
                self.forecast.policy_hash,
            )
            persisted = (
                (root / "features.jsonl").read_text()
                + (root / "actions.jsonl").read_text()
                + (receipts / receipt_name).read_text()
            )
            for forbidden in (
                '"prompt"',
                '"subject"',
                '"target_new"',
                '"progress"',
                '"nll_reduction"',
                '"confirmatory_outcomes_used"',
            ):
                self.assertNotIn(forbidden, persisted)

        poisoned = dict(action)
        poisoned["progress"] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("receipts").mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "confirmatory-test",
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "confirmatory-test",
                ) as action_writer,
            ):
                with self.assertRaises(ContractError):
                    commit_confirmatory_action(
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        receipt_root=root / "receipts",
                        feature=feature,
                        action=poisoned,
                    )


class ConfirmatoryInjectedRunnerTests(unittest.TestCase):
    STATIC_POLICY_HASH = "a" * 64
    FORECAST_POLICY_HASH = "b" * 64

    @staticmethod
    def request(case_id):
        return EditRequest.from_mapping(
            {
                "case_id": case_id,
                "prompt": "{} lives in",
                "subject": f"subject-{case_id}",
                "target_new": "target",
            }
        )

    @classmethod
    def valid_result(cls, request):
        return {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "event_seed": 1,
            "target_token_count": 1,
            "native_c_energy": 16.0,
            "q": SCORE_MIX_Q,
            "probe_direction_count": 6,
            "feature_count": 1,
            "commitment_count": 1,
            "outcome_count": len(CONFIRMATORY_OUTCOME_ACTIONS),
            "direct_z_artifact_sha256": "c" * 64,
            "direct_z_artifact_size": 128,
            "static_policy_hash": cls.STATIC_POLICY_HASH,
            "forecast_policy_hash": cls.FORECAST_POLICY_HASH,
            "action_receipt": {
                "name": f"{'d' * 64}.json",
                "sha256": "e" * 64,
            },
            "rollback_exact": True,
            "pass": True,
        }

    def test_injected_event_runner_contract_and_exact_event_count(self):
        requests = (self.request("one"), self.request("two"))
        writer = _MemoryWriter()
        observed = []

        def injected(*, request, marker):
            observed.append((request.case_id, marker))
            return self.valid_result(request)

        results, abort = run_confirmatory_event_loop(
            requests=requests,
            event_runner=injected,
            event_writer=writer,
            event_kwargs={"marker": "injected"},
            expected_static_policy_hash=self.STATIC_POLICY_HASH,
            expected_forecast_policy_hash=self.FORECAST_POLICY_HASH,
        )
        self.assertIsNone(abort)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            observed,
            [("one", "injected"), ("two", "injected")],
        )
        self.assertEqual(
            [event for event, _payload in writer.rows],
            ["mv1mix_confirmatory_case", "mv1mix_confirmatory_case"],
        )

    def test_injected_runner_schema_violation_aborts_and_stops(self):
        requests = (self.request("one"), self.request("two"))
        writer = _MemoryWriter()
        calls = []

        def invalid(*, request):
            calls.append(request.case_id)
            return {
                "case_id": request.case_id,
                "request_id": request.request_id,
                "feature_count": 1,
                "commitment_count": 1,
                "outcome_count": 5,
                "rollback_exact": True,
                "pass": True,
            }

        results, abort = run_confirmatory_event_loop(
            requests=requests,
            event_runner=invalid,
            event_writer=writer,
            event_kwargs={},
            expected_static_policy_hash=self.STATIC_POLICY_HASH,
            expected_forecast_policy_hash=self.FORECAST_POLICY_HASH,
        )
        self.assertEqual(abort, "ContractError")
        self.assertEqual(calls, ["one"])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["pass"])
        self.assertFalse(results[0]["rollback_exact"])

    def test_injected_runner_wrong_policy_hash_aborts(self):
        request = self.request("one")
        writer = _MemoryWriter()

        def poisoned(*, request):
            result = self.valid_result(request)
            result["static_policy_hash"] = "f" * 64
            return result

        results, abort = run_confirmatory_event_loop(
            requests=(request,),
            event_runner=poisoned,
            event_writer=writer,
            event_kwargs={},
            expected_static_policy_hash=self.STATIC_POLICY_HASH,
            expected_forecast_policy_hash=self.FORECAST_POLICY_HASH,
        )
        self.assertEqual(abort, "ContractError")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["pass"])


if __name__ == "__main__":
    unittest.main()
