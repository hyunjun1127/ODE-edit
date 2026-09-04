from __future__ import annotations

import copy
import json
import math
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from project.run_scripts.ordered_response_barrier_ode import artifacts


def _row(
    case_id: int,
    kind: str,
    prompt_index: int,
    *,
    nll: float,
    predictions: list[int],
    target_ids: list[int],
) -> dict[str, Any]:
    correct = [left == right for left, right in zip(predictions, target_ids, strict=True)]
    prompt_family = kind.removesuffix("_target_new").removesuffix("_target_true")
    target_family = "new" if kind.endswith("_target_new") else "true"
    return {
        "case_id": case_id,
        "kind": kind,
        "prompt_index": prompt_index,
        "prompt": f"SECRET_PROMPT_{case_id}_{prompt_family}_{prompt_index}",
        "target": f"SECRET_TARGET_{case_id}_{target_family}",
        "target_token_ids": target_ids,
        "nll": nll,
        "token_predictions": predictions,
        "token_correct": correct,
        "all_tokens_correct": all(correct),
    }


def _evaluation(case_ids: Sequence[int], *, entry: bool) -> dict[str, list[dict[str, Any]]]:
    result = {kind: [] for kind in artifacts.EVALUATION_KINDS}
    for case_id in case_ids:
        result["rewrite_target_new"].append(
            _row(
                case_id,
                "rewrite_target_new",
                0,
                nll=1.0 if entry else 0.25,
                predictions=[101, 102],
                target_ids=[101, 102],
            )
        )
        result["rewrite_target_true"].append(
            _row(
                case_id,
                "rewrite_target_true",
                0,
                nll=0.5 if entry else 1.25,
                predictions=[201, 0],
                target_ids=[201, 202],
            )
        )
        for prompt_index in range(2):
            result["rephrase_target_new"].append(
                _row(
                    case_id,
                    "rephrase_target_new",
                    prompt_index,
                    nll=1.5 if entry else 0.5 + 0.1 * prompt_index,
                    predictions=[101, 102],
                    target_ids=[101, 102],
                )
            )
            result["rephrase_target_true"].append(
                _row(
                    case_id,
                    "rephrase_target_true",
                    prompt_index,
                    nll=0.5 if entry else 1.5 + 0.1 * prompt_index,
                    predictions=[201, 0],
                    target_ids=[201, 202],
                )
            )
        for prompt_index in range(10):
            target = [201, 202]
            prediction = target if entry or prompt_index != 9 else [201, 0]
            result["locality_target_true"].append(
                _row(
                    case_id,
                    "locality_target_true",
                    prompt_index,
                    nll=0.1 if entry else 0.2,
                    predictions=prediction,
                    target_ids=target,
                )
            )
    return result


def _evaluation_with_canonical_ns(
    case_ids: Sequence[int], *, entry: bool
) -> dict[str, list[dict[str, Any]]]:
    result = _evaluation(case_ids, entry=entry)
    result["locality_target_new"] = [
        _row(
            case_id,
            "locality_target_new",
            prompt_index,
            nll=0.3,
            predictions=[101, 102],
            target_ids=[101, 102],
        )
        for case_id in case_ids
        for prompt_index in range(10)
    ]
    return result


def _request_shas(case_ids: Sequence[int]) -> list[str]:
    return [artifacts.canonical_hash({"case_id": value}) for value in case_ids]


def _semantic(request_count: int = 100) -> dict[str, Any]:
    return {
        "all_strict": False,
        "strict_event_count": request_count,
        "event_count": request_count * 6,
        "request_strict": [False] * request_count,
        "request_strict_count": 0,
        "target_logit_mean": 1.0,
        "target_logit_min": 0.5,
        "maximum_other_logit_mean": 1.5,
        "strict_tie_count": 0,
    }


def _step(arm: str, request_count: int = 100) -> dict[str, Any]:
    optional_values = [0.25] * request_count
    return {
        "arm": arm,
        "sweep": 0,
        "layer": 4,
        "visit_ordinal": 0,
        "built_state_version": 0,
        "resulting_state_version": 1,
        "residual_denominator": 5,
        "command_reference_state_version": 0,
        "build_identity": "5" * 64,
        "residual_sha256": "6" * 64,
        "keys_sha256": "7" * 64,
        "solver_identity": "8" * 64,
        "factor_rank": 100,
        "coefficient_u": 0.5,
        "euler_alpha": 0.125,
        "g": 1.0,
        "r": 2.0,
        "per_request_g": list(optional_values),
        "per_request_r": list(optional_values),
        "potential_before": 2.0,
        "potential_after": 1.5,
        "predicted_reduction": 0.5,
        "actual_reduction": 0.5,
        "discretization_defect": 0.0,
        "entry_sublevel_violation": 0.0,
        "residual_norm_mean_before": 1.0,
        "residual_norm_mean_after": 0.8,
        "response_norm_mean": 0.4,
        "command_norm_mean": 0.2,
        "writer_command_norm_mean": 0.2,
        "predicted_transition_norm_mean": 0.1,
        "actual_transition_norm_mean": 0.1,
        "realization_error_norm_mean": 0.01,
        "model_error_norm_mean": 0.01,
        "per_request_D_R0": list(optional_values),
        "per_request_D_model_R0": list(optional_values),
        "per_request_q_res": list(optional_values),
        "per_request_writer_command_R0_squared": list(optional_values),
        "per_request_response_orthogonal_R0_squared": list(optional_values),
        "per_request_zero_command_cross_response_R0_squared": [None] * request_count,
        "per_request_actual_potential_change": [-0.1] * request_count,
        "per_request_actual_potential_worsened": [False] * request_count,
        "per_request_metric_status": ["DEFINED"] * request_count,
        "direction_frobenius": 2.0,
        "direction_frobenius_squared": 4.0,
        "applied_update_frobenius": 0.25,
        "applied_update_frobenius_squared": 0.0625,
        "resolution_stable_path_increment_frobenius_squared": 0.125,
        "action_geometry_status": "FROBENIUS_ONLY_CREG_NOT_BOUND",
        "native_creg_action_status": "TELEMETRY_WITHHELD",
        "semantic_all_strict": False,
        "semantic_request_count": request_count,
        "semantic_strict_event_count": request_count,
        "semantic_event_count": request_count * 6,
        "semantic_request_strict_count": 0,
        "semantic_tie_count": 0,
        "virtual_state_identity_sha256": "9" * 64,
    }


def _telemetry(arm: str, request_count: int = 100) -> dict[str, Any]:
    if arm == "O":
        return {
            "schema": "orbode.official-bypass.v1",
            "entry_semantic": _semantic(request_count),
            "terminal_semantic": _semantic(request_count),
            "native_compute_z_bypassed_with_shared_fixed_z": True,
            "dynamic_layer_visit_count": 0,
            "factor_build_count": 5,
            "physical_write_count": 1,
            "history_append_count": 0,
            "terminal_net_frobenius": 1.0,
            "terminal_net_frobenius_squared": 1.0,
            "sum_step_action_frobenius_squared": 1.0,
            "resolution_stable_path_frobenius_squared": 1.0,
            "native_creg_action_status": "TELEMETRY_WITHHELD",
        }
    return {
        "schema": "orbode.arm-telemetry.v1",
        "arm": arm,
        "entry_semantic": _semantic(request_count),
        "first_hit": None,
        "terminal_status": "TERMINAL_VALID",
        "terminal_semantic": _semantic(request_count),
        "step_count": 1,
        "zero_action_visit_count": 0,
        "nonzero_action_visit_count": 1,
        "factor_build_count": 1,
        "key_capture_count": 1,
        "terminal_capture_count": 1,
        "jvp_call_count": 1,
        "materialization_count": 1,
        "physical_write_count": 1,
        "history_append_count": 0,
        "anchor_active_request_count": request_count,
        "anchor_zero_request_count": 0,
        "anchor_zero_semantic_miss_count": 0,
        "anchor_zero_semantic_miss_policy": "NONBLOCKING_SCIENTIFIC_OBSERVATION",
        "sum_step_action_frobenius_squared": 0.0625,
        "resolution_stable_path_frobenius_squared": 0.125,
        "terminal_net_frobenius": 0.25,
        "terminal_net_frobenius_squared": 0.0625,
        "native_creg_action_status": "TELEMETRY_WITHHELD",
        "steps": [_step(arm, request_count)],
    }


def _arm(
    arm: str,
    evaluation: Mapping[str, Any],
    *,
    derived: Mapping[str, Any] | None = None,
    request_count: int = 100,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "status": "TERMINAL_VALID",
        "request_count": request_count,
        "endpoint": {
            "arm": arm,
            "status": "TERMINAL_VALID",
            "evaluation": evaluation,
            "selected_weight_endpoint_sha256": "a" * 64,
            "terminal_activation_sha256": "b" * 64,
            "physical_write_count": 1,
            "temporary_observation_materialization_count": 0,
            "history_append_count": 0,
            "inner_history_append_count": 0,
            "inner_cache_mutation_count": 0,
            "semantic_observation": _semantic(request_count),
            "terminal_net_frobenius": 1.0,
            "terminal_net_frobenius_squared": 1.0,
            "terminal_net_frobenius_squared_by_weight": {"layer.weight": 1.0},
        },
        "telemetry": _telemetry(arm, request_count),
        "overlay": {
            "w0_pointer_version_bytes_unchanged": True,
            "physical_inner_write_count": 0,
        },
        "derived_endpoint": derived,
        "fixed_z_identity_sha256": "c" * 64,
        "adapter_ledger": {
            "fixed_z_compute_count": 0,
            "fixed_z_recompute_count": 0,
            "terminal_capture_count": 1,
            "key_capture_count": 20,
            "layer_factorization_count": 20,
            "solve_count": 20,
            "official_endpoint_count": 0,
            "terminal_finalize_count": 1,
            "inner_history_append_count": 0,
            "inner_cache_mutation_count": 0,
            "dynamic_z_count": 0,
        },
        "scientific_promotion": False,
        "wall_seconds": 1.0,
        "jvp_ledger": {
            "jvp_call_count": 20,
            "model_forward_invocation_count": 20,
            "finite_difference_forward_count": 0,
            "temporary_direction_hook_count": 20,
            "attention_backend_switch_count": 0,
            "physical_write_count": 0,
            "wall_seconds": 0.5,
        },
        "model_forward_invocation_count": 40,
        "accounting_reconciliation": {
            "method_state_content_identity_equal": True,
            "primary_endpoint_count": 1,
        },
        "dtype_scope": {
            "model_parameters_and_storage": "torch.float32",
            "numeric_storage_cast_count": 0,
        },
    }


def _round(request_count: int = 100) -> dict[str, Any]:
    case_ids = list(range(10_000, 10_000 + request_count))
    request_shas = _request_shas(case_ids)
    entry = _evaluation(case_ids, entry=True)
    endpoint = _evaluation(case_ids, entry=False)
    derived = {
        "arm": "ORBHit",
        "status": "FIRST_HIT",
        "state_version": 3,
        "factor_count": 3,
        "factor_identities": ["d" * 64] * 3,
        "first_hit": {"sweep": 0, "layer": 6},
        "endpoint": {
            "status": "TERMINAL_VALID",
            "evaluation": endpoint,
            "selected_weight_endpoint_sha256": "e" * 64,
        },
    }
    arms = [
        _arm(arm, endpoint, request_count=request_count)
        for arm in artifacts.PRIMARY_ARM_ORDER
    ]
    arms[3]["derived_endpoint"] = derived
    return {
        "round_index": 0,
        "request_count": request_count,
        "case_ids": case_ids,
        "request_sha256": request_shas,
        "canonical_request_order_sha256": artifacts.canonical_hash(request_shas),
        "sealed_batch_order_digest": "f" * 64,
        "fixed_z": {
            "identity_sha256": "1" * 64,
            "target_context_identity_sha256": "2" * 64,
            "compute_count": request_count,
            "recompute_count": 0,
        },
        "semantic": {
            "inventory_sha256": "3" * 64,
            "event_count": request_count * 6,
            "request_count": request_count,
        },
        "pre_evaluation": entry,
        "runtime_preamble": {
            "schema": "orbode.fast-runtime-preamble.v1",
            "status": "FAST_RUNTIME_PREAMBLE_PASS",
            "request_case_id": str(case_ids[0]),
            "request_sha256": request_shas[0],
            "fixed_z_source": "COHORT_FIXED_Z_COLUMN_0_NO_RECOMPUTE",
            "fixed_z_request_consumption_count": 1,
            "fixed_z_recompute_count": 0,
            "P0_official_scaling": {
                "stock_official_parity": {
                    "mode": "PINNED_STOCK_R_OVER_N_WRAPPER_VS_DIRECT",
                    "dynamic_qcl_one_pass_used_as_stock_oracle": False,
                    "wrapper_selected_weight_endpoint_sha256": "a" * 64,
                    "direct_selected_weight_endpoint_sha256": "a" * 64,
                    "selected_weight_endpoint_sha256_present": True,
                    "selected_weight_endpoint_sha256_equal": True,
                    "wrapper_terminal_activation_sha256": "b" * 64,
                    "direct_terminal_activation_sha256": "b" * 64,
                    "terminal_activation_sha256_present": True,
                    "terminal_activation_sha256_equal": True,
                    "wrapper_semantic_observation_sha256": "c" * 64,
                    "direct_semantic_observation_sha256": "c" * 64,
                    "semantic_observation_sha256_present": True,
                    "semantic_observation_sha256_equal": True,
                    "wrapper_evaluation_sha256": "d" * 64,
                    "direct_evaluation_sha256": "d" * 64,
                    "evaluation_sha256_present": True,
                    "evaluation_sha256_equal": True,
                    "exact_parity": True,
                },
                "dynamic_qcl_one_pass_stock_parity_claim": (
                    "NOT_APPLICABLE_DISTINCT_NUMERICAL_PATH"
                ),
                "residual_scaling_replay": (
                    "WRITER_DEVICE_FP32_DIVISION_THEN_CPU_STORAGE"
                ),
                "residual_scaling_replay_device": "cuda:0",
                "residual_scaling_max_abs_error": 0.0,
                "residual_scaling_max_relative_error": 0.0,
                "right_factor_bitwise_identity": True,
            },
            "outcome_comparison_gate_policy": {
                "schema": "orbode.nonblocking-outcome-comparison-policy.v1",
                "classification": "NONBLOCKING_TELEMETRY_ONLY",
                "blocking_outcome_comparison_count": 0,
                "ours_vs_official_gate": False,
                "ours_vs_ours_gate": False,
                "resolution_match_monotonicity_convergence_gate": False,
                "action_magnitude_match_gate": False,
                "zero_correction_or_official_path_gate": False,
                "official_nonworse_outcome_gate": False,
                "stock_official_o_wrapper_direct_fidelity_gate": True,
                "technical_integrity_gates_retained": True,
                "scientific_selection_influence_count": 0,
            },
            "P1_jvp": {
                "selected_layer": 4,
                "response_selection_status": "FIRST_NUMERICALLY_ACTIVE_RESPONSE",
                "epsilon_grid": [2.0 ** -7, 2.0 ** -8, 2.0 ** -9],
                "primary_epsilon": 2.0 ** -8,
                "absolute_tolerance_formula": (
                    "64*eps_float32*max(1,max_abs(primal))/epsilon"
                ),
                "relative_tolerance": 2.0 ** -5,
                "zero_response_candidates": [],
                "finite_difference": [
                    {
                        "epsilon": epsilon,
                        "maximum_absolute_error": 0.0,
                        "numerical_activity_status": "NUMERICALLY_ACTIVE_RESPONSE",
                        "allclose": True,
                    }
                    for epsilon in (2.0 ** -7, 2.0 ** -8, 2.0 ** -9)
                ],
                "virtual_materialized": {
                    "maximum_absolute_error": 0.0,
                    "numerical_activity_status": "NUMERICALLY_ACTIVE_RESPONSE",
                    "allclose": True,
                },
                "ledger": {"jvp_call_count": 1, "physical_write_count": 0},
            },
            "P2_state_transaction": {
                "entry_l5_keys_sha256": "d" * 64,
                "changed_l5_keys_sha256": "e" * 64,
                "changed_l5_state_version": 1,
                "terminal_changed_observed": True,
                "l5_keys_changed_observed": True,
                "state_effect_comparison_policy": "NONBLOCKING_TELEMETRY_ONLY",
                "state_effect_comparison_gate_count": 0,
                "same_layer_repeated_factor_count": 2,
                "overlay": {
                    "w0_pointer_version_bytes_unchanged": True,
                    "physical_inner_write_count": 0,
                },
                "terminal_identity": {"allclose": True},
                "semantic_logit_identity": {"strict_identity": True},
                "inner_persistent_mutation_count": 0,
                "algorithmic_retry_count": 0,
                "algorithmic_rollback_count": 0,
            },
            "P3_resolution_observation_only": [
                {
                    "N": n,
                    "h": 1.0 / n,
                    "T": 1.0,
                    "status": "TERMINAL_VALID",
                    "selected_weight_endpoint_sha256": "f" * 64,
                    "step_count": n * 5,
                    "terminal_potential": 1.0,
                    "maximum_discretization_defect": 0.0,
                    "sum_discretization_defect": 0.0,
                    "selection_influence_count": 0,
                }
                for n in (2, 4, 8)
            ],
            "P3_orbhit_direct_prefix_identity": {
                "status": "ORBHit_IDENTITY_PASS",
                "orbfh_primary_gate_influence_count": 0,
                "selected_weight_endpoint_sha256_equal": True,
            },
            "w0_pointer_bytes_restore_pass": True,
            "scientific_selection_influence_count": 0,
        },
        "arms": arms,
        "w0_sha256": "4" * 64,
        "w0_pointer_bytes_restore_pass": True,
        "model_forward_invocation_count": 1234,
    }


def _public_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.add(str(key))
            result.update(_public_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_public_keys(item))
    return result


class RawFreeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_ids = list(range(100))
        self.request_sha256 = _request_shas(self.case_ids)
        self.order_sha256 = artifacts.canonical_hash(self.request_sha256)
        self.entry = _evaluation(self.case_ids, entry=True)
        self.current = _evaluation(self.case_ids, entry=False)

    def test_raw_literals_are_removed_and_metrics_are_exact(self) -> None:
        publication = artifacts.reduce_evaluation_payload(
            self.current,
            case_ids=self.case_ids,
            request_sha256=self.request_sha256,
            request_order_sha256=self.order_sha256,
            entry_evaluation=self.entry,
        )
        serialized = json.dumps(publication, sort_keys=True)
        self.assertNotIn("SECRET_PROMPT", serialized)
        self.assertNotIn("SECRET_TARGET", serialized)
        self.assertFalse(_public_keys(publication) & artifacts.PROHIBITED_PUBLIC_FIELDS)
        self.assertEqual(publication["row_count"], 1600)
        self.assertEqual(
            publication["kind_summaries"]["rewrite_target_new"]["row_count"], 100
        )
        self.assertEqual(
            publication["kind_summaries"]["rephrase_target_new"]["row_count"], 200
        )
        self.assertEqual(
            publication["kind_summaries"]["locality_target_true"]["row_count"], 1000
        )
        self.assertEqual(publication["preference"]["rewrite"]["prompt_success_count"], 100)
        self.assertEqual(publication["preference"]["rephrase"]["strict_request_success_count"], 100)
        self.assertEqual(publication["locality"]["prediction_preservation_numerator"], 1900)
        self.assertEqual(publication["locality"]["prediction_preservation_denominator"], 2000)
        checked = artifacts.validate_evaluation_publication(
            publication,
            case_ids=self.case_ids,
            request_sha256=self.request_sha256,
            request_order_sha256=self.order_sha256,
        )
        self.assertEqual(checked["status"], "RAW_FREE_EVALUATION_PASS")

    def test_canonical_ns_uses_prompt_pairs_and_keeps_token_preservation_separate(self) -> None:
        entry = _evaluation_with_canonical_ns(self.case_ids, entry=True)
        current = _evaluation_with_canonical_ns(self.case_ids, entry=False)
        publication = artifacts.reduce_evaluation_payload(
            current,
            case_ids=self.case_ids,
            request_sha256=self.request_sha256,
            request_order_sha256=self.order_sha256,
            entry_evaluation=entry,
        )
        self.assertEqual(publication["schema"], artifacts.EVALUATION_SCHEMA_V2)
        self.assertEqual(publication["row_count"], 2600)
        canonical = publication["locality"]["canonical_ns"]
        self.assertEqual(canonical["predicate"], "target_true_nll < target_new_nll")
        self.assertEqual(canonical["prompt_denominator"], 1000)
        self.assertEqual(canonical["prompt_success_count"], 1000)
        self.assertEqual(canonical["tie_count"], 0)
        self.assertEqual(publication["locality"]["prediction_preservation_denominator"], 2000)
        self.assertEqual(
            artifacts.validate_evaluation_publication(
                publication,
                case_ids=self.case_ids,
                request_sha256=self.request_sha256,
                request_order_sha256=self.order_sha256,
            )["status"],
            "RAW_FREE_EVALUATION_PASS",
        )
        tampered = copy.deepcopy(publication)
        tampered["locality"]["canonical_ns"]["prompt_denominator"] = 1010
        tampered.pop("identity_sha256")
        tampered["identity_sha256"] = artifacts.canonical_hash(tampered)
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.validate_evaluation_publication(
                tampered,
                case_ids=self.case_ids,
                request_sha256=self.request_sha256,
                request_order_sha256=self.order_sha256,
            )

    def test_missing_nonfinite_and_tampered_aggregates_fail(self) -> None:
        missing = copy.deepcopy(self.current)
        missing["rewrite_target_new"].pop()
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_evaluation_payload(
                missing,
                case_ids=self.case_ids,
                request_sha256=self.request_sha256,
                request_order_sha256=self.order_sha256,
            )
        nonfinite = copy.deepcopy(self.current)
        nonfinite["rewrite_target_new"][0]["nll"] = math.nan
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_evaluation_payload(
                nonfinite,
                case_ids=self.case_ids,
                request_sha256=self.request_sha256,
                request_order_sha256=self.order_sha256,
            )
        publication = artifacts.reduce_evaluation_payload(
            self.current,
            case_ids=self.case_ids,
            request_sha256=self.request_sha256,
            request_order_sha256=self.order_sha256,
        )
        publication["kind_summaries"]["rewrite_target_new"]["nll_mean"] += 1.0
        publication.pop("identity_sha256")
        publication["identity_sha256"] = artifacts.canonical_hash(publication)
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.validate_evaluation_publication(
                publication,
                case_ids=self.case_ids,
                request_sha256=self.request_sha256,
                request_order_sha256=self.order_sha256,
            )


class RoundPublicationTests(unittest.TestCase):
    def test_single_request_b1_panel_uses_explicit_denominator(self) -> None:
        source = _round(request_count=1)
        publication = artifacts.reduce_round_payload(
            source, expected_request_count=1
        )
        checked = artifacts.validate_round_publication(
            publication, expected_round_index=0, expected_request_count=1
        )
        self.assertEqual(checked["request_count"], 1)
        self.assertEqual(checked["primary_endpoint_count"], 5)
        self.assertEqual(publication["fixed_z_compute_count"], 1)
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.validate_round_publication(
                publication, expected_round_index=0, expected_request_count=100
            )

    def test_round_primary_panel_and_transaction_are_validated(self) -> None:
        source = _round()
        publication = artifacts.reduce_round_payload(source)
        checked = artifacts.validate_round_publication(publication, expected_round_index=0)
        self.assertEqual(checked["status"], "RAW_FREE_ROUND_PASS")
        self.assertEqual(checked["primary_endpoint_count"], 500)
        self.assertEqual(
            [item["arm"] for item in publication["primary_endpoints"]],
            list(artifacts.PRIMARY_ARM_ORDER),
        )
        self.assertFalse(_public_keys(publication) & artifacts.PROHIBITED_PUBLIC_FIELDS)
        self.assertNotIn("SECRET_PROMPT", json.dumps(publication, sort_keys=True))
        mechanism = publication["primary_endpoints"][1]["mechanism_telemetry"]
        step = mechanism["telemetry"]["steps"][0]
        self.assertEqual(step["per_request_g"], [0.25] * 100)
        self.assertEqual(step["per_request_r"], [0.25] * 100)
        self.assertEqual(step["per_request_D_R0"], [0.25] * 100)
        self.assertEqual(step["per_request_q_res"], [0.25] * 100)
        self.assertEqual(step["applied_update_frobenius_squared"], 0.0625)
        self.assertEqual(mechanism["telemetry"]["terminal_semantic"], _semantic())
        preamble = publication["runtime_preamble"]["facts"]
        self.assertIn("P0_official_scaling", preamble)
        self.assertIn("P1_jvp", preamble)
        self.assertIn("P2_state_transaction", preamble)
        self.assertIn("P3_resolution_observation_only", preamble)

    def test_nested_mechanism_raw_nonfinite_and_request_count_fail(self) -> None:
        leaked = _round()
        leaked["arms"][1]["telemetry"]["steps"][0]["prompt"] = "SECRET_PROMPT"
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_round_payload(leaked)

        nonfinite = _round()
        nonfinite["arms"][1]["telemetry"]["steps"][0]["g"] = math.inf
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_round_payload(nonfinite)

        short = _round()
        short["arms"][1]["telemetry"]["steps"][0]["per_request_q_res"].pop()
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_round_payload(short)

        exception = _round()
        exception["runtime_preamble"]["P3_orbhit_direct_prefix_identity"].update(
            {
                "status": "ORBHit_WITHHELD_TECHNICAL_DIRECT_AUDIT",
                "exception_type": "TechnicalBoundary",
                "exception": "SECRET_PROMPT and SECRET_TARGET must not escape",
            }
        )
        publication = artifacts.reduce_round_payload(exception)
        preamble = publication["runtime_preamble"]["facts"]
        serialized = json.dumps(preamble, sort_keys=True)
        self.assertNotIn("SECRET_PROMPT", serialized)
        self.assertNotIn("SECRET_TARGET", serialized)
        identity = preamble["P3_orbhit_direct_prefix_identity"]
        self.assertTrue(identity["exception_message_redacted"])
        self.assertEqual(len(identity["exception_message_sha256"]), 64)

    def test_outcome_comparisons_cannot_be_reenabled_as_gates(self) -> None:
        source = _round()
        source["runtime_preamble"]["outcome_comparison_gate_policy"][
            "ours_vs_official_gate"
        ] = True
        source["runtime_preamble"]["outcome_comparison_gate_policy"][
            "blocking_outcome_comparison_count"
        ] = 1
        with self.assertRaisesRegex(
            artifacts.ArtifactBoundary, "outcome-comparison gate policy"
        ):
            artifacts.reduce_round_payload(source)

    def test_round_arm_order_w0_and_public_raw_tamper_fail(self) -> None:
        wrong_order = _round()
        wrong_order["arms"][0], wrong_order["arms"][1] = (
            wrong_order["arms"][1],
            wrong_order["arms"][0],
        )
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_round_payload(wrong_order)
        wrong_w0 = _round()
        wrong_w0["w0_pointer_bytes_restore_pass"] = False
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.reduce_round_payload(wrong_w0)
        publication = artifacts.reduce_round_payload(_round())
        publication["prompt"] = "LEAK"
        publication.pop("identity_sha256")
        publication["identity_sha256"] = artifacts.canonical_hash(publication)
        with self.assertRaises(artifacts.ArtifactBoundary):
            artifacts.validate_round_publication(publication, expected_round_index=0)


if __name__ == "__main__":
    unittest.main()
