"""NFE-free scalar analysis for the four-edit capacity/history experiment.

The analyzer compares capacity-aware QP against the native writer separately
inside MEMIT and canonical-history AlphaEdit.  Models are never pooled and no
model-specific policy can be selected after observing results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STREAM_SCHEMA = "ode-edit-capacity-history/v1"
ANALYSIS_SCHEMA = "ode-edit-capacity-history-analysis/v1"
PAIR_SCHEMA = "ode-edit-capacity-history-pair-analysis/v1"
CHECKPOINT_EVENT = "capacity_history_checkpoint"
CHAIN_LENGTH = 4

BRANCH_MEMIT_NATIVE = "memit_native"
BRANCH_MEMIT_QP = "memit_capacity_qp"
BRANCH_ALPHA_NATIVE = "alpha_native_history"
BRANCH_ALPHA_QP = "alpha_capacity_qp_history"
BRANCHES = (
    BRANCH_MEMIT_NATIVE,
    BRANCH_MEMIT_QP,
    BRANCH_ALPHA_NATIVE,
    BRANCH_ALPHA_QP,
)
FAMILY_PAIRS = {
    "memit": (BRANCH_MEMIT_NATIVE, BRANCH_MEMIT_QP),
    "alpha_history": (BRANCH_ALPHA_NATIVE, BRANCH_ALPHA_QP),
}
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")

CURRENT_NONCOLLAPSE_FLOOR = -0.10
UTILITY_FLOOR = 1e-4
PROXY_FLOOR = 1e-8
TECHNICAL_KEYS = (
    "action_committed_before_evaluation",
    "evaluation_firewall_pass",
    "direct_z_once",
    "state_lineage_exact",
    "w0_anchor_exact",
    "precomputed_covariance_only",
    "precomputed_projector_only",
    "alpha_history_exact",
    "capacity_barrier_exact",
    "overloaded_positive_write_suppressed",
    "finite_metrics",
)
CONTRACT_KEYS = (
    "selection_manifest_id",
    "case_order_hash",
    "chain_length",
    "layers",
    "seed",
    "policy_id",
    "policy_parameters_sha256",
    "evaluator_sha256",
    "action_before_evaluation",
    "direct_z_per_branch_edit",
    "history_append_policy",
)


class CapacityHistoryAnalysisError(ValueError):
    """A compact stream differs from the locked experiment contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityHistoryAnalysisError(f"{path}: mapping required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise CapacityHistoryAnalysisError(
            f"{path}: keys differ; missing={sorted(expected_set - set(value))}, "
            f"extra={sorted(set(value) - expected_set)}"
        )


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise CapacityHistoryAnalysisError(f"{path}: finite scalar required")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CapacityHistoryAnalysisError(f"{path}: finite scalar required") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise CapacityHistoryAnalysisError(f"{path}: invalid scalar")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CapacityHistoryAnalysisError(f"{path}: integer >= {minimum} required")
    return value


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapacityHistoryAnalysisError(f"{path}: lowercase SHA-256 required")
    return value


def _normalize_contract(value: Any, path: str) -> dict[str, Any]:
    contract = _mapping(value, path)
    _exact_keys(contract, CONTRACT_KEYS, path)
    if (
        contract["chain_length"] != CHAIN_LENGTH
        or contract["layers"] != [4, 5, 6, 7, 8]
        or contract["seed"] != 41
        or contract["action_before_evaluation"] is not True
        or contract["direct_z_per_branch_edit"] != 1
        or contract["history_append_policy"]
        != "post-accepted-edit-once-history-fixed-within-edit"
        or not isinstance(contract["policy_id"], str)
        or not contract["policy_id"]
    ):
        raise CapacityHistoryAnalysisError(f"{path}: fixed constants differ")
    for key in (
        "selection_manifest_id",
        "case_order_hash",
        "policy_parameters_sha256",
        "evaluator_sha256",
    ):
        _sha256(contract[key], f"{path}.{key}")
    return dict(contract)


def load_stream(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    records: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                wrapper = _mapping(json.loads(line), f"line[{line_number}]")
            except json.JSONDecodeError as exc:
                raise CapacityHistoryAnalysisError(
                    f"line[{line_number}]: invalid JSON"
                ) from exc
            _exact_keys(
                wrapper,
                (
                    "schema_version",
                    "run_id",
                    "sequence",
                    "recorded_at",
                    "event",
                    "payload",
                ),
                f"line[{line_number}]",
            )
            if (
                wrapper["schema_version"] != STREAM_SCHEMA
                or wrapper["event"] != CHECKPOINT_EVENT
                or wrapper["sequence"] != len(records)
            ):
                raise CapacityHistoryAnalysisError(
                    f"line[{line_number}]: stream identity/order differs"
                )
            records.append(_mapping(wrapper["payload"], f"line[{line_number}].payload"))
    return records


def _normalize_record(value: Any, index: int) -> dict[str, Any]:
    path = f"records[{index}]"
    record = _mapping(value, path)
    _exact_keys(
        record,
        (
            "model_alias",
            "edit_id",
            "edit_index",
            "branch",
            "pass",
            "technical",
            "fixed_contract",
            "current_utility",
            "all_edits_mean_utility",
            "prior_mean_utility",
            "prior_success_rate",
            "neighborhood_kl",
            "generation_kl",
            "target_true_nll_delta",
            "capacity_sum",
            "max_layer_share",
            "layer_gini",
            "cumulative_frobenius",
            "native_distance_equivalent_path_length",
            "accepted_path_distance",
            "wall_seconds",
            "proposal_build_count",
            "accepted_round_count",
            "rejected_round_count",
            "first_hit_reached",
            "history_edit_count",
            "overloaded_layer_observations",
            "suppressed_overloaded_observations",
            "capacity_reroute_round_count",
            "capacity_barrier_max_violation",
        ),
        path,
    )
    model_alias = record["model_alias"]
    branch = record["branch"]
    edit_index = _integer(record["edit_index"], f"{path}.edit_index", minimum=1)
    if (
        model_alias not in MODEL_ALIASES
        or branch not in BRANCHES
        or edit_index > CHAIN_LENGTH
        or not isinstance(record["edit_id"], str)
        or not record["edit_id"]
        or type(record["pass"]) is not bool
        or type(record["first_hit_reached"]) is not bool
    ):
        raise CapacityHistoryAnalysisError(f"{path}: identity differs from lock")
    technical = _mapping(record["technical"], f"{path}.technical")
    _exact_keys(technical, TECHNICAL_KEYS, f"{path}.technical")
    technical_values: dict[str, bool] = {}
    for key in TECHNICAL_KEYS:
        if type(technical[key]) is not bool:
            raise CapacityHistoryAnalysisError(f"{path}.technical.{key}: boolean required")
        technical_values[key] = technical[key]

    prior_mean = record["prior_mean_utility"]
    prior_success = record["prior_success_rate"]
    if edit_index == 1:
        if prior_mean is not None or prior_success is not None:
            raise CapacityHistoryAnalysisError(f"{path}: t=1 prior metrics must be null")
        normalized_prior_mean = None
        normalized_prior_success = None
    else:
        normalized_prior_mean = _finite(prior_mean, f"{path}.prior_mean_utility")
        normalized_prior_success = _finite(
            prior_success, f"{path}.prior_success_rate", nonnegative=True
        )
        if normalized_prior_success > 1.0:
            raise CapacityHistoryAnalysisError(f"{path}.prior_success_rate: must be <= 1")

    max_share = _finite(
        record["max_layer_share"], f"{path}.max_layer_share", nonnegative=True
    )
    layer_gini = _finite(record["layer_gini"], f"{path}.layer_gini", nonnegative=True)
    if max_share > 1.0 or layer_gini > 1.0:
        raise CapacityHistoryAnalysisError(f"{path}: concentration metric must be <= 1")
    history_count = _integer(
        record["history_edit_count"], f"{path}.history_edit_count"
    )
    expected_history_count = edit_index if branch in {
        BRANCH_ALPHA_NATIVE,
        BRANCH_ALPHA_QP,
    } else 0
    if history_count != expected_history_count:
        raise CapacityHistoryAnalysisError(f"{path}: history count differs from branch policy")
    overloaded_observations = _integer(
        record["overloaded_layer_observations"],
        f"{path}.overloaded_layer_observations",
    )
    suppressed_observations = _integer(
        record["suppressed_overloaded_observations"],
        f"{path}.suppressed_overloaded_observations",
    )
    reroute_rounds = _integer(
        record["capacity_reroute_round_count"],
        f"{path}.capacity_reroute_round_count",
    )
    barrier_violation = _finite(
        record["capacity_barrier_max_violation"],
        f"{path}.capacity_barrier_max_violation",
        nonnegative=True,
    )
    accepted_rounds = _integer(
        record["accepted_round_count"],
        f"{path}.accepted_round_count",
        minimum=1,
    )
    if (
        suppressed_observations > overloaded_observations
        or reroute_rounds > accepted_rounds
        or (branch in {BRANCH_MEMIT_NATIVE, BRANCH_ALPHA_NATIVE} and any(
            value != 0 for value in (
                overloaded_observations,
                suppressed_observations,
                reroute_rounds,
                barrier_violation,
            )
        ))
    ):
        raise CapacityHistoryAnalysisError(f"{path}: routing diagnostic differs from branch")

    return {
        "model_alias": model_alias,
        "edit_id": record["edit_id"],
        "edit_index": edit_index,
        "branch": branch,
        "pass": record["pass"],
        "technical": technical_values,
        "fixed_contract": _normalize_contract(
            record["fixed_contract"], f"{path}.fixed_contract"
        ),
        "current_utility": _finite(record["current_utility"], f"{path}.current_utility"),
        "all_edits_mean_utility": _finite(
            record["all_edits_mean_utility"], f"{path}.all_edits_mean_utility"
        ),
        "prior_mean_utility": normalized_prior_mean,
        "prior_success_rate": normalized_prior_success,
        "neighborhood_kl": _finite(
            record["neighborhood_kl"], f"{path}.neighborhood_kl", nonnegative=True
        ),
        "generation_kl": _finite(
            record["generation_kl"], f"{path}.generation_kl", nonnegative=True
        ),
        "target_true_nll_delta": _finite(
            record["target_true_nll_delta"], f"{path}.target_true_nll_delta"
        ),
        "capacity_sum": _finite(
            record["capacity_sum"], f"{path}.capacity_sum", nonnegative=True
        ),
        "max_layer_share": max_share,
        "layer_gini": layer_gini,
        "cumulative_frobenius": _finite(
            record["cumulative_frobenius"], f"{path}.cumulative_frobenius", nonnegative=True
        ),
        "native_distance_equivalent_path_length": _finite(
            record["native_distance_equivalent_path_length"],
            f"{path}.native_distance_equivalent_path_length",
            nonnegative=True,
        ),
        "accepted_path_distance": _finite(
            record["accepted_path_distance"],
            f"{path}.accepted_path_distance",
            nonnegative=True,
        ),
        "wall_seconds": _finite(record["wall_seconds"], f"{path}.wall_seconds", nonnegative=True),
        "proposal_build_count": _integer(
            record["proposal_build_count"], f"{path}.proposal_build_count", minimum=1
        ),
        "accepted_round_count": _integer(
            accepted_rounds, f"{path}.accepted_round_count", minimum=1
        ),
        "rejected_round_count": _integer(
            record["rejected_round_count"], f"{path}.rejected_round_count"
        ),
        "first_hit_reached": record["first_hit_reached"],
        "history_edit_count": history_count,
        "overloaded_layer_observations": overloaded_observations,
        "suppressed_overloaded_observations": suppressed_observations,
        "capacity_reroute_round_count": reroute_rounds,
        "capacity_barrier_max_violation": barrier_violation,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CapacityHistoryAnalysisError("mean requires a non-empty vector")
    return math.fsum(values) / len(values)


def _family_effects(
    native: Sequence[Mapping[str, Any]],
    qp: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[float]]:
    current_deltas = [
        float(qp_row["current_utility"]) - float(native_row["current_utility"])
        for native_row, qp_row in zip(native, qp, strict=True)
    ]
    native_prior = [float(row["prior_mean_utility"]) for row in native[1:]]
    qp_prior = [float(row["prior_mean_utility"]) for row in qp[1:]]
    effects = {
        "current_edit_utility_mean_delta": _mean(current_deltas),
        "final_all_edits_utility_delta": float(qp[-1]["all_edits_mean_utility"])
        - float(native[-1]["all_edits_mean_utility"]),
        "final_prior_retention_delta": float(qp[-1]["prior_mean_utility"])
        - float(native[-1]["prior_mean_utility"]),
        "retention_auc_delta": _mean(qp_prior) - _mean(native_prior),
        "prior_success_auc_delta": _mean(
            [float(row["prior_success_rate"]) for row in qp[1:]]
        )
        - _mean([float(row["prior_success_rate"]) for row in native[1:]]),
        "neighborhood_kl_reduction": _mean(
            [float(row["neighborhood_kl"]) for row in native]
        )
        - _mean([float(row["neighborhood_kl"]) for row in qp]),
        "generation_kl_reduction": _mean(
            [float(row["generation_kl"]) for row in native]
        )
        - _mean([float(row["generation_kl"]) for row in qp]),
        "target_true_nll_drift_reduction": _mean(
            [float(row["target_true_nll_delta"]) for row in native]
        )
        - _mean([float(row["target_true_nll_delta"]) for row in qp]),
        "capacity_reduction": float(native[-1]["capacity_sum"])
        - float(qp[-1]["capacity_sum"]),
        "max_layer_share_reduction": float(native[-1]["max_layer_share"])
        - float(qp[-1]["max_layer_share"]),
        "layer_gini_reduction": float(native[-1]["layer_gini"])
        - float(qp[-1]["layer_gini"]),
        "cumulative_frobenius_reduction": float(native[-1]["cumulative_frobenius"])
        - float(qp[-1]["cumulative_frobenius"]),
        "accepted_path_distance_reduction": math.fsum(
            float(row["accepted_path_distance"]) for row in native
        )
        - math.fsum(float(row["accepted_path_distance"]) for row in qp),
    }
    return effects, current_deltas


def analyze_records(
    records: Iterable[Mapping[str, Any]], *, model_alias: str
) -> dict[str, Any]:
    if model_alias not in MODEL_ALIASES:
        raise CapacityHistoryAnalysisError("model is outside the fixed pair")
    normalized = [_normalize_record(record, index) for index, record in enumerate(records)]
    if len(normalized) != CHAIN_LENGTH * len(BRANCHES):
        raise CapacityHistoryAnalysisError("exact 4 edits x 4 branches are required")
    if {record["model_alias"] for record in normalized} != {model_alias}:
        raise CapacityHistoryAnalysisError("model identity differs across stream")
    contract = normalized[0]["fixed_contract"]
    if any(record["fixed_contract"] != contract for record in normalized):
        raise CapacityHistoryAnalysisError("fixed contract differs across checkpoints")

    by_branch: dict[str, list[dict[str, Any]]] = {}
    for branch in BRANCHES:
        rows = sorted(
            (record for record in normalized if record["branch"] == branch),
            key=lambda record: record["edit_index"],
        )
        if (
            [record["edit_index"] for record in rows] != list(range(1, CHAIN_LENGTH + 1))
            or len({record["edit_id"] for record in rows}) != CHAIN_LENGTH
        ):
            raise CapacityHistoryAnalysisError(f"{branch}: edit order/count differs")
        by_branch[branch] = rows
    expected_ids = [row["edit_id"] for row in by_branch[BRANCH_MEMIT_NATIVE]]
    if any([row["edit_id"] for row in by_branch[branch]] != expected_ids for branch in BRANCHES):
        raise CapacityHistoryAnalysisError("branch edit identities/orders differ")

    technical_failures = [
        f"{row['branch']}:t{row['edit_index']}:pass=false"
        for row in normalized
        if not row["pass"]
    ]
    technical_failures.extend(
        f"{row['branch']}:t{row['edit_index']}:{key}=false"
        for row in normalized
        for key, passed in row["technical"].items()
        if not passed
    )
    technical_pass = not technical_failures
    families: dict[str, Any] = {}
    for family, (native_branch, qp_branch) in FAMILY_PAIRS.items():
        native = by_branch[native_branch]
        qp = by_branch[qp_branch]
        effects, current_deltas = _family_effects(native, qp)
        positive_axes = {
            "final_all_edits_utility_delta": effects["final_all_edits_utility_delta"] > UTILITY_FLOOR,
            "final_prior_retention_delta": effects["final_prior_retention_delta"] > UTILITY_FLOOR,
            "retention_auc_delta": effects["retention_auc_delta"] > UTILITY_FLOOR,
            "prior_success_auc_delta": effects["prior_success_auc_delta"] > UTILITY_FLOOR,
            "neighborhood_kl_reduction": effects["neighborhood_kl_reduction"] > PROXY_FLOOR,
            "generation_kl_reduction": effects["generation_kl_reduction"] > PROXY_FLOOR,
            "target_true_nll_drift_reduction": effects["target_true_nll_drift_reduction"] > PROXY_FLOOR,
            "capacity_reduction": effects["capacity_reduction"] > PROXY_FLOOR,
            "max_layer_share_reduction": effects["max_layer_share_reduction"] > PROXY_FLOOR,
            "layer_gini_reduction": effects["layer_gini_reduction"] > PROXY_FLOOR,
            "cumulative_frobenius_reduction": effects["cumulative_frobenius_reduction"] > PROXY_FLOOR,
        }
        noncollapse = effects["current_edit_utility_mean_delta"] >= CURRENT_NONCOLLAPSE_FLOOR
        capacity_signal = any(
            positive_axes[key]
            for key in (
                "capacity_reduction",
                "max_layer_share_reduction",
                "layer_gini_reduction",
            )
        )
        preservation_signal = any(
            positive_axes[key]
            for key in (
                "final_prior_retention_delta",
                "retention_auc_delta",
                "prior_success_auc_delta",
                "neighborhood_kl_reduction",
                "generation_kl_reduction",
                "target_true_nll_drift_reduction",
            )
        )
        gate_pass = technical_pass and noncollapse and (capacity_signal or preservation_signal)
        overloaded_total = sum(row["overloaded_layer_observations"] for row in qp)
        suppressed_total = sum(
            row["suppressed_overloaded_observations"] for row in qp
        )
        routing_diagnostic = {
            "overloaded_layer_observations_total": overloaded_total,
            "suppressed_overloaded_observations_total": suppressed_total,
            "capacity_reroute_round_count_total": sum(
                row["capacity_reroute_round_count"] for row in qp
            ),
            "capacity_barrier_max_violation": max(
                row["capacity_barrier_max_violation"] for row in qp
            ),
            "empirical_overload_observed": overloaded_total > 0,
            "all_observed_positive_overloads_suppressed": (
                overloaded_total == suppressed_total
            ),
        }
        if not technical_pass:
            verdict = "TECHNICAL_INVALID"
        elif not noncollapse:
            verdict = "CAPACITY_QP_HARM_SIGNAL"
        elif capacity_signal or preservation_signal:
            verdict = "CAPACITY_QP_MODEL_SIGNAL"
        else:
            verdict = "CAPACITY_QP_MODEL_NO_SIGNAL"
        families[family] = {
            "native_branch": native_branch,
            "qp_branch": qp_branch,
            "effects": effects,
            "current_edit_deltas": current_deltas,
            "positive_axes": positive_axes,
            "lenient_gate": {
                "current_noncollapse": noncollapse,
                "capacity_signal": capacity_signal,
                "preservation_signal": preservation_signal,
                "model_gate_pass": gate_pass,
            },
            "routing_diagnostic": routing_diagnostic,
            "verdict": verdict,
        }

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "model_alias": model_alias,
        "fixed_contract": contract,
        "technical_validity": {
            "pass": technical_pass,
            "failures": technical_failures,
            "checkpoint_count": len(normalized),
        },
        "families": families,
        "compute": {
            branch: {
                "wall_seconds_total": math.fsum(
                    row["wall_seconds"] for row in by_branch[branch]
                ),
                "proposal_build_count_total": sum(
                    row["proposal_build_count"] for row in by_branch[branch]
                ),
                "accepted_round_count_total": sum(
                    row["accepted_round_count"] for row in by_branch[branch]
                ),
                "rejected_round_count_total": sum(
                    row["rejected_round_count"] for row in by_branch[branch]
                ),
            }
            for branch in BRANCHES
        },
        "verdict": (
            "TECHNICAL_INVALID"
            if not technical_pass
            else "CAPACITY_HISTORY_MODEL_SIGNAL"
            if any(item["lenient_gate"]["model_gate_pass"] for item in families.values())
            else "CAPACITY_HISTORY_MODEL_NO_SIGNAL"
        ),
        "claim_boundary": (
            "same-policy four-edit Motivation signal only; no lifelong, broad "
            "preservation guarantee, or deployable method-superiority claim"
        ),
    }


def analyze_pair(llama: Mapping[str, Any], qwen: Mapping[str, Any]) -> dict[str, Any]:
    analyses = {
        "llama3-8b-inst": _mapping(llama, "llama"),
        "qwen2.5-7b-inst": _mapping(qwen, "qwen"),
    }
    for model, analysis in analyses.items():
        if analysis.get("schema_version") != ANALYSIS_SCHEMA or analysis.get("model_alias") != model:
            raise CapacityHistoryAnalysisError(f"{model}: analysis identity differs")
    if analyses["llama3-8b-inst"]["fixed_contract"] != analyses["qwen2.5-7b-inst"]["fixed_contract"]:
        raise CapacityHistoryAnalysisError("pair fixed contracts differ")
    technical_pass = all(
        analysis["technical_validity"]["pass"] is True for analysis in analyses.values()
    )
    families: dict[str, Any] = {}
    for family in FAMILY_PAIRS:
        llama_family = _mapping(analyses["llama3-8b-inst"]["families"][family], f"llama.{family}")
        qwen_family = _mapping(analyses["qwen2.5-7b-inst"]["families"][family], f"qwen.{family}")
        common_positive_axes = [
            axis
            for axis, passed in llama_family["positive_axes"].items()
            if passed is True and qwen_family["positive_axes"].get(axis) is True
        ]
        noncollapse_both = all(
            item["lenient_gate"]["current_noncollapse"] is True
            for item in (llama_family, qwen_family)
        )
        model_gate_both = all(
            item["lenient_gate"]["model_gate_pass"] is True
            for item in (llama_family, qwen_family)
        )
        family_pass = technical_pass and noncollapse_both and model_gate_both and bool(common_positive_axes)
        families[family] = {
            "models": {
                model: {
                    "verdict": analyses[model]["families"][family]["verdict"],
                    "effects": analyses[model]["families"][family]["effects"],
                    "positive_axes": analyses[model]["families"][family]["positive_axes"],
                    "routing_diagnostic": analyses[model]["families"][family][
                        "routing_diagnostic"
                    ],
                }
                for model in MODEL_ALIASES
            },
            "current_noncollapse_both": noncollapse_both,
            "common_positive_axes": common_positive_axes,
            "family_gate_pass": family_pass,
        }
    passing = [family for family, item in families.items() if item["family_gate_pass"]]
    if not technical_pass:
        verdict = "PAIR_TECHNICAL_INVALID"
    elif len(passing) == len(FAMILY_PAIRS):
        verdict = "CAPACITY_HISTORY_ACTUATOR_COMMON_SIGNAL"
    elif passing:
        verdict = "CAPACITY_HISTORY_FAMILY_CONDITIONED_SIGNAL"
    elif any(not item["current_noncollapse_both"] for item in families.values()):
        verdict = "CAPACITY_HISTORY_HARM_SIGNAL"
    else:
        verdict = "CAPACITY_HISTORY_NO_COMMON_SIGNAL"
    return {
        "schema_version": PAIR_SCHEMA,
        "fixed_contract": analyses["llama3-8b-inst"]["fixed_contract"],
        "technical_pass": technical_pass,
        "families": families,
        "passing_families": passing,
        "verdict": verdict,
        "claim_boundary": (
            "same-policy four-edit model-common directional signal only; no lifelong "
            "or general method-superiority claim"
        ),
    }


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    return _mapping(json.loads(source.read_text(encoding="utf-8")), str(source))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-capacity-history-analysis", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model", allow_abbrev=False)
    model.add_argument("--checkpoints", required=True)
    model.add_argument("--model", required=True, choices=MODEL_ALIASES)
    model.add_argument("--output-json", required=True)
    pair = subparsers.add_parser("pair", allow_abbrev=False)
    pair.add_argument("--llama-analysis", required=True)
    pair.add_argument("--qwen-analysis", required=True)
    pair.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "model":
        result = analyze_records(load_stream(args.checkpoints), model_alias=args.model)
    else:
        result = analyze_pair(_read_json(args.llama_analysis), _read_json(args.qwen_analysis))
    _write_json(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0 if "INVALID" not in result["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
