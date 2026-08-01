"""Scalar-only analysis for the four-edit Motivation preservation diagnostic.

The runner contract is deliberately policy-agnostic.  A candidate policy is
identified by a manifest hash and must be byte-identical across both models;
the analyzer never selects a model-specific policy after seeing outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STREAM_SCHEMA = "ode-edit-microseq/v1"
ANALYSIS_SCHEMA = "ode-edit-microseq-analysis/v1"
PAIR_SCHEMA = "ode-edit-microseq-pair-analysis/v1"
CHECKPOINT_EVENT = "microseq_checkpoint"
CHAIN_LENGTH = 4
BRANCH_NATIVE = "native_sequential_memit"
BRANCH_ODE = "ode_always_refresh_memit"
BRANCHES = (BRANCH_NATIVE, BRANCH_ODE)
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
CURRENT_NONCOLLAPSE_FLOOR = -0.10
UTILITY_FLOOR = 1e-4
PROXY_FLOOR = 1e-6
TECHNICAL_KEYS = (
    "action_committed_before_evaluation",
    "evaluation_firewall_pass",
    "direct_z_once",
    "state_lineage_exact",
    "w0_anchor_exact",
    "precomputed_cache_only",
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
)


class MicroseqAnalysisError(ValueError):
    """A compact scalar stream differs from the fixed Motivation contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MicroseqAnalysisError(f"{path}: mapping required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise MicroseqAnalysisError(
            f"{path}: keys differ; missing={sorted(expected_set - set(value))}, "
            f"extra={sorted(set(value) - expected_set)}"
        )


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise MicroseqAnalysisError(f"{path}: finite scalar required")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MicroseqAnalysisError(f"{path}: finite scalar required") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise MicroseqAnalysisError(f"{path}: invalid scalar")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MicroseqAnalysisError(f"{path}: integer >= {minimum} required")
    return value


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MicroseqAnalysisError(f"{path}: lowercase SHA-256 required")
    return value


def _normalize_contract(value: Any, path: str) -> dict[str, Any]:
    contract = _mapping(value, path)
    _exact_keys(contract, CONTRACT_KEYS, path)
    if (
        contract["chain_length"] != CHAIN_LENGTH
        or contract["layers"] != [4, 5, 6, 7, 8]
        or contract["seed"] != 37
        or contract["action_before_evaluation"] is not True
        or contract["direct_z_per_branch_edit"] != 1
        or not isinstance(contract["policy_id"], str)
        or not contract["policy_id"]
    ):
        raise MicroseqAnalysisError(f"{path}: fixed constants differ from lock")
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
                raise MicroseqAnalysisError(
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
                raise MicroseqAnalysisError(
                    f"line[{line_number}]: stream identity/order differs from lock"
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
            "wall_seconds",
            "proposal_build_count",
            "controlled_nfe",
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
    ):
        raise MicroseqAnalysisError(f"{path}: identity differs from lock")
    technical = _mapping(record["technical"], f"{path}.technical")
    _exact_keys(technical, TECHNICAL_KEYS, f"{path}.technical")
    technical_values: dict[str, bool] = {}
    for key in TECHNICAL_KEYS:
        if type(technical[key]) is not bool:
            raise MicroseqAnalysisError(f"{path}.technical.{key}: boolean required")
        technical_values[key] = technical[key]

    prior_mean = record["prior_mean_utility"]
    prior_success = record["prior_success_rate"]
    if edit_index == 1:
        if prior_mean is not None or prior_success is not None:
            raise MicroseqAnalysisError(f"{path}: t=1 prior metrics must be null")
        normalized_prior_mean = None
        normalized_prior_success = None
    else:
        normalized_prior_mean = _finite(prior_mean, f"{path}.prior_mean_utility")
        normalized_prior_success = _finite(
            prior_success, f"{path}.prior_success_rate", nonnegative=True
        )
        if normalized_prior_success > 1.0:
            raise MicroseqAnalysisError(f"{path}.prior_success_rate: must be <= 1")

    max_share = _finite(
        record["max_layer_share"], f"{path}.max_layer_share", nonnegative=True
    )
    layer_gini = _finite(record["layer_gini"], f"{path}.layer_gini", nonnegative=True)
    if max_share > 1.0 or layer_gini > 1.0:
        raise MicroseqAnalysisError(f"{path}: concentration metric must be <= 1")

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
            record["cumulative_frobenius"],
            f"{path}.cumulative_frobenius",
            nonnegative=True,
        ),
        "native_distance_equivalent_path_length": _finite(
            record["native_distance_equivalent_path_length"],
            f"{path}.native_distance_equivalent_path_length",
            nonnegative=True,
        ),
        "wall_seconds": _finite(
            record["wall_seconds"], f"{path}.wall_seconds", nonnegative=True
        ),
        "proposal_build_count": _integer(
            record["proposal_build_count"], f"{path}.proposal_build_count", minimum=1
        ),
        "controlled_nfe": _integer(
            record["controlled_nfe"], f"{path}.controlled_nfe", minimum=1
        ),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise MicroseqAnalysisError("mean requires a non-empty vector")
    return math.fsum(values) / len(values)


def analyze_records(
    records: Iterable[Mapping[str, Any]], *, model_alias: str
) -> dict[str, Any]:
    if model_alias not in MODEL_ALIASES:
        raise MicroseqAnalysisError("model is outside the fixed pair")
    normalized = [
        _normalize_record(record, index) for index, record in enumerate(records)
    ]
    if len(normalized) != CHAIN_LENGTH * len(BRANCHES):
        raise MicroseqAnalysisError("exact 4 edits x 2 branches are required")
    if {record["model_alias"] for record in normalized} != {model_alias}:
        raise MicroseqAnalysisError("model identity differs across stream")
    contract = normalized[0]["fixed_contract"]
    if any(record["fixed_contract"] != contract for record in normalized):
        raise MicroseqAnalysisError("fixed contract differs across checkpoints")

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
            raise MicroseqAnalysisError(f"{branch}: edit order/count differs from lock")
        by_branch[branch] = rows
    if [row["edit_id"] for row in by_branch[BRANCH_NATIVE]] != [
        row["edit_id"] for row in by_branch[BRANCH_ODE]
    ]:
        raise MicroseqAnalysisError("branch edit identities/orders differ")

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
    native = by_branch[BRANCH_NATIVE]
    ode = by_branch[BRANCH_ODE]
    final_prior_retention_delta = (
        float(ode[-1]["prior_mean_utility"])
        - float(native[-1]["prior_mean_utility"])
    )
    retention_auc_delta = _mean(
        [row["all_edits_mean_utility"] for row in ode]
    ) - _mean([row["all_edits_mean_utility"] for row in native])
    current_deltas = [
        ode_row["current_utility"] - native_row["current_utility"]
        for native_row, ode_row in zip(native, ode, strict=True)
    ]
    effects = {
        "final_prior_retention_delta": final_prior_retention_delta,
        "retention_auc_delta": retention_auc_delta,
        "current_edit_noncollapse_mean": _mean(current_deltas),
        "neighborhood_kl_reduction": _mean(
            [row["neighborhood_kl"] for row in native]
        )
        - _mean([row["neighborhood_kl"] for row in ode]),
        "generation_kl_reduction": _mean(
            [row["generation_kl"] for row in native]
        )
        - _mean([row["generation_kl"] for row in ode]),
        "capacity_reduction": native[-1]["capacity_sum"] - ode[-1]["capacity_sum"],
        "max_layer_share_reduction": (
            native[-1]["max_layer_share"] - ode[-1]["max_layer_share"]
        ),
    }
    positive_axes = {
        "final_prior_retention_delta": effects["final_prior_retention_delta"]
        > UTILITY_FLOOR,
        "retention_auc_delta": effects["retention_auc_delta"] > UTILITY_FLOOR,
        "neighborhood_kl_reduction": effects["neighborhood_kl_reduction"]
        > PROXY_FLOOR,
        "generation_kl_reduction": effects["generation_kl_reduction"] > PROXY_FLOOR,
        "capacity_reduction": effects["capacity_reduction"] > PROXY_FLOOR,
    }
    technical_pass = not technical_failures
    noncollapse = effects["current_edit_noncollapse_mean"] >= CURRENT_NONCOLLAPSE_FLOOR
    any_signal = any(positive_axes.values())
    if not technical_pass:
        verdict = "TECHNICAL_INVALID"
    elif not noncollapse:
        verdict = "MICROSEQ_HARM_SIGNAL"
    elif any_signal:
        verdict = "MICROSEQ_MODEL_SIGNAL"
    else:
        verdict = "MICROSEQ_MODEL_NO_SIGNAL"

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "model_alias": model_alias,
        "fixed_contract": contract,
        "technical_validity": {
            "pass": technical_pass,
            "failures": technical_failures,
            "checkpoint_count": len(normalized),
        },
        "effects": effects,
        "current_edit_deltas": current_deltas,
        "positive_axes": positive_axes,
        "lenient_gate": {
            "current_noncollapse": noncollapse,
            "any_specialization_signal": any_signal,
            "model_gate_pass": technical_pass and noncollapse and any_signal,
        },
        "compute": {
            branch: {
                "wall_seconds_total": math.fsum(
                    row["wall_seconds"] for row in by_branch[branch]
                ),
                "proposal_build_count_total": sum(
                    row["proposal_build_count"] for row in by_branch[branch]
                ),
                "controlled_nfe_total": sum(
                    row["controlled_nfe"] for row in by_branch[branch]
                ),
            }
            for branch in BRANCHES
        },
        "verdict": verdict,
        "claim_boundary": (
            "four-edit Motivation proxy only; no lifelong, downstream-capability, "
            "or method-superiority claim"
        ),
    }


def analyze_pair(llama: Mapping[str, Any], qwen: Mapping[str, Any]) -> dict[str, Any]:
    analyses = {
        "llama3-8b-inst": _mapping(llama, "llama"),
        "qwen2.5-7b-inst": _mapping(qwen, "qwen"),
    }
    for model, analysis in analyses.items():
        if (
            analysis.get("schema_version") != ANALYSIS_SCHEMA
            or analysis.get("model_alias") != model
        ):
            raise MicroseqAnalysisError(f"{model}: analysis identity differs")
    if analyses["llama3-8b-inst"]["fixed_contract"] != analyses["qwen2.5-7b-inst"][
        "fixed_contract"
    ]:
        raise MicroseqAnalysisError("pair fixed contracts differ")
    technical_pass = all(
        analysis["technical_validity"]["pass"] is True
        for analysis in analyses.values()
    )
    noncollapse = all(
        analysis["lenient_gate"]["current_noncollapse"] is True
        for analysis in analyses.values()
    )
    common_positive_axes = [
        axis
        for axis in analyses["llama3-8b-inst"]["positive_axes"]
        if analyses["llama3-8b-inst"]["positive_axes"][axis] is True
        and analyses["qwen2.5-7b-inst"]["positive_axes"].get(axis) is True
    ]
    if not technical_pass:
        verdict = "PAIR_TECHNICAL_INVALID"
    elif not noncollapse:
        verdict = "MICROSEQ_HARM_SIGNAL"
    elif common_positive_axes:
        verdict = "MICROSEQ_LENIENT_SIGNAL"
    else:
        verdict = "MICROSEQ_NO_COMMON_SIGNAL"
    return {
        "schema_version": PAIR_SCHEMA,
        "fixed_contract": analyses["llama3-8b-inst"]["fixed_contract"],
        "models": {
            model: {
                "verdict": analysis["verdict"],
                "effects": analysis["effects"],
                "positive_axes": analysis["positive_axes"],
            }
            for model, analysis in analyses.items()
        },
        "technical_pass": technical_pass,
        "current_noncollapse_both": noncollapse,
        "common_positive_axes": common_positive_axes,
        "verdict": verdict,
        "claim_boundary": (
            "same-policy four-edit Motivation signal only; no lifelong or "
            "method-superiority claim"
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
    parser = argparse.ArgumentParser(prog="ode-edit-microseq-analysis", allow_abbrev=False)
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
        result = analyze_pair(
            _read_json(args.llama_analysis), _read_json(args.qwen_analysis)
        )
    _write_json(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0 if "INVALID" not in result["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
