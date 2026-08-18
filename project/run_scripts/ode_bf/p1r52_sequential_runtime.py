"""P1R52 Repair-R1 Llama Soft and Official AlphaEdit sequential B10x10.

The scientific P1R52 K8 rollout is delegated to the frozen Atomic runtime.
This module owns only inter-B10 persistence, active Historical geometry,
lifetime anchors, terminal-only evaluation, and exact job-terminal W0 restore.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import BATCH_SIZE, COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_backend import capture_committed_history_key_views
from .p1_evaluator import (
    BatchEntryObservationSeal,
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
    load_counterfact_cases_after_freeze,
)
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import _action_frozen_cases, _model_w0_contract, _run_ode_arm
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r29_sequential_preparation import SequentialArmState
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r43_independent_b10x10_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r52_sequential_contract import (
    HISTORY_COUNTS,
    INSTRUCTION_ID,
    METHOD_ID,
    ROUND_COUNT,
    LifetimeAnchor,
    LifetimeAnchorLedger,
    SequentialHRouter,
    assemble_terminal_candidates,
    capture_controller_legal_lifetime_anchors,
    commit_sequential_batch,
    make_history_records,
    scoped_atomic_sequential_adapter,
)
from .p1r52_sequential_scale import P1R52SequentialScale, P1R52_B10X10_SCALE
from .p1r52_official_sequential_baselines import (
    load_official_memit_hparams,
    run_official_memit_apply,
)
from .p1r52_accepted_z_observation import (
    OfficialNativeZCapture,
    evaluate_accepted_z_batch,
    r52_binding,
)
from .scalable_batched_model import build_scalable_capture_plan, build_scalable_objective_plan
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import P1R23_GRID_COUNT, P1R23_LAYER_ORDER, scalable_ordered_request_digest


R52_H_ROLE = "r52-soft-sequential-h"
R52_CONTROL_ROLE = "r52-soft-sequential-alphacache-on-structuralh-off"
NATIVE_ROLE = "native-alphaedit-sequential"
NATIVE_CORRECTED_ROLE = "native-alphaedit-sequential-cache-on-corrected"
MEMIT_ROLE = "official-memit-sequential"
R52_ROLES = (R52_H_ROLE, R52_CONTROL_ROLE)
ROLES = (*R52_ROLES, NATIVE_ROLE)
OFFICIAL_BASELINE_ROLES = (NATIVE_ROLE, NATIVE_CORRECTED_ROLE, MEMIT_ROLE)
EXECUTION_ROLES = (*R52_ROLES, *OFFICIAL_BASELINE_ROLES)
RESULT_NAMES = {
    R52_H_ROLE: "s05-p1r52-llama-soft-sequential-historical-10xb10-tech-r3-v1",
    R52_CONTROL_ROLE: "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb10-v1",
    NATIVE_ROLE: "s05-p1r52-native-alphaedit-sequential-10xb10-v1",
    NATIVE_CORRECTED_ROLE: "s05-p1r52-native-alphaedit-sequential-cache-on-corrected-10xb10-v1",
    MEMIT_ROLE: "s05-p1r52-official-memit-sequential-10xb10-v1",
}

RESULT_NAMES_B100X10 = {
    R52_H_ROLE: "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-v1",
    R52_CONTROL_ROLE: "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-v1",
    NATIVE_CORRECTED_ROLE: "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-v1",
    MEMIT_ROLE: "s05-p1r52-official-memit-sequential-10xb100-v1",
}

RESULT_NAMES_B100X10_TECH_R1 = {
    role: name.removesuffix("-v1") + "-tech-r1-v1"
    for role, name in RESULT_NAMES_B100X10.items()
}

RESULT_NAMES_B100X10_TECH_R2 = {
    role: name.removesuffix("-v1") + "-tech-r2-v1"
    for role, name in RESULT_NAMES_B100X10.items()
}

RESULT_NAMES_B100X10_TECH_R3 = {
    role: name.removesuffix("-v1") + "-tech-r3-v1"
    for role, name in RESULT_NAMES_B100X10.items()
}

RESULT_NAMES_B100X10_TECH_R3_RELEASE_R1 = {
    role: name.removesuffix("-v1") + "-tech-r3-release-r1-v1"
    for role, name in RESULT_NAMES_B100X10.items()
}

RESULT_NAMES_B100X10_ACCEPTED_Z_OBS = {
    role: name.removesuffix("-v1") + "-accepted-z-rephrase-obs-v1"
    for role, name in RESULT_NAMES_B100X10.items()
}

RESULT_NAMES_B100X10_ACCEPTED_Z_OBS_TECH_R1 = {
    role: name.removesuffix("-v1") + "-tech-r1-v1"
    for role, name in RESULT_NAMES_B100X10_ACCEPTED_Z_OBS.items()
}


B1_PROCESS_LOCAL_RECEIPT_PATHS = frozenset(
    {
        ("field_sha256",),
        ("finite_writer_demand", "endpoint_objective_sha256"),
        ("finite_writer_demand", "identity_sha256"),
        ("identity_sha256",),
        ("next_physical_capture_sha256",),
        ("physical_capture_sha256",),
        ("physical_slope", "field_sha256"),
        ("physical_slope", "objective_receipt_sha256"),
        ("target_objective", "identity_sha256"),
        ("target_objective", "model_state_sha256"),
        ("target_update", "identity_sha256"),
        ("target_update", "selected_endpoint", "current_objective_sha256"),
        ("target_update", "selected_endpoint", "identity_sha256"),
        ("target_update", "selected_endpoint", "primary_endpoint_sha256"),
        ("target_update", "writer_demand", "endpoint_objective_sha256"),
        ("target_update", "writer_demand", "identity_sha256"),
    }
)


def _b1_scientific_payload(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Remove only process-local identity cascades from an accepted-k receipt."""

    if isinstance(value, Mapping):
        return {
            key: _b1_scientific_payload(item, (*path, str(key)))
            for key, item in value.items()
            if (*path, str(key)) not in B1_PROCESS_LOCAL_RECEIPT_PATHS
        }
    if isinstance(value, list):
        return [_b1_scientific_payload(item, path) for item in value]
    return value


def _b1_stable_rollout_payload(rollout: Mapping[str, Any]) -> dict[str, Any]:
    initial = {
        key: value
        for key, value in rollout["initial"].items()
        if key not in {"capture_identity_sha256", "identity_sha256"}
    }
    terminal_z8 = {
        key: value
        for key, value in rollout["terminal_z8_oracle"].items()
        if key != "receipt_sha256"
    }
    return {
        "initial": initial,
        "metric": rollout["metric"],
        "terminal_target_sha256": rollout["terminal_target_sha256"],
        "terminal_z8_oracle": terminal_z8,
        "terminal_cumulative_structural_p": rollout["terminal_cumulative_structural_p"],
        "kl_teacher_hash_by_k": rollout["kl_teacher_hash_by_k"],
        "kl_teacher_hash_k8_constant": rollout["kl_teacher_hash_k8_constant"],
        "materializer": rollout["materializer"],
    }


def expected_p1r52_sequential_result_name(
    alias: str,
    role: str,
    *,
    scale: P1R52SequentialScale = P1R52_B10X10_SCALE,
    attempt_suffix: str | None = None,
) -> str:
    if attempt_suffix not in (
        None,
        "tech-r1",
        "tech-r2",
        "tech-r3",
        "tech-r3-release-r1",
        "accepted-z-rephrase-obs",
        "accepted-z-rephrase-obs-tech-r1",
    ):
        raise ODEBFContractError("P1R52 sequential attempt suffix differs")
    if attempt_suffix is not None and scale == P1R52_B10X10_SCALE:
        raise ODEBFContractError("P1R52 B10 sequential attempt suffix is not authorized")
    names = (
        RESULT_NAMES
        if scale == P1R52_B10X10_SCALE
        else RESULT_NAMES_B100X10_ACCEPTED_Z_OBS
        if attempt_suffix == "accepted-z-rephrase-obs"
        else RESULT_NAMES_B100X10_ACCEPTED_Z_OBS_TECH_R1
        if attempt_suffix == "accepted-z-rephrase-obs-tech-r1"
        else RESULT_NAMES_B100X10_TECH_R3_RELEASE_R1
        if attempt_suffix == "tech-r3-release-r1"
        else RESULT_NAMES_B100X10_TECH_R3
        if attempt_suffix == "tech-r3"
        else RESULT_NAMES_B100X10_TECH_R2
        if attempt_suffix == "tech-r2"
        else RESULT_NAMES_B100X10_TECH_R1
        if attempt_suffix == "tech-r1"
        else RESULT_NAMES_B100X10
    )
    if alias != "llama3-8b-inst" or role not in names:
        raise ODEBFContractError("P1R52 sequential result identity differs")
    return names[role]


def structural_h_off_control_receipts(
    receipts: Sequence[Any],
    *,
    alpha_solve_history_width: int,
    anchor_observation_width: int,
    scale: P1R52SequentialScale = P1R52_B10X10_SCALE,
) -> list[dict[str, Any]]:
    """Bind Alpha-cache activity and zero Structural-H action independently."""

    width = int(alpha_solve_history_width)
    anchors = int(anchor_observation_width)
    if width not in scale.history_counts or anchors != width:
        raise ODEBFStateError("Structural-H-off control history/anchor width differs")
    result: list[dict[str, Any]] = []
    for receipt in receipts:
        item = asdict(receipt)
        if int(item["history_width"]) != 0:
            raise ODEBFStateError("Structural-H-off router received decision history")
        result.append(
            {
                **item,
                "status": "ALPHA_CACHE_ON_STRUCTURAL_H_OFF",
                "alpha_solve_history_width": width,
                "alpha_solve_cache_consume_count": width,
                "alpha_solve_cache_append_count": scale.batch_size,
                "structural_h_decision_history_width": 0,
                "structural_h_decision_influence_count": 0,
                "risk_observation_width": width,
                "anchor_observation_width": anchors,
                "observation_ledger_decision_influence_count": 0,
                "counterfactual_h_telemetry_status": "NOT_RECORDED",
                "added_model_forward_count": 0,
                "added_backward_count": 0,
                "added_materialization_count": 0,
            }
        )
    return result


def _weight_values(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        for name, parameter in sorted(parameters.items())
    }


def _snapshot(parameters: Mapping[str, torch.nn.Parameter], *, round_index: int, role: str) -> ArmWeightSnapshot:
    hashes = _hashes(parameters)
    return ArmWeightSnapshot(
        P1Arm.R_BF,
        round_index,
        hashes,
        canonical_hash({"round": round_index, "role": role, "weights": hashes}),
    )


def _percentile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else 0.0


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) != len(values):
        raise ODEBFStateError("sequential observation contains a non-finite value")
    return {
        "count": len(finite),
        "mean": float(statistics.fmean(finite)) if finite else 0.0,
        "median": float(statistics.median(finite)) if finite else 0.0,
        "p90": _percentile(finite, 90.0),
        "raw_max": max(finite, default=0.0),
    }


def _aggregate_metric_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Aggregate already-produced B10 receipts; no model action occurs here."""

    names = ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc")
    metrics: dict[str, Any] = {}
    for name in names:
        rows = [payload[name] for payload in payloads]
        prompt_numerator = sum(int(row["prompt_numerator"]) for row in rows)
        prompt_denominator = sum(int(row["prompt_denominator"]) for row in rows)
        strict_numerator = sum(int(row["strict_request_numerator"]) for row in rows)
        strict_denominator = sum(int(row["strict_request_denominator"]) for row in rows)
        target_new = [
            float(value)
            for row in rows
            for request_values in row["target_new_nll_by_request"]
            for value in request_values
        ]
        target_old = [
            float(value)
            for row in rows
            for request_values in row["target_true_nll_by_request"]
            for value in request_values
        ]
        margin = [right - left for left, right in zip(target_new, target_old, strict=True)]
        metrics[name] = {
            "prompt_numerator": prompt_numerator,
            "prompt_denominator": prompt_denominator,
            "prompt_rate": prompt_numerator / prompt_denominator,
            "strict_request_numerator": strict_numerator,
            "strict_request_denominator": strict_denominator,
            "strict_request_rate": strict_numerator / strict_denominator,
            "prompt_bit_vector_sha256": canonical_hash(
                [row["prompt_bit_vector_sha256"] for row in rows]
            ),
            "strict_request_bit_vector_sha256": canonical_hash(
                [row["strict_request_bit_vector_sha256"] for row in rows]
            ),
            "target_new_nll": _summary(target_new),
            "target_old_nll": _summary(target_old),
            "target_old_minus_new_margin": _summary(margin),
        }
    locality = [payload["legacy_primary"]["metrics"]["locality-preservation"] for payload in payloads]
    local_num = sum(int(row["numerator"]) for row in locality)
    local_den = sum(int(row["denominator"]) for row in locality)
    result = {
        "batch_count": len(payloads),
        "request_count": len(payloads) * batch_size,
        "metrics": metrics,
        "locality": {
            "numerator": local_num,
            "denominator": local_den,
            "rate": local_num / local_den,
            "bit_vector_sha256": canonical_hash([row["bit_vector_sha256"] for row in locality]),
        },
        "canonical_names": {
            "Eff": "rewrite_success",
            "Gen": "paraphrase_success",
            "rephrase_success": "paraphrase_success",
            "rephrase_acc": "paraphrase_acc",
        },
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def _aggregate_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"metrics": {}}
    for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        result["metrics"][name] = {
            "prompt_numerator": left["metrics"][name]["prompt_numerator"]
            - right["metrics"][name]["prompt_numerator"],
            "prompt_rate": left["metrics"][name]["prompt_rate"]
            - right["metrics"][name]["prompt_rate"],
            "strict_request_numerator": left["metrics"][name]["strict_request_numerator"]
            - right["metrics"][name]["strict_request_numerator"],
            "strict_request_rate": left["metrics"][name]["strict_request_rate"]
            - right["metrics"][name]["strict_request_rate"],
            "target_new_nll_mean": left["metrics"][name]["target_new_nll"]["mean"]
            - right["metrics"][name]["target_new_nll"]["mean"],
            "target_old_nll_mean": left["metrics"][name]["target_old_nll"]["mean"]
            - right["metrics"][name]["target_old_nll"]["mean"],
            "target_old_minus_new_margin_mean": left["metrics"][name]["target_old_minus_new_margin"]["mean"]
            - right["metrics"][name]["target_old_minus_new_margin"]["mean"],
        }
    result["locality_rate"] = left["locality"]["rate"] - right["locality"]["rate"]
    result["identity_sha256"] = canonical_hash(result)
    return result


def _evaluate_cohort(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    alias: str,
    role: str,
    cohort_index: int,
    checkpoint_index: int,
    snapshot_sha256: str,
    slots: int,
    expected_batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, Any], float]:
    order = scalable_ordered_request_digest([item.request_sha256 for item in cases])
    freeze = EndpointActionFreeze(
        arm=f"{role}-cohort-{cohort_index:02d}-checkpoint-{checkpoint_index:02d}",
        sequential_batch=checkpoint_index - 1,
        request_order_sha256=order,
        selected_snapshot_sha256=snapshot_sha256,
        fixed_budget_slots_completed=slots,
    )
    started = time.perf_counter()
    receipt = evaluate_counterfact_success_accuracy_batch(
        model,
        tokenizer,
        cases,
        model_alias=alias,
        freeze=freeze,
        expected_batch_size=expected_batch_size,
    ).raw_free_payload()
    elapsed = time.perf_counter() - started
    receipt["cohort_index"] = cohort_index
    receipt["checkpoint_index"] = checkpoint_index
    receipt["evaluation_wall_seconds"] = elapsed
    receipt["endpoint_action_freeze_sha256"] = freeze.identity()
    return receipt, elapsed


def _evaluate_batch_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_path: Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    role: str,
    round_index: int,
    entry_weight_sha256: Mapping[str, str],
    expected_batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, Any], float]:
    """Evaluate one current B10 at its sealed physical entry state."""

    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    parameter_byte_sha256 = canonical_hash(dict(sorted(entry_weight_sha256.items())))
    seal = BatchEntryObservationSeal(
        arm=f"{role}-batch-entry-pre",
        sequential_batch=round_index - 1,
        request_order_sha256=request_order,
        selected_snapshot_sha256=parameter_byte_sha256,
        parameter_byte_sha256=parameter_byte_sha256,
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path,
        requests,
        seal,
        expected_batch_size=expected_batch_size,
    )
    started = time.perf_counter()
    receipt = evaluate_counterfact_success_accuracy_batch(
        model,
        tokenizer,
        cases,
        model_alias=alias,
        freeze=seal,
        expected_batch_size=expected_batch_size,
    ).raw_free_payload()
    elapsed = time.perf_counter() - started
    receipt.update(
        {
            "evaluation_phase": "batch_entry_pre_evaluator",
            "round": round_index,
            "entry_weight_sha256": dict(sorted(entry_weight_sha256.items())),
            "entry_parameter_byte_sha256": parameter_byte_sha256,
            "entry_observation_seal": asdict(seal),
            "entry_observation_seal_sha256": seal.identity(),
            "evaluation_wall_seconds": elapsed,
            "controller_influence_count": 0,
            "routing_influence_count": 0,
            "history_influence_count": 0,
            "anchor_influence_count": 0,
            "backward_count": 0,
            "generation_call_count": 0,
        }
    )
    receipt["entry_pre_identity_sha256"] = canonical_hash(receipt)
    return receipt, elapsed


def _record_evaluator_compute(
    ledger: ComputeLedger,
    receipt: Mapping[str, Any],
    *,
    component: str,
    wall_seconds: float,
) -> None:
    primary = receipt["legacy_primary"]
    ledger.increment("evaluator_forward", int(primary["model_forward_count"]))
    ledger.increment("evaluator_tokens", int(primary["processed_token_count"]))
    ledger.add_time(component, wall_seconds=wall_seconds)


def _flatten(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in values for value in row]


def _evaluation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        row = payload[name]
        new = _flatten(row["target_new_nll_by_request"])
        old = _flatten(row["target_true_nll_by_request"])
        margin = [right - left for left, right in zip(new, old, strict=True)]
        metrics[name] = {
            "prompt_numerator": int(row["prompt_numerator"]),
            "prompt_denominator": int(row["prompt_denominator"]),
            "prompt_rate": float(row["prompt_rate"]),
            "strict_request_numerator": int(row["strict_request_numerator"]),
            "strict_request_denominator": int(row["strict_request_denominator"]),
            "strict_request_rate": float(row["strict_request_rate"]),
            "target_new_nll": _summary(new),
            "target_old_nll": _summary(old),
            "target_old_minus_new_margin": _summary(margin),
            "prompt_bit_vector_sha256": row["prompt_bit_vector_sha256"],
            "strict_request_bit_vector_sha256": row["strict_request_bit_vector_sha256"],
        }
    locality = payload["legacy_primary"]["metrics"]["locality-preservation"]
    result = {
        "metrics": metrics,
        "locality": {
            "numerator": int(locality["numerator"]),
            "denominator": int(locality["denominator"]),
            "rate": int(locality["numerator"]) / int(locality["denominator"]),
            "bit_vector_sha256": locality["bit_vector_sha256"],
        },
        "request_order_sha256": payload["legacy_primary"]["request_order_sha256"],
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def _summary_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return left-minus-right arithmetic on common scalar summaries."""

    result: dict[str, Any] = {"metrics": {}}
    for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        result["metrics"][name] = {
            "prompt_numerator": left["metrics"][name]["prompt_numerator"]
            - right["metrics"][name]["prompt_numerator"],
            "prompt_rate": left["metrics"][name]["prompt_rate"]
            - right["metrics"][name]["prompt_rate"],
            "strict_request_numerator": left["metrics"][name]["strict_request_numerator"]
            - right["metrics"][name]["strict_request_numerator"],
            "strict_request_rate": left["metrics"][name]["strict_request_rate"]
            - right["metrics"][name]["strict_request_rate"],
            "target_new_nll_mean": left["metrics"][name]["target_new_nll"]["mean"]
            - right["metrics"][name]["target_new_nll"]["mean"],
            "target_old_nll_mean": left["metrics"][name]["target_old_nll"]["mean"]
            - right["metrics"][name]["target_old_nll"]["mean"],
            "target_old_minus_new_margin_mean": left["metrics"][name]["target_old_minus_new_margin"]["mean"]
            - right["metrics"][name]["target_old_minus_new_margin"]["mean"],
        }
    result["locality_rate"] = left["locality"]["rate"] - right["locality"]["rate"]
    result["identity_sha256"] = canonical_hash(result)
    return result


def _request_metric_view(payload: Mapping[str, Any], request_index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        row = payload[name]
        bits = [int(value) for value in row["per_request_bits"][request_index]]
        new = [float(value) for value in row["target_new_nll_by_request"][request_index]]
        old = [float(value) for value in row["target_true_nll_by_request"][request_index]]
        result[name] = {
            "bits": bits,
            "correct": int(row["per_request_correct"][request_index]),
            "required": int(row["per_request_required"][request_index]),
            "rate": sum(bits) / len(bits),
            "strict_all_prompts_pass": int(all(bits)),
            "target_new_nll_mean": float(statistics.fmean(new)),
            "target_old_nll_mean": float(statistics.fmean(old)),
            "target_old_minus_new_margin_mean": float(
                statistics.fmean(right - left for left, right in zip(new, old, strict=True))
            ),
        }
    locality = payload["legacy_primary"]["metrics"]["locality-preservation"]
    locality_bits = [int(value) for value in locality["per_case_bits"][request_index]]
    result["locality"] = {
        "bits": locality_bits,
        "correct": sum(locality_bits),
        "required": len(locality_bits),
        "rate": sum(locality_bits) / len(locality_bits),
    }
    return result


def build_pre_post_final_request_rows(
    request_batches: Sequence[Sequence[Mapping[str, Any]]],
    entry_pre: Sequence[Mapping[str, Any]],
    immediate_post: Sequence[Mapping[str, Any]],
    final_w10: Sequence[Mapping[str, Any]],
    *,
    scale: P1R52SequentialScale = P1R52_B10X10_SCALE,
) -> list[dict[str, Any]]:
    """Build aligned raw-free request rows; no evaluator/model action."""

    if not (
        len(request_batches)
        == len(entry_pre)
        == len(immediate_post)
        == len(final_w10)
        == scale.round_count
    ):
        raise ODEBFContractError("sequential pre/post/final cohort inventory differs")
    rows: list[dict[str, Any]] = []
    for round_index, (requests, pre, post, final) in enumerate(
        zip(request_batches, entry_pre, immediate_post, final_w10, strict=True), start=1
    ):
        for payload in (pre, post, final):
            if payload["legacy_primary"]["request_order_sha256"] != scalable_ordered_request_digest(
                [str(item["request_sha256"]) for item in requests]
            ):
                raise ODEBFContractError("sequential pre/post/final request order differs")
        for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
            denominators = {
                tuple(int(value) for value in payload[name]["per_request_required"])
                for payload in (pre, post, final)
            }
            if len(denominators) != 1:
                raise ODEBFContractError("sequential pre/post/final metric denominator differs")
        locality_denominators = {
            tuple(
                int(value)
                for value in payload["legacy_primary"]["metrics"]["locality-preservation"]["per_case_required"]
            )
            for payload in (pre, post, final)
        }
        if len(locality_denominators) != 1:
            raise ODEBFContractError("sequential pre/post/final locality denominator differs")
        for request_index, request in enumerate(requests):
            pre_view = _request_metric_view(pre, request_index)
            post_view = _request_metric_view(post, request_index)
            final_view = _request_metric_view(final, request_index)
            deltas: dict[str, Any] = {}
            for name in (*pre_view.keys(),):
                deltas[name] = {
                    "edit_gain_post_minus_pre_rate": post_view[name]["rate"] - pre_view[name]["rate"],
                    "lifetime_forgetting_final_minus_post_rate": final_view[name]["rate"] - post_view[name]["rate"],
                }
                if name != "locality":
                    deltas[name].update(
                        {
                            "edit_target_new_nll_reduction_pre_minus_post": pre_view[name]["target_new_nll_mean"]
                            - post_view[name]["target_new_nll_mean"],
                            "lifetime_target_new_nll_final_minus_post": final_view[name]["target_new_nll_mean"]
                            - post_view[name]["target_new_nll_mean"],
                            "edit_margin_gain_post_minus_pre": post_view[name]["target_old_minus_new_margin_mean"]
                            - pre_view[name]["target_old_minus_new_margin_mean"],
                            "lifetime_margin_final_minus_post": final_view[name]["target_old_minus_new_margin_mean"]
                            - post_view[name]["target_old_minus_new_margin_mean"],
                        }
                    )
            row = {
                "round": round_index,
                "history_width_at_entry": scale.history_counts[round_index - 1],
                "case_id": int(request["case_id"]),
                "request_index": request_index,
                "request_sha256": str(request["request_sha256"]),
                "entry_pre": pre_view,
                "immediate_post": post_view,
                "final_W10": final_view,
                "deltas": deltas,
            }
            row["identity_sha256"] = canonical_hash(row)
            rows.append(row)
    if len(rows) != scale.request_count:
        raise ODEBFStateError("sequential request table does not match the scale")
    return rows


def build_post_final_request_rows(
    request_batches: Sequence[Sequence[Mapping[str, Any]]],
    immediate_post: Sequence[Mapping[str, Any]],
    final_w10: Sequence[Mapping[str, Any]],
    *,
    scale: P1R52SequentialScale = P1R52_B10X10_SCALE,
) -> list[dict[str, Any]]:
    """Build the amended raw-free request rows without batch-entry metrics."""

    if not (
        len(request_batches)
        == len(immediate_post)
        == len(final_w10)
        == scale.round_count
    ):
        raise ODEBFContractError("sequential post/final cohort inventory differs")
    rows: list[dict[str, Any]] = []
    for round_index, (requests, post, final) in enumerate(
        zip(request_batches, immediate_post, final_w10, strict=True), start=1
    ):
        expected_order = scalable_ordered_request_digest(
            [str(item["request_sha256"]) for item in requests]
        )
        for payload in (post, final):
            if payload["legacy_primary"]["request_order_sha256"] != expected_order:
                raise ODEBFContractError("sequential post/final request order differs")
        for name in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
            denominators = {
                tuple(int(value) for value in payload[name]["per_request_required"])
                for payload in (post, final)
            }
            if len(denominators) != 1:
                raise ODEBFContractError("sequential post/final metric denominator differs")
        for request_index, request in enumerate(requests):
            post_view = _request_metric_view(post, request_index)
            final_view = _request_metric_view(final, request_index)
            forgetting = {
                name: {
                    "lifetime_forgetting_final_minus_post_rate": (
                        final_view[name]["rate"] - post_view[name]["rate"]
                    ),
                    **(
                        {}
                        if name == "locality"
                        else {
                            "lifetime_target_new_nll_final_minus_post": (
                                final_view[name]["target_new_nll_mean"]
                                - post_view[name]["target_new_nll_mean"]
                            ),
                            "lifetime_margin_final_minus_post": (
                                final_view[name]["target_old_minus_new_margin_mean"]
                                - post_view[name]["target_old_minus_new_margin_mean"]
                            ),
                        }
                    ),
                }
                for name in post_view
            }
            row = {
                "round": round_index,
                "history_width_at_entry": scale.history_counts[round_index - 1],
                "case_id": int(request["case_id"]),
                "request_index": request_index,
                "request_sha256": str(request["request_sha256"]),
                "batch_entry_metrics_status": "REMOVED_BY_USER_AMENDMENT",
                "immediate_post": post_view,
                "final_W10": final_view,
                "deltas": forgetting,
            }
            row["identity_sha256"] = canonical_hash(row)
            rows.append(row)
    if len(rows) != scale.request_count:
        raise ODEBFStateError("sequential post/final request table does not match the scale")
    return rows


def _anchor_matrix(anchors: Sequence[LifetimeAnchor]) -> dict[str, np.ndarray]:
    return {
        item.request_sha256: np.asarray(item.target_new_nll_by_context, dtype=np.float64)
        for item in anchors
    }


def _drift_payload(
    immutable: Sequence[LifetimeAnchor],
    entry_observed: Mapping[str, np.ndarray],
    terminal_observed: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    local: list[float] = []
    lifetime: list[float] = []
    per_request: list[dict[str, Any]] = []
    for anchor in immutable:
        request_sha = anchor.request_sha256
        entry = entry_observed[request_sha]
        terminal = terminal_observed[request_sha]
        life_base = np.asarray(anchor.target_new_nll_by_context, dtype=np.float64)
        local_row = np.maximum(terminal - entry, 0.0)
        lifetime_row = np.maximum(terminal - life_base, 0.0)
        local.extend(float(value) for value in local_row)
        lifetime.extend(float(value) for value in lifetime_row)
        per_request.append(
            {
                "request_sha256": request_sha,
                "anchor_round": anchor.round_index,
                "local_max": float(local_row.max(initial=0.0)),
                "lifetime_max": float(lifetime_row.max(initial=0.0)),
                "local_mean": float(local_row.mean()),
                "lifetime_mean": float(lifetime_row.mean()),
            }
        )
    payload = {
        "local": _summary(local),
        "lifetime": _summary(lifetime),
        "per_request": per_request,
        "cohort": {
            str(round_index): {
                "local": _summary([row["local_max"] for row in per_request if row["anchor_round"] == round_index]),
                "lifetime": _summary([row["lifetime_max"] for row in per_request if row["anchor_round"] == round_index]),
            }
            for round_index in sorted({item.round_index for item in immutable})
        },
        "observation_only": True,
        "decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _capture_observation(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    alias: str,
    round_index: int,
    collision_by_request: Mapping[str, str],
    endpoint_weight_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    anchors, receipt = capture_controller_legal_lifetime_anchors(
        model,
        tokenizer,
        requests,
        contexts,
        alias=alias,
        round_index=round_index,
        collision_by_request=collision_by_request,
        endpoint_weight_sha256=endpoint_weight_sha256,
    )
    return _anchor_matrix(anchors), receipt


def _b1_identity_gate(
    case_root: Path,
    *,
    committed_hashes: Mapping[str, str],
    evaluation: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> dict[str, Any]:
    reference_root = Path(
        "/mnt/raid5/janghj/.codex/worktrees/"
        "odeeditsh1-s05-p1r52-rsa-r42safekdc-m1-v1/local/odebf/results/"
        "s05-p1r52-rsa-r42safekdc-m1-independent-b10x10-llama3-8b-inst-soft-repair-r1-v1/"
        "raw/cases/case-01"
    )
    terminal_path = reference_root / "terminal.json"
    accepted_path = reference_root / "raw/ode/p1r52-rsa-r42safekdc-m1-soft/accepted-k8.json"
    if not terminal_path.is_file() or not accepted_path.is_file():
        raise ODEBFStateError("immutable P1R52 B1 identity reference is absent")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    reference_hashes = accepted["materialization"]["effective_bf16_sha256"]
    current_primary = evaluation["legacy_primary"]["metrics"]
    reference_primary = terminal["terminal_four_panel"]["weight"]["primary"]["metrics"]
    legacy_metric_equal = all(
        current_primary[name]["numerator"] == reference_primary[name]["numerator"]
        and current_primary[name]["denominator"] == reference_primary[name]["denominator"]
        for name in ("efficacy", "generalization", "locality-preservation")
    )
    reference_rollout = terminal["rollout"]
    current_accepted_root = case_root / "raw/ode/p1r52-rsa-r42safekdc-m1-soft"
    reference_accepted_root = accepted_path.parent
    accepted_receipts: list[dict[str, Any]] = []
    for step_index in range(1, P1R23_GRID_COUNT + 1):
        current_path = current_accepted_root / f"accepted-k{step_index}.json"
        reference_path = reference_accepted_root / f"accepted-k{step_index}.json"
        if not current_path.is_file() or not reference_path.is_file():
            raise ODEBFStateError("P1R52 B1 accepted receipt inventory differs")
        current_value = json.loads(current_path.read_text(encoding="utf-8"))
        reference_value = json.loads(reference_path.read_text(encoding="utf-8"))
        current_scientific = _b1_scientific_payload(current_value)
        reference_scientific = _b1_scientific_payload(reference_value)
        accepted_receipts.append(
            {
                "step_index": step_index,
                "current_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "scientific_payload_sha256": canonical_hash(current_scientific),
                "scientific_payload_exact": current_scientific == reference_scientific,
            }
        )
    accepted_scientific_exact = all(
        item["scientific_payload_exact"] for item in accepted_receipts
    )
    stable_rollout_equal = (
        _b1_stable_rollout_payload(rollout)
        == _b1_stable_rollout_payload(reference_rollout)
    )
    payload = {
        "reference_terminal_path": str(terminal_path),
        "reference_terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "reference_accepted_k8_path": str(accepted_path),
        "reference_final_bf16_sha256": reference_hashes,
        "sequential_final_bf16_sha256": dict(committed_hashes),
        "final_bf16_exact": dict(committed_hashes) == reference_hashes,
        "legacy_Eff_Gen_Loc_exact": legacy_metric_equal,
        "accepted_k1_k8_receipt_sha256_exact": (
            rollout["accepted_receipt_sha256"]
            == reference_rollout["accepted_receipt_sha256"]
        ),
        "accepted_k1_k8_scientific_payload_exact": accepted_scientific_exact,
        "accepted_k1_k8_receipts": accepted_receipts,
        "process_local_identity_paths_excluded": [
            list(path) for path in sorted(B1_PROCESS_LOCAL_RECEIPT_PATHS)
        ],
        "stable_rollout_payload_exact": stable_rollout_equal,
        "terminal_physical_capture_cross_process_identity": "OBSERVATION_ONLY_DATA_PTR_VERSION_SCOPED",
        "target_direction_amplitude_selection_allocation_materialization_exact": (
            accepted_scientific_exact and stable_rollout_equal
        ),
        "atomic_rewrite_acc": "NOT_RECORDED",
        "atomic_paraphrase_acc": "NOT_RECORDED",
        "sequential_accuracy_nonblocking_extension": True,
        "history_width": 0,
    }
    payload["passed"] = bool(
        payload["final_bf16_exact"]
        and legacy_metric_equal
        and accepted_scientific_exact
        and stable_rollout_equal
    )
    payload["identity_sha256"] = canonical_hash(payload)
    _atomic_write_once(case_root / "b1-atomic-identity.json", payload)
    if not payload["passed"]:
        raise ODEBFStateError("P1R52 sequential B1 atomic identity differs")
    return payload


def run_p1r52_sequential(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    scale: P1R52SequentialScale = P1R52_B10X10_SCALE,
    batch_entry_evaluation_enabled: bool = True,
    accepted_z_observation_enabled: bool = False,
    accepted_z_reference_root: Path | None = None,
) -> dict[str, Any]:
    role_names = RESULT_NAMES if scale == P1R52_B10X10_SCALE else RESULT_NAMES_B100X10
    if alias != "llama3-8b-inst" or role not in role_names:
        raise ODEBFContractError("P1R52 sequential model/role differs")
    if len(stream_batches) != scale.round_count or any(
        len(batch) != scale.batch_size for batch in stream_batches
    ):
        raise ODEBFContractError("P1R52 sequential stream scale differs")
    if scale == P1R52_B10X10_SCALE and (
        stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P1R52 sequential stream identity differs")
    if (
        stream.get("edit_batch_size") != scale.batch_size
        or stream.get("sequential_round_count") != scale.round_count
        or stream.get("logical_edit_count") != scale.request_count
    ):
        raise ODEBFContractError("P1R52 sequential stream geometry differs")
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P1R52 sequential entry W0 differs")
    if accepted_z_observation_enabled and (
        accepted_z_reference_root is None
        or not accepted_z_reference_root.is_dir()
    ):
        raise ODEBFContractError("accepted-z sealed reference root is absent")

    from . import p1_scalable_batched_experiment as experiment

    expected_w0 = _model_w0_contract(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    sequential_state = SequentialArmState(
        role,
        P1R23_LAYER_ORDER,
        maximum_history_records=scale.request_count,
        history_batch_size=scale.batch_size,
    )
    anchor_ledger = LifetimeAnchorLedger.empty(batch_size=scale.batch_size)
    cohort_cases: list[tuple[Any, ...]] = []
    cohort_requests: list[tuple[Mapping[str, Any], ...]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    entry_pre_evaluations: list[dict[str, Any]] = []
    immediate_post_evaluations: list[dict[str, Any]] = []
    accepted_z_observations: list[dict[str, Any]] = []
    prior_commit_hashes = _hashes(touched)
    alphaedit_prior_cache_exit_sha256: str | None = None
    alphaedit_static_projection_identity: str | None = None
    memit_prior_covariance_exit_identity: str | None = None
    memit_hparams = load_official_memit_hparams() if role == MEMIT_ROLE else None
    started = time.perf_counter()
    evaluation_wall = 0.0
    anchor_wall = 0.0
    try:
        for round_index, request_batch in enumerate(stream_batches, start=1):
            requests = tuple(request_batch)
            history_width = (
                sequential_state.history_entry_count()
                if role in R52_ROLES
                else (round_index - 1) * scale.batch_size
            )
            expected_history_width = scale.history_counts[round_index - 1]
            if history_width != expected_history_width:
                raise ODEBFStateError("P1R52 sequential entry history width differs")
            entry_hashes = _hashes(touched)
            if entry_hashes != prior_commit_hashes:
                raise ODEBFStateError("P1R52 prior commit/next entry hash differs")
            entry_contract = _model_w0_contract(touched)
            entry_values = _weight_values(touched)
            entry_receipt = _snapshot(touched, round_index=round_index - 1, role=role)
            prior_requests = tuple(item for batch in cohort_requests for item in batch)
            prior_anchors = tuple(anchor_ledger.anchors)
            prior_anchor_finalized = dict(anchor_ledger.finalized)
            reference_path: Path | None = None
            reference: dict[str, Any] | None = None
            if accepted_z_observation_enabled:
                assert accepted_z_reference_root is not None
                reference_path = (
                    accepted_z_reference_root
                    / "raw"
                    / "batches"
                    / f"b{round_index:02d}"
                    / "terminal.json"
                )
                if not reference_path.is_file():
                    raise ODEBFStateError("accepted-z sealed batch reference is absent")
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
                if (
                    reference.get("role") != role
                    or reference.get("round") != round_index
                    or reference.get("batch_size") != scale.batch_size
                ):
                    raise ODEBFStateError("accepted-z sealed batch reference identity differs")
            entry_observed: dict[str, np.ndarray] = {}
            if accepted_z_observation_enabled:
                assert reference is not None
                entry_anchor_receipt = reference["history_entry_anchor_observation"]
            elif prior_requests:
                anchor_started = time.perf_counter()
                entry_observed, entry_anchor_receipt = _capture_observation(
                    model,
                    tokenizer,
                    prior_requests,
                    contexts,
                    alias=alias,
                    round_index=round_index - 1,
                    collision_by_request=collision_by_request,
                    endpoint_weight_sha256=canonical_hash(entry_hashes),
                )
                anchor_wall += time.perf_counter() - anchor_started
            else:
                entry_anchor_receipt = {"status": "EMPTY_HISTORY", "model_forward_count": 0}

            case_root = raw_root / "batches" / f"b{round_index:02d}"
            case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
            entry_pre: dict[str, Any] | None = None
            if batch_entry_evaluation_enabled:
                entry_pre, pre_wall = _evaluate_batch_entry(
                    model,
                    tokenizer,
                    dataset_path,
                    requests,
                    alias=alias,
                    role=role,
                    round_index=round_index,
                    entry_weight_sha256=entry_hashes,
                    expected_batch_size=scale.batch_size,
                )
                entry_pre_evaluations.append(entry_pre)
                evaluation_wall += pre_wall
                _record_evaluator_compute(
                    job_ledger,
                    entry_pre,
                    component="batch_entry_pre_evaluator",
                    wall_seconds=pre_wall,
                )
            seed_all(COMMON_SEED)
            accepted_z_binding = None
            if role in R52_ROLES:
                request_order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests])
                objective_plan = build_scalable_objective_plan(
                    model,
                    tokenizer,
                    requests,
                    contexts=contexts,
                    request_microbatch_size=min(request_microbatch_size, scale.batch_size),
                    fact_token_strategy=hparams.fact_token,
                )
                capture_plan = build_scalable_capture_plan(
                    tokenizer,
                    requests,
                    contexts=contexts,
                    request_microbatch_size=min(request_microbatch_size, scale.batch_size),
                    fact_token_strategy=hparams.fact_token,
                )
                if objective_plan.request_order_sha256 != request_order or capture_plan.request_order_sha256 != request_order:
                    raise ODEBFContractError("P1R52 sequential objective/capture order differs")
                outer_population = tuple(population_by_sha256[item] for item in theta0_cache.request_order)
                snapshot = _entry_parameter_snapshot_sha256(model, dict(entry_receipt.parameter_sha256))
                counter = ModelForwardCounter(model, job_ledger)
                try:
                    outer_entry_p_cache = build_outer_entry_pretrained_cache(
                        model,
                        tokenizer,
                        outer_population,
                        theta0_cache,
                        outer_entry_snapshot_sha256=snapshot,
                    )
                finally:
                    counter.close()
                arm_state = ArmRuntimeState(
                    P1Arm.R_BF,
                    P1HistoryLedger(
                        layer_order=P1R23_LAYER_ORDER,
                        maximum_records=scale.request_count,
                        batch_size=scale.batch_size,
                    ),
                    ComputeLedger(),
                    entry_receipt,
                    entry_values,
                )
                router = SequentialHRouter(
                    history_width if role == R52_H_ROLE else 0,
                    experiment.solve_p1r43_full_strength_routing,
                )
                with scoped_atomic_sequential_adapter(
                    experiment,
                    sequential_state,
                    router,
                    structural_h_decision_enabled=role == R52_H_ROLE,
                    maximum_history_columns=scale.history_counts[-1],
                ):
                    rollout = _run_ode_arm(
                        model,
                        tokenizer,
                        requests,
                        alias=alias,
                        arm=FixedE8Arm.SOFT,
                        allocation="RS",
                        capture_plan=capture_plan,
                        objective_plan=objective_plan,
                        hparams=hparams,
                        projector=projector,
                        contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        controller_lock=controller_lock,
                        arm_state=arm_state,
                        request_by_sha256=request_by_sha256,
                        population_by_sha256=population_by_sha256,
                        schedule=schedule,
                        outer_entry_p_cache=outer_entry_p_cache,
                        theta0_cache=theta0_cache,
                        touched=touched,
                        base_receipt=entry_receipt,
                        base_values=entry_values,
                        raw_root=case_root / "raw",
                        write_once=_atomic_write_once,
                        p1r24=True,
                        p1r34=True,
                        p1r35=True,
                        p1r52=True,
                    )
                if accepted_z_observation_enabled:
                    accepted_z_binding = r52_binding(
                        role,
                        requests,
                        hparams,
                        rollout["terminal_target"],
                    )
                public = rollout["public"]
                if public["accepted_update_count"] != P1R23_GRID_COUNT or public["tau_final"] != 1.0:
                    raise ODEBFStateError("P1R52 sequential Atomic K8 differs")
                if _model_w0_contract(touched) != entry_contract:
                    raise ODEBFStateError("P1R52 Atomic wrapper did not restore B10 entry")
                candidates, candidate_receipt = assemble_terminal_candidates(entry_values, rollout["terminal_factors"])
                transaction_id = f"p1r52-sequential-b{round_index:02d}-{public['identity_sha256']}"
                records = make_history_records(
                    requests,
                    collision_by_request,
                    version=round_index,
                    terminal_event_sha256=public["identity_sha256"],
                )
                layer_by_name = {
                    f"{hparams.rewrite_module_tmp.format(layer)}.weight": layer
                    for layer in P1R23_LAYER_ORDER
                }
                load = {
                    layer_by_name[name]: float(
                        torch.sum(
                            (
                                candidates[name].to(dtype=torch.float32)
                                - entry_values[name].to(dtype=torch.float32)
                            ) ** 2
                        )
                    )
                    for name in sorted(candidates)
                }
                atomic_payload = public
                h_payload = [asdict(item) for item in router.receipts]
                if role == R52_CONTROL_ROLE:
                    h_payload = structural_h_off_control_receipts(
                        router.receipts,
                        alpha_solve_history_width=history_width,
                        anchor_observation_width=len(anchor_ledger.anchors),
                        scale=scale,
                    )
                anchor_started = time.perf_counter()

                def anchor_factory() -> Any:
                    return capture_controller_legal_lifetime_anchors(
                        model,
                        tokenizer,
                        requests,
                        contexts,
                        alias=alias,
                        round_index=round_index,
                        collision_by_request=collision_by_request,
                        endpoint_weight_sha256=canonical_hash(_hashes(touched)),
                    )

                def history_factory() -> Any:
                    history_started = time.perf_counter()
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        solve_keys, risk_keys, key_identity = capture_committed_history_key_views(
                            model,
                            tokenizer,
                            requests,
                            hparams,
                            projector,
                            contexts,
                            ledger=job_ledger,
                            expected_batch_size=scale.batch_size,
                        )
                    finally:
                        counter.close()
                    history_elapsed = time.perf_counter() - history_started
                    job_ledger.add_time(
                        "post_commit_history_key_capture",
                        wall_seconds=history_elapsed,
                    )
                    prospective = sequential_state.ledger.prospective(
                        transaction_id=transaction_id,
                        expected_version=round_index - 1,
                        records=records,
                        solve_keys_by_layer=solve_keys,
                        risk_keys_by_layer=risk_keys,
                    )
                    capture_receipt = {
                        "post_commit_capture": True,
                        "round": round_index,
                        "request_count": scale.batch_size,
                        "committed_weight_sha256": _hashes(touched),
                        "solve_key_sha256": {
                            str(layer): tensor_sha256(solve_keys[layer])
                            for layer in P1R23_LAYER_ORDER
                        },
                        "risk_key_sha256": {
                            str(layer): tensor_sha256(risk_keys[layer])
                            for layer in P1R23_LAYER_ORDER
                        },
                        "key_identity_sha256": key_identity,
                        "capture_wall_seconds": history_elapsed,
                        "current_batch_in_own_field_count": 0,
                    }
                    capture_receipt["identity_sha256"] = canonical_hash(capture_receipt)
                    return prospective, capture_receipt

                joint_transaction = commit_sequential_batch(
                    touched,
                    candidates,
                    mutation_lock=mutation_lock,
                    transaction_id=transaction_id,
                    history_state=sequential_state,
                    prospective=None,
                    load_increment_by_layer=load,
                    anchor_ledger=anchor_ledger,
                    anchor_factory=anchor_factory,
                    history_factory=history_factory,
                    batch_size=scale.batch_size,
                )
                anchor_wall += time.perf_counter() - anchor_started
                commit = joint_transaction["weight"]
                history_commit = joint_transaction["history"]
                new_anchor_receipt = joint_transaction["anchor_capture"]
                anchor_append = int(joint_transaction["anchor_append_count"])
            else:
                native_z_capture = (
                    OfficialNativeZCapture(
                        role=role,
                        requests=requests,
                        hparams=memit_hparams if role == MEMIT_ROLE else hparams,
                    )
                    if accepted_z_observation_enabled
                    else None
                )
                if role == NATIVE_CORRECTED_ROLE:
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        with (
                            native_z_capture
                            if native_z_capture is not None
                            else contextlib.nullcontext()
                        ):
                            native_payload, _ = run_official_native_apply(
                                model,
                                tokenizer,
                                requests,
                                hparams,
                                touched=touched,
                                reset_cache=round_index == 1,
                                cache_history_width=history_width,
                            )
                    finally:
                        counter.close()
                    cache_contract = native_payload["alphaedit_dynamic_cache_contract"]
                    if not isinstance(cache_contract, Mapping):
                        raise ODEBFStateError("Corrected Native AlphaEdit cache receipt is absent")
                    if round_index > 1:
                        if (
                            cache_contract["entry"]["sha256"]
                            != alphaedit_prior_cache_exit_sha256
                            or not cache_contract["solver_consumed_entry_cache"]
                        ):
                            raise ODEBFStateError("Corrected Native AlphaEdit cache continuity differs")
                    static_identity = canonical_hash(cache_contract["static_projection"])
                    if alphaedit_static_projection_identity is None:
                        alphaedit_static_projection_identity = static_identity
                    elif static_identity != alphaedit_static_projection_identity:
                        raise ODEBFStateError("Corrected Native static projector identity differs")
                    alphaedit_prior_cache_exit_sha256 = cache_contract["exit"]["sha256"]
                elif role == MEMIT_ROLE:
                    if memit_hparams is None:
                        raise ODEBFStateError("Official MEMIT hparams are absent")
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        with (
                            native_z_capture
                            if native_z_capture is not None
                            else contextlib.nullcontext()
                        ):
                            native_payload, _ = run_official_memit_apply(
                                model,
                                tokenizer,
                                requests,
                                memit_hparams,
                                touched=touched,
                            )
                    finally:
                        counter.close()
                    covariance_cache = native_payload["covariance_cache"]
                    if round_index > 1 and (
                        covariance_cache["entry"]["identity_sha256"]
                        != memit_prior_covariance_exit_identity
                        or int(covariance_cache["entry_reused_count"])
                        != len(P1R23_LAYER_ORDER)
                    ):
                        raise ODEBFStateError("Official MEMIT covariance cache continuity differs")
                    memit_prior_covariance_exit_identity = covariance_cache["exit"][
                        "identity_sha256"
                    ]
                else:
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        with (
                            native_z_capture
                            if native_z_capture is not None
                            else contextlib.nullcontext()
                        ):
                            native_payload, _ = run_official_native_apply(
                                model, tokenizer, requests, hparams, touched=touched
                            )
                    finally:
                        counter.close()
                if native_z_capture is not None:
                    accepted_z_binding = native_z_capture.finalize()
                target_backward_count = int(native_payload.get("target_backward_count", 0))
                job_ledger.increment("backward", target_backward_count)
                job_ledger.increment("target_backward", target_backward_count)
                if _model_w0_contract(touched) == entry_contract:
                    raise ODEBFStateError("Official sequential baseline B10 produced no physical transition")
                transaction_id = f"{role}-b{round_index:02d}-{native_payload['identity_sha256']}"
                commit = {
                    "transaction_id": transaction_id,
                    "touched_weights": tuple(sorted(touched)),
                    "commit_count": 1,
                    "rollback_count": 0,
                    "post_commit_verified": True,
                    "parameter_sha256": tuple(sorted(_hashes(touched).items())),
                }
                candidate_receipt = {"native_direct_apply": True, "identity_sha256": native_payload["identity_sha256"]}
                prospective = None
                load = {}
                atomic_payload = native_payload
                h_payload = []
                anchor_started = time.perf_counter()
                try:
                    new_anchors, new_anchor_receipt = capture_controller_legal_lifetime_anchors(
                        model,
                        tokenizer,
                        requests,
                        contexts,
                        alias=alias,
                        round_index=round_index,
                        collision_by_request=collision_by_request,
                        endpoint_weight_sha256=canonical_hash(_hashes(touched)),
                    )
                    anchor_append = anchor_ledger.append_once(transaction_id, new_anchors)
                except BaseException:
                    anchor_ledger.restore(prior_anchors, prior_anchor_finalized)
                    with mutation_lock, torch.no_grad():
                        for name, parameter in touched.items():
                            parameter.copy_(entry_values[name].to(device=parameter.device, dtype=parameter.dtype))
                    if _model_w0_contract(touched) != entry_contract:
                        raise ODEBFStateError("Native sequential anchor rollback differs")
                    raise
                anchor_wall += time.perf_counter() - anchor_started
                history_commit = {
                    "status": (
                        "OFFICIAL_ALPHAEDIT_DYNAMIC_CACHE_C"
                        if role == NATIVE_CORRECTED_ROLE
                        else "STATIC_MEMIT_COVARIANCE_CACHE_ONLY"
                        if role == MEMIT_ROLE
                        else "SUPERSEDED_NATIVE_CACHE_RESET_PER_B10"
                    )
                }

            committed_hashes = _hashes(touched)
            committed_contract = _model_w0_contract(touched)
            if not committed_contract:
                raise ODEBFStateError("P1R52 sequential committed contract is absent")
            if role in OFFICIAL_BASELINE_ROLES and anchor_append != scale.batch_size:
                raise ODEBFStateError("Official baseline sequential anchor append count differs")
            if anchor_append != scale.batch_size:
                raise ODEBFStateError("P1R52 sequential anchor append count differs")
            expected_anchor_count = round_index * scale.batch_size
            if len(anchor_ledger.anchors) != expected_anchor_count:
                raise ODEBFStateError("P1R52 sequential anchor count differs")
            if role in R52_ROLES and sequential_state.history_entry_count() != expected_anchor_count:
                raise ODEBFStateError("P1R52 sequential history count differs")

            snapshot_sha = canonical_hash(committed_hashes)
            loaded_cases, _ = _action_frozen_cases(
                dataset_path,
                requests,
                arm=role,
                selected_snapshot_sha256=snapshot_sha,
                fixed_budget_slots_completed=8 if role in R52_ROLES else 0,
            )
            accepted_z_observation: dict[str, Any] | None = None
            if accepted_z_observation_enabled:
                if (
                    accepted_z_binding is None
                    or reference is None
                    or reference_path is None
                ):
                    raise ODEBFStateError("accepted-z native binding is absent")
                if reference.get("commit_weight_sha256") != committed_hashes:
                    raise ODEBFStateError("accepted-z replay physical commit differs")
                before_observation = _hashes(touched)
                accepted_z_observation, observation_wall = evaluate_accepted_z_batch(
                    model,
                    tokenizer,
                    requests,
                    loaded_cases,
                    role=role,
                    round_index=round_index,
                    binding=accepted_z_binding,
                    committed_weight_sha256=committed_hashes,
                )
                if _hashes(touched) != before_observation:
                    raise ODEBFStateError("accepted-z observation mutated model state")
                accepted_z_observation["sealed_reference_terminal_sha256"] = hashlib.sha256(
                    reference_path.read_bytes()
                ).hexdigest()
                accepted_z_observation["sealed_commit_weight_exact"] = True
                accepted_z_observation["identity_sha256"] = canonical_hash(
                    {
                        key: value
                        for key, value in accepted_z_observation.items()
                        if key != "identity_sha256"
                    }
                )
                accepted_z_observation["receipt_sha256"] = _atomic_write_once(
                    case_root / "accepted-z-rephrase-observation.json",
                    accepted_z_observation,
                )
                accepted_z_observations.append(accepted_z_observation)
                evaluation_wall += observation_wall
                _record_evaluator_compute(
                    job_ledger,
                    accepted_z_observation["scores"],
                    component="accepted_z_rephrase_observation",
                    wall_seconds=observation_wall,
                )
            cohort_cases.append(tuple(loaded_cases))
            cohort_requests.append(requests)
            checkpoint_evaluations: list[dict[str, Any]] = []
            if accepted_z_observation_enabled:
                assert reference is not None
                checkpoint_evaluations = [
                    *reference["prior_history_evaluation"],
                    reference["current_batch_evaluation"],
                ]
                if len(checkpoint_evaluations) != round_index:
                    raise ODEBFStateError("accepted-z sealed checkpoint count differs")
                current_eval = reference["current_batch_evaluation"]
                aggregate = reference["cumulative_evaluation"]
            else:
                for cohort_index, cases in enumerate(cohort_cases, start=1):
                    receipt, wall = _evaluate_cohort(
                        model,
                        tokenizer,
                        cases,
                        alias=alias,
                        role=role,
                        cohort_index=cohort_index,
                        checkpoint_index=round_index,
                        snapshot_sha256=snapshot_sha,
                        slots=8 if role in R52_ROLES else 0,
                        expected_batch_size=scale.batch_size,
                    )
                    checkpoint_evaluations.append(receipt)
                    evaluation_wall += wall
                    _record_evaluator_compute(
                        job_ledger,
                        receipt,
                        component="batch_terminal_checkpoint_evaluator",
                        wall_seconds=wall,
                    )
                current_eval = checkpoint_evaluations[-1]
                aggregate = _aggregate_metric_payloads(
                    checkpoint_evaluations,
                    batch_size=scale.batch_size,
                )
            immediate_post_evaluations.append(current_eval)
            pre_summary = (
                _evaluation_summary(entry_pre)
                if entry_pre is not None
                else "REMOVED_BY_USER_AMENDMENT"
            )
            post_summary = _evaluation_summary(current_eval)
            post_minus_pre = (
                _summary_delta(post_summary, pre_summary)
                if isinstance(pre_summary, Mapping)
                else "NOT_RECORDED_BY_USER_AMENDMENT"
            )

            terminal_observed: dict[str, np.ndarray] = {}
            if accepted_z_observation_enabled:
                assert reference is not None
                terminal_anchor_receipt = reference["history_terminal_anchor_observation"]
                drift = reference["historical_damage"]
            elif prior_requests:
                anchor_started = time.perf_counter()
                terminal_observed, terminal_anchor_receipt = _capture_observation(
                    model,
                    tokenizer,
                    prior_requests,
                    contexts,
                    alias=alias,
                    round_index=round_index,
                    collision_by_request=collision_by_request,
                    endpoint_weight_sha256=snapshot_sha,
                )
                anchor_wall += time.perf_counter() - anchor_started
                drift = _drift_payload(prior_anchors, entry_observed, terminal_observed)
            else:
                terminal_anchor_receipt = {"status": "EMPTY_HISTORY", "model_forward_count": 0}
                drift = {
                    "status": "EMPTY_HISTORY",
                    "local": _summary([]),
                    "lifetime": _summary([]),
                    "per_request": [],
                    "observation_only": True,
                    "decision_influence_count": 0,
                }
                drift["identity_sha256"] = canonical_hash(drift)
            if round_index == 1 and role in R52_ROLES and scale == P1R52_B10X10_SCALE:
                b1_gate = _b1_identity_gate(
                    case_root,
                    committed_hashes=committed_hashes,
                    evaluation=current_eval,
                    rollout=atomic_payload,
                )
            else:
                b1_gate = {
                    "status": (
                        "NOT_APPLICABLE_B100X10_SCALE"
                        if scale != P1R52_B10X10_SCALE
                        else "NOT_APPLICABLE"
                    )
                }

            batch_payload = {
                "schema": f"ode-edit-s05-p1r52-sequential-{scale.scale_id}-terminal/v1",
                "instruction_id": scale.instruction_id,
                "method_id": (
                    METHOD_ID
                    if role == R52_H_ROLE
                    else "P1R52-REPAIR-R1-LLAMA-SOFT-SEQUENTIAL-ALPHACACHE-ON-STRUCTURALH-OFF-V1"
                    if role == R52_CONTROL_ROLE
                    else "OFFICIAL-ALPHAEDIT-SEQUENTIAL-CACHE-ON-CORRECTED-V1"
                    if role == NATIVE_CORRECTED_ROLE
                    else "OFFICIAL-MEMIT-SEQUENTIAL-V1"
                    if role == MEMIT_ROLE
                    else "OFFICIAL-ALPHAEDIT-SEQUENTIAL-CACHE-RESET-PER-B10-SUPERSEDED-V1"
                ),
                "role": role,
                "scale_id": scale.scale_id,
                "batch_size": scale.batch_size,
                "round": round_index,
                "history_width_at_entry": history_width,
                "entry_weight_sha256": entry_hashes,
                "commit_weight_sha256": committed_hashes,
                "next_entry_identity_required": True,
                "interbatch_W0_restore_count": 0,
                "controller_reset_count": 1 if role in R52_ROLES else 0,
                "atomic_or_native": atomic_payload,
                "candidate": candidate_receipt,
                "weight_transaction": asdict(commit) if not isinstance(commit, dict) else commit,
                "history_transaction": asdict(history_commit) if not isinstance(history_commit, dict) else history_commit,
                "anchor_capture": new_anchor_receipt,
                "anchor_append_count": anchor_append,
                "active_history_count": sequential_state.history_entry_count() if role in R52_ROLES else 0,
                "alpha_solve_history_width": history_width if role in R52_ROLES else 0,
                "alpha_solve_cache_consume_count": history_width if role in R52_ROLES else 0,
                "alpha_solve_cache_append_count": scale.batch_size if role in R52_ROLES else 0,
                "historical_control_status": (
                    "ALPHA_CACHE_ON_STRUCTURAL_H_OFF"
                    if role == R52_CONTROL_ROLE
                    else "HISTORICAL_ACTIVE"
                    if role == R52_H_ROLE
                    else "OFFICIAL_ALPHAEDIT_DYNAMIC_CACHE_C_ON"
                    if role == NATIVE_CORRECTED_ROLE
                    else "OFFICIAL_MEMIT_STATIC_COVARIANCE_CACHE"
                    if role == MEMIT_ROLE
                    else "SUPERSEDED_NATIVE_CACHE_RESET_PER_B10"
                ),
                "official_baseline_cache": (
                    native_payload.get("alphaedit_dynamic_cache_contract")
                    if role == NATIVE_CORRECTED_ROLE
                    else native_payload.get("covariance_cache")
                    if role == MEMIT_ROLE
                    else None
                ),
                "structural_h_decision_history_width": history_width if role == R52_H_ROLE else 0,
                "structural_h_decision_influence_count": 0 if role == R52_CONTROL_ROLE else sum(
                    int(item["status"] == "H_ACTIVE_CERTIFIED") for item in h_payload
                ),
                "risk_observation_width": history_width if role == R52_CONTROL_ROLE else 0,
                "physical_weight_persistence": True,
                "observation_ledger_decision_influence_count": 0,
                "control_added_model_forward_count": 0,
                "control_added_backward_count": 0,
                "control_added_materialization_count": 0,
                "lifetime_anchor_count": len(anchor_ledger.anchors),
                "structural_h_routing": h_payload,
                "current_batch_evaluation": current_eval,
                "current_batch_immediate_post_summary": post_summary,
                "current_batch_post_minus_pre": post_minus_pre,
                "prior_history_evaluation": checkpoint_evaluations[:-1],
                "cumulative_evaluation": aggregate,
                "history_entry_anchor_observation": entry_anchor_receipt,
                "history_terminal_anchor_observation": terminal_anchor_receipt,
                "historical_damage": drift,
                "B1_atomic_identity": b1_gate,
                "action_freeze": True,
                "heldout_controller_influence_count": 0,
                "batch_entry_pre_controller_routing_history_anchor_influence": [0, 0, 0, 0],
                "batch_entry_pre_backward_generation_count": [0, 0],
                "B1_cross_arm_entry_equality_required": round_index == 1,
                "B1_cross_arm_entry_equality_status": (
                    "PENDING_PAIRED_TERMINAL_INTEGRITY" if round_index == 1 else "NOT_APPLICABLE"
                ),
                "batch_entry_metrics_status": (
                    "RECORDED_LEGACY_SCOPE"
                    if batch_entry_evaluation_enabled
                    else "REMOVED_BY_USER_AMENDMENT"
                ),
                "accepted_z_rephrase_observation": accepted_z_observation,
                "sealed_W_evaluator_receipts_reused": accepted_z_observation_enabled,
                "duplicate_W_evaluator_model_forward_count": 0,
                "retry_count": 0,
                "backtracking_count": 0,
            }
            if batch_entry_evaluation_enabled:
                batch_payload["batch_entry_pre_evaluation"] = entry_pre
                batch_payload["batch_entry_pre_summary"] = pre_summary
            batch_payload["identity_sha256"] = canonical_hash(batch_payload)
            batch_sha = _atomic_write_once(case_root / "terminal.json", batch_payload)
            checkpoint_rows.append(
                {
                    "round": round_index,
                    "terminal_sha256": batch_sha,
                    "history_width_at_entry": history_width,
                    "commit_weight_sha256": committed_hashes,
                    "entry_pre_summary": pre_summary,
                    "immediate_post_summary": post_summary,
                    "post_minus_pre": post_minus_pre,
                    "cumulative_evaluation": aggregate,
                    "historical_damage": drift,
                }
            )
            stages.record(
                f"post_sequential_{role}_b{round_index:02d}",
                {
                    "round": round_index,
                    "history_width_at_entry": history_width,
                    "commit_weight_sha256": committed_hashes,
                    "W0_restored": False,
                    "active_history_count": sequential_state.history_entry_count() if role in R52_ROLES else 0,
                    "anchor_count": len(anchor_ledger.anchors),
                },
            )
            prior_commit_hashes = committed_hashes

        final_snapshot_sha = canonical_hash(prior_commit_hashes)
        if accepted_z_observation_enabled:
            assert accepted_z_reference_root is not None
            sealed_terminal_path = accepted_z_reference_root / "terminal.json"
            sealed_terminal = json.loads(
                sealed_terminal_path.read_text(encoding="utf-8")
            )
            final_evaluations = sealed_terminal[
                f"final_{scale.final_label}_cohort_receipts"
            ]
            final_b100 = sealed_terminal[f"final_{scale.final_label}"]
            if len(final_evaluations) != scale.round_count:
                raise ODEBFStateError("accepted-z sealed final cohort count differs")
        else:
            final_evaluations = []
            for cohort_index, cases in enumerate(cohort_cases, start=1):
                receipt, wall = _evaluate_cohort(
                    model,
                    tokenizer,
                    cases,
                    alias=alias,
                    role=f"{role}-{scale.final_label}-terminal",
                    cohort_index=cohort_index,
                    checkpoint_index=scale.round_count,
                    snapshot_sha256=final_snapshot_sha,
                    slots=8 if role in R52_ROLES else 0,
                    expected_batch_size=scale.batch_size,
                )
                final_evaluations.append(receipt)
                evaluation_wall += wall
                _record_evaluator_compute(
                    job_ledger,
                    receipt,
                    component=f"final_W10_{scale.final_label}_evaluator",
                    wall_seconds=wall,
                )
            final_b100 = _aggregate_metric_payloads(
                final_evaluations,
                batch_size=scale.batch_size,
            )
        immediate_post_all = _aggregate_metric_payloads(
            immediate_post_evaluations,
            batch_size=scale.batch_size,
        )
        if batch_entry_evaluation_enabled:
            true_entry_pre_all: dict[str, Any] | str = _aggregate_metric_payloads(
                entry_pre_evaluations,
                batch_size=scale.batch_size,
            )
            request_comparison_rows = build_pre_post_final_request_rows(
                cohort_requests,
                entry_pre_evaluations,
                immediate_post_evaluations,
                final_evaluations,
                scale=scale,
            )
            cohort_comparison_rows = [
                {
                    "round": round_index,
                    "batch_age_at_W10": scale.round_count - round_index,
                    "history_width_at_entry": scale.history_counts[round_index - 1],
                    "entry_pre": _evaluation_summary(pre),
                    "immediate_post": _evaluation_summary(post),
                    "final_W10": _evaluation_summary(final),
                    "post_minus_pre": _summary_delta(
                        _evaluation_summary(post), _evaluation_summary(pre)
                    ),
                    "final_minus_post": _summary_delta(
                        _evaluation_summary(final), _evaluation_summary(post)
                    ),
                }
                for round_index, (pre, post, final) in enumerate(
                    zip(
                        entry_pre_evaluations,
                        immediate_post_evaluations,
                        final_evaluations,
                        strict=True,
                    ),
                    start=1,
                )
            ]
        else:
            true_entry_pre_all = "REMOVED_BY_USER_AMENDMENT"
            request_comparison_rows = build_post_final_request_rows(
                cohort_requests,
                immediate_post_evaluations,
                final_evaluations,
                scale=scale,
            )
            cohort_comparison_rows = [
                {
                    "round": round_index,
                    "batch_age_at_W10": scale.round_count - round_index,
                    "history_width_at_entry": scale.history_counts[round_index - 1],
                    "batch_entry_metrics_status": "REMOVED_BY_USER_AMENDMENT",
                    "immediate_post": _evaluation_summary(post),
                    "final_W10": _evaluation_summary(final),
                    "final_minus_post": _summary_delta(
                        _evaluation_summary(final), _evaluation_summary(post)
                    ),
                }
                for round_index, (post, final) in enumerate(
                    zip(immediate_post_evaluations, final_evaluations, strict=True),
                    start=1,
                )
            ]
    finally:
        terminal_restore = _restore_exact_w0(
            touched,
            base_values,
            mutation_lock=mutation_lock,
            expected_contract=expected_w0,
        )
        if any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched):
            raise ODEBFStateError("P1R52 sequential terminal W0 pointer differs")

    checkpoint_table = {
        "schema": f"ode-edit-s05-p1r52-sequential-{scale.scale_id}-pre-post-checkpoints/v1",
        "role": role,
        "scale_id": scale.scale_id,
        "rows": checkpoint_rows,
        "row_count": len(checkpoint_rows),
    }
    checkpoint_table["identity_sha256"] = canonical_hash(checkpoint_table)
    checkpoint_filename = (
        "b1-b10-pre-post-checkpoints.json"
        if batch_entry_evaluation_enabled
        else "b1-b10-post-final-checkpoints.json"
    )
    checkpoint_table_sha = _atomic_write_once(destination / checkpoint_filename, checkpoint_table)
    request_table = {
        "schema": f"ode-edit-s05-p1r52-sequential-{scale.scale_id}-pre-post-final-requests/v1",
        "role": role,
        "scale_id": scale.scale_id,
        "rows": request_comparison_rows,
        "row_count": len(request_comparison_rows),
    }
    request_table["identity_sha256"] = canonical_hash(request_table)
    request_filename = (
        "entry-pre-immediate-post-final-w10-requests.json"
        if batch_entry_evaluation_enabled
        else "immediate-post-final-w10-requests.json"
    )
    request_table_sha = _atomic_write_once(destination / request_filename, request_table)
    cohort_table = {
        "schema": f"ode-edit-s05-p1r52-sequential-{scale.scale_id}-batch-age-cohorts/v1",
        "role": role,
        "scale_id": scale.scale_id,
        "rows": cohort_comparison_rows,
        "row_count": len(cohort_comparison_rows),
    }
    cohort_table["identity_sha256"] = canonical_hash(cohort_table)
    cohort_filename = (
        "batch-age-pre-post-final-cohorts.json"
        if batch_entry_evaluation_enabled
        else "batch-age-post-final-cohorts.json"
    )
    cohort_table_sha = _atomic_write_once(destination / cohort_filename, cohort_table)
    aggregate_table = {
        "schema": f"ode-edit-s05-p1r52-sequential-{scale.scale_id}-post-final-aggregate/v1",
        "role": role,
        "scale_id": scale.scale_id,
        f"immediate_post_B{scale.batch_size}_panels": immediate_post_all,
        f"final_W10_{scale.final_label}": final_b100,
        "final_W10_minus_immediate_post": _aggregate_delta(
            final_b100, immediate_post_all
        ),
        "batch_entry_metrics_status": (
            "RECORDED_LEGACY_SCOPE"
            if batch_entry_evaluation_enabled
            else "REMOVED_BY_USER_AMENDMENT"
        ),
    }
    if batch_entry_evaluation_enabled:
        aggregate_table[f"true_entry_pre_B{scale.batch_size}_panels"] = true_entry_pre_all
        aggregate_table["immediate_post_minus_entry_pre"] = _aggregate_delta(
            immediate_post_all, true_entry_pre_all
        )
    aggregate_table["identity_sha256"] = canonical_hash(aggregate_table)
    aggregate_filename = (
        "true-entry-post-final-aggregate.json"
        if batch_entry_evaluation_enabled
        else "post-final-aggregate.json"
    )
    aggregate_table_sha = _atomic_write_once(destination / aggregate_filename, aggregate_table)

    accepted_z_table_sha: str | None = None
    accepted_z_table_path: Path | None = None
    if accepted_z_observation_enabled:
        if len(accepted_z_observations) != scale.round_count:
            raise ODEBFStateError("accepted-z observation round count differs")
        accepted_z_table = {
            "schema": "ode-edit-s05-p1r52-b100-accepted-z-rephrase-observations/v1",
            "role": role,
            "rows": accepted_z_observations,
            "row_count": len(accepted_z_observations),
            "batch_entry_W_metrics_count": 0,
            "imputation_count": 0,
        }
        accepted_z_table["identity_sha256"] = canonical_hash(accepted_z_table)
        accepted_z_table_path = destination / "accepted-z-rephrase-observations.json"
        accepted_z_table_sha = _atomic_write_once(
            accepted_z_table_path, accepted_z_table
        )

    terminal = {
        "schema": f"ode-edit-s05-p1r52-sequential-10xb{scale.batch_size}-terminal/v1",
        "instruction_id": scale.instruction_id,
        "method_id": (
            METHOD_ID
            if role == R52_H_ROLE
            else "P1R52-REPAIR-R1-LLAMA-SOFT-SEQUENTIAL-ALPHACACHE-ON-STRUCTURALH-OFF-V1"
            if role == R52_CONTROL_ROLE
            else "OFFICIAL-ALPHAEDIT-SEQUENTIAL-CACHE-ON-CORRECTED-V1"
            if role == NATIVE_CORRECTED_ROLE
            else "OFFICIAL-MEMIT-SEQUENTIAL-V1"
            if role == MEMIT_ROLE
            else "OFFICIAL-ALPHAEDIT-SEQUENTIAL-CACHE-RESET-PER-B10-SUPERSEDED-V1"
        ),
        "source_head": source_head,
        "alias": alias,
        "role": role,
        "scale_id": scale.scale_id,
        "round_count": scale.round_count,
        "batch_size": scale.batch_size,
        "request_count": scale.request_count,
        "checkpoint_rows": checkpoint_rows,
        f"immediate_post_all_B{scale.batch_size}": immediate_post_all,
        f"final_{scale.final_label}": final_b100,
        f"final_{scale.final_label}_cohort_receipts": final_evaluations,
        "immediate_post_receipts": immediate_post_evaluations,
        "batch_entry_metrics_status": (
            "RECORDED_LEGACY_SCOPE"
            if batch_entry_evaluation_enabled
            else "REMOVED_BY_USER_AMENDMENT"
        ),
        "comparison_tables": {
            "checkpoint": {
                "path": str(destination / checkpoint_filename),
                "sha256": checkpoint_table_sha,
                "rows": len(checkpoint_rows),
            },
            "request": {
                "path": str(destination / request_filename),
                "sha256": request_table_sha,
                "rows": len(request_comparison_rows),
            },
            "cohort": {
                "path": str(destination / cohort_filename),
                "sha256": cohort_table_sha,
                "rows": len(cohort_comparison_rows),
            },
            "aggregate": {
                "path": str(destination / aggregate_filename),
                "sha256": aggregate_table_sha,
                "rows": 1,
            },
        },
        "history_widths": list(scale.history_counts),
        "terminal_active_history_count": sequential_state.history_entry_count() if role in R52_ROLES else 0,
        "terminal_official_method_cache_history_count": (
            scale.request_count if role == NATIVE_CORRECTED_ROLE else 0
        ),
        "official_method_cache_kind": (
            "DYNAMIC_ALPHAEDIT_KEY_OUTER_PRODUCT_CACHE_C"
            if role == NATIVE_CORRECTED_ROLE
            else "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE"
            if role == MEMIT_ROLE
            else "NONE_OR_SUPERSEDED"
        ),
        "terminal_lifetime_anchor_count": len(anchor_ledger.anchors),
        "interbatch_W0_restore_count": 0,
        "terminal_W0_restore_count": 1,
        "terminal_W0_restore": terminal_restore,
        "action_freeze_checkpoint_count": scale.round_count,
        "batch_entry_pre_evaluator_count": len(entry_pre_evaluations),
        "batch_entry_pre_evaluator_forward_count": sum(
            int(item["legacy_primary"]["model_forward_count"])
            for item in entry_pre_evaluations
        ),
        "batch_entry_pre_evaluator_token_count": sum(
            int(item["legacy_primary"]["processed_token_count"])
            for item in entry_pre_evaluations
        ),
        "accepted_z_observation_enabled": accepted_z_observation_enabled,
        "accepted_z_observation_count": len(accepted_z_observations),
        "accepted_z_observation_model_forward_count": sum(
            int(item["model_forward_count"]) for item in accepted_z_observations
        ),
        "accepted_z_observation_processed_token_count": sum(
            int(item["processed_token_count"]) for item in accepted_z_observations
        ),
        "accepted_z_observation_added_backward_count": 0,
        "accepted_z_observation_added_generation_call_count": 0,
        "accepted_z_observation_action_influence_count": 0,
        "sealed_W_evaluator_receipts_reused": accepted_z_observation_enabled,
        "duplicate_W_evaluator_model_forward_count": 0,
        "accepted_z_observation_table": (
            {
                "path": str(accepted_z_table_path),
                "sha256": accepted_z_table_sha,
                "rows": len(accepted_z_observations),
            }
            if accepted_z_table_path is not None
            else None
        ),
        "batch_entry_pre_evaluator_backward_count": 0,
        "batch_entry_pre_evaluator_generation_count": 0,
        "batch_entry_pre_evaluator_controller_routing_history_anchor_influence": [0, 0, 0, 0],
        "B1_cross_arm_entry_equality_gate": "REQUIRED_IN_PAIRED_TERMINAL_INTEGRITY",
        f"full_{scale.final_label}_terminal_evaluation_count": 1,
        "inner_K_step_heldout_evaluation_count": 0,
        "evaluator_controller_influence_count": 0,
        "accuracy_amendment_added_model_forward_count": 0,
        "accuracy_amendment_added_backward_count": 0,
        "accuracy_amendment_added_generation_count": 0,
        "evaluation_wall_seconds": evaluation_wall,
        "anchor_capture_wall_seconds": anchor_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    if batch_entry_evaluation_enabled:
        terminal[f"true_entry_pre_all_B{scale.batch_size}"] = true_entry_pre_all
        terminal["entry_pre_receipts"] = entry_pre_evaluations
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"ode-edit-s05-p1r52-sequential-10xb{scale.batch_size}-manifest/v1",
        "source_head": source_head,
        "role": role,
        "terminal_sha256": terminal_sha,
        "batch_terminal_sha256": [row["terminal_sha256"] for row in checkpoint_rows],
        "comparison_table_sha256": {
            checkpoint_filename: checkpoint_table_sha,
            request_filename: request_table_sha,
            cohort_filename: cohort_table_sha,
            aggregate_filename: aggregate_table_sha,
        },
        "scale_id": scale.scale_id,
        "round_count": scale.round_count,
        "batch_size": scale.batch_size,
        "request_count": scale.request_count,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_SEQUENTIAL_CELL_TERMINAL",
        "role": role,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "round_count": scale.round_count,
        "request_count": scale.request_count,
        "W0_restored": True,
    }


__all__ = [
    "EXECUTION_ROLES",
    "MEMIT_ROLE",
    "NATIVE_CORRECTED_ROLE",
    "NATIVE_ROLE",
    "OFFICIAL_BASELINE_ROLES",
    "RESULT_NAMES",
    "RESULT_NAMES_B100X10",
    "RESULT_NAMES_B100X10_TECH_R1",
    "RESULT_NAMES_B100X10_TECH_R2",
    "RESULT_NAMES_B100X10_TECH_R3",
    "RESULT_NAMES_B100X10_TECH_R3_RELEASE_R1",
    "R52_CONTROL_ROLE",
    "R52_H_ROLE",
    "ROLES",
    "build_pre_post_final_request_rows",
    "expected_p1r52_sequential_result_name",
    "run_p1r52_sequential",
    "structural_h_off_control_receipts",
]
