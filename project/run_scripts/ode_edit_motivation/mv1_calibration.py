"""MV-1 C0 inference-only predictive-heterogeneity calibration.

This runner reuses the pinned MV-0 runtime and EasyEdit bridge.  It writes
only case IDs, hashes, scalar utilities, and resource metadata below
``local/``.  It never persists prompts, targets, full logits, weights, or
generations, and it never recomputes covariance/projector artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    canonical_json,
)
from .diagnostic_math import c_squared_norm, exact_top1_stop, smooth_target_utility
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .hooks import (
    TemporaryExactMemitApplication,
    TemporaryLowRankApplication,
)
from .manifests import (
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    CounterFactSelectionManifest,
    fixed_model_spec,
    generate_counterfact_selection,
    load_counterfact_requests,
    preflight_fixed_artifacts,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    REPOSITORY_ROOT,
    MV0Error,
    RollbackError,
    SanitizedJsonlWriter,
    TeacherForcedResult,
    _bridge_pins,
    _covariance_specs,
    _event_seed,
    _file_sha256,
    _freeze_contexts,
    _git_runtime_state,
    _load_hparams,
    _load_verified_covariances,
    _local_run_directory,
    _relative_provenance,
    _safe_payload,
    _silence_upstream,
    _weight_hashes,
    _write_json_exclusive,
)


FRACTIONS = (1.0 / 256.0, 1.0 / 64.0, 1.0 / 16.0)
PROBE_RATIO = 0.25
NEAR_TIE_TOLERANCE = 0.0
PILOT_CASE_COUNT = 3
CALIBRATION_CASE_COUNT = 20
ACTION_UNIFORM = "uniform"


class MV1Error(MV0Error):
    """Fail-closed MV-1 contract violation."""


@dataclass(frozen=True, slots=True)
class RewriteMetrics:
    utility: float
    context_utility: tuple[float, ...]
    nll: float
    context_nll: tuple[float, ...]
    exact_margin_min: float
    exact_satisfied: bool
    target_token_count: int
    logits_hash: str

    def compact(self) -> dict[str, Any]:
        return {
            "utility": self.utility,
            "context_utility": list(self.context_utility),
            "nll": self.nll,
            "context_nll": list(self.context_nll),
            "exact_margin_min": self.exact_margin_min,
            "exact_satisfied": self.exact_satisfied,
            "target_token_count": self.target_token_count,
            "logits_hash": self.logits_hash,
        }


@dataclass(frozen=True, slots=True)
class ExactTeacherBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    input_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionDirection:
    action_id: str
    proposal: MemitFactorProposal
    c_squared_norm: float

    def __post_init__(self) -> None:
        if not self.action_id or self.c_squared_norm <= 0:
            raise ContractError("action direction requires a positive C norm")
        if not math.isfinite(self.c_squared_norm):
            raise ContractError("action direction C norm must be finite")


def _cpu_factor(factor: LowRankFactor) -> LowRankFactor:
    return LowRankFactor(
        weight_name=factor.weight_name,
        left=factor.left.detach().cpu().float(),
        right=factor.right.detach().cpu().float(),
        expected_weight_sha256=factor.expected_weight_sha256,
        native_update_transposed=factor.native_update_transposed,
    )


def proposal_c_energy(
    proposal: MemitFactorProposal,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> float:
    """Return block-diagonal C energy without dense update materialization."""

    total = 0.0
    for factor in proposal.factors:
        layer = layer_by_weight.get(factor.weight_name)
        if layer is None or layer not in covariance_by_layer:
            raise ContractError(f"missing covariance mapping for {factor.weight_name}")
        value = float(
            c_squared_norm(
                _cpu_factor(factor),
                covariance_by_layer[layer].detach().cpu().float(),
            )
        )
        if not math.isfinite(value) or value <= 0:
            raise ContractError(f"non-positive C energy for {factor.weight_name}")
        total += value
    if not math.isfinite(total) or total <= 0:
        raise ContractError("proposal C energy must be positive")
    return total


def _proposal_with_factors(
    source: MemitFactorProposal,
    factors: Sequence[LowRankFactor],
    *,
    solver_suffix: str,
) -> MemitFactorProposal:
    return MemitFactorProposal(
        snapshot=source.snapshot,
        factors=tuple(factors),
        semantics=source.semantics,
        solver_name=f"{source.solver_name}/{solver_suffix}",
        residual_denominator=source.residual_denominator,
    )


def scale_proposal(
    proposal: MemitFactorProposal,
    scale: float,
    *,
    solver_suffix: str,
) -> MemitFactorProposal:
    if not math.isfinite(scale):
        raise ContractError("proposal scale must be finite")
    return _proposal_with_factors(
        proposal,
        tuple(factor.scaled(scale) for factor in proposal.factors),
        solver_suffix=solver_suffix,
    )


def build_unit_c_actions(
    synchronous: MemitFactorProposal,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> tuple[ActionDirection, ...]:
    """Build five single-layer and one uniform unit-C directions."""

    if not synchronous.is_synchronous:
        raise ContractError("MV-1 actions require a synchronous proposal")
    if len(synchronous.factors) < 2:
        raise ContractError("MV-1 requires at least two editable layers")
    unit_factors: list[LowRankFactor] = []
    actions: list[ActionDirection] = []
    for factor in synchronous.factors:
        energy = proposal_c_energy(
            _proposal_with_factors(
                synchronous,
                (factor,),
                solver_suffix="energy-probe",
            ),
            covariance_by_layer,
            layer_by_weight,
        )
        unit = factor.scaled(1.0 / math.sqrt(energy))
        unit_factors.append(unit)
        layer = layer_by_weight[factor.weight_name]
        actions.append(
            ActionDirection(
                action_id=f"layer_{layer}",
                proposal=_proposal_with_factors(
                    synchronous,
                    (unit,),
                    solver_suffix=f"unit-c-layer-{layer}",
                ),
                c_squared_norm=1.0,
            )
        )
    uniform_scale = 1.0 / math.sqrt(len(unit_factors))
    uniform = _proposal_with_factors(
        synchronous,
        tuple(factor.scaled(uniform_scale) for factor in unit_factors),
        solver_suffix="unit-c-uniform",
    )
    uniform_energy = proposal_c_energy(uniform, covariance_by_layer, layer_by_weight)
    if not math.isclose(uniform_energy, 1.0, rel_tol=2e-5, abs_tol=2e-5):
        raise ContractError(f"uniform action C energy is {uniform_energy}, expected 1")
    actions.append(
        ActionDirection(
            action_id=ACTION_UNIFORM,
            proposal=uniform,
            c_squared_norm=uniform_energy,
        )
    )
    return tuple(actions)


def assert_exact_action_contract(
    synchronous: MemitFactorProposal,
    ordered: MemitFactorProposal,
    actions: Sequence[ActionDirection],
    *,
    expected_factor_names: Sequence[str],
    expected_action_ids: Sequence[str],
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> None:
    """Fail closed unless proposals and actions match the locked C0 set."""

    factor_names = tuple(expected_factor_names)
    action_ids = tuple(expected_action_ids)
    if (
        tuple(factor.weight_name for factor in synchronous.factors) != factor_names
        or tuple(factor.weight_name for factor in ordered.factors) != factor_names
    ):
        raise MV1Error("MEMIT proposal factors do not match the exact layer set")
    if tuple(action.action_id for action in actions) != action_ids:
        raise MV1Error("runtime action set differs from the locked finite action set")
    if len(actions) != len(factor_names) + 1 or action_ids[-1] != ACTION_UNIFORM:
        raise MV1Error("locked action set must be one action per layer plus uniform")
    for index, action in enumerate(actions):
        expected_names = (
            factor_names
            if index == len(factor_names)
            else (factor_names[index],)
        )
        if tuple(factor.weight_name for factor in action.proposal.factors) != expected_names:
            raise MV1Error("action proposal factors do not match its locked action")
        observed_energy = proposal_c_energy(
            action.proposal,
            covariance_by_layer,
            layer_by_weight,
        )
        if (
            not math.isclose(observed_energy, 1.0, rel_tol=2e-5, abs_tol=2e-5)
            or not math.isclose(
                action.c_squared_norm,
                observed_energy,
                rel_tol=2e-5,
                abs_tol=2e-5,
            )
        ):
            raise MV1Error("runtime action direction is not unit C-normalized")


def select_analytic_action(
    scores: Mapping[str, float],
    *,
    fallback_action: str = ACTION_UNIFORM,
    near_tie_tolerance: float = NEAR_TIE_TOLERANCE,
) -> str:
    """Choose from a finite action set with deterministic fallback."""

    if not scores or fallback_action not in scores:
        raise ContractError("analytic action scores must include the fallback")
    if not math.isfinite(near_tie_tolerance) or near_tie_tolerance < 0:
        raise ContractError("near-tie tolerance must be finite and non-negative")
    normalized: dict[str, float] = {}
    for action_id, value in scores.items():
        if not action_id or not math.isfinite(float(value)):
            raise ContractError("analytic scores require finite named actions")
        normalized[action_id] = float(value)
    best = max(normalized.values())
    tied = sorted(
        action_id
        for action_id, value in normalized.items()
        if best - value <= near_tie_tolerance
    )
    return fallback_action if fallback_action in tied else tied[0]


def build_exact_teacher_batch(
    tokenizer: Any,
    request: EditRequest,
    contexts: Any,
) -> ExactTeacherBatch:
    """Tokenize prefix+target once and prove the target suffix identity."""

    if getattr(tokenizer, "padding_side", None) != "right":
        raise MV1Error("MV-1 teacher forcing requires right padding")
    target_text = request.to_easyedit()["target_new"]
    target_ids = tokenizer.encode(
        target_text,
        add_special_tokens=False,
    )
    if not target_ids:
        raise MV1Error("rewrite target tokenization is empty")
    normalized_targets = tuple(int(value) for value in target_ids)
    templates = [item for group in contexts.templates for item in group]
    input_rows: list[tuple[int, ...]] = []
    for context in templates:
        prefix = context.format(request.prompt).format(request.subject)
        full_ids = tuple(
            int(value)
            for value in tokenizer.encode(
                prefix + target_text,
                add_special_tokens=True,
            )
        )
        if (
            len(full_ids) <= len(normalized_targets)
            or full_ids[-len(normalized_targets) :] != normalized_targets
        ):
            raise MV1Error(
                "prefix+target tokenization does not preserve the target suffix"
            )
        model_input = full_ids[:-1]
        if len(model_input) < len(normalized_targets):
            raise MV1Error("teacher-forced input is shorter than its target")
        input_rows.append(model_input)
    if not input_rows:
        raise MV1Error("teacher-forced context set is empty")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise MV1Error("teacher-forced tokenizer has no valid pad token")
    max_length = max(len(row) for row in input_rows)
    input_ids = torch.full(
        (len(input_rows), max_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(input_rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    return ExactTeacherBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=torch.tensor(normalized_targets, dtype=torch.long),
        input_lengths=tuple(len(row) for row in input_rows),
    )


def teacher_forced_rewrite_exact(
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
) -> tuple[TeacherForcedResult, torch.Tensor]:
    """Evaluate exact causal target positions after suffix-ID verification."""

    batch = build_exact_teacher_batch(runtime.tokenizer, request, contexts)
    device = next(runtime.model.parameters()).device
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
    }
    selected: list[torch.Tensor] = []
    with torch.inference_mode():
        logits = runtime.model(**inputs).logits
        target_count = int(batch.target_ids.numel())
        for row, length in enumerate(batch.input_lengths):
            start = length - target_count
            selected.append(logits[row, start:length, :].float())
    selected_logits = torch.stack(selected, dim=0)
    targets = batch.target_ids.to(selected_logits.device).expand(
        selected_logits.shape[0], -1
    )
    losses = torch.nn.functional.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(selected_logits.shape[:2])
    return (
        TeacherForcedResult(
            logits=selected_logits.detach().cpu().contiguous(),
            nll=float(losses.mean().detach().cpu()),
            context_nll=tuple(
                float(value) for value in losses.mean(dim=1).detach().cpu()
            ),
            target_token_count=target_count,
        ),
        batch.target_ids,
    )


def rewrite_metrics(
    teacher: TeacherForcedResult,
    target_ids: torch.Tensor,
) -> RewriteMetrics:
    logits = teacher.logits.float()
    if (
        logits.ndim != 3
        or target_ids.ndim != 1
        or logits.shape[1] != target_ids.numel()
        or teacher.target_token_count != target_ids.numel()
    ):
        raise MV1Error("teacher-forced logits/target layout mismatch")
    targets = target_ids.expand(logits.shape[0], -1)
    utility = smooth_target_utility(logits, targets, temperature=1.0)
    if utility.shape != logits.shape[:2] or not bool(torch.isfinite(utility).all()):
        raise MV1Error("smooth utility has an invalid shape or value")
    exact = exact_top1_stop(logits, targets)
    return RewriteMetrics(
        utility=float(utility.mean()),
        context_utility=tuple(float(value) for value in utility.mean(dim=1)),
        nll=float(teacher.nll),
        context_nll=tuple(float(value) for value in teacher.context_nll),
        exact_margin_min=min(exact.margins),
        exact_satisfied=exact.all_satisfied,
        target_token_count=int(target_ids.numel()),
        logits_hash=teacher.logits_hash,
    )


def _state_identity(runtime: FixedModelRuntime) -> str:
    payload = {
        "training": bool(runtime.model.training),
        "use_cache": getattr(runtime.model.config, "use_cache", None),
        "padding_side": getattr(runtime.tokenizer, "padding_side", None),
        "requires_grad": [
            [name, bool(parameter.requires_grad)]
            for name, parameter in runtime.model.named_parameters()
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _reset_rng(cpu_state: torch.Tensor, cuda_state: torch.Tensor) -> None:
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, 0)


def _evaluate_branch(
    *,
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
    target_ids: torch.Tensor,
    proposal: MemitFactorProposal | None,
    exact_application: bool,
    base_hashes: Mapping[str, str],
    state_identity: str,
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
) -> RewriteMetrics:
    _reset_rng(cpu_rng, cuda_rng)
    rng_before = rng_state_hash()
    manager: contextlib.AbstractContextManager[Any]
    if proposal is None:
        manager = contextlib.nullcontext()
    elif exact_application:
        manager = TemporaryExactMemitApplication(runtime.model, proposal)
    else:
        manager = TemporaryLowRankApplication(runtime.model, proposal)
    with manager:
        teacher, observed_target_ids = teacher_forced_rewrite_exact(
            runtime,
            request,
            contexts,
        )
        if not torch.equal(observed_target_ids, target_ids):
            raise MV1Error("teacher-forced branch target identity changed")
        metrics = rewrite_metrics(teacher, observed_target_ids)
    if _weight_hashes(runtime.model, base_hashes) != dict(base_hashes):
        raise RollbackError("MV-1 branch did not restore editable weights")
    if _state_identity(runtime) != state_identity:
        raise RollbackError("MV-1 branch changed non-weight runtime state")
    if rng_state_hash() != rng_before:
        raise RollbackError("MV-1 inference branch changed RNG state")
    return metrics


def _metric_difference(
    after: RewriteMetrics,
    before: RewriteMetrics,
) -> dict[str, Any]:
    if (
        len(after.context_utility) != len(before.context_utility)
        or len(after.context_nll) != len(before.context_nll)
    ):
        raise MV1Error("branch context denominator changed")
    return {
        "progress": after.utility - before.utility,
        "context_progress": [
            right - left
            for left, right in zip(before.context_utility, after.context_utility)
        ],
        "nll_reduction": before.nll - after.nll,
        "context_nll_reduction": [
            left - right
            for left, right in zip(before.context_nll, after.context_nll)
        ],
        "exact_margin_min": after.exact_margin_min,
        "exact_satisfied": after.exact_satisfied,
    }


def _fraction_label(value: float) -> str:
    denominator = round(1.0 / value)
    if not math.isclose(value, 1.0 / denominator, rel_tol=0.0, abs_tol=1e-15):
        raise ContractError("fraction must have an exact reciprocal label")
    return f"q_1_{denominator}"


def _feature_hash(payload: Mapping[str, Any]) -> str:
    safe = _safe_payload(payload)
    return hashlib.sha256(canonical_json(safe).encode("utf-8")).hexdigest()


def _layer_by_weight(hparams: Any) -> dict[str, int]:
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": int(layer)
        for layer in hparams.layers
    }


def commit_feature_actions(
    *,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    receipt_root: Path,
    case_id: str,
    request_id: str,
    feature_records: Mapping[float, Mapping[str, Any]],
    committed_actions: Mapping[float, Mapping[str, Any]],
) -> tuple[str, str]:
    """Durably commit the supplied actions before later probes or outcomes."""

    if set(feature_records) != set(committed_actions) or not feature_records:
        raise ContractError("feature/action commitments must cover the same fractions")
    receipt_entries = []
    for fraction in sorted(feature_records):
        feature = feature_records[fraction]
        action = committed_actions[fraction]
        if (
            feature.get("case_id") != case_id
            or feature.get("request_id") != request_id
            or action.get("case_id") != case_id
            or action.get("request_id") != request_id
            or action.get("feature_hash") != feature.get("feature_hash")
        ):
            raise ContractError("feature/action commitment identity mismatch")
        feature_writer.write("mv1_feature", feature)
        action_writer.write("mv1_action_commitment", action)
        receipt_entries.append(
            {
                "case_id": case_id,
                "request_id": request_id,
                "fraction": fraction,
                "feature_hash": feature["feature_hash"],
                "commitment_hash": action["commitment_hash"],
                "action_id": action["action_id"],
            }
        )
    feature_writer.sync()
    action_writer.sync()
    receipt_payload = {
        "schema_version": "ode-edit-mv1-action-receipt/v1",
        "case_id": case_id,
        "request_id": request_id,
        "entries": receipt_entries,
        "durability": "features-and-actions-flush+fsync-before-exclusive-receipt",
    }
    receipt_identity = {
        "case_id": case_id,
        "request_id": request_id,
        "fractions": sorted(feature_records),
    }
    receipt_name = (
        hashlib.sha256(
            canonical_json(receipt_identity).encode("utf-8")
        ).hexdigest()
        + ".json"
    )
    receipt_path = receipt_root / receipt_name
    _write_json_exclusive(receipt_path, receipt_payload)
    return receipt_name, _file_sha256(receipt_path)


def _run_event(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    direct_z_root: Path,
    receipt_root: Path,
    request: EditRequest,
    seed: int,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    outcome_writer: SanitizedJsonlWriter,
) -> dict[str, Any]:
    event_seed = _event_seed(seed, request.case_id)
    seed_runtime(event_seed)
    if (
        len(request.request_id) != 64
        or any(char not in "0123456789abcdef" for char in request.request_id)
    ):
        raise MV1Error("request_id must be a full canonical SHA-256 digest")
    layer_by_weight = _layer_by_weight(hparams)
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    baseline_teacher, target_ids = teacher_forced_rewrite_exact(
        runtime,
        request,
        contexts,
    )
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    with _silence_upstream():
        direct_z = bridge.load_or_compute_direct_z(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            local_cache_root=direct_z_root,
            cache_path=f"{request.request_id}.pt",
        )
        synchronous = bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
        )
        ordered = bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )
    synchronous.assert_same_entry_snapshot(ordered)
    expected_factor_names = tuple(layer_by_weight)
    if _weight_hashes(runtime.model, weight_names) != base_hashes:
        raise RollbackError("proposal construction changed editable weights")
    native_energy = proposal_c_energy(
        ordered,
        covariance_moments,
        layer_by_weight,
    )
    unit_actions = build_unit_c_actions(
        synchronous,
        covariance_moments,
        layer_by_weight,
    )
    expected_action_ids = tuple(
        [*(f"layer_{int(layer)}" for layer in hparams.layers), ACTION_UNIFORM]
    )
    assert_exact_action_contract(
        synchronous,
        ordered,
        unit_actions,
        expected_factor_names=expected_factor_names,
        expected_action_ids=expected_action_ids,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    raw_layer_c_norms = {
        f"layer_{layer_by_weight[factor.weight_name]}": math.sqrt(
            proposal_c_energy(
                _proposal_with_factors(
                    synchronous,
                    (factor,),
                    solver_suffix="raw-layer-energy",
                ),
                covariance_moments,
                layer_by_weight,
            )
        )
        for factor in synchronous.factors
    }
    state_identity = _state_identity(runtime)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()

    probe_cache: dict[tuple[str, float], RewriteMetrics] = {}

    def evaluate_scaled(
        action: ActionDirection,
        signed_distance: float,
    ) -> RewriteMetrics:
        key = (action.action_id, round(signed_distance, 15))
        if key not in probe_cache:
            proposal = scale_proposal(
                action.proposal,
                signed_distance,
                solver_suffix=f"distance-{signed_distance:.12g}",
            )
            probe_cache[key] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=proposal,
                exact_application=False,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return probe_cache[key]

    feature_records: dict[float, dict[str, Any]] = {}
    committed_actions: dict[float, dict[str, Any]] = {}
    action_receipts: dict[str, dict[str, str]] = {}
    for fraction in FRACTIONS:
        distance = math.sqrt(fraction * native_energy)
        epsilon = PROBE_RATIO * distance
        scores: dict[str, float] = {}
        context_scores: dict[str, list[float]] = {}
        for action in unit_actions:
            plus = evaluate_scaled(action, epsilon)
            minus = evaluate_scaled(action, -epsilon)
            scores[action.action_id] = (plus.utility - minus.utility) / (
                2.0 * epsilon
            )
            context_scores[action.action_id] = [
                (right - left) / (2.0 * epsilon)
                for left, right in zip(
                    minus.context_utility,
                    plus.context_utility,
                )
            ]
        feature_payload = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "fraction": fraction,
            "fraction_label": _fraction_label(fraction),
            "native_c_energy": native_energy,
            "operational_c_distance": distance,
            "probe_c_distance": epsilon,
            "action_scores": scores,
            "context_action_scores": context_scores,
            "layer_factor_c_norms": raw_layer_c_norms,
            "feature_policy": "inference-only-central-fd/temperature-1/mean-context-token",
        }
        feature_hash = _feature_hash(feature_payload)
        feature_payload["feature_hash"] = feature_hash
        action_id = select_analytic_action(scores)
        commitment_payload = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "fraction": fraction,
            "fraction_label": _fraction_label(fraction),
            "feature_hash": feature_hash,
            "action_id": action_id,
            "controller": "finite-action-argmax/static-fallback-uniform",
            "near_tie_tolerance": NEAR_TIE_TOLERANCE,
        }
        commitment_hash = _feature_hash(commitment_payload)
        commitment_payload["commitment_hash"] = commitment_hash
        feature_records[fraction] = feature_payload
        committed_actions[fraction] = commitment_payload
        receipt_name, receipt_sha256 = commit_feature_actions(
            feature_writer=feature_writer,
            action_writer=action_writer,
            receipt_root=receipt_root,
            case_id=request.case_id,
            request_id=request.request_id,
            feature_records={fraction: feature_payload},
            committed_actions={fraction: commitment_payload},
        )
        action_receipts[_fraction_label(fraction)] = {
            "name": receipt_name,
            "sha256": receipt_sha256,
        }

    # Operational outcomes are opened only after every feature/action commit.
    operational_cache: dict[tuple[str, float], RewriteMetrics] = {}

    def evaluate_operational(
        action: ActionDirection,
        distance: float,
    ) -> RewriteMetrics:
        key = (action.action_id, round(distance, 15))
        if key not in operational_cache:
            proposal = scale_proposal(
                action.proposal,
                distance,
                solver_suffix=f"operational-distance-{distance:.12g}",
            )
            operational_cache[key] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=proposal,
                exact_application=False,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return operational_cache[key]

    outcome_count = 0
    for fraction in FRACTIONS:
        distance = float(feature_records[fraction]["operational_c_distance"])
        for action in unit_actions:
            after = evaluate_operational(action, distance)
            outcome_writer.write(
                "mv1_operational_outcome",
                {
                    "case_id": request.case_id,
                    "request_id": request.request_id,
                    "fraction": fraction,
                    "fraction_label": _fraction_label(fraction),
                    "action_id": action.action_id,
                    "feature_hash": feature_records[fraction]["feature_hash"],
                    "commitment_hash": committed_actions[fraction][
                        "commitment_hash"
                    ],
                    "selected_by_controller": (
                        committed_actions[fraction]["action_id"]
                        == action.action_id
                    ),
                    "operational_c_distance": distance,
                    "operational_c_energy": distance * distance,
                    "budget_validation": "scaled-unit-c",
                    **_metric_difference(after, baseline),
                    "status": "completed",
                    "rollback_exact": True,
                },
            )
            outcome_count += 1

        global_alpha = scale_proposal(
            ordered,
            math.sqrt(fraction),
            solver_suffix=f"global-alpha-{_fraction_label(fraction)}",
        )
        global_after = _evaluate_branch(
            runtime=runtime,
            request=request,
            contexts=contexts,
            target_ids=target_ids,
            proposal=global_alpha,
            exact_application=False,
            base_hashes=base_hashes,
            state_identity=state_identity,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
        )
        outcome_writer.write(
            "mv1_contextual_outcome",
            {
                "case_id": request.case_id,
                "request_id": request.request_id,
                "fraction": fraction,
                "fraction_label": _fraction_label(fraction),
                "action_id": "ordered_global_alpha",
                "commitment_hash": committed_actions[fraction][
                    "commitment_hash"
                ],
                "operational_c_distance": distance,
                "operational_c_energy": distance * distance,
                "budget_validation": "ordered-energy-times-global-alpha-squared",
                **_metric_difference(global_after, baseline),
                "status": "completed",
                "rollback_exact": True,
            },
        )
        outcome_count += 1

    native_after = _evaluate_branch(
        runtime=runtime,
        request=request,
        contexts=contexts,
        target_ids=target_ids,
        proposal=ordered,
        exact_application=True,
        base_hashes=base_hashes,
        state_identity=state_identity,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
    )
    replay = _evaluate_branch(
        runtime=runtime,
        request=request,
        contexts=contexts,
        target_ids=target_ids,
        proposal=None,
        exact_application=False,
        base_hashes=base_hashes,
        state_identity=state_identity,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
    )
    outcome_writer.write(
        "mv1_reference_outcome",
        {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "action_id": "native_memit_full",
            "c_distance": math.sqrt(native_energy),
            **_metric_difference(native_after, baseline),
            "status": "completed",
            "rollback_exact": True,
        },
    )
    outcome_writer.write(
        "mv1_replay_outcome",
        {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "action_id": "no_op_replay",
            "c_distance": 0.0,
            **_metric_difference(replay, baseline),
            "logits_hash_equal": replay.logits_hash == baseline.logits_hash,
            "status": "completed",
            "rollback_exact": True,
        },
    )
    outcome_count += 2
    if _weight_hashes(runtime.model, weight_names) != base_hashes:
        raise RollbackError("MV-1 event did not restore the base state")
    return {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "event_seed": event_seed,
        "target_token_count": baseline.target_token_count,
        "native_c_energy": native_energy,
        "feature_count": len(FRACTIONS),
        "commitment_count": len(FRACTIONS),
        "outcome_count": outcome_count,
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_artifact_size": direct_z.artifact.size,
        "action_receipts": action_receipts,
        "rollback_exact": True,
        "pass": True,
    }


def _selection_cases(
    selection: CounterFactSelectionManifest,
    case_count: int,
) -> tuple[str, ...]:
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < 1
        or case_count > CALIBRATION_CASE_COUNT
    ):
        raise ContractError("MV-1 C0 case count must be in [1, 20]")
    return selection.calibration[:case_count]


def _mv1_slurm_state(
    model_alias: str,
    run_id: str,
    case_count: int,
) -> dict[str, Any]:
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial Slurm identity is forbidden")
    expected = {
        ("llama3-8b-inst", "mv1_llama_c0p_v1", PILOT_CASE_COUNT): (
            "odeedit_mv1_c0p_pair_v1",
            "devbox",
        ),
        ("qwen2.5-7b-inst", "mv1_qwen_c0p_v1", PILOT_CASE_COUNT): (
            "odeedit_mv1_c0p_pair_v1",
            "devbox",
        ),
        ("llama3-8b-inst", "mv1_llama_c0_v1", CALIBRATION_CASE_COUNT): (
            "odeedit_mv1_c0_pair_v1",
            "devbox",
        ),
        ("qwen2.5-7b-inst", "mv1_qwen_c0_v1", CALIBRATION_CASE_COUNT): (
            "odeedit_mv1_c0_pair_v1",
            "devbox",
        ),
    }.get((model_alias, run_id, case_count))
    if (
        expected is None
        or values["job_name"] != expected[0]
        or values["node"] != expected[1]
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("Slurm identity is outside the audited MV-1 C0 envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
    }


def run_mv1_calibration(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    case_count: int = PILOT_CASE_COUNT,
    seed: int = 17,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    """Run one model's C0 pilot or full calibration split."""

    started = time.perf_counter()
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    slurm_state = _mv1_slurm_state(model_alias, run_id, case_count)
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_counterfact_selection(root, seed=selection_seed)
    case_ids = _selection_cases(selection, case_count)
    requests = load_counterfact_requests(root, case_ids)
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("bridge provenance is outside the fixed full manifest")

    with offline_environment():
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(seed)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("model loader returned a different fixed model spec")
        contexts = _freeze_contexts(bridge, runtime, seed=seed)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        outcomes_path = destination / "outcomes.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        _write_json_exclusive(
            manifest_path,
            {
                "schema_version": "ode-edit-mv1-c0-manifest/v1",
                "run_id": run_id,
                "ode_edit_git": git_state,
                "slurm": slurm_state,
                "model": runtime.metadata(),
                "hparams_relative_path": spec.hparams_path,
                "selection": selection.to_dict(),
                "selected_split": "calibration",
                "selected_case_ids": list(case_ids),
                "selected_request_ids": [request.request_id for request in requests],
                "contexts": {
                    "manifest_id": contexts.manifest_id,
                    "source": contexts.source,
                    "group_sizes": [len(group) for group in contexts.templates],
                    "raw_templates_persisted": False,
                },
                "provenance_id": provenance.manifest_id,
                "fixed_files": _relative_provenance(provenance, root),
                "fractions": list(FRACTIONS),
                "probe_ratio": PROBE_RATIO,
                "action_set": [
                    *(f"layer_{layer}" for layer in spec.layers),
                    ACTION_UNIFORM,
                ],
                "decision_policy": "score-before-outcome finite-action argmax",
                "action_receipt_policy": (
                    "per-event exclusive receipt after feature/action fsync"
                ),
                "utility_policy": "temperature=1 mean-context-token smooth target margin",
                "projector_policy": "sha256-and-size-verify-only; never-deserialized",
                "covariance_policy": "verified-read-only; recompute-and-download-blocked",
                "direct_z_policy": "computed-once-and-frozen-under-this-local-run",
                "artifact_firewall": (
                    "structured JSON contains case IDs/full request hashes/scalars "
                    "only; activation-derived direct_z remains local-only"
                ),
            },
        )
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root,
            runtime=runtime,
            specs=covariance_specs,
        )
        direct_z_root = destination / "direct_z"
        receipt_root = destination / "action_receipts"
        receipt_root.mkdir(mode=0o700)
        torch.cuda.reset_peak_memory_stats(0)
        event_results: list[dict[str, Any]] = []
        abort_failure_type: str | None = None
        with (
            SanitizedJsonlWriter(
                features_path,
                run_id,
                schema_version="ode-edit-mv1-c0/v1",
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path,
                run_id,
                schema_version="ode-edit-mv1-c0/v1",
            ) as action_writer,
            SanitizedJsonlWriter(
                outcomes_path,
                run_id,
                schema_version="ode-edit-mv1-c0/v1",
            ) as outcome_writer,
            SanitizedJsonlWriter(
                events_path,
                run_id,
                schema_version="ode-edit-mv1-c0/v1",
            ) as event_writer,
        ):
            for request in requests:
                try:
                    result = _run_event(
                        runtime=runtime,
                        bridge=bridge,
                        hparams=hparams,
                        contexts=contexts,
                        covariance_specs=covariance_specs,
                        covariance_moments=covariance_moments,
                        direct_z_root=direct_z_root,
                        receipt_root=receipt_root,
                        request=request,
                        seed=seed,
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        outcome_writer=outcome_writer,
                    )
                except Exception as exc:
                    result = {
                        "case_id": request.case_id,
                        "request_id": request.request_id,
                        "failure_type": type(exc).__name__,
                        "failure_class": "fatal_calibration_contract_or_runtime",
                        "rollback_exact": False,
                        "pass": False,
                    }
                    abort_failure_type = type(exc).__name__
                event_results.append(result)
                event_writer.write("mv1_case", result)
                if abort_failure_type is not None:
                    break

        elapsed = time.perf_counter() - started
        planned = len(requests)
        attempted = len(event_results)
        summary = {
            "schema_version": "ode-edit-mv1-c0-summary/v1",
            "run_id": run_id,
            "model_alias": model_alias,
            "slurm": slurm_state,
            "provenance_id": provenance.manifest_id,
            "selection_manifest_id": selection.manifest_id,
            "context_id": contexts.manifest_id,
            "run_status": (
                "aborted" if abort_failure_type is not None else "completed"
            ),
            "abort_failure_type": abort_failure_type,
            "planned_case_count": planned,
            "attempted_case_count": attempted,
            "not_run_due_to_abort_count": planned - attempted,
            "pass_count": sum(bool(item["pass"]) for item in event_results),
            "failure_count": planned - sum(
                bool(item["pass"]) for item in event_results
            ),
            "all_pass": bool(
                abort_failure_type is None
                and attempted == planned
                and all(bool(item["pass"]) for item in event_results)
            ),
            "case_results": [
                {
                    "case_id": item["case_id"],
                    "status": "attempted",
                    "pass": bool(item["pass"]),
                }
                for item in event_results
            ]
            + [
                {
                    "case_id": case_id,
                    "status": "not_run_due_to_abort",
                    "pass": False,
                }
                for case_id in case_ids[attempted:]
            ],
            "feature_count": sum(
                int(item.get("feature_count", 0)) for item in event_results
            ),
            "commitment_count": sum(
                int(item.get("commitment_count", 0)) for item in event_results
            ),
            "outcome_count": sum(
                int(item.get("outcome_count", 0)) for item in event_results
            ),
            "all_rollbacks_exact": bool(
                event_results
                and all(bool(item["rollback_exact"]) for item in event_results)
            ),
            "covariance_files_loaded": list(loaded_covariances),
            "projector_files_loaded": [],
            "direct_z_artifact_count": (
                len(tuple(direct_z_root.glob("*.pt")))
                if direct_z_root.is_dir()
                else 0
            ),
            "action_receipt_count": len(tuple(receipt_root.glob("*.json"))),
            "resource": {
                "wall_seconds": elapsed,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "visible_gpu_count": 1,
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "features_sha256": _file_sha256(features_path),
                "actions_sha256": _file_sha256(actions_path),
                "outcomes_sha256": _file_sha256(outcomes_path),
                "events_sha256": _file_sha256(events_path),
                "action_receipts": {
                    path.name: _file_sha256(path)
                    for path in sorted(receipt_root.glob("*.json"))
                },
            },
            "git_output_written": False,
        }
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def _case_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cases must be an integer") from exc
    if count not in (PILOT_CASE_COUNT, CALIBRATION_CASE_COUNT):
        raise argparse.ArgumentTypeError("cases must be exactly 3 or 20")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-mv1-c0",
        description="Inference-only MV-1 C0 predictive-heterogeneity calibration",
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cases", type=_case_count, default=PILOT_CASE_COUNT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_mv1_calibration(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
            case_count=args.cases,
            seed=args.seed,
            selection_seed=args.selection_seed,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
