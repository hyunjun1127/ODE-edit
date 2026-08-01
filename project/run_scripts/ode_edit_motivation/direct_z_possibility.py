"""Direct-z realization possibility diagnostic.

This runner is intentionally a mechanism probe, not a method claim.  It
freezes EasyEdit's native direct-z once at W0, constructs six precommitted
realizations, and measures hidden-state fidelity, rewrite/paraphrase utility,
and held-out next-token KL separately.  EasyEdit, Wikipedia moments, and all
fixed artifacts are read-only; hooks and experiment logic live in ODE-edit.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    EditRequest,
    MemitFactorProposal,
    canonical_json,
    sha256_bytes,
)
from .direct_z_fidelity import (
    aggregate_sequence_spill,
    aggregate_shared_delta_fidelity,
    combine_unit_c_single_layer_proposals,
    make_batch_position_patch,
    solve_five_direction_ridge,
    unwrap_layer_output,
)
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage, proposal_direction_hash
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .hooks import (
    ForwardCapture,
    TemporaryExactMemitApplication,
    TemporaryLowRankApplication,
    TensorHashRuntimeError,
    assert_snapshot_current,
    assert_tensor_sha256_device_parity,
)
from .manifests import (
    COUNTERFACT_RELATIVE_PATH,
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    _iter_top_level_json_objects,
    fixed_model_spec,
    generate_counterfact_selection,
    load_counterfact_requests,
    preflight_fixed_artifacts,
    scan_counterfact_case_ids,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    RollbackError,
    SanitizedJsonlWriter,
    TeacherForcedResult,
    _bridge_pins,
    _covariance_specs,
    _event_seed,
    _freeze_contexts,
    _git_runtime_state,
    _load_hparams,
    _load_verified_covariances,
    _local_run_directory,
    _safe_payload,
    _silence_upstream,
    _weight_hashes,
    _write_json_exclusive,
)
from .mv1_calibration import (
    ACTION_UNIFORM,
    MV1Error,
    RewriteMetrics,
    _feature_hash,
    _layer_by_weight,
    _state_identity,
    assert_exact_action_contract,
    build_exact_teacher_batch,
    build_unit_c_actions,
    proposal_c_energy,
    rewrite_metrics,
    scale_proposal,
)
from .quarter_step_refresh import (
    LAYERS,
    QSTEP_K,
    REFRESHED_PREFIX,
    _assert_step_energy,
    _build_policy_path,
    _build_score_mix_action,
    _capture_source_snapshot,
    _combine_steps,
    _commit_action as _commit_qstep_action,
    _proposal_panel,
)


DIRECTZ_RUN_SEED = 31
DIRECTZ_CASE_COUNT = 8
DIRECTZ_RANK_START = 132
DIRECTZ_RANK_STOP = 140
DIRECTZ_HOP_FRACTION = 0.25
DIRECTZ_PROBE_FRACTION = 1.0 / 64.0
DIRECTZ_BOOTSTRAP_SEED = 20260801
DIRECTZ_BOOTSTRAP_RESAMPLES = 4000
DIRECTZ_JOB_NAME = "odeedit_dzf_pair_v2"
DIRECTZ_RUN_IDS = MappingProxyType(
    {
        "llama3-8b-inst": "dzf_llama_p0_v2",
        "qwen2.5-7b-inst": "dzf_qwen_p0_v2",
    }
)

NO_OP = "no_op_replay"
ORACLE_DO_Z = "oracle_do_z"
NATIVE_ORDERED = "native_ordered_full"
NATIVE_C_MATCHED = "native_alpha_c_matched"
BF_REFRESHED_K4 = "bf_current_refreshed_k4"
SYNC_Z_CONE = "sync_z_cone_oracle"
DIRECTZ_BRANCH_ORDER = (
    NO_OP,
    ORACLE_DO_Z,
    NATIVE_ORDERED,
    NATIVE_C_MATCHED,
    BF_REFRESHED_K4,
    SYNC_Z_CONE,
)

SELECTION_SCHEMA = "ode-edit-direct-z-possibility-selection/v2"
MANIFEST_SCHEMA = "ode-edit-direct-z-possibility-manifest/v1"
STREAM_SCHEMA = "ode-edit-direct-z-possibility-stream/v1"
RECEIPT_SCHEMA = "ode-edit-direct-z-possibility-receipt/v2"
SUMMARY_SCHEMA = "ode-edit-direct-z-possibility-summary/v1"
EVENT_SCHEMA = "ode-edit-direct-z-possibility-event/v2"


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _execution_envelope(model_alias: str, run_id: str) -> None:
    if model_alias not in DIRECTZ_RUN_IDS or DIRECTZ_RUN_IDS[model_alias] != run_id:
        raise MV1Error("direct-z possibility model/run identity differs from lock")


def _slurm_state(model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial direct-z possibility Slurm identity is forbidden")
    if (
        values["job_name"] != DIRECTZ_JOB_NAME
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("direct-z possibility Slurm identity differs from lock")
    return {"under_slurm": True, **values}


def generate_directz_selection(easyedit_root: str | Path) -> dict[str, Any]:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(root, seed=DEFAULT_SELECTION_SEED)
    all_case_ids = scan_counterfact_case_ids(root / COUNTERFACT_RELATIVE_PATH)
    ranked = tuple(
        sorted(
            all_case_ids,
            key=lambda case_id: (
                hashlib.sha256(
                    DEFAULT_SELECTION_SEED.encode("utf-8")
                    + b"\0"
                    + case_id.encode("utf-8")
                ).digest(),
                case_id,
            ),
        )
    )
    if (
        len(ranked) != canonical.source_row_count
        or ranked[:100] != canonical.ordered_case_ids
    ):
        raise ContractError("direct-z selection disagrees with canonical rank")
    selected = ranked[DIRECTZ_RANK_START:DIRECTZ_RANK_STOP]
    if len(selected) != DIRECTZ_CASE_COUNT or set(selected).intersection(ranked[:132]):
        raise ContractError("direct-z selection is not fresh [132:140]")
    payload = {
        "schema_version": SELECTION_SCHEMA,
        "seed": DEFAULT_SELECTION_SEED,
        "source_sha256": canonical.source_sha256,
        "source_size": canonical.source_size,
        "source_row_count": canonical.source_row_count,
        "rank_start": DIRECTZ_RANK_START,
        "rank_stop": DIRECTZ_RANK_STOP,
        "case_ids": list(selected),
        "prior_order_hash": sha256_bytes(
            canonical_json(list(ranked[:DIRECTZ_RANK_START])).encode("utf-8")
        ),
        "order_hash": sha256_bytes(canonical_json(list(selected)).encode("utf-8")),
    }
    payload["manifest_id"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


def _valid_eval_text(name: str, value: Any, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ContractError(f"{name} is not a bounded non-empty string")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise ContractError(f"{name} contains a control character")
    return value


@dataclass(frozen=True, slots=True)
class HeldoutEvaluationCase:
    case_id: str
    paraphrase_prompts: tuple[str, ...]
    preservation_prompts: tuple[str, ...]
    payload_hash: str


def load_heldout_evaluation_case(
    easyedit_root: str | Path,
    request: EditRequest,
) -> HeldoutEvaluationCase:
    """Load evaluation-only text after the durable action receipt."""

    source = Path(easyedit_root).expanduser().resolve(strict=True) / COUNTERFACT_RELATIVE_PATH
    for blob in _iter_top_level_json_objects(source):
        row = json.loads(blob)
        if str(row.get("case_id")) != request.case_id:
            continue
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise ContractError("held-out row lacks requested_rewrite")
        if (
            rewrite.get("prompt") != request.prompt
            or rewrite.get("subject") != request.subject
            or str(rewrite.get("target_new", {}).get("str", "")).strip()
            != request.target_new
        ):
            raise ContractError("held-out row does not match committed request")
        paraphrases = tuple(
            _valid_eval_text("paraphrase prompt", item)
            for item in row.get("paraphrase_prompts", ())
        )
        neighborhoods = tuple(
            _valid_eval_text("neighborhood prompt", item)
            for item in row.get("neighborhood_prompts", ())
        )
        generations = tuple(
            _valid_eval_text("generation prompt", item)
            for item in row.get("generation_prompts", ())
        )
        if len(paraphrases) < 2 or len(neighborhoods) < 4 or len(generations) < 4:
            raise ContractError("held-out prompt counts are below the fixed schema")
        selected_paraphrases = paraphrases[:2]
        preservation = (*neighborhoods[:4], *generations[:4])
        private_payload = {
            "case_id": request.case_id,
            "paraphrase_prompts": list(selected_paraphrases),
            "preservation_prompts": list(preservation),
        }
        return HeldoutEvaluationCase(
            case_id=request.case_id,
            paraphrase_prompts=selected_paraphrases,
            preservation_prompts=preservation,
            payload_hash=sha256_bytes(canonical_json(private_payload).encode("utf-8")),
        )
    raise ContractError("selected CounterFact evaluation row is missing")


@dataclass(slots=True)
class TeacherCapture:
    rewrite: RewriteMetrics
    sequence_states: torch.Tensor
    subject_states: torch.Tensor
    attention_mask: torch.Tensor
    subject_positions: tuple[int, ...]


def _exact_templates(request: EditRequest, contexts: Any) -> tuple[str, ...]:
    templates = tuple(
        str(item).format(request.prompt)
        for group in contexts.templates
        for item in group
    )
    if not templates or templates[0] != request.prompt:
        raise ContractError("first exact context is not the canonical request")
    if any(template.count("{}") != 1 for template in templates):
        raise ContractError("exact context template contract changed")
    return templates


def _subject_positions(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    templates: Sequence[str],
    subject: str,
) -> tuple[int, ...]:
    return tuple(
        int(
            bindings.compute_z.find_fact_lookup_idx(
                template,
                subject,
                runtime.tokenizer,
                hparams.fact_token,
                verbose=False,
            )
        )
        for template in templates
    )


def _capture_teacher(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    contexts: Any,
    request: EditRequest,
    patch_delta: torch.Tensor | None = None,
) -> TeacherCapture:
    batch = build_exact_teacher_batch(runtime.tokenizer, request, contexts)
    templates = _exact_templates(request, contexts)
    if len(templates) != batch.input_ids.shape[0]:
        raise ContractError("exact context and teacher batch counts differ")
    positions = _subject_positions(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        templates=templates,
        subject=request.subject,
    )
    layer_name = str(hparams.layer_module_tmp).format(int(hparams.layers[-1]))
    patch = None
    if patch_delta is not None:
        patch = make_batch_position_patch(
            layer_name,
            positions,
            patch_delta,
            mode="add",
        )
    device = next(runtime.model.parameters()).device
    inputs = {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
    }
    with torch.inference_mode(), ForwardCapture(
        runtime.model,
        layer_name,
        retain_input=False,
        retain_output=True,
        edit_output=patch,
    ) as capture:
        logits = runtime.model(**inputs).logits
    sequence = unwrap_layer_output(capture.output).detach().cpu().float().contiguous()
    if sequence.ndim != 3 or sequence.shape[:2] != batch.input_ids.shape:
        raise ContractError("captured z-layer sequence has an invalid layout")
    normalized_positions = tuple(
        position if position >= 0 else sequence.shape[1] + position
        for position in positions
    )
    if any(position < 0 or position >= sequence.shape[1] for position in normalized_positions):
        raise ContractError("subject lookup position is outside the exact batch")
    row_indices = torch.arange(sequence.shape[0], dtype=torch.long)
    subject_states = sequence[
        row_indices,
        torch.tensor(normalized_positions, dtype=torch.long),
    ].contiguous()
    target_count = int(batch.target_ids.numel())
    selected = []
    for row, length in enumerate(batch.input_lengths):
        start = length - target_count
        selected.append(logits[row, start:length, :].float())
    selected_logits = torch.stack(selected).detach().cpu().contiguous()
    targets = batch.target_ids.expand(selected_logits.shape[0], -1)
    losses = torch.nn.functional.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(selected_logits.shape[:2])
    teacher = TeacherForcedResult(
        logits=selected_logits,
        nll=float(losses.mean()),
        context_nll=tuple(float(value) for value in losses.mean(dim=1)),
        target_token_count=target_count,
    )
    return TeacherCapture(
        rewrite=rewrite_metrics(teacher, batch.target_ids),
        sequence_states=sequence,
        subject_states=subject_states,
        attention_mask=batch.attention_mask.detach().cpu().contiguous(),
        subject_positions=normalized_positions,
    )


def _concrete_patch(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    prompts: Sequence[str],
    subject: str,
    delta: torch.Tensor,
) -> Any:
    positions: list[int] = []
    values: list[torch.Tensor] = []
    delta_cpu = delta.detach().cpu().float()
    for prompt in prompts:
        offset = prompt.find(subject)
        if offset < 0:
            positions.append(0)
            values.append(torch.zeros_like(delta_cpu))
            continue
        template = f"{prompt[:offset]}{{}}{prompt[offset + len(subject):]}"
        position = _subject_positions(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            templates=(template,),
            subject=subject,
        )[0]
        positions.append(position)
        values.append(delta_cpu)
    layer_name = str(hparams.layer_module_tmp).format(int(hparams.layers[-1]))
    return make_batch_position_patch(
        layer_name,
        positions,
        torch.stack(values),
        mode="add",
    )


def _teacher_forced_prompts(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    prompts: Sequence[str],
    target_text: str,
    subject: str,
    patch_delta: torch.Tensor | None = None,
) -> RewriteMetrics:
    tokenizer = runtime.tokenizer
    normalized_target = target_text if target_text.startswith(" ") else f" {target_text}"
    target_ids = tuple(
        int(value) for value in tokenizer.encode(normalized_target, add_special_tokens=False)
    )
    if not target_ids or not prompts:
        raise MV1Error("held-out teacher-forcing input is empty")
    rows: list[tuple[int, ...]] = []
    for prompt in prompts:
        full = tuple(
            int(value)
            for value in tokenizer.encode(
                prompt + normalized_target,
                add_special_tokens=True,
            )
        )
        if len(full) <= len(target_ids) or full[-len(target_ids) :] != target_ids:
            raise MV1Error("held-out prompt does not preserve target suffix")
        rows.append(full[:-1])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise MV1Error("tokenizer has no valid pad token")
    maximum = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    layer_name = str(hparams.layer_module_tmp).format(int(hparams.layers[-1]))
    patch = (
        None
        if patch_delta is None
        else _concrete_patch(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            prompts=prompts,
            subject=subject,
            delta=patch_delta,
        )
    )
    device = next(runtime.model.parameters()).device
    manager = (
        contextlib.nullcontext()
        if patch is None
        else ForwardCapture(
            runtime.model,
            layer_name,
            retain_input=False,
            retain_output=False,
            edit_output=patch,
        )
    )
    selected: list[torch.Tensor] = []
    with torch.inference_mode(), manager:
        logits = runtime.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits
        for index, length in enumerate(map(len, rows)):
            start = length - len(target_ids)
            selected.append(logits[index, start:length, :].float())
    selected_logits = torch.stack(selected).detach().cpu().contiguous()
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    targets = target_tensor.expand(selected_logits.shape[0], -1)
    losses = torch.nn.functional.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(selected_logits.shape[:2])
    teacher = TeacherForcedResult(
        logits=selected_logits,
        nll=float(losses.mean()),
        context_nll=tuple(float(value) for value in losses.mean(dim=1)),
        target_token_count=len(target_ids),
    )
    return rewrite_metrics(teacher, target_tensor)


def _next_token_logits(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    prompts: Sequence[str],
    subject: str,
    patch_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if not prompts:
        raise ContractError("preservation prompt set is empty")
    tokenizer = runtime.tokenizer
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ContractError("preservation KL requires right padding")
    layer_name = str(hparams.layer_module_tmp).format(int(hparams.layers[-1]))
    patch = (
        None
        if patch_delta is None
        else _concrete_patch(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            prompts=prompts,
            subject=subject,
            delta=patch_delta,
        )
    )
    manager = (
        contextlib.nullcontext()
        if patch is None
        else ForwardCapture(
            runtime.model,
            layer_name,
            retain_input=False,
            retain_output=False,
            edit_output=patch,
        )
    )
    device = next(runtime.model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode(), manager:
        logits = runtime.model(input_ids=input_ids, attention_mask=attention_mask).logits
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths, :].detach().cpu().float().contiguous()


def _kl_from_w0(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ContractError("preservation logits changed shape")
    reference_log = torch.log_softmax(reference.float(), dim=-1)
    candidate_log = torch.log_softmax(candidate.float(), dim=-1)
    value = torch.sum(reference_log.exp() * (reference_log - candidate_log), dim=-1).mean()
    result = _finite("held-out KL", value.item())
    if result < -1e-6:
        raise ContractError("held-out KL is materially negative")
    return max(0.0, result)


def _proposal_frobenius_norm(proposal: MemitFactorProposal | None) -> float:
    if proposal is None:
        return 0.0
    squared = torch.zeros((), dtype=torch.float64)
    for factor in proposal.factors:
        left = factor.left.detach().cpu().to(dtype=torch.float64)
        right = factor.right.detach().cpu().to(dtype=torch.float64)
        squared = squared + torch.sum((left.T @ left) * (right.T @ right))
    result = float(torch.sqrt(squared.clamp_min(0.0)))
    return _finite("proposal Frobenius norm", result)


def _restore_rng(cpu_rng: torch.Tensor, cuda_rng: torch.Tensor) -> None:
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device=0)


@dataclass(slots=True)
class BranchObservation:
    teacher: TeacherCapture
    paraphrase: RewriteMetrics
    preservation_logits: torch.Tensor


def _observe_branch(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    contexts: Any,
    request: EditRequest,
    heldout: HeldoutEvaluationCase,
    patch_delta: torch.Tensor | None = None,
) -> BranchObservation:
    return BranchObservation(
        teacher=_capture_teacher(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
            patch_delta=patch_delta,
        ),
        paraphrase=_teacher_forced_prompts(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            prompts=heldout.paraphrase_prompts,
            target_text=request.target_new,
            subject=request.subject,
            patch_delta=patch_delta,
        ),
        preservation_logits=_next_token_logits(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            prompts=heldout.preservation_prompts,
            subject=request.subject,
            patch_delta=patch_delta,
        ),
    )


def _evaluate_arm(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    contexts: Any,
    request: EditRequest,
    heldout: HeldoutEvaluationCase,
    origin_snapshot: Any,
    base_hashes: Mapping[str, str],
    base_state_identity: str,
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    proposal: MemitFactorProposal | None = None,
    exact_application: bool = False,
    patch_delta: torch.Tensor | None = None,
) -> BranchObservation:
    assert_snapshot_current(runtime.model, origin_snapshot)
    _restore_rng(cpu_rng, cuda_rng)
    manager: contextlib.AbstractContextManager[Any]
    if proposal is None:
        manager = contextlib.nullcontext()
    elif exact_application:
        manager = TemporaryExactMemitApplication(runtime.model, proposal)
    else:
        manager = TemporaryLowRankApplication(runtime.model, proposal)
    try:
        with manager:
            result = _observe_branch(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                contexts=contexts,
                request=request,
                heldout=heldout,
                patch_delta=patch_delta,
            )
    finally:
        _restore_rng(cpu_rng, cuda_rng)
        assert_snapshot_current(runtime.model, origin_snapshot)
    if (
        _weight_hashes(runtime.model, tuple(base_hashes)) != dict(base_hashes)
        or _state_identity(runtime) != base_state_identity
    ):
        raise RollbackError("direct-z arm did not restore W0/model state")
    return result


def _outcome_payload(
    *,
    case_id: str,
    request_id: str,
    arm_id: str,
    branch: BranchObservation,
    baseline: BranchObservation,
    base_teacher: TeacherCapture,
    shared_delta: torch.Tensor,
    canonical_target: torch.Tensor,
    endpoint_c_energy: float,
    endpoint_frobenius_norm: float,
    nfe: int,
) -> dict[str, Any]:
    fidelity = aggregate_shared_delta_fidelity(
        base_teacher.subject_states,
        branch.teacher.subject_states,
        shared_delta,
        canonical_index=0,
        canonical_absolute_target=canonical_target,
    )
    spill = aggregate_sequence_spill(
        base_teacher.sequence_states,
        branch.teacher.sequence_states,
        shared_delta,
        base_teacher.subject_positions,
        canonical_index=0,
        canonical_absolute_target=canonical_target,
        attention_mask=base_teacher.attention_mask,
    )
    heldout_kl = _kl_from_w0(
        baseline.preservation_logits,
        branch.preservation_logits,
    )
    payload = {
        "case_id": case_id,
        "request_id": request_id,
        "arm_id": arm_id,
        "case_pass": True,
        "arm_success": True,
        "technical_pass": True,
        "rollback_exact": True,
        "firewall_pass": True,
        "receipt_before_outcome": True,
        "z_residual_ratio": fidelity.canonical_target_error_relative,
        "generated_delta_error_mean": fidelity.generated_mean_target_error_relative,
        "generated_delta_error_worst": fidelity.generated_worst_target_error_relative,
        "delta_gain": fidelity.canonical_gain,
        "delta_cosine": fidelity.canonical_cosine,
        "off_token_spill_ratio": spill.global_off_token_spill_rms / spill.delta_l2,
        "output_progress": branch.teacher.rewrite.utility
        - baseline.teacher.rewrite.utility,
        "output_nll_reduction": baseline.teacher.rewrite.nll
        - branch.teacher.rewrite.nll,
        "exact_margin_min": branch.teacher.rewrite.exact_margin_min,
        "exact_satisfied": branch.teacher.rewrite.exact_satisfied,
        "paraphrase_nll_reduction": baseline.paraphrase.nll - branch.paraphrase.nll,
        "heldout_kl": heldout_kl,
        "preservation_score": -heldout_kl,
        "endpoint_c_energy": endpoint_c_energy,
        "endpoint_frobenius_norm": endpoint_frobenius_norm,
        "nfe": nfe,
    }
    _safe_payload(payload)
    return payload


def _run_event(
    *,
    root: Path,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    direct_z_root: Path,
    receipt_root: Path,
    request: EditRequest,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    outcome_writer: SanitizedJsonlWriter,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_runtime(_event_seed(DIRECTZ_RUN_SEED, request.case_id))
    layer_by_weight = _layer_by_weight(hparams)
    if tuple(int(layer) for layer in hparams.layers) != LAYERS:
        raise MV1Error("direct-z possibility requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    base_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()
    bindings = bridge.load()
    origin_snapshot = _capture_source_snapshot(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        weight_names=weight_names,
        provenance_id=bindings.provenance.manifest_id,
    )
    w0_teacher = _capture_teacher(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
    )
    target_ids = build_exact_teacher_batch(runtime.tokenizer, request, contexts).target_ids
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
        b0 = bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
        )
        ordered_w0 = bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )
    b0.assert_same_entry_snapshot(ordered_w0)
    target_lineage = FrozenTargetLineage.start(
        direct_z=direct_z,
        origin_snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    target_lineage.assert_authorizes(
        direct_z=direct_z,
        snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    canonical_target = direct_z.values[:, 0].detach().cpu().float()
    shared_delta = canonical_target - w0_teacher.subject_states[0]
    delta_l2 = float(torch.linalg.vector_norm(shared_delta))
    if not math.isfinite(delta_l2) or delta_l2 <= 0.0:
        raise ContractError("frozen direct-z delta norm is invalid")

    native_energy = proposal_c_energy(ordered_w0, covariance_moments, layer_by_weight)
    native_distance = math.sqrt(native_energy)
    hop_distance = native_distance * DIRECTZ_HOP_FRACTION
    hop_energy = native_energy / float(QSTEP_K * QSTEP_K)
    probe_distance = native_distance * DIRECTZ_PROBE_FRACTION
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (native_energy, native_distance, hop_distance, hop_energy, probe_distance)
    ):
        raise ContractError("direct-z C-distance budget is invalid")
    w0_unit = build_unit_c_actions(b0, covariance_moments, layer_by_weight)
    assert_exact_action_contract(
        b0,
        ordered_w0,
        w0_unit,
        expected_factor_names=weight_names,
        expected_action_ids=tuple(f"layer_{layer}" for layer in LAYERS)
        + (ACTION_UNIFORM,),
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    base_cpu_rng = torch.get_rng_state().clone()
    base_cuda_rng = torch.cuda.get_rng_state(0).clone()
    w0_panel = _proposal_panel(
        runtime=runtime,
        request=request,
        contexts=contexts,
        target_ids=target_ids,
        actions=w0_unit,
        epsilon=probe_distance,
        base_hashes=base_hashes,
        state_identity=base_state_identity,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
        panel_label="direct-z-w0-b0",
    )
    decision0, action0 = _build_score_mix_action(
        synchronous=b0,
        unit_actions=w0_unit,
        scores=w0_panel.scores,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    common_step = scale_proposal(
        action0.proposal,
        hop_distance,
        solver_suffix="direct-z-bf-common-step-1",
    )
    _assert_step_energy(
        common_step,
        expected=hop_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    refreshed_path = _build_policy_path(
        policy=REFRESHED_PREFIX,
        runtime=runtime,
        bridge=bridge,
        hparams=hparams,
        contexts=contexts,
        covariance_specs=covariance_specs,
        covariance_moments=covariance_moments,
        request=request,
        direct_z=direct_z,
        target_ids=target_ids,
        origin_snapshot=origin_snapshot,
        target_lineage=target_lineage,
        b0=b0,
        w0_unit=w0_unit,
        decision0=decision0,
        action0=action0,
        common_step=common_step,
        hop_distance=hop_distance,
        hop_energy=hop_energy,
        probe_distance=probe_distance,
        weight_names=weight_names,
        layer_by_weight=layer_by_weight,
        provenance_id=bindings.provenance.manifest_id,
        base_hashes=base_hashes,
        base_state_identity=base_state_identity,
        base_cpu_rng=base_cpu_rng,
        base_cuda_rng=base_cuda_rng,
    )
    if len(refreshed_path.proposals) != QSTEP_K:
        raise ContractError("BF refreshed path did not produce K=4")
    bf_endpoint = _combine_steps(
        b0,
        refreshed_path.proposals,
        solver_suffix="direct-z-bf-refreshed-k4",
    )
    bf_energy = proposal_c_energy(bf_endpoint, covariance_moments, layer_by_weight)
    native_c_matched = scale_proposal(
        ordered_w0,
        math.sqrt(bf_energy / native_energy),
        solver_suffix="direct-z-native-c-matched-to-bf",
    )
    matched_energy = proposal_c_energy(
        native_c_matched,
        covariance_moments,
        layer_by_weight,
    )
    if not math.isclose(matched_energy, bf_energy, rel_tol=3e-5, abs_tol=3e-5):
        raise ContractError("native C-matched arm differs from BF endpoint budget")

    responses: list[torch.Tensor] = []
    for action in w0_unit[: len(LAYERS)]:
        plus = scale_proposal(
            action.proposal,
            probe_distance,
            solver_suffix=f"direct-z-cone-plus-{action.action_id}",
        )
        minus = scale_proposal(
            action.proposal,
            -probe_distance,
            solver_suffix=f"direct-z-cone-minus-{action.action_id}",
        )
        with TemporaryLowRankApplication(runtime.model, plus):
            plus_state = _capture_teacher(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                contexts=contexts,
                request=request,
            ).subject_states[0]
        with TemporaryLowRankApplication(runtime.model, minus):
            minus_state = _capture_teacher(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                contexts=contexts,
                request=request,
            ).subject_states[0]
        responses.append((plus_state - minus_state) / (2.0 * probe_distance))
    response_matrix = torch.stack(responses, dim=1)
    zcone_solution = solve_five_direction_ridge(
        response_matrix,
        shared_delta,
        radius=math.sqrt(bf_energy),
        ridge=1e-6,
        max_iterations=5000,
        tolerance=1e-8,
    )
    if not zcone_solution.converged:
        raise ContractError("z-cone projected ridge solver did not converge")
    zcone = combine_unit_c_single_layer_proposals(
        b0,
        tuple(action.proposal for action in w0_unit[: len(LAYERS)]),
        zcone_solution.coefficients,
        solver_suffix="direct-z-z-cone-oracle",
    )
    # The five unit-C blocks are C-orthogonal.  Keep exact zero coefficients
    # (useful diagnostic sparsity) without routing them through the generic
    # positive-factor energy helper, which intentionally rejects zero blocks.
    zcone_energy = float(
        sum(coefficient * coefficient for coefficient in zcone_solution.coefficients)
    )
    if zcone_energy > bf_energy * (1.0 + 5e-5) + 5e-5:
        raise ContractError("z-cone arm exceeds BF endpoint C-energy cap")
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("direct-z construction did not restore W0/RNG")

    endpoint_energies = {
        NO_OP: 0.0,
        ORACLE_DO_Z: 0.0,
        NATIVE_ORDERED: native_energy,
        NATIVE_C_MATCHED: matched_energy,
        BF_REFRESHED_K4: bf_energy,
        SYNC_Z_CONE: zcone_energy,
    }
    proposals = {
        NATIVE_ORDERED: ordered_w0,
        NATIVE_C_MATCHED: native_c_matched,
        BF_REFRESHED_K4: bf_endpoint,
        SYNC_Z_CONE: zcone,
    }
    feature: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "target_identity": {
            "origin_lineage_id": target_lineage.lineage_id,
            "direct_z_tensor_sha256": target_lineage.direct_z_tensor_sha256,
            "direct_z_artifact_sha256": target_lineage.direct_z_artifact_sha256,
            "target_token_sha256": target_lineage.target_token_sha256,
            "direct_z_compute_count": 1,
        },
        "native_c_energy": native_energy,
        "bf_endpoint_c_energy": bf_energy,
        "hop_c_energy": hop_energy,
        "probe_c_distance": probe_distance,
        "direct_z_delta_l2": delta_l2,
        "bf_layer_weights": list(refreshed_path.layer_weights),
        "zcone_solver": zcone_solution.to_dict(),
        "controller_policy": (
            "BF refreshed K=4 uses only exact rewrite central probes; z-cone uses "
            "W0 five-direction canonical activation responses under the BF endpoint cap"
        ),
    }
    feature["feature_hash"] = _feature_hash(feature)
    action: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(DIRECTZ_BRANCH_ORDER),
        "path_action_hashes": {
            NO_OP: [],
            ORACLE_DO_Z: [target_lineage.direct_z_tensor_sha256],
            NATIVE_ORDERED: [proposal_direction_hash(ordered_w0)],
            NATIVE_C_MATCHED: [proposal_direction_hash(native_c_matched)],
            BF_REFRESHED_K4: [
                proposal_direction_hash(item) for item in refreshed_path.proposals
            ],
            SYNC_Z_CONE: [proposal_direction_hash(zcone)],
        },
        "per_hop_c_energy": hop_energy,
        "endpoint_c_energies": endpoint_energies,
        "evaluation_text_access": "forbidden-until-receipt",
    }
    action["commitment_hash"] = _feature_hash(action)
    receipt_name, receipt_hash = _commit_qstep_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action,
        expected_branch_order=DIRECTZ_BRANCH_ORDER,
        feature_event="direct_z_possibility_feature",
        action_event="direct_z_possibility_action_commitment",
        receipt_schema=RECEIPT_SCHEMA,
    )

    heldout = load_heldout_evaluation_case(root, request)
    baseline = _evaluate_arm(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
        heldout=heldout,
        origin_snapshot=origin_snapshot,
        base_hashes=base_hashes,
        base_state_identity=base_state_identity,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
    )
    if baseline.teacher.rewrite.logits_hash != w0_teacher.rewrite.logits_hash:
        raise RollbackError("post-receipt W0 teacher baseline differs from precommit W0")
    observations: dict[str, BranchObservation] = {NO_OP: baseline}
    observations[ORACLE_DO_Z] = _evaluate_arm(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
        heldout=heldout,
        origin_snapshot=origin_snapshot,
        base_hashes=base_hashes,
        base_state_identity=base_state_identity,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
        patch_delta=shared_delta,
    )
    for arm_id in (NATIVE_ORDERED, NATIVE_C_MATCHED, BF_REFRESHED_K4, SYNC_Z_CONE):
        observations[arm_id] = _evaluate_arm(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
            heldout=heldout,
            origin_snapshot=origin_snapshot,
            base_hashes=base_hashes,
            base_state_identity=base_state_identity,
            cpu_rng=base_cpu_rng,
            cuda_rng=base_cuda_rng,
            proposal=proposals[arm_id],
            exact_application=arm_id in (NATIVE_ORDERED, NATIVE_C_MATCHED),
        )
    if tuple(observations) != DIRECTZ_BRANCH_ORDER:
        raise ContractError("direct-z operational branch order differs from lock")
    for arm_id in DIRECTZ_BRANCH_ORDER:
        endpoint_proposal = proposals.get(arm_id)
        payload = _outcome_payload(
            case_id=request.case_id,
            request_id=request.request_id,
            arm_id=arm_id,
            branch=observations[arm_id],
            baseline=baseline,
            base_teacher=w0_teacher,
            shared_delta=shared_delta,
            canonical_target=canonical_target,
            endpoint_c_energy=endpoint_energies[arm_id],
            endpoint_frobenius_norm=_proposal_frobenius_norm(endpoint_proposal),
            nfe=3,
        )
        outcome_writer.write("direct_z_possibility_outcome", payload)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("direct-z event did not restore W0/RNG")
    return {
        "schema_version": EVENT_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "direct_z_compute_count": 1,
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(DIRECTZ_BRANCH_ORDER),
        "receipt": {"name": receipt_name, "sha256": receipt_hash},
        "evaluation_payload_sha256": heldout.payload_hash,
        "technical": {
            "same_frozen_direct_z": True,
            "precomputed_covariance_only": True,
            "receipt_before_outcome": True,
            "rollback_exact": True,
            "firewall_pass": True,
        },
        "wall_seconds": time.perf_counter() - started,
        "pass": True,
    }


def _validate_event(raw: Mapping[str, Any], request: EditRequest) -> dict[str, Any]:
    result = dict(raw)
    if (
        result.get("schema_version") != EVENT_SCHEMA
        or result.get("case_id") != request.case_id
        or result.get("request_id") != request.request_id
        or result.get("direct_z_compute_count") != 1
        or result.get("feature_count") != 1
        or result.get("commitment_count") != 1
        or result.get("outcome_count") != len(DIRECTZ_BRANCH_ORDER)
        or not result.get("pass")
        or not all(result.get("technical", {}).values())
    ):
        raise ContractError("direct-z event failed its technical gate")
    return result


def run_direct_z_possibility(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    _execution_envelope(model_alias, run_id)
    _slurm = _slurm_state(model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    _git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_directz_selection(root)
    requests = load_counterfact_requests(root, selection["case_ids"])
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not {record.path for record in bridge_provenance.files}.issubset(
        {record.path for record in provenance.files}
    ):
        raise ContractError("direct-z bridge provenance is outside fixed manifest")
    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(DIRECTZ_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("direct-z model loader returned another fixed model")
        contexts = _freeze_contexts(bridge, runtime, seed=DIRECTZ_RUN_SEED)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        outcomes_path = destination / "outcomes.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        direct_z_root = destination / "direct_z"
        receipt_root = destination / "action_receipts"
        receipt_root.mkdir(mode=0o700)
        config_payload = {
            "layers": list(LAYERS),
            "case_count": DIRECTZ_CASE_COUNT,
            "rank_slice": [DIRECTZ_RANK_START, DIRECTZ_RANK_STOP],
            "arm_order": list(DIRECTZ_BRANCH_ORDER),
            "k": QSTEP_K,
            "hop_fraction": DIRECTZ_HOP_FRACTION,
            "probe_fraction": DIRECTZ_PROBE_FRACTION,
            "run_seed": DIRECTZ_RUN_SEED,
        }
        config_sha256 = sha256_bytes(canonical_json(config_payload).encode("utf-8"))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "case_count": DIRECTZ_CASE_COUNT,
            "arm_order": list(DIRECTZ_BRANCH_ORDER),
            "selection_sha256": selection["manifest_id"],
            "config_sha256": config_sha256,
            "bootstrap_seed": DIRECTZ_BOOTSTRAP_SEED,
            "bootstrap_resamples": DIRECTZ_BOOTSTRAP_RESAMPLES,
            "claim_boundary": "possibility_only_not_method_superiority",
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, _loaded_covariances = _load_verified_covariances(
            root=root,
            runtime=runtime,
            specs=covariance_specs,
        )
        torch.cuda.reset_peak_memory_stats(0)
        results: list[dict[str, Any]] = []
        abort_failure_type: str | None = None
        with (
            SanitizedJsonlWriter(features_path, run_id, schema_version=STREAM_SCHEMA) as feature_writer,
            SanitizedJsonlWriter(actions_path, run_id, schema_version=STREAM_SCHEMA) as action_writer,
            SanitizedJsonlWriter(outcomes_path, run_id, schema_version=STREAM_SCHEMA) as outcome_writer,
            SanitizedJsonlWriter(events_path, run_id, schema_version=STREAM_SCHEMA) as event_writer,
        ):
            for request in requests:
                try:
                    raw = _run_event(
                        root=root,
                        runtime=runtime,
                        bridge=bridge,
                        hparams=hparams,
                        contexts=contexts,
                        covariance_specs=covariance_specs,
                        covariance_moments=covariance_moments,
                        direct_z_root=direct_z_root,
                        receipt_root=receipt_root,
                        request=request,
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        outcome_writer=outcome_writer,
                    )
                    result = _validate_event(raw, request)
                except Exception as exc:
                    print(
                        f"DIRECTZ_LOCAL_TRACEBACK_BEGIN type={type(exc).__name__}",
                        file=sys.stderr,
                    )
                    for frame, line_number in traceback.walk_tb(exc.__traceback__):
                        print(
                            f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                            file=sys.stderr,
                        )
                    if isinstance(exc, TensorHashRuntimeError):
                        print(
                            "DIRECTZ_LOCAL_TENSOR_HASH_DIAGNOSTIC "
                            f"phase={exc.phase} category={exc.category}",
                            file=sys.stderr,
                        )
                    print("DIRECTZ_LOCAL_TRACEBACK_END", file=sys.stderr)
                    result = {
                        "schema_version": EVENT_SCHEMA,
                        "case_id": request.case_id,
                        "request_id": request.request_id,
                        "failure_type": type(exc).__name__,
                        "pass": False,
                    }
                    abort_failure_type = type(exc).__name__
                results.append(result)
                event_writer.write("direct_z_possibility_case", result)
                if abort_failure_type is not None:
                    break
            sequences = {
                "features": feature_writer.sequence,
                "actions": action_writer.sequence,
                "outcomes": outcome_writer.sequence,
                "events": event_writer.sequence,
            }
        planned = len(requests)
        attempted = len(results)
        pass_count = sum(bool(result.get("pass", False)) for result in results)
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        exact_counts = bool(
            abort_failure_type is None
            and sequences
            == {
                "features": planned,
                "actions": planned,
                "outcomes": planned * len(DIRECTZ_BRANCH_ORDER),
                "events": planned,
            }
            and receipt_count == planned
        )
        all_rollbacks_exact = bool(
            abort_failure_type is None
            and all(
                bool(result.get("technical", {}).get("rollback_exact"))
                for result in results
            )
        )
        firewall_pass = bool(
            abort_failure_type is None
            and all(
                bool(result.get("technical", {}).get("firewall_pass"))
                for result in results
            )
        )
        receipt_before_outcome = bool(
            abort_failure_type is None
            and all(
                bool(result.get("technical", {}).get("receipt_before_outcome"))
                for result in results
            )
        )
        direct_z_once_per_case = bool(
            abort_failure_type is None
            and all(result.get("direct_z_compute_count") == 1 for result in results)
        )
        all_pass_value = bool(pass_count == planned and exact_counts)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "planned_case_count": planned,
            "attempted_case_count": attempted,
            "pass_case_count": pass_count,
            "failed_case_count": planned - pass_count,
            "outcome_count": sequences["outcomes"],
            "arm_order": list(DIRECTZ_BRANCH_ORDER),
            "selection_sha256": selection["manifest_id"],
            "config_sha256": config_sha256,
            "all_rollbacks_exact": all_rollbacks_exact,
            "firewall_pass": firewall_pass,
            "receipt_before_outcome": receipt_before_outcome,
            "direct_z_once_per_case": direct_z_once_per_case,
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {
            **summary,
            "all_pass": all_pass_value,
            "output_directory": str(destination),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-direct-z-possibility",
        description="Measure direct-z realization fidelity, utility, and preservation.",
        allow_abbrev=False,
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_direct_z_possibility(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"direct-z possibility failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "run_id": summary["run_id"],
                "all_pass": summary["all_pass"],
                "output_directory": summary["output_directory"],
            }
        )
    )
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
