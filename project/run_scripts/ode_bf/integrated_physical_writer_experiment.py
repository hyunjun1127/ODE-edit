"""Standalone single-arm P1R14 rollout and post-freeze experiment shell."""

from __future__ import annotations

import hashlib
import math
import json
import os
import resource
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .common_cold_coordinate import CommonColdScaleMetric
from .common_cold_coordinate import CommonColdScale, common_cold_bootstrap
from .contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    MODEL_ALIASES,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from .cold_start_target import capture_cold_z_base, cold_lookup_positions
from .functional import WaypointFactor, tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_GRID_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
    PHYSICAL_WRITER_MICROBATCH_GROUPS,
    PhysicalWriterComputePhase,
    PhysicalWriterForwardRole,
    PhysicalWriterGroupLedger,
    PhysicalWriterStepStatus,
    solve_integrated_physical_writer,
)
from .integrated_physical_writer_runtime import (
    IntegratedFactorAccumulator,
    accumulate_functional_p_vjp,
    accumulate_writer_vjp_microbatches,
    build_integrated_physical_field_from_keys,
    build_integrated_technical_problem,
    functional_p_vjp_microbatch,
    shared_dynamic_target_factor_group,
    writer_edit_transport_vjp_microbatch,
)
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .p1_replay import evaluate_next_token_log_probs, samplewise_teacher_kl
from .request_digest import ordered_request_digest_v1


R14_EXPERIMENT_SCHEMA = "ode-edit-s05-integrated-physical-writer-experiment/v1"


class IntegratedOutputRootCollision(FileExistsError):
    """The exact create-once P1R14 result root already exists."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise ODEBFStateError("integrated raw-free receipt is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ODEBFStateError("integrated raw-free temporary receipt collides")
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


class IntegratedStageRecorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ordinal = 0
        self.records: list[dict[str, Any]] = []

    def record(self, name: str, payload: Mapping[str, Any]) -> str:
        if not name or "/" in name or ".." in name:
            raise ODEBFContractError("integrated stage name differs")
        path = self.root / f"stage-{self.ordinal:04d}-{name}.json"
        value = {
            "schema": f"{R14_EXPERIMENT_SCHEMA}-stage",
            "ordinal": self.ordinal,
            "name": name,
            **dict(payload),
        }
        sha256 = _atomic_write_once(path, value)
        self.records.append(
            {
                "ordinal": self.ordinal,
                "name": name,
                "path": str(path.relative_to(self.root.parent)),
                "sha256": sha256,
            }
        )
        self.ordinal += 1
        return sha256


@dataclass(frozen=True, slots=True)
class IntegratedActionFreeze:
    request_order_sha256: str
    selected_snapshot_sha256: str
    accepted_transition_count: int
    attempted_field_count: int
    terminal_status: str
    action_frozen: bool = True

    def identity(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(slots=True)
class IntegratedRolloutResult:
    final_target: torch.Tensor
    factors_by_weight: dict[str, tuple[WaypointFactor, ...]]
    accepted_transition_count: int
    attempted_field_count: int
    terminal_status: str
    final_tau: float
    field_receipts: tuple[dict[str, Any], ...]
    transition_receipts: tuple[dict[str, Any], ...]
    compute_receipt: dict[str, Any]
    action_freeze: IntegratedActionFreeze

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": R14_EXPERIMENT_SCHEMA,
            "final_target_sha256": tensor_sha256(self.final_target),
            "factor_count_by_weight": [
                [name, len(factors)]
                for name, factors in sorted(self.factors_by_weight.items())
            ],
            "accepted_transition_count": self.accepted_transition_count,
            "attempted_field_count": self.attempted_field_count,
            "terminal_status": self.terminal_status,
            "final_tau": self.final_tau,
            "field_receipts": list(self.field_receipts),
            "transition_receipts": list(self.transition_receipts),
            "compute_receipt": self.compute_receipt,
            "action_freeze": {
                **asdict(self.action_freeze),
                "identity_sha256": self.action_freeze.identity(),
            },
        }


def _allocation_receipt(
    velocity: Sequence[float],
    applied_theta: Sequence[float],
) -> dict[str, Any]:
    value = np.asarray(tuple(float(item) for item in velocity), dtype=np.float64)
    theta = np.asarray(tuple(float(item) for item in applied_theta), dtype=np.float64)
    if value.shape != (5,) or theta.shape != (5,) or np.any(value < 0.0):
        raise ODEBFContractError("integrated allocation vector differs")
    total = float(value.sum())
    shares = value / total if total > 0.0 else np.zeros_like(value)
    hhi = float(shares @ shares)
    nonzero = shares[shares > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
    neff = float(1.0 / hhi) if hhi > 0.0 else 0.0
    ordered = sorted(
        zip(PHYSICAL_WRITER_LAYER_ORDER, shares, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return {
        "velocity": value.tolist(),
        "applied_theta": theta.tolist(),
        "share": shares.tolist(),
        "top1": [ordered[0][0], float(ordered[0][1])],
        "top2": [ordered[1][0], float(ordered[1][1])],
        "hhi": hhi,
        "entropy": entropy,
        "effective_layer_count": neff,
        "active_mask": [bool(item > 0.0) for item in value],
        "h_application_count": 1,
    }


def _record_vjp_groups(
    ledger: PhysicalWriterGroupLedger,
    *,
    step_index: int,
    writer_receipts: Sequence[Any],
    p_receipts: Sequence[Any],
) -> None:
    for role, prefix, receipts in (
        (PhysicalWriterForwardRole.WRITER_VJP, "writer", writer_receipts),
        (PhysicalWriterForwardRole.P_VJP, "p", p_receipts),
    ):
        for receipt in receipts:
            ordinal = int(
                receipt.context_ordinal
                if role is PhysicalWriterForwardRole.WRITER_VJP
                else receipt.microbatch_ordinal
            )
            ledger.record(
                group_id=f"step-{step_index:02d}-{prefix}-{ordinal}",
                phase=PhysicalWriterComputePhase.PRODUCTION,
                role=role,
                step_index=step_index,
                microbatch_ordinal=ordinal,
                model_forward_calls=receipt.model_forward_calls,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=receipt.autograd_backend_invocations,
                backward_calls=receipt.backward_calls,
                processed_tokens=receipt.processed_token_count,
            )


def run_integrated_virtual_rollout(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    p_anchor_microbatches: Sequence[Sequence[Mapping[str, Any]]],
    p_teacher_log_probs: Sequence[torch.Tensor],
    p_entry_kl: Sequence[torch.Tensor],
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    joint_entry_target: torch.Tensor,
    scale_metric: CommonColdScaleMetric,
    lookup_positions: Sequence[int],
    target_layer_name: str,
    residual_tolerance: float,
    write_stage: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> IntegratedRolloutResult:
    """Run the single Dynamic-BG K8 arm without mutating a model parameter."""

    batch = tuple(requests)
    order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in batch]
    )
    if (
        len(batch) != BATCH_SIZE
        or len(p_anchor_microbatches) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or len(p_teacher_log_probs) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or len(p_entry_kl) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or any(len(item) != BATCH_SIZE for item in p_anchor_microbatches)
        or len(lookup_positions) != BATCH_SIZE * PHYSICAL_WRITER_MICROBATCH_GROUPS
        or scale_metric.request_order_sha256 != order
    ):
        raise ODEBFContractError("integrated rollout inventory differs")
    target = joint_entry_target.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    metric_norm_squared = tuple(
        float(item) ** 2 for item in scale_metric.per_request_z_base_norm
    )
    ledger = PhysicalWriterGroupLedger()
    accumulator = IntegratedFactorAccumulator()
    fields: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    accepted = 0
    attempted = 0
    terminal_status = "INTEGRATED_PHYSICAL_WRITER_TAU_COMPLETE"
    for step in range(PHYSICAL_WRITER_GRID_COUNT):
        cumulative = accumulator.factors_by_weight()
        keys, current, residual, desired, target_group = (
            shared_dynamic_target_factor_group(
                model,
                tokenizer,
                batch,
                contexts,
                target_state=target,
                target_layer_name=target_layer_name,
                lookup_positions=lookup_positions,
                rewrite_module_template=hparams.rewrite_module_tmp,
                fact_token_strategy=hparams.fact_token,
                cumulative_factors_by_weight=cumulative,
                scale_metric=scale_metric,
            )
        )
        ledger.record(
            group_id=f"step-{step:02d}-target-factor",
            phase=PhysicalWriterComputePhase.PRODUCTION,
            role=PhysicalWriterForwardRole.TARGET_FACTOR,
            step_index=step,
            microbatch_ordinal=None,
            model_forward_calls=target_group.model_forward_calls,
            physical_microbatch_graphs=1,
            autograd_backend_invocations=target_group.backward_calls,
            backward_calls=target_group.backward_calls,
            processed_tokens=target_group.processed_token_count,
        )
        # Prior per-layer key parity is a CPU/source gate.  Production uses the
        # single shared multi-hook group above and never pays a five-forward
        # endpoint family merely for observability.
        parity_payload: dict[str, Any] = {
            "policy": "PRECHECKPOINT_SINGLE_GROUP_PRIOR_KEY_PARITY",
            "production_forward_family_count": 0,
            "decision_influence_count": 0,
        }
        field, field_build = build_integrated_physical_field_from_keys(
            model,
            hparams=hparams,
            projector=projector,
            precomputed_keys=keys,
            target_state=target,
            terminal_current_z=current,
            request_order_sha256=order,
            accepted_waypoint=step,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            residual_tolerance=residual_tolerance,
        )
        if (
            field_build.residual_sha256 != tensor_sha256(residual)
            or field_build.target_state_sha256 != tensor_sha256(target)
            or field_build.terminal_current_z_sha256 != tensor_sha256(current)
        ):
            raise ODEBFContractError("integrated target/field identity differs")
        writer_micro = tuple(
            writer_edit_transport_vjp_microbatch(
                model,
                tokenizer,
                batch,
                contexts,
                context_ordinal=ordinal,
                field=field,
                cumulative_factors_by_weight=cumulative,
                target_layer_name=target_layer_name,
                lookup_positions=lookup_positions,
                desired_target=desired,
                metric_norm_squared=metric_norm_squared,
            )
            for ordinal in range(PHYSICAL_WRITER_MICROBATCH_GROUPS)
        )
        writer = accumulate_writer_vjp_microbatches(writer_micro)
        p_micro = tuple(
            functional_p_vjp_microbatch(
                model,
                tokenizer,
                p_anchor_microbatches[ordinal],
                microbatch_ordinal=ordinal,
                field=field,
                cumulative_factors_by_weight=cumulative,
                teacher_log_probs=p_teacher_log_probs[ordinal],
                entry_kl=p_entry_kl[ordinal],
            )
            for ordinal in range(PHYSICAL_WRITER_MICROBATCH_GROUPS)
        )
        p_derivative, p_receipt = accumulate_functional_p_vjp(p_micro)
        _record_vjp_groups(
            ledger,
            step_index=step,
            writer_receipts=writer_micro,
            p_receipts=p_micro,
        )
        technical = build_integrated_technical_problem(
            field,
            a_edit=writer.a_edit,
            committed_load_by_layer={layer: 0.0 for layer in PHYSICAL_WRITER_LAYER_ORDER},
            controller_lock=controller_lock,
        )
        pre_guard: dict[str, Any] = {}

        def observe(value: Mapping[str, Any]) -> None:
            if pre_guard:
                raise ODEBFContractError("integrated pre-guard receipt repeated")
            pre_guard.update(dict(value))

        routing = solve_integrated_physical_writer(
            technical.problem,
            a_edit=writer.a_edit,
            a_transport=writer.a_transport,
            current_nohook_loss=writer.loss_new,
            goal_loss=target_group.goal_target_new_nll,
            step_index=step,
            functional_p_derivative=p_derivative,
            structural_p_matrix_raw=technical.structural_p_matrix_raw,
            pre_guard_observer=observe,
        )
        if not pre_guard:
            raise ODEBFContractError("integrated pre-guard receipt is absent")
        attempted += 1
        field_payload = {
            "schema": f"{R14_EXPERIMENT_SCHEMA}-field",
            "step_index": step,
            "tau_before": step * PHYSICAL_WRITER_H,
            "target_factor": target_group.raw_free_payload(),
            "field": field_build.raw_free_payload(),
            "writer_vjp": writer.raw_free_payload(),
            "functional_p_vjp": p_receipt,
            "technical_problem": technical.raw_free_payload(),
            "pre_guard": pre_guard,
            "prior_key_parity": parity_payload,
            "routing": routing.raw_free_payload(),
            "routing_sha256": routing.identity_sha256,
            "overlay_guard_objective_termination_influence_count": 0,
            "hard_h_p_budget_veto_count": 0,
        }
        field_payload["identity_sha256"] = canonical_hash(field_payload)
        fields.append(field_payload)
        if write_stage is not None:
            write_stage(f"field-{step:04d}", field_payload)
        if routing.status is PhysicalWriterStepStatus.NO_PHYSICAL_W_ONLY_DIRECTION:
            terminal_status = routing.status.value
            break
        factor_receipt = accumulator.append(
            field, routing.applied_theta, step_index=step
        )
        accepted += 1
        target = desired.detach().to(device="cpu", dtype=torch.float32).contiguous()
        transition = {
            "schema": f"{R14_EXPERIMENT_SCHEMA}-transition",
            "step_index": step,
            "tau_after": accepted * PHYSICAL_WRITER_H,
            "status": routing.status.value,
            "target_sha256": tensor_sha256(target),
            "routing": routing.raw_free_payload(),
            "allocation": _allocation_receipt(
                routing.velocity, routing.applied_theta
            ),
            "factor_accumulation": factor_receipt.raw_free_payload(),
            "target_clock_advance_count": 1,
            "weight_clock_advance_count": 1,
            "scientific_retry_count": 0,
            "first_hit_decision_influence_count": 0,
        }
        transition["identity_sha256"] = canonical_hash(transition)
        transitions.append(transition)
        if write_stage is not None:
            write_stage(f"transition-{step + 1:04d}", transition)
    if accepted == PHYSICAL_WRITER_GRID_COUNT:
        accumulator.assert_complete_k8()
    compute = ledger.validate_production(
        fixed_online_forward_groups=0,
        completed_steps=accepted,
        attempted_field_count=attempted,
    )
    snapshot_sha = canonical_hash(
        {
            "target_sha256": tensor_sha256(target),
            "factor_step_receipt_sha256": [
                item.identity_sha256 for item in accumulator.receipts
            ],
            "accepted": accepted,
            "attempted": attempted,
            "terminal_status": terminal_status,
        }
    )
    freeze = IntegratedActionFreeze(
        order,
        snapshot_sha,
        accepted,
        attempted,
        terminal_status,
    )
    if write_stage is not None:
        write_stage(
            "action-freeze",
            {
                "schema": f"{R14_EXPERIMENT_SCHEMA}-action-freeze",
                **asdict(freeze),
                "identity_sha256": freeze.identity(),
                "factor_step_receipt_sha256": [
                    item.identity_sha256 for item in accumulator.receipts
                ],
                "heldout_controller_access_count": 0,
            },
        )
    ledger.freeze_actions(freeze.identity())
    return IntegratedRolloutResult(
        target,
        accumulator.factors_by_weight(),
        accepted,
        attempted,
        terminal_status,
        accepted * PHYSICAL_WRITER_H,
        tuple(fields),
        tuple(transitions),
        compute,
        freeze,
    )


def _validate_source_freeze(repo_root: Path, source_head: str) -> None:
    if len(source_head) != 40 or any(item not in "0123456789abcdef" for item in source_head):
        raise ODEBFContractError("integrated source head differs")
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if observed != source_head or tracked:
        raise ODEBFContractError("integrated source freeze differs")


def run_integrated_physical_writer_experiment(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
    run_attempt_id: str,
) -> dict[str, Any]:
    """Execute the standalone one-arm P1R14 validation lifecycle."""

    from .alpha_backend import fresh_contexts_twice, load_original_bf16, seed_all
    from .artifacts import ODEBFArtifactGuard
    from .integrated_physical_writer_selection import (
        build_r14_fresh_seal,
        load_r14_controller_requests,
    )
    from .p1_integrated_physical_writer_panel import (
        P1R14_NUMERICAL_LOCK_FILE,
        P1R14_SOURCE_MANIFEST_FILE,
        load_and_validate_p1r14_numerical_lock,
        load_and_validate_p1r14_seals,
        load_and_validate_p1r14_source_manifest,
        p1r14_forecast,
        validate_p1r14_attempt_output_namespace,
        validate_p1r14_source_closure,
    )

    root = repo_root.resolve(strict=True)
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("integrated experiment alias differs")
    _validate_source_freeze(root, source_head)
    namespace = validate_p1r14_attempt_output_namespace(
        repo_root=root,
        alias=alias,
        output_root=output_root,
        run_attempt_id=run_attempt_id,
    )
    destination = Path(namespace["result_root"])
    if destination.exists() or destination.is_symlink():
        raise IntegratedOutputRootCollision("integrated result root is create-once")
    try:
        destination.mkdir(mode=0o700, parents=True)
    except FileExistsError as exc:
        raise IntegratedOutputRootCollision("integrated result root is create-once") from exc
    raw_root = destination / "raw"
    stages_root = raw_root / "stages"
    stages_root.mkdir(mode=0o700, parents=True)
    stages = IntegratedStageRecorder(stages_root)
    started = time.time()
    locks = root / "project" / "run_scripts" / "ode_bf" / "locks"
    guard = ODEBFArtifactGuard(
        root,
        locks / "p0_artifact_lock.json",
        alias,
        require_held_ode_alloc=False,
    )
    artifact_receipt = guard.preflight()
    exclusion, seal = load_and_validate_p1r14_seals(locks)
    rebuilt_seal = build_r14_fresh_seal(guard.base_guard.dataset, exclusion)
    if rebuilt_seal["root_digest"] != seal["root_digest"]:
        raise ODEBFContractError("integrated fresh seal dataset replay differs")
    source_closure = validate_p1r14_source_closure(root)
    source_manifest, source_manifest_sha = load_and_validate_p1r14_source_manifest(
        root,
        locks / P1R14_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    numerical, numerical_sha = load_and_validate_p1r14_numerical_lock(
        locks / P1R14_NUMERICAL_LOCK_FILE,
        exclusion_root=exclusion["root_digest"],
        fresh_seal_root=seal["root_digest"],
        artifact_lock_root=artifact_receipt.lock_sha256,
    )
    forecast = p1r14_forecast()
    if not forecast.fits:
        raise ODEBFContractError("integrated resource forecast differs")
    requests = load_r14_controller_requests(guard.base_guard.dataset, seal)
    anchor_population = load_r14_controller_requests(
        guard.base_guard.dataset, seal, population="functional_p_anchors"
    )
    anchor_groups = tuple(
        tuple(anchor_population[start : start + BATCH_SIZE])
        for start in range(0, len(anchor_population), BATCH_SIZE)
    )
    if len(anchor_groups) != PHYSICAL_WRITER_MICROBATCH_GROUPS:
        raise ODEBFContractError("integrated P anchor grouping differs")
    stages.record(
        "preflight",
        {
            "source_head": source_head,
            "run_attempt_id": namespace["run_attempt_id"],
            "run_attempt_result_name": namespace["result_name"],
            "run_attempt_namespace_identity_sha256": namespace[
                "identity_sha256"
            ],
            "artifact_lock_sha256": artifact_receipt.lock_sha256,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest["root_digest"],
            "source_closure_sha256": source_closure["identity_sha256"],
            "numerical_lock_sha256": numerical_sha,
            "fresh_seal_root": seal["root_digest"],
            "historical_exclusion_root": exclusion["root_digest"],
            "request_order_sha256": seal["batch_ordered_request_digest_v1"],
            "functional_p_anchor_order_sha256": seal[
                "functional_p_anchor_order_digest"
            ],
            "heldout_field_open_count": 0,
            "model_access_count_before_seal_validation": 0,
        },
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ODEBFContractError("integrated runtime requires one visible GPU")
    free_bytes, allocatable_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    if int(free_bytes) < 32 * 1024**3 or int(allocatable_bytes) < 32 * 1024**3:
        raise ODEBFContractError("integrated live GPU capacity differs")
    stages.record(
        "runtime-capacity",
        {
            "visible_device_count": 1,
            "device_name_sha256": hashlib.sha256(
                str(properties.name).encode("utf-8")
            ).hexdigest(),
            "physical_total_bytes": int(properties.total_memory),
            "allocatable_total_bytes": int(allocatable_bytes),
            "free_bytes": int(free_bytes),
            "forecast": forecast.raw_free_payload(),
        },
    )
    seed_all(COMMON_SEED)
    model, tokenizer, hparams = load_original_bf16(guard)
    contexts, context_sha = fresh_contexts_twice(
        model, tokenizer, seed=COMMON_SEED
    )
    if tuple(len(group) for group in contexts) != (1, 5):
        raise ODEBFContractError("integrated context inventory differs")
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    if order != seal["batch_ordered_request_digest_v1"]:
        raise ODEBFContractError("integrated runtime request order differs")
    z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
    lookup_positions = cold_lookup_positions(
        tokenizer,
        requests,
        contexts,
        fact_token_strategy=hparams.fact_token,
    )
    metric = CommonColdScaleMetric.from_z_base(
        z_base, order, CommonColdScale.BATCH_GLOBAL
    )
    bootstrap_ledger = ComputeLedger()
    joint_target, bootstrap = common_cold_bootstrap(
        model,
        tokenizer,
        requests,
        contexts,
        target_layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        z_base=z_base,
        metric=metric,
        ledger=bootstrap_ledger,
    )
    stages.record(
        "model-context-bootstrap",
        {
            "parameter_count": sum(item.numel() for item in model.parameters()),
            "parameter_dtype": "torch.bfloat16",
            "context_contract_sha256": seal["context_contract_sha256"],
            "context_sha256": context_sha,
            "target_layer_name_sha256": hashlib.sha256(
                target_layer_name.encode("utf-8")
            ).hexdigest(),
            "lookup_position_sha256": canonical_hash(list(lookup_positions)),
            "z_base_sha256": tensor_sha256(z_base),
            "scale_metric": metric.raw_free_payload(),
            "bootstrap": bootstrap,
            "bootstrap_compute": bootstrap_ledger.raw_free_payload(),
            "native_or_direct_z_controller_access_count": 0,
        },
    )
    teacher_groups: list[torch.Tensor] = []
    entry_kl_groups: list[torch.Tensor] = []
    teacher_receipts: list[dict[str, Any]] = []
    for ordinal, anchors in enumerate(anchor_groups):
        teacher, receipt = evaluate_next_token_log_probs(model, tokenizer, anchors)
        entry_kl = samplewise_teacher_kl(teacher, teacher)
        teacher_groups.append(teacher)
        entry_kl_groups.append(entry_kl)
        teacher_receipts.append(
            {
                "ordinal": ordinal,
                "order_sha256": receipt.request_order_sha256,
                "teacher_sha256": tensor_sha256(teacher),
                "entry_kl_sha256": tensor_sha256(entry_kl),
                "model_forward_count": receipt.model_forward_count,
                "processed_token_count": receipt.processed_token_count,
            }
        )
    stages.record(
        "functional-p-entry",
        {
            "anchor_count": 60,
            "groups": teacher_receipts,
            "historical_h_item_count": 0,
            "historical_h_compute_count": 0,
        },
    )
    projector = torch.load(guard.projector, map_location="cpu", weights_only=True)
    if (
        not isinstance(projector, torch.Tensor)
        or projector.dtype is not torch.float32
        or projector.ndim != 3
        or projector.shape[0] != len(hparams.layers)
    ):
        raise ODEBFContractError("integrated projector tensor differs")
    covariance_registry = PinnedCovarianceRegistry(
        easyedit_root=guard.easyedit_root,
        covariance_spec=guard.base_guard.spec["covariance"],
    )
    touched = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": dict(
            model.named_parameters()
        )[f"{hparams.rewrite_module_tmp.format(layer)}.weight"]
        for layer in hparams.layers
    }
    w0_sha = {
        name: tensor_sha256(parameter)
        for name, parameter in sorted(touched.items())
    }
    mutation_lock = threading.RLock()
    controller_lock = P1ControllerLock()
    rollout = run_integrated_virtual_rollout(
        model,
        tokenizer,
        requests=requests,
        contexts=contexts,
        p_anchor_microbatches=anchor_groups,
        p_teacher_log_probs=tuple(teacher_groups),
        p_entry_kl=tuple(entry_kl_groups),
        hparams=hparams,
        projector=projector,
        covariance_registry=covariance_registry,
        projector_sha256=guard.spec["projector_sha256"],
        controller_lock=controller_lock,
        joint_entry_target=joint_target,
        scale_metric=metric,
        lookup_positions=lookup_positions,
        target_layer_name=target_layer_name,
        residual_tolerance=controller_lock.residual_tolerance,
        write_stage=stages.record,
    )
    if any(
        tensor_sha256(touched[name]) != digest
        for name, digest in w0_sha.items()
    ):
        raise ODEBFStateError("integrated virtual rollout changed W0")
    stages.record("rollout-terminal-prefix", rollout.raw_free_payload())
    from .integrated_physical_writer_terminal import run_integrated_terminal_panel

    terminal = run_integrated_terminal_panel(
        model,
        tokenizer,
        model_alias=alias,
        requests=requests,
        dataset_path=guard.base_guard.dataset,
        hparams=hparams,
        target_layer_name=target_layer_name,
        final_target=rollout.final_target,
        factors_by_weight=rollout.factors_by_weight,
        accepted_transition_count=rollout.accepted_transition_count,
        action_freeze=rollout.action_freeze,
        p_anchor_microbatches=anchor_groups,
        p_teacher_log_probs=tuple(teacher_groups),
        p_entry_kl=tuple(entry_kl_groups),
        projector=projector,
        contexts=contexts,
        mutation_lock=mutation_lock,
        residual_tolerance=controller_lock.residual_tolerance,
    )
    final_w0_sha = {
        name: tensor_sha256(parameter)
        for name, parameter in sorted(touched.items())
    }
    if final_w0_sha != w0_sha:
        raise ODEBFStateError("integrated final W0 restore differs")
    guard.assert_unchanged()
    terminal_sha = stages.record(
        "terminal-panel", terminal.raw_free_payload()
    )
    elapsed = time.time() - started
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "schema": f"{R14_EXPERIMENT_SCHEMA}-terminal",
        "status": "INTEGRATED_PHYSICAL_WRITER_EXPERIMENT_TERMINAL",
        "model_alias": alias,
        "source_head": source_head,
        "method_id": numerical["method_id"],
        "fresh_seal_root": seal["root_digest"],
        "accepted_transition_count": rollout.accepted_transition_count,
        "attempted_field_count": rollout.attempted_field_count,
        "tau_final": rollout.final_tau,
        "trajectory_status": rollout.terminal_status,
        "action_freeze_sha256": rollout.action_freeze.identity(),
        "terminal_panel_sha256": terminal_sha,
        "final_w0_restore_exact": True,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "scientific_retry_count": 0,
        "first_hit_decision_influence_count": 0,
        "hard_h_p_budget_veto_count": 0,
        "scientific_promotion_authorized": False,
        "wall_seconds": elapsed,
        "host_maxrss_kib": maxrss,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "stage_receipts": list(stages.records),
    }
    result["identity_sha256"] = canonical_hash(result)
    result_sha = _atomic_write_once(destination / "terminal.json", result)
    members: list[dict[str, Any]] = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json":
            members.append(
                {
                    "path": str(path.relative_to(destination)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    manifest = {
        "schema": f"{R14_EXPERIMENT_SCHEMA}-result-manifest",
        "terminal_sha256": result_sha,
        "members": members,
        "member_root": canonical_hash(members),
    }
    manifest["root_digest"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": result["status"],
        "terminal_sha256": result_sha,
        "manifest_sha256": manifest_sha,
        "manifest_root": manifest["root_digest"],
        "accepted_transition_count": rollout.accepted_transition_count,
        "tau_final": rollout.final_tau,
        "trajectory_status": rollout.terminal_status,
        "final_w0_restore_exact": True,
        "scientific_promotion_authorized": False,
    }


__all__ = [
    "IntegratedOutputRootCollision",
    "IntegratedActionFreeze",
    "IntegratedRolloutResult",
    "run_integrated_virtual_rollout",
    "run_integrated_physical_writer_experiment",
]
