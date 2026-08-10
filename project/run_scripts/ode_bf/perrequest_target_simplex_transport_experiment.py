"""Single-arm Stage-A rollout for the P1R15 integrated method repair.

Stage A intentionally binds the outcome-selected R13 request seal for
mechanistic comparison.  This module contains no Stage-B selector, salt,
case identity, Native controller input, factorial arm, or retry path.
"""

from __future__ import annotations

import hashlib
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

from .contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    MODEL_ALIASES,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from .functional import WaypointFactor, tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_GRID_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
    PHYSICAL_WRITER_MICROBATCH_GROUPS,
    PhysicalWriterComputePhase,
    PhysicalWriterForwardRole,
    PhysicalWriterGroupLedger,
)
from .integrated_physical_writer_experiment import IntegratedActionFreeze
from .integrated_physical_writer_runtime import (
    IntegratedFactorAccumulator,
    accumulate_functional_p_vjp,
    accumulate_writer_vjp_microbatches,
    build_integrated_physical_field_from_keys,
    build_integrated_technical_problem,
    functional_p_vjp_microbatch,
    writer_edit_transport_vjp_microbatch,
)
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .perrequest_target_simplex_transport import (
    P1R15_METHOD_ID,
    P1R15_SCHEMA,
    SimplexStepStatus,
    TargetStepStatus,
    solve_perrequest_simplex_transport,
)
from .perrequest_target_simplex_transport_runtime import (
    perrequest_dynamic_target_factor_group,
)
from .request_digest import ordered_request_digest_v1


P1R15_STAGE_A_R13_SEAL_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)
P1R15_STAGE_A_R13_REQUEST_ORDER = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)
P1R15_STAGE_A_R13_CHECKPOINT = (
    "e6facd2d5dfae12d3c094b51981ad99951174109"
)
P1R15_STAGE_A_KIND = "OUTCOME_SELECTED_R13_MECHANISTIC_REGRESSION"


class P1R15OutputRootCollision(FileExistsError):
    """The exact create-once Stage-A result root already exists."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise ODEBFStateError("P1R15 raw-free receipt is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ODEBFStateError("P1R15 temporary receipt collides")
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


class P1R15StageRecorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ordinal = 0
        self.records: list[dict[str, Any]] = []

    def record(self, name: str, payload: Mapping[str, Any]) -> str:
        if not name or "/" in name or ".." in name:
            raise ODEBFContractError("P1R15 stage name differs")
        path = self.root / f"stage-{self.ordinal:04d}-{name}.json"
        value = {
            "schema": f"{P1R15_SCHEMA}-stage",
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


@dataclass(slots=True)
class P1R15StageARolloutResult:
    final_target: torch.Tensor
    factors_by_weight: dict[str, tuple[WaypointFactor, ...]]
    accepted_transition_count: int
    attempted_full_field_count: int
    target_group_count: int
    terminal_status: str
    terminal_component: str
    final_tau: float
    field_receipts: tuple[dict[str, Any], ...]
    transition_receipts: tuple[dict[str, Any], ...]
    compute_receipt: dict[str, Any]
    action_freeze: IntegratedActionFreeze

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": f"{P1R15_SCHEMA}-stage-a-rollout",
            "method_id": P1R15_METHOD_ID,
            "stage_a_kind": P1R15_STAGE_A_KIND,
            "stage_a_seal_root": P1R15_STAGE_A_R13_SEAL_ROOT,
            "stage_a_request_order_sha256": P1R15_STAGE_A_R13_REQUEST_ORDER,
            "stage_b_material_count": 0,
            "final_target_sha256": tensor_sha256(self.final_target),
            "factor_count_by_weight": [
                [name, len(factors)]
                for name, factors in sorted(self.factors_by_weight.items())
            ],
            "accepted_transition_count": self.accepted_transition_count,
            "attempted_full_field_count": self.attempted_full_field_count,
            "target_group_count": self.target_group_count,
            "terminal_status": self.terminal_status,
            "terminal_component": self.terminal_component,
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
    velocity: Sequence[float], applied_theta: Sequence[float]
) -> dict[str, Any]:
    value = np.asarray(tuple(float(item) for item in velocity), dtype=np.float64)
    theta = np.asarray(tuple(float(item) for item in applied_theta), dtype=np.float64)
    if (
        value.shape != (5,)
        or theta.shape != (5,)
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or not np.array_equal(theta, PHYSICAL_WRITER_H * value)
    ):
        raise ODEBFContractError("P1R15 allocation identity differs")
    speed = float(value.sum())
    share = value / speed if speed > 0.0 else np.zeros_like(value)
    hhi = float(share @ share)
    ordered = sorted(
        zip(PHYSICAL_WRITER_LAYER_ORDER, share, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return {
        "velocity": value.tolist(),
        "applied_theta": theta.tolist(),
        "speed": speed,
        "alpha": None if speed == 0.0 else share.tolist(),
        "alpha_status": (
            "UNDEFINED_ZERO_SPEED" if speed == 0.0 else "DEFINED_POSITIVE_SPEED"
        ),
        "top1": [ordered[0][0], float(ordered[0][1])],
        "top2": [ordered[1][0], float(ordered[1][1])],
        "hhi": hhi,
        "effective_layer_count": 0.0 if hhi == 0.0 else 1.0 / hhi,
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


def _partial_compute_receipt(
    ledger: PhysicalWriterGroupLedger,
    *,
    accepted: int,
    attempted_full: int,
    target_groups: int,
) -> dict[str, Any]:
    records = ledger.records
    payload = {
        "schema": f"{P1R15_SCHEMA}-partial-compute-ledger",
        "accepted_transition_count": accepted,
        "attempted_full_field_count": attempted_full,
        "target_group_count": target_groups,
        "production_forward_group_count": len(records),
        "model_forward_call_count": sum(item.model_forward_calls for item in records),
        "backward_call_count": sum(item.backward_calls for item in records),
        "processed_token_count": sum(item.processed_tokens for item in records),
        "terminal_evaluation_count": 0,
        "records": [item.raw_free_payload() for item in records],
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def run_p1r15_stage_a_virtual_rollout(
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
    z_base: torch.Tensor,
    lookup_positions: Sequence[int],
    target_layer_name: str,
    residual_tolerance: float,
    write_stage: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> P1R15StageARolloutResult:
    """Run one P1R15 Dynamic target trajectory without parameter mutation."""

    batch = tuple(requests)
    order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in batch]
    )
    if (
        len(batch) != BATCH_SIZE
        or order != P1R15_STAGE_A_R13_REQUEST_ORDER
        or len(p_anchor_microbatches) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or len(p_teacher_log_probs) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or len(p_entry_kl) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or any(len(group) != BATCH_SIZE for group in p_anchor_microbatches)
        or len(lookup_positions) != BATCH_SIZE * PHYSICAL_WRITER_MICROBATCH_GROUPS
        or z_base.ndim != 2
        or z_base.shape[1] != BATCH_SIZE
    ):
        raise ODEBFContractError("P1R15 Stage-A rollout inventory differs")
    target = z_base.detach().to(device="cpu", dtype=torch.float32).contiguous()
    base_norm = torch.linalg.vector_norm(
        z_base.detach().to(device="cpu", dtype=torch.float64), dim=0
    )
    metric_norm_squared = tuple(float(item) ** 2 for item in base_norm)
    cumulative_g_path = (0.0,) * BATCH_SIZE
    ledger = PhysicalWriterGroupLedger()
    accumulator = IntegratedFactorAccumulator()
    fields: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    accepted = 0
    attempted_full = 0
    target_groups = 0
    terminal_status = "P1R15_FIXED_E8_TAU_COMPLETE"
    terminal_component = "NONE"
    for step in range(PHYSICAL_WRITER_GRID_COUNT):
        cumulative = accumulator.factors_by_weight()
        keys, current, residual, desired, target_group = (
            perrequest_dynamic_target_factor_group(
                model,
                tokenizer,
                batch,
                contexts,
                target_state=target,
                z_base=z_base,
                target_layer_name=target_layer_name,
                lookup_positions=lookup_positions,
                rewrite_module_template=hparams.rewrite_module_tmp,
                fact_token_strategy=hparams.fact_token,
                cumulative_factors_by_weight=cumulative,
                step_index=step,
                cumulative_g_path_before=cumulative_g_path,
            )
        )
        target_groups += 1
        ledger.record(
            group_id=f"step-{step:02d}-target-factor",
            phase=PhysicalWriterComputePhase.PRODUCTION,
            role=PhysicalWriterForwardRole.TARGET_FACTOR,
            step_index=step,
            microbatch_ordinal=None,
            model_forward_calls=target_group.model_forward_calls,
            physical_microbatch_graphs=1,
            autograd_backend_invocations=target_group.autograd_backend_invocations,
            backward_calls=target_group.backward_calls,
            processed_tokens=target_group.processed_token_count,
        )
        if target_group.target_step.status is TargetStepStatus.TARGET_NO_DESCENT_DIRECTION:
            terminal_status = target_group.target_step.status.value
            terminal_component = target_group.target_step.status.value
            boundary = {
                "schema": f"{P1R15_SCHEMA}-target-boundary",
                "step_index": step,
                "tau_before": accepted * PHYSICAL_WRITER_H,
                "target_factor": target_group.raw_free_payload(),
                "target_clock_advance_count": 0,
                "weight_clock_advance_count": 0,
                "candidate_count": 0,
                "scientific_retry_count": 0,
            }
            boundary["identity_sha256"] = canonical_hash(boundary)
            fields.append(boundary)
            if write_stage is not None:
                write_stage(f"target-boundary-{step:04d}", boundary)
            break

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
        if field_build.residual_sha256 != tensor_sha256(residual):
            raise ODEBFContractError("P1R15 field residual identity differs")
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
        attempted_full += 1
        technical = build_integrated_technical_problem(
            field,
            a_edit=writer.a_edit,
            committed_load_by_layer={layer: 0.0 for layer in PHYSICAL_WRITER_LAYER_ORDER},
            controller_lock=controller_lock,
        )
        pre_guard: dict[str, Any] = {}

        def observe(value: Mapping[str, Any]) -> None:
            if pre_guard:
                raise ODEBFContractError("P1R15 pre-guard receipt repeated")
            pre_guard.update(dict(value))

        routing = solve_perrequest_simplex_transport(
            technical.problem,
            a_edit=writer.a_edit,
            a_transport=writer.a_transport,
            current_nohook_loss=writer.loss_new,
            goal_loss=target_group.goal_target_new_nll,
            transport_loss_current=writer.loss_transport,
            step_index=step,
            functional_p_derivative=p_derivative,
            structural_p_matrix_raw=technical.structural_p_matrix_raw,
            pre_guard_observer=observe,
        )
        if not pre_guard:
            raise ODEBFContractError("P1R15 pre-guard receipt is absent")
        field_payload = {
            "schema": f"{P1R15_SCHEMA}-field",
            "step_index": step,
            "tau_before": accepted * PHYSICAL_WRITER_H,
            "target_factor": target_group.raw_free_payload(),
            "field": field_build.raw_free_payload(),
            "writer_vjp": writer.raw_free_payload(),
            "functional_p_vjp": p_receipt,
            "technical_problem": technical.raw_free_payload(),
            "pre_guard": pre_guard,
            "routing": routing.raw_free_payload(),
            "routing_sha256": routing.identity_sha256,
            "overlay_guard_objective_termination_influence_count": 0,
            "hard_h_p_budget_veto_count": 0,
            "positive_layer_mask_access_count": 0,
        }
        field_payload["identity_sha256"] = canonical_hash(field_payload)
        fields.append(field_payload)
        if write_stage is not None:
            write_stage(f"field-{step:04d}", field_payload)
        if routing.status is SimplexStepStatus.NO_JOINT_PHYSICAL_TRANSPORT_DIRECTION:
            terminal_status = routing.status.value
            terminal_component = routing.failure_component.value
            break
        factor_receipt = accumulator.append(
            field, routing.applied_theta, step_index=step
        )
        accepted += 1
        target = desired.detach().to(device="cpu", dtype=torch.float32).contiguous()
        cumulative_g_path = target_group.target_step.cumulative_g_path_after
        transition = {
            "schema": f"{P1R15_SCHEMA}-transition",
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
    if target_groups == attempted_full:
        compute = ledger.validate_production(
            fixed_online_forward_groups=0,
            completed_steps=accepted,
            attempted_field_count=attempted_full,
        )
    else:
        compute = _partial_compute_receipt(
            ledger,
            accepted=accepted,
            attempted_full=attempted_full,
            target_groups=target_groups,
        )
    snapshot_sha = canonical_hash(
        {
            "target_sha256": tensor_sha256(target),
            "factor_step_receipt_sha256": [
                item.identity_sha256 for item in accumulator.receipts
            ],
            "accepted": accepted,
            "attempted_full": attempted_full,
            "target_groups": target_groups,
            "terminal_status": terminal_status,
            "terminal_component": terminal_component,
        }
    )
    freeze = IntegratedActionFreeze(
        order,
        snapshot_sha,
        accepted,
        target_groups,
        terminal_status,
    )
    if write_stage is not None:
        write_stage(
            "action-freeze",
            {
                "schema": f"{P1R15_SCHEMA}-action-freeze",
                **asdict(freeze),
                "terminal_component": terminal_component,
                "identity_sha256": freeze.identity(),
                "heldout_controller_access_count": 0,
                "stage_b_material_count": 0,
            },
        )
    ledger.freeze_actions(freeze.identity())
    return P1R15StageARolloutResult(
        target,
        accumulator.factors_by_weight(),
        accepted,
        attempted_full,
        target_groups,
        terminal_status,
        terminal_component,
        accepted * PHYSICAL_WRITER_H,
        tuple(fields),
        tuple(transitions),
        compute,
        freeze,
    )


def _validate_source_freeze(repo_root: Path, source_head: str) -> None:
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ODEBFContractError("P1R15 source head differs")
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
        raise ODEBFContractError("P1R15 source freeze differs")


def run_p1r15_stage_a_experiment(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
) -> dict[str, Any]:
    """Execute the single-arm, reused-R13-seal Stage-A lifecycle."""

    from .alpha_backend import fresh_contexts_twice, load_original_bf16, seed_all
    from .artifacts import ODEBFArtifactGuard, load_rooted_json
    from .cold_start_target import capture_cold_z_base, cold_lookup_positions
    from .integrated_physical_writer_selection import (
        load_r14_controller_requests,
        verify_r14_fresh_seal,
        verify_r14_historical_exclusion,
    )
    from .integrated_physical_writer_terminal import run_integrated_terminal_panel
    from .p1_perrequest_target_simplex_transport_panel import (
        P1R15_NUMERICAL_LOCK_FILE,
        P1R15_P_ANCHOR_SEAL_FILE,
        P1R15_P_ANCHOR_SEAL_ROOT,
        P1R15_SOURCE_MANIFEST_FILE,
        expected_p1r15_stage_a_context_sha256,
        load_and_validate_p1r15_numerical_lock,
        load_and_validate_p1r15_source_manifest,
        load_and_validate_p1r15_stage_a_seal,
        load_common_cold_requests,
        p1r15_stage_a_forecast,
        validate_p1r15_source_closure,
        validate_p1r15_stage_a_output_root,
    )
    from .p1_replay import evaluate_next_token_log_probs, samplewise_teacher_kl

    root = repo_root.resolve(strict=True)
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R15 experiment alias differs")
    _validate_source_freeze(root, source_head)
    namespace = validate_p1r15_stage_a_output_root(
        repo_root=root, alias=alias, output_root=output_root
    )
    destination = output_root.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise P1R15OutputRootCollision("P1R15 result root is create-once")
    try:
        destination.mkdir(mode=0o700, parents=True)
    except FileExistsError as exc:
        raise P1R15OutputRootCollision("P1R15 result root is create-once") from exc
    stages_root = destination / "raw" / "stages"
    stages_root.mkdir(mode=0o700, parents=True)
    stages = P1R15StageRecorder(stages_root)
    started = time.time()
    locks = root / "project" / "run_scripts" / "ode_bf" / "locks"
    guard = ODEBFArtifactGuard(
        root,
        locks / "p0_artifact_lock.json",
        alias,
        require_held_ode_alloc=False,
    )
    artifact = guard.preflight()
    stage_a_seal, stage_a_seal_sha = load_and_validate_p1r15_stage_a_seal(locks)
    numerical, numerical_sha = load_and_validate_p1r15_numerical_lock(
        locks / P1R15_NUMERICAL_LOCK_FILE
    )
    source_closure = validate_p1r15_source_closure(root)
    source_manifest, source_manifest_sha = load_and_validate_p1r15_source_manifest(
        root,
        locks / P1R15_SOURCE_MANIFEST_FILE,
        source_head=source_head,
    )
    historical, _ = load_rooted_json(locks / "p1r14_historical_exclusion.json")
    historical = verify_r14_historical_exclusion(historical)
    anchor_seal, anchor_seal_sha = load_rooted_json(
        locks / P1R15_P_ANCHOR_SEAL_FILE
    )
    anchor_seal = verify_r14_fresh_seal(
        anchor_seal, exclusion_root=historical["root_digest"]
    )
    if anchor_seal["root_digest"] != P1R15_P_ANCHOR_SEAL_ROOT:
        raise ODEBFContractError("P1R15 P-anchor seal differs")
    forecast = p1r15_stage_a_forecast(alias)
    if not forecast.fits:
        raise ODEBFContractError("P1R15 resource forecast differs")
    requests = load_common_cold_requests(
        Path(guard.base_guard.dataset), stage_a_seal
    )
    anchors = load_r14_controller_requests(
        guard.base_guard.dataset,
        anchor_seal,
        population="functional_p_anchors",
    )
    anchor_groups = tuple(
        tuple(anchors[start : start + BATCH_SIZE])
        for start in range(0, len(anchors), BATCH_SIZE)
    )
    if len(anchor_groups) != PHYSICAL_WRITER_MICROBATCH_GROUPS:
        raise ODEBFContractError("P1R15 P-anchor grouping differs")
    stages.record(
        "preflight",
        {
            "source_head": source_head,
            "method_id": P1R15_METHOD_ID,
            "output_namespace": namespace,
            "artifact_lock_sha256": artifact.lock_sha256,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_root": source_manifest["root_digest"],
            "source_closure_sha256": source_closure["identity_sha256"],
            "numerical_lock_sha256": numerical_sha,
            "numerical_lock_root": numerical["root_digest"],
            "stage_a_seal_sha256": stage_a_seal_sha,
            "stage_a_seal_root": stage_a_seal["root_digest"],
            "stage_a_request_order_sha256": P1R15_STAGE_A_R13_REQUEST_ORDER,
            "p_anchor_seal_sha256": anchor_seal_sha,
            "p_anchor_seal_root": anchor_seal["root_digest"],
            "stage_b_material_count": 0,
            "stage_b_access_count": 0,
            "model_access_count_before_stage_a_seal_validation": 0,
            "heldout_access_count_before_action_freeze": 0,
        },
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ODEBFContractError("P1R15 runtime requires one visible GPU")
    free_bytes, allocatable_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    if int(free_bytes) < 32 * 1024**3 or int(allocatable_bytes) < 32 * 1024**3:
        raise ODEBFContractError("P1R15 live GPU capacity differs")
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
    contexts, context_sha = fresh_contexts_twice(model, tokenizer, seed=COMMON_SEED)
    if tuple(len(group) for group in contexts) != (1, 5):
        raise ODEBFContractError("P1R15 context geometry differs")
    expected_context_sha = expected_p1r15_stage_a_context_sha256(
        stage_a_seal, alias
    )
    if context_sha != expected_context_sha:
        raise ODEBFContractError("P1R15 Stage-A R13 context identity differs")
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    if order != P1R15_STAGE_A_R13_REQUEST_ORDER:
        raise ODEBFContractError("P1R15 runtime request order differs")
    z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
    lookup_positions = cold_lookup_positions(
        tokenizer,
        requests,
        contexts,
        fact_token_strategy=hparams.fact_token,
    )
    stages.record(
        "model-context-entry",
        {
            "parameter_count": sum(item.numel() for item in model.parameters()),
            "parameter_dtype": "torch.bfloat16",
            "context_sha256": context_sha,
            "stage_a_r13_context_sha256": expected_context_sha,
            "context_exact_equal_to_stage_a_r13": True,
            "base_model_revision": artifact.base_model_revision,
            "target_layer_name_sha256": hashlib.sha256(
                target_layer_name.encode("utf-8")
            ).hexdigest(),
            "lookup_position_sha256": canonical_hash(list(lookup_positions)),
            "z_base_sha256": tensor_sha256(z_base),
            "joint_entry_target_sha256": tensor_sha256(z_base),
            "separate_target_only_bootstrap_count": 0,
            "native_or_direct_z_controller_access_count": 0,
        },
    )
    teacher_groups: list[torch.Tensor] = []
    entry_kl_groups: list[torch.Tensor] = []
    teacher_receipts: list[dict[str, Any]] = []
    for ordinal, group in enumerate(anchor_groups):
        teacher, receipt = evaluate_next_token_log_probs(model, tokenizer, group)
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
            "anchor_count": len(anchors),
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
        raise ODEBFContractError("P1R15 projector tensor differs")
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
    controller_lock = P1ControllerLock()
    mutation_lock = threading.RLock()
    rollout = run_p1r15_stage_a_virtual_rollout(
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
        z_base=z_base,
        lookup_positions=lookup_positions,
        target_layer_name=target_layer_name,
        residual_tolerance=controller_lock.residual_tolerance,
        write_stage=stages.record,
    )
    if any(tensor_sha256(touched[name]) != digest for name, digest in w0_sha.items()):
        raise ODEBFStateError("P1R15 virtual rollout changed W0")
    stages.record("rollout-terminal-prefix", rollout.raw_free_payload())
    terminal_freeze = rollout.action_freeze
    terminal_freeze_adapter_count = 0
    if rollout.accepted_transition_count == 0:
        terminal_freeze = IntegratedActionFreeze(
            rollout.action_freeze.request_order_sha256,
            rollout.action_freeze.selected_snapshot_sha256,
            0,
            1,
            "NO_PHYSICAL_W_ONLY_DIRECTION",
        )
        terminal_freeze_adapter_count = 1
    terminal = run_integrated_terminal_panel(
        model,
        tokenizer,
        model_alias=alias,
        requests=requests,
        dataset_path=Path(guard.base_guard.dataset),
        hparams=hparams,
        target_layer_name=target_layer_name,
        final_target=rollout.final_target,
        factors_by_weight=rollout.factors_by_weight,
        accepted_transition_count=rollout.accepted_transition_count,
        action_freeze=terminal_freeze,
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
        raise ODEBFStateError("P1R15 final W0 restore differs")
    guard.assert_unchanged()
    terminal_sha = stages.record(
        "terminal-panel",
        {
            **terminal.raw_free_payload(),
            "terminal_freeze_adapter_count": terminal_freeze_adapter_count,
            "authoritative_trajectory_status": rollout.terminal_status,
            "authoritative_trajectory_component": rollout.terminal_component,
        },
    )
    result = {
        "schema": f"{P1R15_SCHEMA}-stage-a-terminal",
        "status": "P1R15_STAGE_A_EXPERIMENT_TERMINAL",
        "model_alias": alias,
        "source_head": source_head,
        "method_id": P1R15_METHOD_ID,
        "stage_a_kind": P1R15_STAGE_A_KIND,
        "stage_a_seal_root": stage_a_seal["root_digest"],
        "stage_a_request_order_sha256": order,
        "stage_b_material_count": 0,
        "stage_b_access_count": 0,
        "accepted_transition_count": rollout.accepted_transition_count,
        "attempted_full_field_count": rollout.attempted_full_field_count,
        "target_group_count": rollout.target_group_count,
        "tau_final": rollout.final_tau,
        "trajectory_status": rollout.terminal_status,
        "trajectory_component": rollout.terminal_component,
        "action_freeze_sha256": rollout.action_freeze.identity(),
        "terminal_panel_sha256": terminal_sha,
        "final_w0_restore_exact": True,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "scientific_retry_count": 0,
        "hard_h_p_budget_veto_count": 0,
        "scientific_promotion_authorized": False,
        "wall_seconds": time.time() - started,
        "host_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
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
        "schema": f"{P1R15_SCHEMA}-stage-a-result-manifest",
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
        "trajectory_component": rollout.terminal_component,
        "final_w0_restore_exact": True,
        "scientific_promotion_authorized": False,
    }


__all__ = [
    "P1R15_STAGE_A_KIND",
    "P1R15_STAGE_A_R13_CHECKPOINT",
    "P1R15_STAGE_A_R13_REQUEST_ORDER",
    "P1R15_STAGE_A_R13_SEAL_ROOT",
    "P1R15StageARolloutResult",
    "P1R15OutputRootCollision",
    "P1R15StageRecorder",
    "run_p1r15_stage_a_virtual_rollout",
    "run_p1r15_stage_a_experiment",
]
