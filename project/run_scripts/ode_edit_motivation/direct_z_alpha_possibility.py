"""AlphaEdit-geometry follow-up for the direct-z possibility diagnostic.

This is an atomic mechanism probe, not a lifelong editor benchmark.  It
replays the exact MEMIT-v2 frozen direct-z artifacts, uses the precomputed
AlphaEdit projector read-only, and compares genuine P-inside-solve Alpha
geometry, a same-Alpha-base post-hoc BP ablation, and ODE/BF refresh at one
case-specific C budget inherited from the completed MEMIT run.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from .alphaedit_factors import compare_low_rank_proposals
from .alphaedit_proposal_adapter import AlphaEditProposalAdapter
from .alphaedit_reference import AlphaEditSolverConfig, load_alphaedit_solver_config
from .contracts import ContractError, EditRequest, MemitFactorProposal, canonical_json, sha256_bytes
from .direct_z_alpha_replay import (
    AlphaReplayArtifact,
    ModelReplayPreflight,
    load_alpha_replay_lock,
    preflight_model_replay,
)
from .direct_z_fidelity import (
    combine_unit_c_single_layer_proposals,
    solve_five_direction_ridge,
)
from .direct_z_possibility import (
    DIRECTZ_BOOTSTRAP_RESAMPLES,
    DIRECTZ_BOOTSTRAP_SEED,
    DIRECTZ_CASE_COUNT,
    DIRECTZ_PROBE_FRACTION,
    DIRECTZ_RUN_SEED,
    NO_OP,
    ORACLE_DO_Z,
    _capture_teacher,
    _evaluate_arm,
    _outcome_payload,
    _proposal_frobenius_norm,
    generate_directz_selection,
    load_heldout_evaluation_case,
)
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage, proposal_direction_hash
from .gpu_runtime import FixedModelRuntime, load_fixed_model, offline_environment, rng_state_hash, seed_runtime
from .hooks import TemporaryLowRankApplication, TensorHashRuntimeError, assert_tensor_sha256_device_parity
from .manifests import MODEL_SPECS, fixed_model_spec, load_counterfact_requests, preflight_fixed_artifacts
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    RollbackError,
    SanitizedJsonlWriter,
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
    _feature_hash,
    _layer_by_weight,
    _state_identity,
    assert_exact_action_contract,
    build_exact_teacher_batch,
    build_unit_c_actions,
    proposal_c_energy,
    scale_proposal,
)
from .projector_adapter import AlphaEditProjectorBank
from .quarter_step_refresh import (
    LAYERS,
    QSTEP_K,
    REFRESHED_PREFIX,
    _assert_step_energy,
    _build_policy_path,
    _build_score_mix_action,
    _combine_steps,
    _commit_action as _commit_qstep_action,
    _capture_source_snapshot,
    _proposal_panel,
)


ALPHA_DIRECTZ_JOB_NAME = "odeedit_dzf_alpha_pair_v1"
ALPHA_DIRECTZ_RUN_IDS = MappingProxyType(
    {
        "llama3-8b-inst": "dzf_alpha_llama_p0_v1",
        "qwen2.5-7b-inst": "dzf_alpha_qwen_p0_v1",
    }
)

ALPHA_GENUINE_FULL = "alpha_genuine_ordered_full"
ALPHA_GENUINE_MATCHED = "alpha_genuine_ordered_c_matched"
ALPHA_POSTHOC_FULL = "alpha_posthoc_bp_full"
ALPHA_POSTHOC_MATCHED = "alpha_posthoc_bp_c_matched"
ALPHA_BF_REFRESHED = "alpha_bf_genuine_refreshed_k4"
ALPHA_SYNC_CONE = "alpha_sync_z_cone_genuine"
ALPHA_DIRECTZ_BRANCH_ORDER = (
    NO_OP,
    ORACLE_DO_Z,
    ALPHA_GENUINE_FULL,
    ALPHA_GENUINE_MATCHED,
    ALPHA_POSTHOC_FULL,
    ALPHA_POSTHOC_MATCHED,
    ALPHA_BF_REFRESHED,
    ALPHA_SYNC_CONE,
)

ALPHA_MANIFEST_SCHEMA = "ode-edit-direct-z-alpha-paired-manifest/v1"
ALPHA_STREAM_SCHEMA = "ode-edit-direct-z-alpha-paired-stream/v1"
ALPHA_RECEIPT_SCHEMA = "ode-edit-direct-z-alpha-paired-receipt/v1"
ALPHA_EVENT_SCHEMA = "ode-edit-direct-z-alpha-paired-event/v1"
ALPHA_SUMMARY_SCHEMA = "ode-edit-direct-z-alpha-paired-summary/v1"


def _execution_envelope(model_alias: str, run_id: str) -> None:
    if (
        model_alias not in ALPHA_DIRECTZ_RUN_IDS
        or ALPHA_DIRECTZ_RUN_IDS[model_alias] != run_id
    ):
        raise MV1Error("Alpha direct-z model/run identity differs from lock")


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
        raise MV1Error("partial Alpha direct-z Slurm identity is forbidden")
    if (
        values["job_name"] != ALPHA_DIRECTZ_JOB_NAME
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("Alpha direct-z Slurm identity differs from lock")
    return {"under_slurm": True, **values}


def _finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ContractError(f"{name} must be finite and positive")
    return result


def _proposal_is_effectively_zero(proposal: MemitFactorProposal) -> bool:
    """Return whether every low-rank block represents an exact zero update."""

    factors = tuple(proposal.factors)
    if not factors:
        raise ContractError("zero-proposal check requires at least one factor")
    return all(
        int(torch.count_nonzero(factor.left).detach().cpu().item()) == 0
        or int(torch.count_nonzero(factor.right).detach().cpu().item()) == 0
        for factor in factors
    )


def _zero_safe_right_leak_ratio(
    adapter: AlphaEditProposalAdapter,
    proposal: MemitFactorProposal,
) -> float:
    """Assign the mathematically natural leak ratio zero to a zero update."""

    if _proposal_is_effectively_zero(proposal):
        return 0.0
    observed = float(adapter.proposal_right_leak(proposal).leak_ratio)
    if not math.isfinite(observed) or observed < 0.0:
        raise ContractError("right-projector leak ratio is invalid")
    return observed


def _bf_endpoint_receipt_binding(
    *,
    proposal_hash: str,
    endpoint_scale_from_raw: float,
    raw_path_c_energy: float,
    evaluation_endpoint_c_energy: float,
) -> dict[str, Any]:
    """Build the explicit BF endpoint record copied into the action receipt."""

    if (
        not isinstance(proposal_hash, str)
        or len(proposal_hash) != 64
        or any(character not in "0123456789abcdef" for character in proposal_hash)
    ):
        raise ContractError("BF evaluation endpoint proposal hash is invalid")
    scale = _finite_positive("BF endpoint scale", endpoint_scale_from_raw)
    raw_energy = _finite_positive("BF raw path C energy", raw_path_c_energy)
    endpoint_energy = _finite_positive(
        "BF evaluation endpoint C energy", evaluation_endpoint_c_energy
    )
    if not math.isclose(
        raw_energy * scale * scale,
        endpoint_energy,
        rel_tol=5e-5,
        abs_tol=1e-10,
    ):
        raise ContractError("BF endpoint receipt scale does not reproduce its C energy")
    return {
        "proposal_direction_hash": proposal_hash,
        "endpoint_scale_from_raw": scale,
        "raw_path_c_energy": raw_energy,
        "evaluation_endpoint_c_energy": endpoint_energy,
    }


def _aggregate_event_technical_provenance(
    results: Sequence[Mapping[str, Any]],
    *,
    planned_case_count: int,
    key: str,
) -> bool:
    """Aggregate one construction claim from completed event provenance only."""

    return bool(
        planned_case_count > 0
        and len(results) == planned_case_count
        and all(
            result.get("pass") is True
            and result.get("technical", {}).get(key) is True
            for result in results
        )
    )


def _assert_reference_energy(
    proposal: MemitFactorProposal,
    *,
    reference: float,
    covariance_moments: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> float:
    observed = proposal_c_energy(proposal, covariance_moments, layer_by_weight)
    if not math.isclose(observed, reference, rel_tol=5e-5, abs_tol=1e-10):
        raise ContractError("Alpha endpoint differs from the locked MEMIT C energy")
    return observed


def _match_reference_energy(
    proposal: MemitFactorProposal,
    *,
    source_energy: float,
    reference: float,
    suffix: str,
) -> MemitFactorProposal:
    source = _finite_positive("source C energy", source_energy)
    target = _finite_positive("reference C energy", reference)
    return scale_proposal(
        proposal,
        math.sqrt(target / source),
        solver_suffix=suffix,
    )


def _anchor_payload(
    *,
    record: AlphaReplayArtifact,
    replay: ModelReplayPreflight,
    origin_snapshot: Any,
    target_lineage: FrozenTargetLineage,
    alpha_config: AlphaEditSolverConfig,
) -> dict[str, Any]:
    payload = {
        "source_run_id": replay.model.source_run_id,
        "source_manifest_sha256": replay.model.source_manifest_identity.sha256,
        "source_summary_sha256": replay.model.source_summary_identity.sha256,
        "model_id": origin_snapshot.model_id,
        "context_id": origin_snapshot.context_id,
        "request_ids": list(origin_snapshot.request_ids),
        "memit_target_hparams_sha256": origin_snapshot.hparams_sha256,
        "origin_snapshot_id": origin_snapshot.snapshot_id,
        "origin_state_id": origin_snapshot.state_id,
        "origin_parameter_hashes": {
            item.name: item.sha256 for item in origin_snapshot.parameters
        },
        "direct_z_artifact_sha256": record.artifact_identity.sha256,
        "direct_z_artifact_size": record.artifact_identity.size,
        "direct_z_tensor_sha256": record.tensor_sha256,
        "target_token_sha256": record.target_token_sha256,
        "origin_lineage_id": record.origin_lineage_id,
        "alpha_solver_config_id": alpha_config.config_id,
        "alpha_solver_yaml_sha256": alpha_config.yaml_sha256,
        "cross_solver_frozen_target_anchor": True,
        "direct_z_compute_count": 0,
        "direct_z_load_count": 1,
    }
    if (
        target_lineage.lineage_id != record.origin_lineage_id
        or target_lineage.direct_z_tensor_sha256 != record.tensor_sha256
        or target_lineage.direct_z_artifact_sha256
        != record.artifact_identity.sha256
        or target_lineage.direct_z_artifact_size != record.artifact_identity.size
        or target_lineage.target_token_sha256 != record.target_token_sha256
    ):
        raise ContractError("cross-solver frozen-target anchor differs from replay lock")
    payload["target_anchor_id"] = _feature_hash(payload)
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
    replay: ModelReplayPreflight,
    alpha_config: AlphaEditSolverConfig,
    projector_bank: AlphaEditProjectorBank,
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
        raise ContractError("Alpha direct-z requires exact layers 4--8")
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
    record = replay.record_for_request(request.request_id)
    if record.case_id != request.case_id:
        raise ContractError("replay case/request join differs from lock")
    replay.assert_current()
    source_direct_z_root = replay.source_run_root / "direct_z"
    with _silence_upstream():
        direct_z = bridge.load_or_compute_direct_z(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            local_cache_root=source_direct_z_root,
            cache_path=f"{request.request_id}.pt",
            expected_identity=record.artifact_identity,
        )
    replay.assert_current()
    if direct_z.tensor_sha256 != record.tensor_sha256:
        raise ContractError("loaded direct-z tensor differs from replay lock")
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
    target_anchor = _anchor_payload(
        record=record,
        replay=replay,
        origin_snapshot=origin_snapshot,
        target_lineage=target_lineage,
        alpha_config=alpha_config,
    )
    canonical_target = direct_z.values[:, 0].detach().cpu().float()
    shared_delta = canonical_target - w0_teacher.subject_states[0]
    delta_l2 = _finite_positive(
        "direct-z delta norm",
        torch.linalg.vector_norm(shared_delta).item(),
    )

    adapter = AlphaEditProposalAdapter(
        bridge=bridge,
        config=alpha_config,
        projector_bank=projector_bank,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        request=request,
        memit_hparams=hparams,
        contexts=contexts,
        model_id=runtime.spec.snapshot_name,
        direct_z=direct_z,
        target_token_ids=target_ids,
        provenance_id=bindings.provenance.manifest_id,
    )
    with _silence_upstream():
        genuine_build = adapter.propose_ordered(
            origin_lineage=target_lineage,
            construction="genuine-p-inside-solve",
            solver_suffix="alpha-genuine-ordered-full",
        )
        posthoc_build = adapter.propose_ordered(
            origin_lineage=target_lineage,
            construction="posthoc-unprojected-alpha-base-at-p",
            solver_suffix="alpha-posthoc-ordered-full",
        )
        synchronous_build = adapter.propose_synchronous(
            frozen_target_lineage=target_lineage,
            construction="genuine-p-inside-solve",
            solver_suffix="alpha-genuine-w0",
        )
    genuine_full = genuine_build.proposal
    posthoc_full = posthoc_build.proposal
    b0 = synchronous_build.proposal
    genuine_full.assert_same_entry_snapshot(posthoc_full)
    genuine_full.assert_same_entry_snapshot(b0)

    reference_energy = _finite_positive(
        "locked MEMIT reference C energy",
        record.reference_c_energy,
    )
    genuine_natural_energy = proposal_c_energy(
        genuine_full, covariance_moments, layer_by_weight
    )
    posthoc_natural_energy = proposal_c_energy(
        posthoc_full, covariance_moments, layer_by_weight
    )
    genuine_matched = _match_reference_energy(
        genuine_full,
        source_energy=genuine_natural_energy,
        reference=reference_energy,
        suffix="alpha-genuine-c-matched-to-memit-bf-v2",
    )
    posthoc_matched = _match_reference_energy(
        posthoc_full,
        source_energy=posthoc_natural_energy,
        reference=reference_energy,
        suffix="alpha-posthoc-c-matched-to-memit-bf-v2",
    )
    genuine_matched_energy = _assert_reference_energy(
        genuine_matched,
        reference=reference_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    posthoc_matched_energy = _assert_reference_energy(
        posthoc_matched,
        reference=reference_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )

    hop_distance = math.sqrt(reference_energy) / float(QSTEP_K)
    hop_energy = reference_energy / float(QSTEP_K * QSTEP_K)
    probe_distance = math.sqrt(reference_energy) * DIRECTZ_PROBE_FRACTION
    w0_unit = build_unit_c_actions(b0, covariance_moments, layer_by_weight)
    assert_exact_action_contract(
        b0,
        genuine_full,
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
        panel_label="direct-z-alpha-genuine-w0",
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
        solver_suffix="direct-z-alpha-bf-common-step-1",
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
        bridge=adapter,
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
        raise ContractError("Alpha BF refreshed path did not produce K=4")
    bf_raw = _combine_steps(
        b0,
        refreshed_path.proposals,
        solver_suffix="direct-z-alpha-bf-raw-k4",
    )
    bf_raw_energy = proposal_c_energy(bf_raw, covariance_moments, layer_by_weight)
    bf_endpoint_scale = math.sqrt(
        reference_energy / _finite_positive("BF raw path C energy", bf_raw_energy)
    )
    bf_endpoint = _match_reference_energy(
        bf_raw,
        source_energy=bf_raw_energy,
        reference=reference_energy,
        suffix="direct-z-alpha-bf-endpoint-c-matched-to-memit-bf-v2",
    )
    bf_energy = _assert_reference_energy(
        bf_endpoint,
        reference=reference_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )

    responses: list[torch.Tensor] = []
    for action in w0_unit[: len(LAYERS)]:
        plus = scale_proposal(
            action.proposal,
            probe_distance,
            solver_suffix=f"direct-z-alpha-cone-plus-{action.action_id}",
        )
        minus = scale_proposal(
            action.proposal,
            -probe_distance,
            solver_suffix=f"direct-z-alpha-cone-minus-{action.action_id}",
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
    zcone_solution = solve_five_direction_ridge(
        torch.stack(responses, dim=1),
        shared_delta,
        radius=math.sqrt(reference_energy),
        ridge=1e-6,
        max_iterations=5000,
        tolerance=1e-8,
    )
    zero_cone_coefficients = all(
        float(coefficient) == 0.0 for coefficient in zcone_solution.coefficients
    )
    if not zcone_solution.converged and not zero_cone_coefficients:
        raise ContractError("Alpha z-cone projected ridge solver did not converge")
    zcone = combine_unit_c_single_layer_proposals(
        b0,
        tuple(action.proposal for action in w0_unit[: len(LAYERS)]),
        zcone_solution.coefficients,
        solver_suffix="direct-z-alpha-genuine-sync-cone",
    )
    zcone_energy = float(
        sum(coefficient * coefficient for coefficient in zcone_solution.coefficients)
    )
    if not math.isfinite(zcone_energy) or zcone_energy < 0.0:
        raise ContractError("Alpha z-cone energy is invalid")
    if zcone_energy > reference_energy * (1.0 + 5e-5) + 1e-10:
        raise ContractError("Alpha z-cone exceeds the locked reference C cap")

    genuine_leak = adapter.proposal_right_leak(genuine_full)
    posthoc_leak = adapter.proposal_right_leak(posthoc_full)
    bf_leak = adapter.proposal_right_leak(bf_endpoint)
    zcone_zero_proposal = _proposal_is_effectively_zero(zcone)
    zcone_leak_ratio = _zero_safe_right_leak_ratio(adapter, zcone)
    geometry_comparison = compare_low_rank_proposals(genuine_full, posthoc_full)
    all_builds = adapter.builds
    max_solve_residual = max(build.max_solve_residual for build in all_builds)
    genuine_ordered_build_count = sum(
        build.construction == "genuine-p-inside-solve"
        and build.proposal.residual_denominator is None
        for build in all_builds
    )
    genuine_synchronous_build_count = sum(
        build.construction == "genuine-p-inside-solve"
        and build.proposal.residual_denominator == len(LAYERS)
        for build in all_builds
    )
    posthoc_ordered_build_count = sum(
        build.construction == "posthoc-unprojected-alpha-base-at-p"
        and build.proposal.residual_denominator is None
        for build in all_builds
    )
    exact_build_count = len(all_builds) == QSTEP_K + 2
    genuine_alpha_ordered_solve_once = bool(
        exact_build_count
        and genuine_ordered_build_count == 1
        and sum(build is genuine_build for build in all_builds) == 1
        and genuine_build.construction == "genuine-p-inside-solve"
        and genuine_build.proposal.residual_denominator is None
    )
    genuine_alpha_bf_all_hops_genuine = bool(
        exact_build_count
        and len(refreshed_path.proposals) == QSTEP_K
        and proposal_direction_hash(refreshed_path.proposals[0])
        == proposal_direction_hash(common_step)
        and genuine_synchronous_build_count == QSTEP_K
        and sum(build is synchronous_build for build in all_builds) == 1
        and synchronous_build.construction == "genuine-p-inside-solve"
    )
    posthoc_arms_projection_only = bool(
        exact_build_count
        and posthoc_ordered_build_count == 1
        and sum(build is posthoc_build for build in all_builds) == 1
        and posthoc_build.construction == "posthoc-unprojected-alpha-base-at-p"
        and posthoc_build.proposal is posthoc_full
    )
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("Alpha direct-z construction did not restore W0/RNG")
    replay.assert_current()
    projector_bank.assert_stat_current()

    endpoint_energies = {
        NO_OP: 0.0,
        ORACLE_DO_Z: 0.0,
        ALPHA_GENUINE_FULL: genuine_natural_energy,
        ALPHA_GENUINE_MATCHED: genuine_matched_energy,
        ALPHA_POSTHOC_FULL: posthoc_natural_energy,
        ALPHA_POSTHOC_MATCHED: posthoc_matched_energy,
        ALPHA_BF_REFRESHED: bf_energy,
        ALPHA_SYNC_CONE: zcone_energy,
    }
    proposals = {
        ALPHA_GENUINE_FULL: genuine_full,
        ALPHA_GENUINE_MATCHED: genuine_matched,
        ALPHA_POSTHOC_FULL: posthoc_full,
        ALPHA_POSTHOC_MATCHED: posthoc_matched,
        ALPHA_BF_REFRESHED: bf_endpoint,
        ALPHA_SYNC_CONE: zcone,
    }
    leak_ratios = {
        NO_OP: 0.0,
        ORACLE_DO_Z: 0.0,
        ALPHA_GENUINE_FULL: genuine_leak.leak_ratio,
        ALPHA_GENUINE_MATCHED: genuine_leak.leak_ratio,
        ALPHA_POSTHOC_FULL: posthoc_leak.leak_ratio,
        ALPHA_POSTHOC_MATCHED: posthoc_leak.leak_ratio,
        ALPHA_BF_REFRESHED: bf_leak.leak_ratio,
        ALPHA_SYNC_CONE: zcone_leak_ratio,
    }
    constructions = {
        NO_OP: "no_weight_write",
        ORACLE_DO_Z: "weight_free_direct_z_ceiling",
        ALPHA_GENUINE_FULL: "genuine_alphaedit_isolated_ordered_solve",
        ALPHA_GENUINE_MATCHED: "genuine_alphaedit_isolated_ordered_solve_c_matched",
        ALPHA_POSTHOC_FULL: "posthoc_unprojected_alpha_base_right_projection",
        ALPHA_POSTHOC_MATCHED: "posthoc_unprojected_alpha_base_right_projection_c_matched",
        ALPHA_BF_REFRESHED: "genuine_alphaedit_refreshed_k4_endpoint_c_matched",
        ALPHA_SYNC_CONE: "genuine_alphaedit_sync_cone_diagnostic",
    }
    feature: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "target_anchor": target_anchor,
        "target_identity": {
            "origin_lineage_id": target_lineage.lineage_id,
            "target_anchor_id": target_anchor["target_anchor_id"],
            "direct_z_tensor_sha256": target_lineage.direct_z_tensor_sha256,
            "target_token_sha256": target_lineage.target_token_sha256,
        },
        "reference_c_energy": reference_energy,
        "genuine_natural_c_energy": genuine_natural_energy,
        "posthoc_natural_c_energy": posthoc_natural_energy,
        "bf_raw_path_c_energy": bf_raw_energy,
        "bf_endpoint_c_energy": bf_energy,
        "bf_endpoint_scale_from_raw": bf_endpoint_scale,
        "hop_c_energy": hop_energy,
        "probe_c_distance": probe_distance,
        "direct_z_delta_l2": delta_l2,
        "genuine_vs_posthoc": geometry_comparison.to_dict(),
        "endpoint_right_projector_violation": {
            "genuine": genuine_leak.leak_ratio,
            "posthoc": posthoc_leak.leak_ratio,
            "bf": bf_leak.leak_ratio,
            "cone": zcone_leak_ratio,
        },
        "zcone_zero_proposal": zcone_zero_proposal,
        "max_alpha_solve_residual": max_solve_residual,
        "alpha_build_count": len(all_builds),
        "genuine_alpha_ordered_build_count": genuine_ordered_build_count,
        "genuine_alpha_synchronous_build_count": genuine_synchronous_build_count,
        "posthoc_alpha_ordered_build_count": posthoc_ordered_build_count,
        "bf_layer_weights": list(refreshed_path.layer_weights),
        "zcone_solver": zcone_solution.to_dict(),
        "cache_policy": "isolated-zero-per-case-no-accumulation",
        "controller_policy": (
            "K=4 numerical refresh inside one atomic edit; genuine Alpha directions; "
            "endpoint direction normalized to the paired MEMIT-v2 BF C energy"
        ),
    }
    feature["feature_hash"] = _feature_hash(feature)
    action: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(ALPHA_DIRECTZ_BRANCH_ORDER),
        "path_action_hashes": {
            NO_OP: [],
            ORACLE_DO_Z: [target_lineage.direct_z_tensor_sha256],
            ALPHA_GENUINE_FULL: [proposal_direction_hash(genuine_full)],
            ALPHA_GENUINE_MATCHED: [proposal_direction_hash(genuine_matched)],
            ALPHA_POSTHOC_FULL: [proposal_direction_hash(posthoc_full)],
            ALPHA_POSTHOC_MATCHED: [proposal_direction_hash(posthoc_matched)],
            ALPHA_BF_REFRESHED: [
                proposal_direction_hash(item) for item in refreshed_path.proposals
            ],
            ALPHA_SYNC_CONE: [proposal_direction_hash(zcone)],
            "bf_evaluation_endpoint_binding": _bf_endpoint_receipt_binding(
                proposal_hash=proposal_direction_hash(bf_endpoint),
                endpoint_scale_from_raw=bf_endpoint_scale,
                raw_path_c_energy=bf_raw_energy,
                evaluation_endpoint_c_energy=bf_energy,
            ),
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
        expected_branch_order=ALPHA_DIRECTZ_BRANCH_ORDER,
        feature_event="direct_z_alpha_feature",
        action_event="direct_z_alpha_action_commitment",
        receipt_schema=ALPHA_RECEIPT_SCHEMA,
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
        raise RollbackError("Alpha post-receipt W0 differs from precommit W0")
    observations = {NO_OP: baseline}
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
    for arm_id in ALPHA_DIRECTZ_BRANCH_ORDER[2:]:
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
        )
    if tuple(observations) != ALPHA_DIRECTZ_BRANCH_ORDER:
        raise ContractError("Alpha direct-z branch order differs from lock")
    nfe = {
        NO_OP: 1,
        ORACLE_DO_Z: 1,
        ALPHA_GENUINE_FULL: 5,
        ALPHA_GENUINE_MATCHED: 5,
        ALPHA_POSTHOC_FULL: 5,
        ALPHA_POSTHOC_MATCHED: 5,
        ALPHA_BF_REFRESHED: 24,
        ALPHA_SYNC_CONE: 15,
    }
    genuine_arms = {
        ALPHA_GENUINE_FULL,
        ALPHA_GENUINE_MATCHED,
        ALPHA_BF_REFRESHED,
        ALPHA_SYNC_CONE,
    }
    posthoc_arms = {ALPHA_POSTHOC_FULL, ALPHA_POSTHOC_MATCHED}
    for arm_id in ALPHA_DIRECTZ_BRANCH_ORDER:
        proposal = proposals.get(arm_id)
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
            endpoint_frobenius_norm=_proposal_frobenius_norm(proposal),
            nfe=nfe[arm_id],
        )
        payload.update(
            {
                "arm_construction": constructions[arm_id],
                "target_anchor_id": target_anchor["target_anchor_id"],
                "frozen_target_replay_exact": True,
                "projector_precomputed_only": True,
                "projector_integrity_exact": True,
                "genuine_alpha_solver_used": arm_id in genuine_arms,
                "posthoc_b_at_p_only": arm_id in posthoc_arms,
                "reference_c_energy": reference_energy,
                "endpoint_right_projector_violation_ratio": leak_ratios[arm_id],
            }
        )
        _safe_payload(payload)
        outcome_writer.write("direct_z_alpha_outcome", payload)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("Alpha direct-z event did not restore W0/RNG")
    replay.assert_current()
    projector_bank.assert_stat_current()
    return {
        "schema_version": ALPHA_EVENT_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "direct_z_compute_count": 0,
        "direct_z_load_count": 1,
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(ALPHA_DIRECTZ_BRANCH_ORDER),
        "receipt": {"name": receipt_name, "sha256": receipt_hash},
        "evaluation_payload_sha256": heldout.payload_hash,
        "alpha_build_count": len(all_builds),
        "genuine_alpha_ordered_build_count": genuine_ordered_build_count,
        "genuine_alpha_synchronous_build_count": genuine_synchronous_build_count,
        "posthoc_alpha_ordered_build_count": posthoc_ordered_build_count,
        "bf_hop_count": len(refreshed_path.proposals),
        "zcone_zero_proposal": zcone_zero_proposal,
        "technical": {
            "frozen_target_replay_exact": True,
            "cross_solver_target_anchor_exact": True,
            "precomputed_covariance_only": True,
            "projector_precomputed_only": True,
            "projector_integrity_exact": True,
            "isolated_cache_c_zero": True,
            "receipt_before_outcome": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "genuine_alpha_ordered_solve_once_per_case": (
                genuine_alpha_ordered_solve_once
            ),
            "genuine_alpha_bf_all_hops_genuine": (
                genuine_alpha_bf_all_hops_genuine
            ),
            "posthoc_arms_projection_only": posthoc_arms_projection_only,
        },
        "wall_seconds": time.perf_counter() - started,
        "pass": True,
    }


def _validate_event(raw: Mapping[str, Any], request: EditRequest) -> dict[str, Any]:
    result = dict(raw)
    if (
        result.get("schema_version") != ALPHA_EVENT_SCHEMA
        or result.get("case_id") != request.case_id
        or result.get("request_id") != request.request_id
        or result.get("direct_z_compute_count") != 0
        or result.get("direct_z_load_count") != 1
        or result.get("feature_count") != 1
        or result.get("commitment_count") != 1
        or result.get("outcome_count") != len(ALPHA_DIRECTZ_BRANCH_ORDER)
        or not result.get("pass")
        or not all(result.get("technical", {}).values())
    ):
        raise ContractError("Alpha direct-z event failed its technical gate")
    return result


def run_direct_z_alpha_possibility(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    _execution_envelope(model_alias, run_id)
    _slurm_state(model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    _git_runtime_state()
    fixed_provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_directz_selection(root)
    replay_lock = load_alpha_replay_lock()
    if (
        replay_lock.selection_sha256 != selection["manifest_id"]
        or replay_lock.rank_slice
        != (int(selection["rank_start"]), int(selection["rank_stop"]))
        or replay_lock.case_order != tuple(selection["case_ids"])
    ):
        raise ContractError("Alpha replay lock differs from direct-z v2 selection")
    requests = load_counterfact_requests(root, selection["case_ids"])
    replay = preflight_model_replay(
        output,
        model_alias,
        selection["case_ids"],
        replay_lock=replay_lock,
    )
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not {record.path for record in bridge_provenance.files}.issubset(
        {record.path for record in fixed_provenance.files}
    ):
        raise ContractError("Alpha direct-z bridge provenance is outside fixed manifest")
    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        alpha_config, alpha_reference = load_alphaedit_solver_config(
            root,
            model_alias,
            memit_hparams=hparams,
        )
        seed_runtime(DIRECTZ_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("Alpha direct-z model loader returned another fixed model")
        contexts = _freeze_contexts(bridge, runtime, seed=DIRECTZ_RUN_SEED)
        projector_bank = AlphaEditProjectorBank.open(root, spec)
        destination = _local_run_directory(output, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        outcomes_path = destination / "outcomes.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        receipt_root = destination / "action_receipts"
        receipt_root.mkdir(mode=0o700)
        config_payload = {
            "layers": list(LAYERS),
            "case_count": DIRECTZ_CASE_COUNT,
            "rank_slice": list(replay_lock.rank_slice),
            "arm_order": list(ALPHA_DIRECTZ_BRANCH_ORDER),
            "k": QSTEP_K,
            "probe_fraction": DIRECTZ_PROBE_FRACTION,
            "run_seed": DIRECTZ_RUN_SEED,
            "budget_reference": "paired-memit-v2-bf-endpoint-c-energy",
            "cache_policy": "isolated-zero-per-case-no-accumulation",
        }
        config_sha256 = sha256_bytes(canonical_json(config_payload).encode("utf-8"))
        manifest = {
            "schema_version": ALPHA_MANIFEST_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "case_count": DIRECTZ_CASE_COUNT,
            "arm_order": list(ALPHA_DIRECTZ_BRANCH_ORDER),
            "selection_sha256": selection["manifest_id"],
            "config_sha256": config_sha256,
            "bootstrap_seed": DIRECTZ_BOOTSTRAP_SEED,
            "bootstrap_resamples": DIRECTZ_BOOTSTRAP_RESAMPLES,
            "paired_memit_run_id": replay.model.source_run_id,
            "paired_memit_manifest_sha256": replay.model.source_manifest_identity.sha256,
            "replay_lock_id": replay_lock.lock_id,
            "frozen_target_mode": "replay_memit_v2_direct_z_no_recompute",
            "alpha_solver": alpha_config.to_dict(),
            "alpha_reference_manifest_id": alpha_reference.manifest_id,
            "projector": projector_bank.metadata(),
            "claim_boundary": (
                "atomic_geometry_signal_only_not_lifelong_or_method_superiority"
            ),
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
            SanitizedJsonlWriter(features_path, run_id, schema_version=ALPHA_STREAM_SCHEMA) as feature_writer,
            SanitizedJsonlWriter(actions_path, run_id, schema_version=ALPHA_STREAM_SCHEMA) as action_writer,
            SanitizedJsonlWriter(outcomes_path, run_id, schema_version=ALPHA_STREAM_SCHEMA) as outcome_writer,
            SanitizedJsonlWriter(events_path, run_id, schema_version=ALPHA_STREAM_SCHEMA) as event_writer,
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
                        replay=replay,
                        alpha_config=alpha_config,
                        projector_bank=projector_bank,
                        receipt_root=receipt_root,
                        request=request,
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        outcome_writer=outcome_writer,
                    )
                    result = _validate_event(raw, request)
                except Exception as exc:
                    print(
                        f"ALPHA_DIRECTZ_LOCAL_TRACEBACK_BEGIN type={type(exc).__name__}",
                        file=sys.stderr,
                    )
                    for frame, line_number in traceback.walk_tb(exc.__traceback__):
                        print(
                            f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                            file=sys.stderr,
                        )
                    if isinstance(exc, TensorHashRuntimeError):
                        print(
                            "ALPHA_DIRECTZ_LOCAL_TENSOR_HASH_DIAGNOSTIC "
                            f"phase={exc.phase} category={exc.category}",
                            file=sys.stderr,
                        )
                    print("ALPHA_DIRECTZ_LOCAL_TRACEBACK_END", file=sys.stderr)
                    result = {
                        "schema_version": ALPHA_EVENT_SCHEMA,
                        "case_id": request.case_id,
                        "request_id": request.request_id,
                        "failure_type": type(exc).__name__,
                        "pass": False,
                    }
                    abort_failure_type = type(exc).__name__
                results.append(result)
                event_writer.write("direct_z_alpha_case", result)
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
                "outcomes": planned * len(ALPHA_DIRECTZ_BRANCH_ORDER),
                "events": planned,
            }
            and receipt_count == planned
        )
        all_rollbacks_exact = bool(
            abort_failure_type is None
            and all(result.get("technical", {}).get("rollback_exact") for result in results)
        )
        firewall_pass = bool(
            abort_failure_type is None
            and all(result.get("technical", {}).get("firewall_pass") for result in results)
        )
        receipt_before_outcome = bool(
            abort_failure_type is None
            and all(
                result.get("technical", {}).get("receipt_before_outcome")
                for result in results
            )
        )
        frozen_target_replay_exact = bool(
            abort_failure_type is None
            and all(
                result.get("technical", {}).get("frozen_target_replay_exact")
                for result in results
            )
        )
        direct_z_recompute_count_total = sum(
            int(result.get("direct_z_compute_count", 0)) for result in results
        )
        frozen_target_load_once_per_case = bool(
            abort_failure_type is None
            and all(result.get("direct_z_load_count") == 1 for result in results)
        )
        replay.assert_current()
        projector_bank.assert_hash_current()
        alpha_reference.assert_current()
        summary = {
            "schema_version": ALPHA_SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "planned_case_count": planned,
            "attempted_case_count": attempted,
            "pass_case_count": pass_count,
            "failed_case_count": planned - pass_count,
            "outcome_count": sequences["outcomes"],
            "arm_order": list(ALPHA_DIRECTZ_BRANCH_ORDER),
            "selection_sha256": selection["manifest_id"],
            "config_sha256": config_sha256,
            "paired_memit_run_id": replay.model.source_run_id,
            "paired_memit_manifest_sha256": replay.model.source_manifest_identity.sha256,
            "replay_lock_id": replay_lock.lock_id,
            "all_rollbacks_exact": all_rollbacks_exact,
            "firewall_pass": firewall_pass,
            "receipt_before_outcome": receipt_before_outcome,
            "frozen_target_replay_exact": frozen_target_replay_exact,
            "frozen_target_load_once_per_case": frozen_target_load_once_per_case,
            "direct_z_recompute_count_total": direct_z_recompute_count_total,
            "projector_precomputed_only": True,
            "projector_integrity_exact": True,
            "genuine_alpha_ordered_solve_once_per_case": (
                _aggregate_event_technical_provenance(
                    results,
                    planned_case_count=planned,
                    key="genuine_alpha_ordered_solve_once_per_case",
                )
            ),
            "genuine_alpha_bf_all_hops_genuine": (
                _aggregate_event_technical_provenance(
                    results,
                    planned_case_count=planned,
                    key="genuine_alpha_bf_all_hops_genuine",
                )
            ),
            "posthoc_arms_projection_only": (
                _aggregate_event_technical_provenance(
                    results,
                    planned_case_count=planned,
                    key="posthoc_arms_projection_only",
                )
            ),
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {
            **summary,
            "all_pass": bool(pass_count == planned and exact_counts),
            "output_directory": str(destination),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-direct-z-alpha-possibility",
        description="Replay direct-z under genuine and post-hoc AlphaEdit geometry.",
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
        summary = run_direct_z_alpha_possibility(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"Alpha direct-z possibility failed: {type(exc).__name__}", file=sys.stderr)
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
