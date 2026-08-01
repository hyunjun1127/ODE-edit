"""Reusable outcome-free direction gate over four-hop edit trajectories."""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import ContractError, EditRequest, MemitFactorProposal, SnapshotManifest
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage
from .gpu_runtime import FixedModelRuntime
from .hooks import TemporaryLowRankApplication, assert_snapshot_current
from .mv0_fidelity import RollbackError, _silence_upstream, _weight_hashes
from .mv1_calibration import (
    ActionDirection,
    _state_identity,
    build_unit_c_actions,
    proposal_c_energy,
    scale_proposal,
)
from .quarter_step_refresh import (
    COMMON_STEP_1,
    LAYERS,
    QSTEP_K,
    BuiltPath,
    _assert_step_energy,
    _build_score_mix_action,
    _capture_source_snapshot,
    _combine_steps,
    _proposal_panel,
    _rebind_origin_actions,
    _rebind_origin_proposal,
    _set_rng,
)
from .trajectory import per_layer_c_cosines


GATED_PREFIX = "gated"
GATE_RELATIVE_MARGIN = 0.02

ProposalTransform = Callable[
    [MemitFactorProposal, str],
    tuple[MemitFactorProposal, Mapping[str, Any]],
]


def normalized_predicted_advantage(refreshed: float, fixed: float) -> float:
    values = (float(refreshed), float(fixed))
    if any(not math.isfinite(value) for value in values):
        raise ContractError("direction-gate predicted scores must be finite")
    denominator = max(abs(values[0]), abs(values[1]), 1e-12)
    return (values[0] - values[1]) / denominator


def choose_refreshed_direction(
    refreshed: float,
    fixed: float,
    *,
    relative_margin: float = GATE_RELATIVE_MARGIN,
) -> tuple[bool, float]:
    margin = float(relative_margin)
    if not math.isfinite(margin) or margin < 0.0 or margin >= 1.0:
        raise ContractError("direction-gate margin must be in [0,1)")
    advantage = normalized_predicted_advantage(refreshed, fixed)
    # Exact/near ties prefer the transported basis.  This makes refresh earn
    # its extra proposal/probe cost and suppresses noise-limited state changes.
    return advantage > margin, advantage


def adaptive_lineage_label(selection: str, step_index: int) -> str:
    """Map one gate choice onto the existing quarter-step vocabulary."""

    if selection not in {"refreshed", "fixed"} or step_index not in range(2, 5):
        raise ContractError("adaptive lineage choice/step is outside the lock")
    return f"{selection}_step_{step_index}"


def build_adaptive_gated_path(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    request: EditRequest,
    direct_z: Any,
    target_ids: torch.Tensor,
    origin_snapshot: SnapshotManifest,
    target_lineage: FrozenTargetLineage,
    b0: MemitFactorProposal,
    w0_unit: Sequence[ActionDirection],
    decision0: Any,
    common_step: MemitFactorProposal,
    hop_distance: float,
    hop_energy: float,
    probe_distance: float,
    weight_names: Sequence[str],
    layer_by_weight: Mapping[str, int],
    provenance_id: str,
    base_hashes: Mapping[str, str],
    base_state_identity: str,
    base_cpu_rng: torch.Tensor,
    base_cuda_rng: torch.Tensor,
    proposal_transform: ProposalTransform | None = None,
    relative_margin: float = GATE_RELATIVE_MARGIN,
) -> BuiltPath:
    """Choose refreshed vs transported direction at each current state.

    Both candidate bases are evaluated at the *same* gated trajectory state
    using the same D/64 central-probe contract.  Only the selected action is
    applied.  No operational endpoint outcome is available until all gate
    choices and lineage receipts have been committed by the caller.
    """

    assert_snapshot_current(runtime.model, origin_snapshot)
    _set_rng(base_cpu_rng, base_cuda_rng)
    proposals: list[MemitFactorProposal] = []
    descendants: list[SnapshotManifest] = []
    lineages: list[FrozenTargetLineage] = []
    panels: list[Mapping[str, Any]] = []
    layer_weights: list[Mapping[str, float]] = []
    c_cosines: list[Mapping[str, float]] = []
    step_energies: list[float] = []
    endpoint_energies: list[float] = []
    current_source = origin_snapshot
    current_lineage = target_lineage
    try:
        with ExitStack() as stack:
            for step_index in range(1, QSTEP_K + 1):
                if step_index == 1:
                    step_proposal = common_step
                    selection = "shared_w0"
                    selected_weights = dict(decision0.layer_weights)
                    panel_payload: Mapping[str, Any] = {
                        "shared_w0_panel": True,
                        "selection": selection,
                        "relative_margin": relative_margin,
                    }
                    cosine_payload: Mapping[str, float] = {
                        f"layer_{layer}": 1.0 for layer in LAYERS
                    }
                else:
                    with _silence_upstream():
                        refreshed = bridge.propose_synchronous_memit_factors(
                            runtime.model,
                            runtime.tokenizer,
                            (request,),
                            hparams,
                            contexts,
                            direct_z,
                            covariance_specs,
                            model_id=runtime.spec.snapshot_name,
                            frozen_target_lineage=current_lineage,
                        )
                    transform_payload: Mapping[str, Any] = {}
                    if proposal_transform is not None:
                        refreshed, transform_payload = proposal_transform(
                            refreshed,
                            f"gated-refreshed-step-{step_index}",
                        )
                    fixed = _rebind_origin_proposal(
                        b0,
                        descendant_snapshot=current_source,
                        lineage=current_lineage,
                        solver_suffix=f"gated-fixed-w{step_index - 1}",
                    )
                    refreshed_unit = build_unit_c_actions(
                        refreshed, covariance_moments, layer_by_weight
                    )
                    fixed_unit = _rebind_origin_actions(
                        w0_unit,
                        descendant_snapshot=current_source,
                        lineage=current_lineage,
                        solver_suffix=f"gated-fixed-w{step_index - 1}",
                    )
                    state_hashes = _weight_hashes(runtime.model, weight_names)
                    state_identity = _state_identity(runtime)
                    state_cpu_rng = torch.get_rng_state().clone()
                    state_cuda_rng = torch.cuda.get_rng_state(0).clone()
                    refreshed_panel = _proposal_panel(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        target_ids=target_ids,
                        actions=refreshed_unit,
                        epsilon=probe_distance,
                        base_hashes=state_hashes,
                        state_identity=state_identity,
                        cpu_rng=state_cpu_rng,
                        cuda_rng=state_cuda_rng,
                        panel_label=f"gated-refreshed-w{step_index - 1}",
                    )
                    fixed_panel = _proposal_panel(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        target_ids=target_ids,
                        actions=fixed_unit,
                        epsilon=probe_distance,
                        base_hashes=state_hashes,
                        state_identity=state_identity,
                        cpu_rng=state_cpu_rng,
                        cuda_rng=state_cuda_rng,
                        panel_label=f"gated-fixed-w{step_index - 1}",
                    )
                    refreshed_decision, refreshed_action = _build_score_mix_action(
                        synchronous=refreshed,
                        unit_actions=refreshed_unit,
                        scores=refreshed_panel.scores,
                        covariance_moments=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
                    fixed_decision, fixed_action = _build_score_mix_action(
                        synchronous=fixed,
                        unit_actions=fixed_unit,
                        scores=fixed_panel.scores,
                        covariance_moments=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
                    use_refreshed, advantage = choose_refreshed_direction(
                        refreshed_decision.predicted_score,
                        fixed_decision.predicted_score,
                        relative_margin=relative_margin,
                    )
                    selection = "refreshed" if use_refreshed else "fixed"
                    selected_action = refreshed_action if use_refreshed else fixed_action
                    selected_decision = (
                        refreshed_decision if use_refreshed else fixed_decision
                    )
                    step_proposal = scale_proposal(
                        selected_action.proposal,
                        hop_distance,
                        solver_suffix=f"adaptive-gate-{selection}-step-{step_index}",
                    )
                    selected_weights = dict(selected_decision.layer_weights)
                    panel_payload = {
                        "selection": selection,
                        "relative_margin": relative_margin,
                        "normalized_predicted_advantage": advantage,
                        "refreshed_predicted_score": (
                            refreshed_decision.predicted_score
                        ),
                        "fixed_predicted_score": fixed_decision.predicted_score,
                        "refreshed_panel": refreshed_panel.to_dict(),
                        "fixed_panel": fixed_panel.to_dict(),
                        "proposal_transform": dict(transform_payload),
                    }
                    cosine_payload = per_layer_c_cosines(
                        b0,
                        refreshed,
                        covariance_by_layer=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )

                observed_energy = _assert_step_energy(
                    step_proposal,
                    expected=hop_energy,
                    covariance_moments=covariance_moments,
                    layer_by_weight=layer_by_weight,
                )
                application = stack.enter_context(
                    TemporaryLowRankApplication(runtime.model, step_proposal)
                )
                child_source = _capture_source_snapshot(
                    runtime=runtime,
                    request=request,
                    contexts=contexts,
                    hparams=hparams,
                    weight_names=weight_names,
                    provenance_id=provenance_id,
                )
                current_lineage = current_lineage.derive_quarter_step(
                    parent_snapshot=current_source,
                    child_snapshot=child_source,
                    application=application,
                    label=(
                        COMMON_STEP_1
                        if step_index == 1
                        else adaptive_lineage_label(selection, step_index)
                    ),
                )
                current_lineage.assert_authorizes(
                    direct_z=direct_z,
                    snapshot=child_source,
                    target_token_ids=target_ids,
                )
                proposals.append(step_proposal)
                descendants.append(child_source)
                lineages.append(current_lineage)
                panels.append(panel_payload)
                layer_weights.append(selected_weights)
                c_cosines.append(cosine_payload)
                step_energies.append(observed_energy)
                endpoint = _combine_steps(
                    b0,
                    proposals,
                    solver_suffix=f"adaptive-gate-prefix-{step_index}",
                )
                endpoint_energies.append(
                    proposal_c_energy(endpoint, covariance_moments, layer_by_weight)
                )
                current_source = child_source
    finally:
        _set_rng(base_cpu_rng, base_cuda_rng)
        assert_snapshot_current(runtime.model, origin_snapshot)
        if (
            _weight_hashes(runtime.model, weight_names) != dict(base_hashes)
            or _state_identity(runtime) != base_state_identity
        ):
            raise RollbackError("adaptive gated path did not restore W0")
    return BuiltPath(
        policy=GATED_PREFIX,
        proposals=tuple(proposals),
        descendant_snapshots=tuple(descendants),
        lineages=tuple(lineages),
        panels=tuple(panels),
        layer_weights=tuple(layer_weights),
        c_cosines=tuple(c_cosines),
        step_energies=tuple(step_energies),
        endpoint_energies=tuple(endpoint_energies),
    )
