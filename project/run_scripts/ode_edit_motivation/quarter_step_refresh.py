"""Preregistered K=4 native-distance Motivation diagnostic.

This is an independent follow-up to MV-2, not a mutation or reinterpretation
of its q=1/256 result.  For each fresh case it freezes one W0 direct-z target,
defines native C-distance ``D`` from the ordered MEMIT update, and constructs
four-hop trajectories with an exact per-hop distance ``D/4``.  The local
controller probe remains fixed at ``D/64`` so treatment size and sensor radius
do not change together.

The three operational paths are:

* refreshed proposal direction plus refreshed coefficients at every state;
* fixed W0 proposal basis with refreshed coefficients at every state;
* fixed W0 proposal basis and fixed W0 coefficients.

All path proposals, hashes, C budgets, and lineage receipts are durably
committed before checkpoint outcomes are evaluated.  A one-shot native ordered
update and its naive four-way fixed split form the application-noise control.
Heavy tensors and direct-z artifacts remain under the Git-ignored local tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
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
    TemporaryLowRankApplication,
    TensorHashRuntimeError,
    assert_snapshot_current,
    assert_tensor_sha256_device_parity,
    capture_snapshot,
)
from .manifests import (
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    CounterFactSelectionManifest,
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
from .mv1_calibration import (
    ACTION_UNIFORM,
    ActionDirection,
    MV1Error,
    _evaluate_branch,
    _feature_hash,
    _layer_by_weight,
    _metric_difference,
    _state_identity,
    assert_exact_action_contract,
    build_unit_c_actions,
    proposal_c_energy,
    rewrite_metrics,
    scale_proposal,
    teacher_forced_rewrite_exact,
)
from .mv1_score_mix import (
    EXPECTED_LAYER_COUNT,
    _assert_outcome_free,
    build_score_mix_direction,
    derive_score_mix_decision,
)
from .trajectory import (
    MATCHED_C_ABS_TOL,
    MATCHED_C_REL_TOL,
    build_central_probe_panel,
    evaluate_temporary_trajectory,
    per_layer_c_cosines,
)


QSTEP_JOB_NAME = "odeedit_qstep_pair_v1"
QSTEP_RUN_IDS = MappingProxyType(
    {
        "llama3-8b-inst": "qstep4_llama_f0_v1",
        "qwen2.5-7b-inst": "qstep4_qwen_f0_v1",
    }
)
QSTEP_RUN_SEED = 17
QSTEP_K = 4
QSTEP_HOP_FRACTION = 0.25
QSTEP_TOTAL_Q = 1.0
QSTEP_PROBE_FRACTION = 1.0 / 64.0
QSTEP_CASE_COUNT = 12
QSTEP_RANK_START = 112
QSTEP_RANK_STOP = 124
QSTEP_BOOTSTRAP_SEED = 20260802
QSTEP_BOOTSTRAP_RESAMPLES = 4000
QSTEP_PRACTICAL_EFFECT_FLOOR = 1e-4
LAYERS = (4, 5, 6, 7, 8)
LAYER_ACTIONS = tuple(f"layer_{layer}" for layer in LAYERS)
PROBE_ACTIONS = (*LAYER_ACTIONS, ACTION_UNIFORM)

NO_OP_REPLAY = "no_op_replay"
COMMON_STEP_1 = "common_step_1"
REFRESHED_PREFIX = "refreshed"
COEFFICIENT_PREFIX = "coefficient"
FIXED_PREFIX = "fixed"
NATIVE_ORDERED_FULL = "native_ordered_full"
NATIVE_ORDERED_SPLIT4 = "native_ordered_split4"


def _checkpoint_arm(prefix: str, step: int) -> str:
    return f"{prefix}_step_{step}"


QSTEP_BRANCH_ORDER = (
    NO_OP_REPLAY,
    COMMON_STEP_1,
    *(
        arm
        for step in range(2, QSTEP_K + 1)
        for arm in (
            _checkpoint_arm(REFRESHED_PREFIX, step),
            _checkpoint_arm(COEFFICIENT_PREFIX, step),
            _checkpoint_arm(FIXED_PREFIX, step),
        )
    ),
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
)

QSTEP_SELECTION_SCHEMA = "ode-edit-quarter-step-selection/v1"
QSTEP_MANIFEST_SCHEMA = "ode-edit-quarter-step-manifest/v1"
QSTEP_STREAM_SCHEMA = "ode-edit-quarter-step/v1"
QSTEP_RECEIPT_SCHEMA = "ode-edit-quarter-step-receipt/v1"
QSTEP_SUMMARY_SCHEMA = "ode-edit-quarter-step-summary/v1"
QSTEP_EVENT_SCHEMA = "ode-edit-quarter-step-event/v1"


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


def _full_hash(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class QuarterStepSelectionManifest:
    seed: str
    source_sha256: str
    source_size: int
    source_row_count: int
    canonical_first100_order_hash: str
    prior_mv2_order_hash: str
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.seed != DEFAULT_SELECTION_SEED:
            raise ContractError("quarter-step selection requires canonical seed")
        _full_hash("source hash", self.source_sha256)
        _full_hash("first100 order hash", self.canonical_first100_order_hash)
        _full_hash("prior MV-2 order hash", self.prior_mv2_order_hash)
        if (
            isinstance(self.source_size, bool)
            or not isinstance(self.source_size, int)
            or self.source_size <= 0
            or isinstance(self.source_row_count, bool)
            or not isinstance(self.source_row_count, int)
            or self.source_row_count < QSTEP_RANK_STOP
            or len(self.case_ids) != QSTEP_CASE_COUNT
            or len(set(self.case_ids)) != QSTEP_CASE_COUNT
        ):
            raise ContractError("quarter-step selection metadata is invalid")

    @property
    def order_hash(self) -> str:
        return sha256_bytes(canonical_json(list(self.case_ids)).encode("utf-8"))

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(
            canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": QSTEP_SELECTION_SCHEMA,
            "seed": self.seed,
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "rank_slice": {
                "start": QSTEP_RANK_START,
                "stop": QSTEP_RANK_STOP,
                "count": QSTEP_CASE_COUNT,
            },
            "canonical_first100_order_hash": self.canonical_first100_order_hash,
            "prior_mv2_order_hash": self.prior_mv2_order_hash,
            "case_ids": list(self.case_ids),
            "order_hash": self.order_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def _salted_rank(case_id: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            DEFAULT_SELECTION_SEED.encode("utf-8")
            + b"\0"
            + case_id.encode("utf-8")
        ).digest(),
        case_id,
    )


def select_quarter_step_cases(
    all_case_ids: Sequence[str],
    *,
    canonical: CounterFactSelectionManifest,
) -> QuarterStepSelectionManifest:
    if (
        canonical.seed != DEFAULT_SELECTION_SEED
        or len(canonical.ordered_case_ids) != 100
    ):
        raise ContractError("quarter-step selection requires canonical first100")
    normalized = tuple(str(case_id) for case_id in all_case_ids)
    if (
        len(normalized) != canonical.source_row_count
        or len(set(normalized)) != len(normalized)
    ):
        raise ContractError("quarter-step case-ID source is incomplete or duplicated")
    ranked = tuple(sorted(normalized, key=_salted_rank))
    first100 = ranked[:100]
    prior_mv2 = ranked[100:112]
    selected = ranked[QSTEP_RANK_START:QSTEP_RANK_STOP]
    if first100 != canonical.ordered_case_ids:
        raise ContractError("canonical first100 differs from salted rank lock")
    if set(selected).intersection((*first100, *prior_mv2)):
        raise ContractError("quarter-step fresh cases overlap prior evidence")
    return QuarterStepSelectionManifest(
        seed=DEFAULT_SELECTION_SEED,
        source_sha256=canonical.source_sha256,
        source_size=canonical.source_size,
        source_row_count=canonical.source_row_count,
        canonical_first100_order_hash=canonical.order_hash,
        prior_mv2_order_hash=sha256_bytes(
            canonical_json(list(prior_mv2)).encode("utf-8")
        ),
        case_ids=selected,
    )


def generate_quarter_step_selection(
    easyedit_root: str | Path,
) -> QuarterStepSelectionManifest:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(root, seed=DEFAULT_SELECTION_SEED)
    all_case_ids = scan_counterfact_case_ids(
        root / "data/counterfact/counterfact.json"
    )
    return select_quarter_step_cases(all_case_ids, canonical=canonical)


def _execution_envelope(model_alias: str, run_id: str) -> None:
    if model_alias not in QSTEP_RUN_IDS or run_id != QSTEP_RUN_IDS[model_alias]:
        raise MV1Error("quarter-step model/run identity differs from precommit")


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
        raise MV1Error("partial quarter-step Slurm identity is forbidden")
    if (
        values["job_name"] != QSTEP_JOB_NAME
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("quarter-step Slurm identity differs from fixed envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
    }


def _proposal_panel(
    *,
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
    target_ids: torch.Tensor,
    actions: Sequence[ActionDirection],
    epsilon: float,
    base_hashes: Mapping[str, str],
    state_identity: str,
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    panel_label: str,
) -> Any:
    def evaluate(action: ActionDirection, signed_distance: float) -> Any:
        return _evaluate_branch(
            runtime=runtime,
            request=request,
            contexts=contexts,
            target_ids=target_ids,
            proposal=scale_proposal(
                action.proposal,
                signed_distance,
                solver_suffix=f"qstep-{panel_label}-probe-{signed_distance:.12g}",
            ),
            exact_application=False,
            base_hashes=base_hashes,
            state_identity=state_identity,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
        )

    return build_central_probe_panel(
        actions,
        epsilon=epsilon,
        evaluate=evaluate,
        expected_action_ids=PROBE_ACTIONS,
    )


def _build_score_mix_action(
    *,
    synchronous: MemitFactorProposal,
    unit_actions: Sequence[ActionDirection],
    scores: Mapping[str, float],
    covariance_moments: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> tuple[Any, ActionDirection]:
    decision = derive_score_mix_decision(
        {action_id: scores[action_id] for action_id in LAYER_ACTIONS},
        layers=LAYERS,
    )
    action = build_score_mix_direction(
        synchronous=synchronous,
        unit_actions=unit_actions,
        decision=decision,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    return decision, action


def _set_rng(cpu_rng: torch.Tensor, cuda_rng: torch.Tensor) -> None:
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device=0)


def _capture_source_snapshot(
    *,
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
    hparams: Any,
    weight_names: Sequence[str],
    provenance_id: str,
) -> SnapshotManifest:
    return capture_snapshot(
        runtime.model,
        model_id=runtime.spec.snapshot_name,
        requests=(request,),
        context_id=contexts.manifest_id,
        hparams=hparams,
        weight_names=weight_names,
        provenance_ids=(provenance_id,),
    )


def _rebind_origin_proposal(
    proposal: MemitFactorProposal,
    *,
    descendant_snapshot: SnapshotManifest,
    lineage: FrozenTargetLineage,
    solver_suffix: str,
) -> MemitFactorProposal:
    """Rebind exact W0 factor bytes to one verified trajectory descendant."""

    if not isinstance(proposal, MemitFactorProposal):
        raise ContractError("fixed proposal transport requires MemitFactorProposal")
    lineage.assert_terminal_state(descendant_snapshot)
    if (
        proposal.snapshot.model_id != descendant_snapshot.model_id
        or proposal.snapshot.context_id != descendant_snapshot.context_id
        or proposal.snapshot.request_ids != descendant_snapshot.request_ids
        or proposal.snapshot.hparams_sha256 != descendant_snapshot.hparams_sha256
        or proposal.snapshot.state_id != lineage.origin_state_id
    ):
        raise ContractError("fixed proposal is not bound to trajectory W0")
    before_hash = proposal_direction_hash(proposal)
    factors = tuple(
        LowRankFactor(
            weight_name=factor.weight_name,
            left=factor.left,
            right=factor.right,
            expected_weight_sha256=descendant_snapshot.parameter(
                factor.weight_name
            ).sha256,
            native_update_transposed=factor.native_update_transposed,
        )
        for factor in proposal.factors
    )
    rebound_snapshot = SnapshotManifest(
        model_id=descendant_snapshot.model_id,
        context_id=descendant_snapshot.context_id,
        request_ids=descendant_snapshot.request_ids,
        hparams_sha256=descendant_snapshot.hparams_sha256,
        parameters=descendant_snapshot.parameters,
        provenance_ids=tuple(
            sorted({*descendant_snapshot.provenance_ids, lineage.lineage_id})
        ),
    )
    rebound = MemitFactorProposal(
        snapshot=rebound_snapshot,
        factors=factors,
        semantics=proposal.semantics,
        solver_name=f"{proposal.solver_name}/qstep-fixed/{solver_suffix}",
        residual_denominator=proposal.residual_denominator,
    )
    if proposal_direction_hash(rebound) != before_hash:
        raise ContractError("fixed proposal transport changed factor tensor bytes")
    return rebound


def _rebind_origin_actions(
    actions: Sequence[ActionDirection],
    *,
    descendant_snapshot: SnapshotManifest,
    lineage: FrozenTargetLineage,
    solver_suffix: str,
) -> tuple[ActionDirection, ...]:
    return tuple(
        ActionDirection(
            action_id=action.action_id,
            proposal=_rebind_origin_proposal(
                action.proposal,
                descendant_snapshot=descendant_snapshot,
                lineage=lineage,
                solver_suffix=f"{solver_suffix}-{action.action_id}",
            ),
            c_squared_norm=action.c_squared_norm,
        )
        for action in actions
    )


def _assert_step_energy(
    proposal: MemitFactorProposal,
    *,
    expected: float,
    covariance_moments: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> float:
    observed = proposal_c_energy(proposal, covariance_moments, layer_by_weight)
    if not math.isclose(
        observed,
        expected,
        rel_tol=MATCHED_C_REL_TOL,
        abs_tol=MATCHED_C_ABS_TOL,
    ):
        raise ContractError("quarter-step proposal failed matched per-hop C energy")
    return observed


def _combine_steps(
    origin: MemitFactorProposal,
    steps: Sequence[MemitFactorProposal],
    *,
    solver_suffix: str,
) -> MemitFactorProposal:
    """Represent the endpoint displacement as one low-rank proposal."""

    if not steps:
        raise ContractError("endpoint combination requires at least one step")
    origin_by_name = {factor.weight_name: factor for factor in origin.factors}
    step_maps = [
        {factor.weight_name: factor for factor in proposal.factors}
        for proposal in steps
    ]
    if any(not set(mapping).issubset(origin_by_name) for mapping in step_maps):
        raise ContractError("trajectory step contains an unknown factor")
    factors: list[LowRankFactor] = []
    for weight_name in tuple(factor.weight_name for factor in origin.factors):
        source = origin_by_name[weight_name]
        # score_mix deliberately omits zero-weight layers.  Missing factors are
        # exact zero increments, so the endpoint contains only observed pieces.
        pieces = [
            mapping[weight_name]
            for mapping in step_maps
            if weight_name in mapping
        ]
        if not pieces:
            continue
        if any(
            piece.native_update_transposed != source.native_update_transposed
            for piece in pieces
        ):
            raise ContractError("trajectory factor orientation changed")
        factors.append(
            LowRankFactor(
                weight_name=weight_name,
                left=torch.cat(tuple(piece.left for piece in pieces), dim=1),
                right=torch.cat(tuple(piece.right for piece in pieces), dim=1),
                expected_weight_sha256=origin.snapshot.parameter(weight_name).sha256,
                native_update_transposed=source.native_update_transposed,
            )
        )
    return MemitFactorProposal(
        snapshot=origin.snapshot,
        factors=tuple(factors),
        semantics=origin.semantics,
        solver_name=f"{origin.solver_name}/qstep-endpoint/{solver_suffix}",
        residual_denominator=origin.residual_denominator,
    )


@dataclass(slots=True)
class BuiltPath:
    policy: str
    proposals: tuple[MemitFactorProposal, ...]
    descendant_snapshots: tuple[SnapshotManifest, ...]
    lineages: tuple[FrozenTargetLineage, ...]
    panels: tuple[Mapping[str, Any], ...]
    layer_weights: tuple[Mapping[str, float], ...]
    c_cosines: tuple[Mapping[str, float], ...]
    step_energies: tuple[float, ...]
    endpoint_energies: tuple[float, ...]


def _build_policy_path(
    *,
    policy: str,
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
    action0: ActionDirection,
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
    proposal_transform: Callable[
        [MemitFactorProposal, str],
        tuple[MemitFactorProposal, Mapping[str, Any]],
    ]
    | None = None,
) -> BuiltPath:
    if policy not in {REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX}:
        raise ContractError("quarter-step path policy is outside the lock")
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
                    decision = decision0
                    panel_payload: Mapping[str, Any] = {"shared_w0_panel": True}
                    cosine_payload: Mapping[str, float] = {
                        f"layer_{layer}": 1.0 for layer in LAYERS
                    }
                    lineage_label = COMMON_STEP_1
                elif policy == REFRESHED_PREFIX:
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
                            f"{policy}-step-{step_index}",
                        )
                    refreshed_unit = build_unit_c_actions(
                        refreshed, covariance_moments, layer_by_weight
                    )
                    state_hashes = _weight_hashes(runtime.model, weight_names)
                    state_identity = _state_identity(runtime)
                    state_cpu_rng = torch.get_rng_state().clone()
                    state_cuda_rng = torch.cuda.get_rng_state(0).clone()
                    panel = _proposal_panel(
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
                        panel_label=f"{policy}-w{step_index - 1}",
                    )
                    decision, action = _build_score_mix_action(
                        synchronous=refreshed,
                        unit_actions=refreshed_unit,
                        scores=panel.scores,
                        covariance_moments=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
                    step_proposal = scale_proposal(
                        action.proposal,
                        hop_distance,
                        solver_suffix=f"qstep-{policy}-step-{step_index}",
                    )
                    panel_payload = panel.to_dict()
                    if proposal_transform is not None:
                        panel_payload = {
                            **panel_payload,
                            "proposal_transform": dict(transform_payload),
                        }
                    cosine_payload = per_layer_c_cosines(
                        b0,
                        refreshed,
                        covariance_by_layer=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
                    lineage_label = f"refreshed_step_{step_index}"
                elif policy == COEFFICIENT_PREFIX:
                    transported_b0 = _rebind_origin_proposal(
                        b0,
                        descendant_snapshot=current_source,
                        lineage=current_lineage,
                        solver_suffix=f"{policy}-w{step_index - 1}",
                    )
                    transported_unit = _rebind_origin_actions(
                        w0_unit,
                        descendant_snapshot=current_source,
                        lineage=current_lineage,
                        solver_suffix=f"{policy}-w{step_index - 1}",
                    )
                    state_hashes = _weight_hashes(runtime.model, weight_names)
                    state_identity = _state_identity(runtime)
                    state_cpu_rng = torch.get_rng_state().clone()
                    state_cuda_rng = torch.cuda.get_rng_state(0).clone()
                    panel = _proposal_panel(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        target_ids=target_ids,
                        actions=transported_unit,
                        epsilon=probe_distance,
                        base_hashes=state_hashes,
                        state_identity=state_identity,
                        cpu_rng=state_cpu_rng,
                        cuda_rng=state_cuda_rng,
                        panel_label=f"{policy}-w{step_index - 1}",
                    )
                    decision, action = _build_score_mix_action(
                        synchronous=transported_b0,
                        unit_actions=transported_unit,
                        scores=panel.scores,
                        covariance_moments=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
                    step_proposal = scale_proposal(
                        action.proposal,
                        hop_distance,
                        solver_suffix=f"qstep-{policy}-step-{step_index}",
                    )
                    panel_payload = panel.to_dict()
                    cosine_payload = {
                        f"layer_{layer}": 1.0 for layer in LAYERS
                    }
                    lineage_label = f"coefficient_step_{step_index}"
                else:
                    fixed_action = _rebind_origin_proposal(
                        action0.proposal,
                        descendant_snapshot=current_source,
                        lineage=current_lineage,
                        solver_suffix=f"{policy}-w{step_index - 1}",
                    )
                    decision = decision0
                    step_proposal = scale_proposal(
                        fixed_action,
                        hop_distance,
                        solver_suffix=f"qstep-{policy}-step-{step_index}",
                    )
                    panel_payload = {"probe_evaluation_count": 0}
                    cosine_payload = {
                        f"layer_{layer}": 1.0 for layer in LAYERS
                    }
                    lineage_label = f"fixed_step_{step_index}"

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
                    label=lineage_label,
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
                layer_weights.append(dict(decision.layer_weights))
                c_cosines.append(cosine_payload)
                step_energies.append(observed_energy)
                endpoint = _combine_steps(
                    b0,
                    proposals,
                    solver_suffix=f"{policy}-prefix-{step_index}",
                )
                endpoint_energies.append(
                    proposal_c_energy(endpoint, covariance_moments, layer_by_weight)
                )
                current_source = child_source
    finally:
        _set_rng(base_cpu_rng, base_cuda_rng)
        assert_snapshot_current(runtime.model, origin_snapshot)
        if (
            _weight_hashes(runtime.model, weight_names) != base_hashes
            or _state_identity(runtime) != base_state_identity
        ):
            raise RollbackError("quarter-step path construction did not restore W0")
    return BuiltPath(
        policy=policy,
        proposals=tuple(proposals),
        descendant_snapshots=tuple(descendants),
        lineages=tuple(lineages),
        panels=tuple(panels),
        layer_weights=tuple(layer_weights),
        c_cosines=tuple(c_cosines),
        step_energies=tuple(step_energies),
        endpoint_energies=tuple(endpoint_energies),
    )


def _build_native_split_path(
    *,
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
    hparams: Any,
    ordered_w0: MemitFactorProposal,
    origin_snapshot: SnapshotManifest,
    target_lineage: FrozenTargetLineage,
    direct_z: Any,
    target_ids: torch.Tensor,
    covariance_moments: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    native_energy: float,
    weight_names: Sequence[str],
    provenance_id: str,
    base_hashes: Mapping[str, str],
    base_state_identity: str,
    base_cpu_rng: torch.Tensor,
    base_cuda_rng: torch.Tensor,
) -> BuiltPath:
    hop_energy = native_energy / 16.0
    proposals: list[MemitFactorProposal] = []
    descendants: list[SnapshotManifest] = []
    lineages: list[FrozenTargetLineage] = []
    step_energies: list[float] = []
    endpoint_energies: list[float] = []
    current_source = origin_snapshot
    current_lineage = target_lineage
    assert_snapshot_current(runtime.model, origin_snapshot)
    _set_rng(base_cpu_rng, base_cuda_rng)
    try:
        with ExitStack() as stack:
            for step_index in range(1, QSTEP_K + 1):
                rebound = _rebind_origin_proposal(
                    ordered_w0,
                    descendant_snapshot=current_source,
                    lineage=current_lineage,
                    solver_suffix=f"native-w{step_index - 1}",
                )
                step_proposal = scale_proposal(
                    rebound,
                    QSTEP_HOP_FRACTION,
                    solver_suffix=f"qstep-native-step-{step_index}",
                )
                step_energies.append(
                    _assert_step_energy(
                        step_proposal,
                        expected=hop_energy,
                        covariance_moments=covariance_moments,
                        layer_by_weight=layer_by_weight,
                    )
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
                    label=f"native_step_{step_index}",
                )
                current_lineage.assert_authorizes(
                    direct_z=direct_z,
                    snapshot=child_source,
                    target_token_ids=target_ids,
                )
                proposals.append(step_proposal)
                descendants.append(child_source)
                lineages.append(current_lineage)
                endpoint = _combine_steps(
                    ordered_w0,
                    proposals,
                    solver_suffix=f"native-prefix-{step_index}",
                )
                endpoint_energies.append(
                    proposal_c_energy(endpoint, covariance_moments, layer_by_weight)
                )
                current_source = child_source
    finally:
        _set_rng(base_cpu_rng, base_cuda_rng)
        assert_snapshot_current(runtime.model, origin_snapshot)
        if (
            _weight_hashes(runtime.model, weight_names) != base_hashes
            or _state_identity(runtime) != base_state_identity
        ):
            raise RollbackError("native split construction did not restore W0")
    return BuiltPath(
        policy="native",
        proposals=tuple(proposals),
        descendant_snapshots=tuple(descendants),
        lineages=tuple(lineages),
        panels=tuple({"probe_evaluation_count": 0} for _ in range(QSTEP_K)),
        layer_weights=tuple({} for _ in range(QSTEP_K)),
        c_cosines=tuple({} for _ in range(QSTEP_K)),
        step_energies=tuple(step_energies),
        endpoint_energies=tuple(endpoint_energies),
    )


def _evaluate_prefix(
    *,
    model: torch.nn.Module,
    origin_snapshot: SnapshotManifest,
    proposals: Sequence[MemitFactorProposal],
    descendant_snapshots: Sequence[SnapshotManifest],
    prefix_length: int,
    evaluator: Callable[[], Any],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
) -> Any:
    if (
        prefix_length <= 0
        or prefix_length > len(proposals)
        or len(proposals) != len(descendant_snapshots)
    ):
        raise ContractError("trajectory prefix is outside the committed sequence")
    assert_snapshot_current(model, origin_snapshot)
    _set_rng(cpu_rng, cuda_rng)
    try:
        with ExitStack() as stack:
            for index in range(prefix_length):
                stack.enter_context(
                    TemporaryLowRankApplication(model, proposals[index])
                )
                assert_snapshot_current(model, descendant_snapshots[index])
            result = evaluator()
    finally:
        _set_rng(cpu_rng, cuda_rng)
        assert_snapshot_current(model, origin_snapshot)
    return result


def _path_feature(path: BuiltPath) -> dict[str, Any]:
    return {
        "policy": path.policy,
        "proposal_hashes": [proposal_direction_hash(item) for item in path.proposals],
        "descendant_state_ids": [item.state_id for item in path.descendant_snapshots],
        "lineage_ids": [item.lineage_id for item in path.lineages],
        "panels": list(path.panels),
        "layer_weights": list(path.layer_weights),
        "w0_to_current_c_cosines": list(path.c_cosines),
        "step_c_energies": list(path.step_energies),
        "endpoint_c_energies": list(path.endpoint_energies),
    }


def _commit_action(
    *,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    receipt_root: Path,
    feature: Mapping[str, Any],
    action: Mapping[str, Any],
    expected_branch_order: Sequence[str] = QSTEP_BRANCH_ORDER,
    feature_event: str = "quarter_step_feature",
    action_event: str = "quarter_step_action_commitment",
    receipt_schema: str = QSTEP_RECEIPT_SCHEMA,
) -> tuple[str, str]:
    locked_branch_order = tuple(str(item) for item in expected_branch_order)
    if (
        not locked_branch_order
        or len(set(locked_branch_order)) != len(locked_branch_order)
        or any(not item for item in locked_branch_order)
        or not feature_event
        or not action_event
        or not receipt_schema
    ):
        raise ContractError("feature/action commitment envelope is invalid")
    _assert_outcome_free(feature)
    _assert_outcome_free(action)
    if (
        feature["case_id"] != action["case_id"]
        or feature["request_id"] != action["request_id"]
        or action["feature_hash"] != feature["feature_hash"]
        or tuple(action["branch_order"]) != locked_branch_order
        or _feature_hash(
            {key: value for key, value in feature.items() if key != "feature_hash"}
        )
        != feature["feature_hash"]
        or _feature_hash(
            {
                key: value
                for key, value in action.items()
                if key != "commitment_hash"
            }
        )
        != action["commitment_hash"]
    ):
        raise ContractError("quarter-step feature/action commitment mismatch")
    _safe_payload(feature)
    _safe_payload(action)
    feature_writer.write(feature_event, feature)
    action_writer.write(action_event, action)
    feature_writer.sync()
    action_writer.sync()
    identity = {
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
        "origin_lineage_id": feature["target_identity"]["origin_lineage_id"],
    }
    receipt_name = sha256_bytes(canonical_json(identity).encode("utf-8")) + ".json"
    receipt = {
        "schema_version": receipt_schema,
        **identity,
        "branch_order": list(locked_branch_order),
        "path_action_hashes": action["path_action_hashes"],
        "per_hop_c_energy": action["per_hop_c_energy"],
        "all_paths_committed_before_outcomes": True,
        "durability": "feature+actions-write+flush+fsync-before-exclusive-receipt",
    }
    _write_json_exclusive(receipt_root / receipt_name, receipt)
    return receipt_name, _file_sha256(receipt_root / receipt_name)


def _run_quarter_step_event(
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
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    outcome_writer: SanitizedJsonlWriter,
    analysis_case_writer: SanitizedJsonlWriter,
) -> dict[str, Any]:
    """Build, commit, and only then evaluate one fresh quarter-step event."""

    event_started = time.perf_counter()
    event_seed = _event_seed(QSTEP_RUN_SEED, request.case_id)
    seed_runtime(event_seed)
    _full_hash("quarter-step request ID", request.request_id)
    layer_by_weight = _layer_by_weight(hparams)
    if (
        tuple(int(layer) for layer in hparams.layers) != LAYERS
        or len(layer_by_weight) != EXPECTED_LAYER_COUNT
    ):
        raise MV1Error("quarter-step diagnostic requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    base_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()

    baseline_teacher, target_ids = teacher_forced_rewrite_exact(
        runtime, request, contexts
    )
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("quarter-step baseline changed W0/RNG")

    bindings = bridge.load()
    origin_source_snapshot = _capture_source_snapshot(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        weight_names=weight_names,
        provenance_id=bindings.provenance.manifest_id,
    )
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
        origin_snapshot=origin_source_snapshot,
        target_token_ids=target_ids,
    )
    target_lineage.assert_authorizes(
        direct_z=direct_z,
        snapshot=origin_source_snapshot,
        target_token_ids=target_ids,
    )

    native_energy = proposal_c_energy(
        ordered_w0, covariance_moments, layer_by_weight
    )
    native_distance = math.sqrt(native_energy)
    hop_distance = native_distance * QSTEP_HOP_FRACTION
    hop_energy = native_energy / float(QSTEP_K * QSTEP_K)
    probe_distance = native_distance * QSTEP_PROBE_FRACTION
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (
            native_energy,
            native_distance,
            hop_distance,
            hop_energy,
            probe_distance,
        )
    ):
        raise MV1Error("quarter-step C-distance budget is invalid")

    w0_unit = build_unit_c_actions(b0, covariance_moments, layer_by_weight)
    assert_exact_action_contract(
        b0,
        ordered_w0,
        w0_unit,
        expected_factor_names=weight_names,
        expected_action_ids=PROBE_ACTIONS,
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
        panel_label="w0-b0",
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
        solver_suffix="qstep-common-step-1",
    )
    _assert_step_energy(
        common_step,
        expected=hop_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )

    path_kwargs = {
        "runtime": runtime,
        "bridge": bridge,
        "hparams": hparams,
        "contexts": contexts,
        "covariance_specs": covariance_specs,
        "covariance_moments": covariance_moments,
        "request": request,
        "direct_z": direct_z,
        "target_ids": target_ids,
        "origin_snapshot": origin_source_snapshot,
        "target_lineage": target_lineage,
        "b0": b0,
        "w0_unit": w0_unit,
        "decision0": decision0,
        "action0": action0,
        "common_step": common_step,
        "hop_distance": hop_distance,
        "hop_energy": hop_energy,
        "probe_distance": probe_distance,
        "weight_names": weight_names,
        "layer_by_weight": layer_by_weight,
        "provenance_id": bindings.provenance.manifest_id,
        "base_hashes": base_hashes,
        "base_state_identity": base_state_identity,
        "base_cpu_rng": base_cpu_rng,
        "base_cuda_rng": base_cuda_rng,
    }
    refreshed_path = _build_policy_path(policy=REFRESHED_PREFIX, **path_kwargs)
    coefficient_path = _build_policy_path(policy=COEFFICIENT_PREFIX, **path_kwargs)
    fixed_path = _build_policy_path(policy=FIXED_PREFIX, **path_kwargs)
    native_path = _build_native_split_path(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        ordered_w0=ordered_w0,
        origin_snapshot=origin_source_snapshot,
        target_lineage=target_lineage,
        direct_z=direct_z,
        target_ids=target_ids,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
        native_energy=native_energy,
        weight_names=weight_names,
        provenance_id=bindings.provenance.manifest_id,
        base_hashes=base_hashes,
        base_state_identity=base_state_identity,
        base_cpu_rng=base_cpu_rng,
        base_cuda_rng=base_cuda_rng,
    )
    paths = {
        REFRESHED_PREFIX: refreshed_path,
        COEFFICIENT_PREFIX: coefficient_path,
        FIXED_PREFIX: fixed_path,
    }
    if any(len(path.proposals) != QSTEP_K for path in (*paths.values(), native_path)):
        raise ContractError("quarter-step path length differs from K=4")
    common_hash = proposal_direction_hash(common_step)
    common_states = {
        path.descendant_snapshots[0].state_id for path in paths.values()
    }
    if (
        any(proposal_direction_hash(path.proposals[0]) != common_hash for path in paths.values())
        or len(common_states) != 1
        or any(
            not math.isclose(
                energy,
                hop_energy,
                rel_tol=MATCHED_C_REL_TOL,
                abs_tol=MATCHED_C_ABS_TOL,
            )
            for path in (*paths.values(), native_path)
            for energy in path.step_energies
        )
    ):
        raise ContractError("quarter-step paths do not share the matched first hop")
    for label, endpoint_energy in (
        (FIXED_PREFIX, fixed_path.endpoint_energies[-1]),
        (NATIVE_ORDERED_SPLIT4, native_path.endpoint_energies[-1]),
    ):
        if not math.isclose(
            endpoint_energy,
            native_energy,
            rel_tol=2e-4,
            abs_tol=2e-4,
        ):
            raise ContractError(f"{label} full endpoint differs from native C budget")
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("quarter-step feature construction did not restore W0/RNG")

    target_identity = {
        "model_id": target_lineage.model_id,
        "context_id": target_lineage.context_id,
        "request_ids": list(target_lineage.request_ids),
        "origin_lineage_id": target_lineage.lineage_id,
        "direct_z_tensor_sha256": target_lineage.direct_z_tensor_sha256,
        "direct_z_artifact_sha256": target_lineage.direct_z_artifact_sha256,
        "target_token_sha256": target_lineage.target_token_sha256,
        "target_token_shape": list(target_lineage.target_token_shape),
        "target_token_dtype": target_lineage.target_token_dtype,
        "direct_z_compute_count": 1,
    }
    feature: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "k": QSTEP_K,
        "hop_fraction": QSTEP_HOP_FRACTION,
        "total_q": QSTEP_TOTAL_Q,
        "native_c_energy": native_energy,
        "native_c_distance": native_distance,
        "per_hop_c_energy": hop_energy,
        "hop_c_distance": hop_distance,
        "probe_c_distance": probe_distance,
        "target_identity": target_identity,
        "origin_lineage": target_lineage.to_dict(),
        "w0_panel": w0_panel.to_dict(),
        "w0_layer_weights": decision0.layer_weights,
        "w0_proposals": {
            "synchronous_b0": proposal_direction_hash(b0),
            "ordered_native": proposal_direction_hash(ordered_w0),
            "common_step": common_hash,
        },
        "paths": {
            **{policy: _path_feature(path) for policy, path in paths.items()},
            "native_split4": _path_feature(native_path),
        },
        "compute_plan": {
            "direct_z_compute_count": 1,
            "proposal_build_count": 5,
            "probe_panel_count": 7,
            "probe_evaluation_count": 7 * 2 * len(PROBE_ACTIONS),
            "operational_branch_count": len(QSTEP_BRANCH_ORDER),
            "controlled_nfe": 1 + 7 * 2 * len(PROBE_ACTIONS) + len(QSTEP_BRANCH_ORDER),
            "nfe_scope": "controlled teacher-forced rewrite forwards only",
        },
    }
    feature["feature_hash"] = _feature_hash(feature)
    path_action_hashes = {
        policy: [proposal_direction_hash(item) for item in path.proposals]
        for policy, path in paths.items()
    }
    path_action_hashes["native_split4"] = [
        proposal_direction_hash(item) for item in native_path.proposals
    ]
    path_action_hashes[NATIVE_ORDERED_FULL] = [
        proposal_direction_hash(ordered_w0)
    ]
    action: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(QSTEP_BRANCH_ORDER),
        "path_action_hashes": path_action_hashes,
        "per_hop_c_energy": hop_energy,
        "path_endpoint_c_energies": {
            policy: list(path.endpoint_energies)
            for policy, path in paths.items()
        },
        "native_split_endpoint_c_energies": list(native_path.endpoint_energies),
        "lineage_ids": {
            policy: [lineage.lineage_id for lineage in path.lineages]
            for policy, path in paths.items()
        },
        "native_split_lineage_ids": [
            lineage.lineage_id for lineage in native_path.lineages
        ],
        "matched_per_hop_c": True,
        "common_first_hop_exact": True,
        "best_hop_selection_forbidden": True,
        "action_policy": (
            "D=sqrt(ordered-native-C-energy); four D/4 hops; frozen W0 target; "
            "fixed D/64 controller probes; refresh/fixed policies committed before outcomes"
        ),
    }
    action["commitment_hash"] = _feature_hash(action)
    receipt_name, receipt_hash = _commit_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action,
    )

    # No outcome-dependent decision is permitted above this durability point.
    def evaluate_frozen_target() -> Any:
        teacher, observed_target_ids = teacher_forced_rewrite_exact(
            runtime, request, contexts
        )
        target_lineage.assert_target_tokens(observed_target_ids)
        return rewrite_metrics(teacher, observed_target_ids)

    after_by_arm: dict[str, Any] = {}
    after_by_arm[NO_OP_REPLAY] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=origin_source_snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
    )
    after_by_arm[COMMON_STEP_1] = _evaluate_prefix(
        model=runtime.model,
        origin_snapshot=origin_source_snapshot,
        proposals=refreshed_path.proposals,
        descendant_snapshots=refreshed_path.descendant_snapshots,
        prefix_length=1,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
    )
    for step_index in range(2, QSTEP_K + 1):
        for policy, path in paths.items():
            after_by_arm[_checkpoint_arm(policy, step_index)] = _evaluate_prefix(
                model=runtime.model,
                origin_snapshot=origin_source_snapshot,
                proposals=path.proposals,
                descendant_snapshots=path.descendant_snapshots,
                prefix_length=step_index,
                evaluator=evaluate_frozen_target,
                cpu_rng=base_cpu_rng,
                cuda_rng=base_cuda_rng,
            )
    after_by_arm[NATIVE_ORDERED_FULL] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=origin_source_snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
        first_proposal=ordered_w0,
    )
    after_by_arm[NATIVE_ORDERED_SPLIT4] = _evaluate_prefix(
        model=runtime.model,
        origin_snapshot=origin_source_snapshot,
        proposals=native_path.proposals,
        descendant_snapshots=native_path.descendant_snapshots,
        prefix_length=QSTEP_K,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
    )
    if tuple(after_by_arm) != QSTEP_BRANCH_ORDER:
        raise ContractError("quarter-step operational branch order differs from lock")

    endpoint_energy_by_arm: dict[str, float] = {
        NO_OP_REPLAY: 0.0,
        COMMON_STEP_1: refreshed_path.endpoint_energies[0],
        NATIVE_ORDERED_FULL: native_energy,
        NATIVE_ORDERED_SPLIT4: native_path.endpoint_energies[-1],
    }
    cumulative_distance_by_arm: dict[str, float] = {
        NO_OP_REPLAY: 0.0,
        COMMON_STEP_1: hop_distance,
        NATIVE_ORDERED_FULL: native_distance,
        NATIVE_ORDERED_SPLIT4: native_distance,
    }
    step_index_by_arm: dict[str, int] = {
        NO_OP_REPLAY: 0,
        COMMON_STEP_1: 1,
        NATIVE_ORDERED_FULL: QSTEP_K,
        NATIVE_ORDERED_SPLIT4: QSTEP_K,
    }
    for step_index in range(2, QSTEP_K + 1):
        for policy, path in paths.items():
            arm_id = _checkpoint_arm(policy, step_index)
            endpoint_energy_by_arm[arm_id] = path.endpoint_energies[step_index - 1]
            cumulative_distance_by_arm[arm_id] = step_index * hop_distance
            step_index_by_arm[arm_id] = step_index

    for arm_id in QSTEP_BRANCH_ORDER:
        after = after_by_arm[arm_id]
        outcome = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
            "receipt_sha256": receipt_hash,
            "action_id": arm_id,
            "branch_index": QSTEP_BRANCH_ORDER.index(arm_id),
            "step_index": step_index_by_arm[arm_id],
            "progress": _finite(
                f"{arm_id} progress", after.utility - baseline.utility
            ),
            "endpoint_c_energy": endpoint_energy_by_arm[arm_id],
            "cumulative_path_distance": cumulative_distance_by_arm[arm_id],
            "per_hop_c_energy": (
                0.0 if arm_id == NO_OP_REPLAY else hop_energy
            ),
            "success": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "controlled_nfe": 1,
            **_metric_difference(after, baseline),
        }
        _safe_payload(outcome)
        outcome_writer.write("quarter_step_outcome", outcome)

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("quarter-step event did not restore W0/RNG")
    event_wall_seconds = time.perf_counter() - event_started
    analysis_case = {
        "model_alias": runtime.spec.alias,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "pass": True,
        "technical": {
            "exact_panel": True,
            "lineage_exact": True,
            "matched_per_hop_c": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
            "native_split_control": True,
        },
        "branch_order": list(QSTEP_BRANCH_ORDER),
        "arms": {
            arm_id: {
                "progress": _finite(
                    f"{arm_id} analysis progress",
                    after_by_arm[arm_id].utility - baseline.utility,
                ),
                "endpoint_c_energy": endpoint_energy_by_arm[arm_id],
                "cumulative_path_distance": cumulative_distance_by_arm[arm_id],
                "step_index": step_index_by_arm[arm_id],
                "nfe": 1,
                "success": True,
            }
            for arm_id in QSTEP_BRANCH_ORDER
        },
        "budgets": {
            "native_c_energy": native_energy,
            "per_hop_c_energy": hop_energy,
            "native_c_distance": native_distance,
            "hop_c_distance": hop_distance,
            "probe_c_distance": probe_distance,
        },
        "lineage": {
            "origin_state_id": origin_source_snapshot.state_id,
            "target_token_sha256": target_lineage.target_token_sha256,
            "direct_z_tensor_sha256": target_lineage.direct_z_tensor_sha256,
            "context_id": target_lineage.context_id,
            "final_lineage_ids": {
                policy: path.lineages[-1].lineage_id
                for policy, path in paths.items()
            },
            "native_split_final_lineage_id": native_path.lineages[-1].lineage_id,
        },
        "geometry": {
            "w0_to_current_c_cosines": {
                policy: list(path.c_cosines) for policy, path in paths.items()
            }
        },
        "compute": {
            "controlled_nfe": 1 + 7 * 2 * len(PROBE_ACTIONS) + len(QSTEP_BRANCH_ORDER),
            "proposal_build_count": 5,
            "probe_panel_count": 7,
            "wall_seconds": event_wall_seconds,
        },
    }
    _safe_payload(analysis_case)
    analysis_case_writer.write("quarter_step_analysis_case", analysis_case)
    return {
        "schema_version": QSTEP_EVENT_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "event_seed": event_seed,
        "target_token_count": int(target_ids.numel()),
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_artifact_size": direct_z.artifact.size,
        "direct_z_compute_count": 1,
        "origin_lineage_id": target_lineage.lineage_id,
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(QSTEP_BRANCH_ORDER),
        "analysis_case_count": 1,
        "receipt": {"name": receipt_name, "sha256": receipt_hash},
        "technical": dict(analysis_case["technical"]),
        "compute": dict(analysis_case["compute"]),
        "pass": True,
    }


def _validate_execution_mode(
    *,
    slurm_state: Mapping[str, Any],
    model_loader: Callable[[str], FixedModelRuntime],
    event_runner: Callable[..., Mapping[str, Any]],
) -> bool:
    under_slurm = slurm_state.get("under_slurm")
    if under_slurm is True:
        if model_loader is not load_fixed_model or event_runner is not _run_quarter_step_event:
            raise MV1Error("production quarter-step run forbids injected seams")
        return True
    if under_slurm is not False:
        raise MV1Error("quarter-step Slurm state is malformed")
    if model_loader is load_fixed_model or event_runner is _run_quarter_step_event:
        raise MV1Error("non-Slurm quarter-step run requires both test seams")
    return False


def _validate_event_result(
    raw: Mapping[str, Any], *, request: EditRequest
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "case_id",
        "request_id",
        "event_seed",
        "target_token_count",
        "direct_z_artifact_sha256",
        "direct_z_artifact_size",
        "direct_z_compute_count",
        "origin_lineage_id",
        "feature_count",
        "commitment_count",
        "outcome_count",
        "analysis_case_count",
        "receipt",
        "technical",
        "compute",
        "pass",
    }
    result = dict(raw)
    if set(result) != expected_fields:
        raise ContractError("quarter-step event-result schema differs from lock")
    if (
        result["schema_version"] != QSTEP_EVENT_SCHEMA
        or result["case_id"] != request.case_id
        or result["request_id"] != request.request_id
        or result["direct_z_compute_count"] != 1
        or result["feature_count"] != 1
        or result["commitment_count"] != 1
        or result["outcome_count"] != len(QSTEP_BRANCH_ORDER)
        or result["analysis_case_count"] != 1
        or result["pass"] is not True
    ):
        raise ContractError("quarter-step event result violates fixed counts")
    for field in ("event_seed", "target_token_count", "direct_z_artifact_size"):
        if (
            isinstance(result[field], bool)
            or not isinstance(result[field], int)
            or result[field] <= 0
        ):
            raise ContractError(f"quarter-step event {field} must be positive")
    _full_hash("direct-z artifact hash", result["direct_z_artifact_sha256"])
    _full_hash("origin lineage ID", result["origin_lineage_id"])
    required_technical = {
        "exact_panel",
        "lineage_exact",
        "matched_per_hop_c",
        "rollback_exact",
        "firewall_pass",
        "receipt_before_outcome",
        "native_split_control",
    }
    technical = result["technical"]
    if (
        not isinstance(technical, Mapping)
        or set(technical) != required_technical
        or any(technical[key] is not True for key in required_technical)
    ):
        raise ContractError("quarter-step technical gates are incomplete")
    receipt = result["receipt"]
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != {"name", "sha256"}
        or not isinstance(receipt["name"], str)
        or len(receipt["name"]) != 69
        or not receipt["name"].endswith(".json")
    ):
        raise ContractError("quarter-step receipt identity is malformed")
    _full_hash("quarter-step receipt hash", receipt["sha256"])
    compute = result["compute"]
    if (
        not isinstance(compute, Mapping)
        or set(compute)
        != {
            "controlled_nfe",
            "proposal_build_count",
            "probe_panel_count",
            "wall_seconds",
        }
        or compute["controlled_nfe"]
        != 1 + 7 * 2 * len(PROBE_ACTIONS) + len(QSTEP_BRANCH_ORDER)
        or compute["proposal_build_count"] != 5
        or compute["probe_panel_count"] != 7
        or _finite("quarter-step wall seconds", compute["wall_seconds"]) < 0.0
    ):
        raise ContractError("quarter-step compute accounting differs from lock")
    return result


def run_quarter_step_event_loop(
    *,
    requests: Sequence[EditRequest],
    event_runner: Callable[..., Mapping[str, Any]],
    event_writer: SanitizedJsonlWriter,
    event_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    results: list[dict[str, Any]] = []
    abort_failure_type: str | None = None
    for request in requests:
        try:
            raw = event_runner(request=request, **event_kwargs)
            if not isinstance(raw, Mapping):
                raise ContractError("quarter-step event runner must return a mapping")
            result = _validate_event_result(raw, request=request)
        except Exception as exc:
            print(
                f"QSTEP_LOCAL_TRACEBACK_BEGIN type={type(exc).__name__}",
                file=sys.stderr,
            )
            for frame, line_number in traceback.walk_tb(exc.__traceback__):
                print(
                    f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                    file=sys.stderr,
                )
            if isinstance(exc, TensorHashRuntimeError):
                print(
                    "QSTEP_LOCAL_TENSOR_HASH_DIAGNOSTIC "
                    f"phase={exc.phase} category={exc.category} "
                    f"dtype={exc.dtype} shape={','.join(map(str, exc.shape))} "
                    f"device={exc.device} numel={exc.numel}",
                    file=sys.stderr,
                )
            print("QSTEP_LOCAL_TRACEBACK_END", file=sys.stderr)
            result = {
                "case_id": request.case_id,
                "request_id": request.request_id,
                "failure_type": type(exc).__name__,
                "failure_class": "fatal_quarter_step_contract_or_runtime",
                "feature_count": 0,
                "commitment_count": 0,
                "outcome_count": 0,
                "analysis_case_count": 0,
                "rollback_exact": False,
                "pass": False,
            }
            abort_failure_type = type(exc).__name__
        results.append(result)
        event_writer.write("quarter_step_case", result)
        if abort_failure_type is not None:
            break
    return results, abort_failure_type


def failure_analysis_case(
    *, request: EditRequest, model_alias: str, context_id: str
) -> dict[str, Any]:
    if model_alias not in MODEL_SPECS:
        raise ContractError("failure analysis model alias is invalid")
    context_id = _full_hash("failure analysis context ID", context_id)

    def sentinel(label: str) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "schema_version": QSTEP_EVENT_SCHEMA,
                    "status": "not_observed_due_to_fatal_abort",
                    "case_id": request.case_id,
                    "request_id": request.request_id,
                    "label": label,
                }
            ).encode("utf-8")
        )

    return {
        "model_alias": model_alias,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "pass": False,
        "technical": {
            "exact_panel": False,
            "lineage_exact": False,
            "matched_per_hop_c": False,
            "rollback_exact": False,
            "firewall_pass": False,
            "receipt_before_outcome": False,
            "native_split_control": False,
        },
        "branch_order": list(QSTEP_BRANCH_ORDER),
        "arms": {
            arm_id: {
                "progress": 0.0,
                "endpoint_c_energy": 0.0,
                "cumulative_path_distance": 0.0,
                "step_index": 0,
                "nfe": 0,
                "success": False,
            }
            for arm_id in QSTEP_BRANCH_ORDER
        },
        "budgets": {
            "native_c_energy": 0.0,
            "per_hop_c_energy": 0.0,
            "native_c_distance": 0.0,
            "hop_c_distance": 0.0,
            "probe_c_distance": 0.0,
        },
        "lineage": {
            "origin_state_id": sentinel("unobserved-origin-state"),
            "target_token_sha256": sentinel("unobserved-target-token"),
            "direct_z_tensor_sha256": sentinel("unobserved-direct-z"),
            "context_id": context_id,
            "final_lineage_ids": {
                policy: sentinel(f"unobserved-{policy}-lineage")
                for policy in (REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX)
            },
            "native_split_final_lineage_id": sentinel(
                "unobserved-native-lineage"
            ),
        },
        "geometry": {
            "w0_to_current_c_cosines": {
                policy: []
                for policy in (REFRESHED_PREFIX, COEFFICIENT_PREFIX, FIXED_PREFIX)
            }
        },
        "compute": {
            "controlled_nfe": 0,
            "proposal_build_count": 0,
            "probe_panel_count": 0,
            "wall_seconds": 0.0,
        },
    }


def run_quarter_step_refresh(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    event_runner: Callable[..., Mapping[str, Any]] = _run_quarter_step_event,
) -> dict[str, Any]:
    """Run one fixed model on fresh canonical salted ranks [112:124]."""

    started = time.perf_counter()
    _execution_envelope(model_alias, run_id)
    slurm = _slurm_state(model_alias, run_id)
    _validate_execution_mode(
        slurm_state=slurm,
        model_loader=model_loader,
        event_runner=event_runner,
    )
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_quarter_step_selection(root)
    requests = load_counterfact_requests(root, selection.case_ids)
    if len(requests) != QSTEP_CASE_COUNT:
        raise ContractError("quarter-step request count differs from fresh lock")
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("quarter-step bridge provenance is outside fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(QSTEP_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("quarter-step model loader returned a different spec")
        contexts = _freeze_contexts(bridge, runtime, seed=QSTEP_RUN_SEED)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        outcomes_path = destination / "outcomes.jsonl"
        analysis_cases_path = destination / "analysis_cases.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        direct_z_root = destination / "direct_z"
        receipt_root = destination / "action_receipts"
        receipt_root.mkdir(mode=0o700)
        manifest = {
            "schema_version": QSTEP_MANIFEST_SCHEMA,
            "run_id": run_id,
            "ode_edit_git": git_state,
            "slurm": slurm,
            "model": runtime.metadata(),
            "hparams_relative_path": spec.hparams_path,
            "selection": selection.to_dict(),
            "selected_case_ids": list(selection.case_ids),
            "selected_request_ids": [request.request_id for request in requests],
            "contexts": {
                "manifest_id": contexts.manifest_id,
                "source": contexts.source,
                "group_sizes": [len(group) for group in contexts.templates],
                "raw_templates_persisted": False,
            },
            "provenance_id": provenance.manifest_id,
            "fixed_files": _relative_provenance(provenance, root),
            "constants": {
                "layers": list(LAYERS),
                "k": QSTEP_K,
                "hop_fraction": QSTEP_HOP_FRACTION,
                "total_q": QSTEP_TOTAL_Q,
                "probe_fraction_of_native_distance": QSTEP_PROBE_FRACTION,
                "run_seed": QSTEP_RUN_SEED,
                "case_count": QSTEP_CASE_COUNT,
                "bootstrap_seed": QSTEP_BOOTSTRAP_SEED,
                "bootstrap_resamples": QSTEP_BOOTSTRAP_RESAMPLES,
                "practical_effect_floor": QSTEP_PRACTICAL_EFFECT_FLOOR,
                "branch_order": list(QSTEP_BRANCH_ORDER),
            },
            "target_policy": (
                "direct-z computed once at W0; tensor/token/context frozen and "
                "descendant use authorized only by exact four-hop lineage"
            ),
            "distance_policy": (
                "D=sqrt(ordered native C energy); each operational hop D/4; "
                "four-hop path length D; controller probe fixed at D/64"
            ),
            "covariance_policy": (
                "run-start pinned immutable cache; miss/recompute/download blocked"
            ),
            "projector_policy": "preflight identity only; never deserialized",
            "information_firewall": (
                "only canonical request plus rewrite contexts; evaluation prompts, "
                "generations, weights, logits, activations excluded"
            ),
            "artifact_policy": (
                "Git-excluded tensors; streams contain IDs, hashes, finite scalars, "
                "and compact metadata only"
            ),
            "expected_counts": {
                "features": QSTEP_CASE_COUNT,
                "actions": QSTEP_CASE_COUNT,
                "outcomes": QSTEP_CASE_COUNT * len(QSTEP_BRANCH_ORDER),
                "analysis_cases": QSTEP_CASE_COUNT,
                "events": QSTEP_CASE_COUNT,
                "receipts": QSTEP_CASE_COUNT,
            },
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root, runtime=runtime, specs=covariance_specs
        )
        torch.cuda.reset_peak_memory_stats(0)
        with (
            SanitizedJsonlWriter(
                features_path, run_id, schema_version=QSTEP_STREAM_SCHEMA
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path, run_id, schema_version=QSTEP_STREAM_SCHEMA
            ) as action_writer,
            SanitizedJsonlWriter(
                outcomes_path, run_id, schema_version=QSTEP_STREAM_SCHEMA
            ) as outcome_writer,
            SanitizedJsonlWriter(
                analysis_cases_path, run_id, schema_version=QSTEP_STREAM_SCHEMA
            ) as analysis_case_writer,
            SanitizedJsonlWriter(
                events_path, run_id, schema_version=QSTEP_STREAM_SCHEMA
            ) as event_writer,
        ):
            results, abort_failure_type = run_quarter_step_event_loop(
                requests=requests,
                event_runner=event_runner,
                event_writer=event_writer,
                event_kwargs={
                    "runtime": runtime,
                    "bridge": bridge,
                    "hparams": hparams,
                    "contexts": contexts,
                    "covariance_specs": covariance_specs,
                    "covariance_moments": covariance_moments,
                    "direct_z_root": direct_z_root,
                    "receipt_root": receipt_root,
                    "feature_writer": feature_writer,
                    "action_writer": action_writer,
                    "outcome_writer": outcome_writer,
                    "analysis_case_writer": analysis_case_writer,
                },
            )
            while analysis_case_writer.sequence < len(requests):
                request = requests[analysis_case_writer.sequence]
                analysis_case_writer.write(
                    "quarter_step_analysis_case",
                    failure_analysis_case(
                        request=request,
                        model_alias=model_alias,
                        context_id=contexts.manifest_id,
                    ),
                )
            stream_sequences = {
                "features": feature_writer.sequence,
                "actions": action_writer.sequence,
                "outcomes": outcome_writer.sequence,
                "analysis_cases": analysis_case_writer.sequence,
                "events": event_writer.sequence,
            }
        planned = len(requests)
        attempted = len(results)
        pass_count = sum(bool(result["pass"]) for result in results)
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        exact_counts = bool(
            abort_failure_type is None
            and stream_sequences
            == {
                "features": planned,
                "actions": planned,
                "outcomes": planned * len(QSTEP_BRANCH_ORDER),
                "analysis_cases": planned,
                "events": planned,
            }
            and receipt_count == planned
        )
        summary = {
            "schema_version": QSTEP_SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "slurm": slurm,
            "provenance_id": provenance.manifest_id,
            "selection_manifest_id": selection.manifest_id,
            "context_id": contexts.manifest_id,
            "run_status": "aborted" if abort_failure_type else "completed",
            "abort_failure_type": abort_failure_type,
            "planned_case_count": planned,
            "attempted_case_count": attempted,
            "not_run_due_to_abort_count": planned - attempted,
            "pass_count": pass_count,
            "failure_count": planned - pass_count,
            "all_pass": bool(
                abort_failure_type is None
                and attempted == planned
                and pass_count == planned
                and exact_counts
            ),
            "case_results": [
                {
                    "case_id": result["case_id"],
                    "status": "attempted",
                    "pass": bool(result["pass"]),
                }
                for result in results
            ]
            + [
                {
                    "case_id": case_id,
                    "status": "not_run_due_to_abort",
                    "pass": False,
                }
                for case_id in tuple(request.case_id for request in requests)[attempted:]
            ],
            "constants": manifest["constants"],
            "stream_sequences": stream_sequences,
            "expected_counts_exact": exact_counts,
            "all_rollbacks_exact": bool(
                results
                and all(
                    bool(result.get("technical", {}).get("rollback_exact", False))
                    for result in results
                )
            ),
            "covariance_files_loaded": list(loaded_covariances),
            "projector_files_loaded": [],
            "direct_z_artifact_count": (
                len(tuple(direct_z_root.glob("*.pt")))
                if direct_z_root.is_dir()
                else 0
            ),
            "resource": {
                "wall_seconds": time.perf_counter() - started,
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
                "analysis_cases_sha256": _file_sha256(analysis_cases_path),
                "events_sha256": _file_sha256(events_path),
                "receipts": {
                    path.name: _file_sha256(path)
                    for path in sorted(receipt_root.glob("*.json"))
                },
            },
            "git_output_written": False,
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-quarter-step-refresh",
        description="Run the locked native-distance four-hop Motivation diagnostic.",
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
        summary = run_quarter_step_refresh(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
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
