"""Precommitted minimal MV-2 stale-versus-refreshed diagnostic.

The runner uses the fixed Motivation models, layers 4--8, q=1/256, seed 17,
and h=1/2 on the exact canonical salted ranks [100:112].  Those constants are
not CLI options.  Per event it computes direct-z once at W0, freezes its
value/token/context identity, applies one outcome-blind partial score-mix
action to obtain W1, then compares three equal-C continuation policies:

* refreshed W1 directions with refreshed W1 coefficients;
* transported W0 directions rescored for refreshed W1 coefficients;
* transported W0 directions with fixed W0 coefficients.

All operational branch decisions and their hashes are durably committed before
the six outcome branches are evaluated.  EasyEdit remains read-only; pinned
covariance caches may be loaded but never recomputed, and projectors are never
deserialized.
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
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage
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
    PROBE_RATIO,
    SCORE_MIX_Q,
    _assert_outcome_free,
    build_score_mix_direction,
    derive_score_mix_decision,
)
from .trajectory import (
    MATCHED_C_ABS_TOL,
    MATCHED_C_REL_TOL,
    action_direction_hash,
    build_central_probe_panel,
    evaluate_temporary_trajectory,
    per_layer_c_cosines,
    proposal_direction_hash,
    transport_fixed_actions,
    transport_fixed_proposal,
    validate_matched_c_budget,
)


MV2_JOB_NAME = "odeedit_mv2refresh_pair_v1"
MV2_RUN_IDS = MappingProxyType(
    {
        "llama3-8b-inst": "mv2refresh_llama_e0_v1",
        "qwen2.5-7b-inst": "mv2refresh_qwen_e0_v1",
    }
)
MV2_HASH_DIAGNOSTIC_ID = "hashdiag-v1"
MV2_RUN_SEED = 17
MV2_STEP_SIZE = 0.5
MV2_Q = SCORE_MIX_Q
MV2_CASE_COUNT = 12
MV2_RANK_START = 100
MV2_RANK_STOP = 112
MV2_BOOTSTRAP_SEED = 20260801
MV2_BOOTSTRAP_RESAMPLES = 4000
MV2_PRACTICAL_EFFECT_FLOOR = 1e-4
LAYERS = (4, 5, 6, 7, 8)
LAYER_ACTIONS = tuple(f"layer_{layer}" for layer in LAYERS)
PROBE_ACTIONS = (*LAYER_ACTIONS, ACTION_UNIFORM)

NO_OP_REPLAY = "no_op_replay"
H0_SHAM = "h0_sham"
PARTIAL_JOINT = "partial_joint"
REFRESHED_DIRECTION_REFRESHED_COEFFICIENT = (
    "refreshed_direction_refreshed_coefficient"
)
FIXED_DIRECTION_REFRESHED_COEFFICIENT = (
    "fixed_direction_refreshed_coefficient"
)
FIXED_DIRECTION_FIXED_COEFFICIENT = "fixed_direction_fixed_coefficient"
CONTINUATION_ACTIONS = (
    REFRESHED_DIRECTION_REFRESHED_COEFFICIENT,
    FIXED_DIRECTION_REFRESHED_COEFFICIENT,
    FIXED_DIRECTION_FIXED_COEFFICIENT,
)
MV2_BRANCH_ORDER = (
    NO_OP_REPLAY,
    H0_SHAM,
    PARTIAL_JOINT,
    *CONTINUATION_ACTIONS,
)

MV2_SELECTION_SCHEMA = "ode-edit-mv2-refresh-selection/v1"
MV2_MANIFEST_SCHEMA = "ode-edit-mv2-refresh-manifest/v1"
MV2_STREAM_SCHEMA = "ode-edit-mv2-refresh/v1"
MV2_RECEIPT_SCHEMA = "ode-edit-mv2-refresh-receipt/v1"
MV2_SUMMARY_SCHEMA = "ode-edit-mv2-refresh-summary/v1"
MV2_EVENT_SCHEMA = "ode-edit-mv2-refresh-event/v1"

_FEATURE_FIELDS = frozenset(
    {
        "case_id",
        "request_id",
        "q",
        "h",
        "b0",
        "partial_c_energy",
        "second_step_c_energy",
        "probe_c_distance",
        "target_identity",
        "lineage",
        "panels",
        "layer_weights",
        "proposal_direction_hashes",
        "w0_w1_c_cosine",
        "compute_plan",
        "feature_hash",
    }
)
_ACTION_FIELDS = frozenset(
    {
        "case_id",
        "request_id",
        "q",
        "h",
        "feature_hash",
        "branch_order",
        "branch_action_hashes",
        "branch_c_energies",
        "matched_second_c",
        "matched_c_rel_tol",
        "matched_c_abs_tol",
        "lineage_ids",
        "action_policy",
        "commitment_hash",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "request_id",
        "event_seed",
        "target_token_count",
        "direct_z_artifact_sha256",
        "direct_z_artifact_size",
        "direct_z_compute_count",
        "origin_lineage_id",
        "partial_lineage_id",
        "feature_count",
        "commitment_count",
        "outcome_count",
        "analysis_case_count",
        "receipt",
        "technical",
        "compute",
        "pass",
    }
)


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
class MV2SelectionManifest:
    """ID-only exact canonical ranks [100:112]."""

    seed: str
    source_sha256: str
    source_size: int
    source_row_count: int
    canonical_first100_order_hash: str
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.seed != DEFAULT_SELECTION_SEED:
            raise ContractError("MV-2 requires the canonical selection seed")
        _full_hash("MV-2 source hash", self.source_sha256)
        _full_hash(
            "MV-2 first100 order hash",
            self.canonical_first100_order_hash,
        )
        if (
            isinstance(self.source_size, bool)
            or not isinstance(self.source_size, int)
            or self.source_size <= 0
            or isinstance(self.source_row_count, bool)
            or not isinstance(self.source_row_count, int)
            or self.source_row_count < MV2_RANK_STOP
            or len(self.case_ids) != MV2_CASE_COUNT
            or len(set(self.case_ids)) != MV2_CASE_COUNT
        ):
            raise ContractError("MV-2 selection metadata is invalid")

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
            "schema_version": MV2_SELECTION_SCHEMA,
            "seed": self.seed,
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "rank_slice": {
                "start": MV2_RANK_START,
                "stop": MV2_RANK_STOP,
                "count": MV2_CASE_COUNT,
            },
            "canonical_first100_order_hash": self.canonical_first100_order_hash,
            "case_ids": list(self.case_ids),
            "order_hash": self.order_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def _salted_rank(case_id: str, seed: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            seed.encode("utf-8") + b"\0" + case_id.encode("utf-8")
        ).digest(),
        case_id,
    )


def select_mv2_cases(
    all_case_ids: Sequence[str],
    *,
    canonical: CounterFactSelectionManifest,
) -> MV2SelectionManifest:
    """Verify the existing first100 lock, then return the next exact 12 IDs."""

    if (
        canonical.seed != DEFAULT_SELECTION_SEED
        or len(canonical.calibration) != 20
        or len(canonical.confirmatory) != 60
        or len(canonical.untouched) != 20
    ):
        raise ContractError("MV-2 requires the canonical 20/60/20 selection")
    normalized = tuple(str(case_id) for case_id in all_case_ids)
    if (
        len(normalized) != canonical.source_row_count
        or len(set(normalized)) != len(normalized)
    ):
        raise ContractError("MV-2 case-ID source is incomplete or duplicated")
    ranked = tuple(
        sorted(
            normalized,
            key=lambda case_id: _salted_rank(case_id, DEFAULT_SELECTION_SEED),
        )
    )
    first100 = ranked[:MV2_RANK_START]
    if first100 != canonical.ordered_case_ids:
        raise ContractError("canonical first100 differs from the salted rank lock")
    next12 = ranked[MV2_RANK_START:MV2_RANK_STOP]
    if set(next12).intersection(first100):
        raise ContractError("MV-2 next12 overlaps the existing first100")
    return MV2SelectionManifest(
        seed=DEFAULT_SELECTION_SEED,
        source_sha256=canonical.source_sha256,
        source_size=canonical.source_size,
        source_row_count=canonical.source_row_count,
        canonical_first100_order_hash=canonical.order_hash,
        case_ids=next12,
    )


def generate_mv2_selection(easyedit_root: str | Path) -> MV2SelectionManifest:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(
        root,
        seed=DEFAULT_SELECTION_SEED,
    )
    all_case_ids = scan_counterfact_case_ids(
        root / "data/counterfact/counterfact.json"
    )
    return select_mv2_cases(all_case_ids, canonical=canonical)


def _execution_envelope(model_alias: str, run_id: str) -> None:
    if model_alias not in MV2_RUN_IDS or run_id != MV2_RUN_IDS[model_alias]:
        raise MV1Error("MV-2 model/run identity differs from the precommit")


def _technical_case_limit(model_alias: str) -> int | None:
    """Return the exact diagnostic limit without exposing a scientific knob."""

    diagnostic_id = os.environ.get("ODEEDIT_MV2_TECHNICAL_DIAGNOSTIC")
    raw_limit = os.environ.get("ODEEDIT_MV2_TECHNICAL_MAX_CASES")
    if diagnostic_id is None and raw_limit is None:
        return None
    if (
        diagnostic_id != MV2_HASH_DIAGNOSTIC_ID
        or raw_limit != "1"
        or model_alias != "llama3-8b-inst"
        or os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    ):
        raise MV1Error("MV-2 technical case limit is outside the diagnostic lock")
    return 1


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
        raise MV1Error("partial MV-2 Slurm identity is forbidden")
    if (
        values["job_name"] != MV2_JOB_NAME
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("MV-2 Slurm identity differs from the fixed envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
    }


def _validate_execution_mode(
    *,
    slurm_state: Mapping[str, Any],
    model_loader: Callable[[str], FixedModelRuntime],
    event_runner: Callable[..., Mapping[str, Any]],
) -> bool:
    under_slurm = slurm_state.get("under_slurm")
    if under_slurm is True:
        if model_loader is not load_fixed_model or event_runner is not _run_mv2_event:
            raise MV1Error("production MV-2 forbids injected seams")
        return True
    if under_slurm is not False:
        raise MV1Error("MV-2 Slurm state is malformed")
    if model_loader is load_fixed_model or event_runner is _run_mv2_event:
        raise MV1Error("non-Slurm MV-2 requires both injected test seams")
    return False


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
                solver_suffix=f"mv2-{panel_label}-probe-{signed_distance:.12g}",
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


def commit_mv2_action(
    *,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    receipt_root: Path,
    feature: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[str, str]:
    """Durably commit all six branch decisions before operational outcomes."""

    if set(feature) != _FEATURE_FIELDS or set(action) != _ACTION_FIELDS:
        raise ContractError("MV-2 feature/action schema differs from the lock")
    _assert_outcome_free(feature)
    _assert_outcome_free(action)
    if (
        feature["case_id"] != action["case_id"]
        or feature["request_id"] != action["request_id"]
        or feature["q"] != MV2_Q
        or feature["h"] != MV2_STEP_SIZE
        or action["q"] != MV2_Q
        or action["h"] != MV2_STEP_SIZE
        or action["feature_hash"] != feature["feature_hash"]
        or tuple(action["branch_order"]) != MV2_BRANCH_ORDER
        or tuple(action["branch_action_hashes"]) != MV2_BRANCH_ORDER
        or tuple(action["branch_c_energies"]) != MV2_BRANCH_ORDER
        or action["matched_second_c"] is not True
    ):
        raise ContractError("MV-2 feature/action identity mismatch")
    if (
        _feature_hash(
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
        raise ContractError("MV-2 feature/action hash mismatch")
    _safe_payload(feature)
    _safe_payload(action)
    feature_writer.write("mv2_refresh_feature", feature)
    action_writer.write("mv2_refresh_action_commitment", action)
    feature_writer.sync()
    action_writer.sync()

    receipt_identity = {
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
        "origin_lineage_id": action["lineage_ids"]["origin"],
        "partial_lineage_id": action["lineage_ids"]["partial"],
    }
    receipt_name = (
        sha256_bytes(canonical_json(receipt_identity).encode("utf-8")) + ".json"
    )
    receipt_payload = {
        "schema_version": MV2_RECEIPT_SCHEMA,
        **receipt_identity,
        "q": MV2_Q,
        "h": MV2_STEP_SIZE,
        "branch_order": list(MV2_BRANCH_ORDER),
        "branch_action_hashes": action["branch_action_hashes"],
        "branch_c_energies": action["branch_c_energies"],
        "matched_second_c": True,
        "durability": (
            "feature+six-branch-action-write+flush+fsync-before-exclusive-receipt"
        ),
        "operational_branches_observed_before_commitment": False,
    }
    _write_json_exclusive(receipt_root / receipt_name, receipt_payload)
    return receipt_name, _file_sha256(receipt_root / receipt_name)


def _run_mv2_event(
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
    event_started = time.perf_counter()
    event_seed = _event_seed(MV2_RUN_SEED, request.case_id)
    seed_runtime(event_seed)
    if (
        len(request.request_id) != 64
        or any(character not in "0123456789abcdef" for character in request.request_id)
    ):
        raise MV1Error("MV-2 request ID must be a canonical SHA-256")
    layer_by_weight = _layer_by_weight(hparams)
    if (
        tuple(int(layer) for layer in hparams.layers) != LAYERS
        or len(layer_by_weight) != EXPECTED_LAYER_COUNT
    ):
        raise MV1Error("MV-2 requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    base_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()

    baseline_teacher, target_ids = teacher_forced_rewrite_exact(
        runtime,
        request,
        contexts,
    )
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("MV-2 baseline changed the W0 runtime")

    bindings = bridge.load()
    origin_source_snapshot = capture_snapshot(
        runtime.model,
        model_id=runtime.spec.snapshot_name,
        requests=(request,),
        context_id=contexts.manifest_id,
        hparams=hparams,
        weight_names=weight_names,
        provenance_ids=(bindings.provenance.manifest_id,),
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
        ordered_w0,
        covariance_moments,
        layer_by_weight,
    )
    operational_b0 = MV2_Q * native_energy
    distance = math.sqrt(operational_b0)
    epsilon = PROBE_RATIO * distance
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (native_energy, operational_b0, distance, epsilon)
    ):
        raise MV1Error("MV-2 operational/probe budget is invalid")
    w0_unit = build_unit_c_actions(
        b0,
        covariance_moments,
        layer_by_weight,
    )
    assert_exact_action_contract(
        b0,
        ordered_w0,
        w0_unit,
        expected_factor_names=weight_names,
        expected_action_ids=PROBE_ACTIONS,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    w0_panel = _proposal_panel(
        runtime=runtime,
        request=request,
        contexts=contexts,
        target_ids=target_ids,
        actions=w0_unit,
        epsilon=epsilon,
        base_hashes=base_hashes,
        state_identity=base_state_identity,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
        panel_label="w0-b0",
    )
    decision0, action0 = _build_score_mix_action(
        synchronous=b0,
        unit_actions=w0_unit,
        scores=w0_panel.scores,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )

    # h=0 invokes the same descendant-target refresh machinery at byte-identical W0.
    h0_lineage = target_lineage.derive_h0(
        snapshot=origin_source_snapshot,
    )
    h0_action_hash = h0_lineage.hops[0].action_hash
    with _silence_upstream():
        h0_b0 = bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
            frozen_target_lineage=h0_lineage,
        )
    if proposal_direction_hash(h0_b0) != proposal_direction_hash(b0):
        raise MV1Error("h=0 refresh changed the W0 proposal direction")
    h0_unit = build_unit_c_actions(
        h0_b0,
        covariance_moments,
        layer_by_weight,
    )
    assert_exact_action_contract(
        h0_b0,
        ordered_w0,
        h0_unit,
        expected_factor_names=weight_names,
        expected_action_ids=PROBE_ACTIONS,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    w0_action_hashes = {
        action.action_id: action_direction_hash(action) for action in w0_unit
    }
    h0_action_hashes = {
        action.action_id: action_direction_hash(action) for action in h0_unit
    }
    if h0_action_hashes != w0_action_hashes:
        raise MV1Error("h=0 refresh action bytes/C-energy differ at W0")
    h0_verification = {
        "mode": "proposal-and-unit-action-hash-no-duplicate-probe",
        "proposal_direction_equal": True,
        "unit_action_hashes_equal": True,
        "unit_action_hashes": h0_action_hashes,
        "probe_evaluation_count": 0,
    }

    partial_distance = MV2_STEP_SIZE * distance
    second_distance = (1.0 - MV2_STEP_SIZE) * distance
    partial_energy = partial_distance * partial_distance
    second_energy = second_distance * second_distance
    full_step_proposal = scale_proposal(
        action0.proposal,
        distance,
        solver_suffix="mv2-full-joint-reference",
    )
    partial_proposal = scale_proposal(
        full_step_proposal,
        MV2_STEP_SIZE,
        solver_suffix="mv2-partial-joint-h-1-2",
    )
    partial_action_hash = proposal_direction_hash(partial_proposal)
    if not math.isclose(
        proposal_c_energy(
            partial_proposal,
            covariance_moments,
            layer_by_weight,
        ),
        partial_energy,
        rel_tol=MATCHED_C_REL_TOL,
        abs_tol=MATCHED_C_ABS_TOL,
    ):
        raise MV1Error("MV-2 partial action failed its C-budget gate")

    with TemporaryLowRankApplication(runtime.model, partial_proposal) as partial_apply:
        w1_source_snapshot = capture_snapshot(
            runtime.model,
            model_id=runtime.spec.snapshot_name,
            requests=(request,),
            context_id=contexts.manifest_id,
            hparams=hparams,
            weight_names=weight_names,
            provenance_ids=(bindings.provenance.manifest_id,),
        )
        partial_lineage = target_lineage.derive_partial(
            parent_snapshot=origin_source_snapshot,
            child_snapshot=w1_source_snapshot,
            application=partial_apply,
            full_step_proposal=full_step_proposal,
        )
        if partial_lineage.hops[0].action_hash != partial_action_hash:
            raise MV1Error("partial lineage action differs from the applied proposal")
        partial_lineage.assert_authorizes(
            direct_z=direct_z,
            snapshot=w1_source_snapshot,
            target_token_ids=target_ids,
        )
        with _silence_upstream():
            b1 = bridge.propose_synchronous_memit_factors(
                runtime.model,
                runtime.tokenizer,
                (request,),
                hparams,
                contexts,
                direct_z,
                covariance_specs,
                model_id=runtime.spec.snapshot_name,
                frozen_target_lineage=partial_lineage,
            )
        w1_hashes = _weight_hashes(runtime.model, weight_names)
        w1_state_identity = _state_identity(runtime)
        w1_cpu_rng = torch.get_rng_state().clone()
        w1_cuda_rng = torch.cuda.get_rng_state(0).clone()

        b1_unit = build_unit_c_actions(
            b1,
            covariance_moments,
            layer_by_weight,
        )
        w1_refreshed_panel = _proposal_panel(
            runtime=runtime,
            request=request,
            contexts=contexts,
            target_ids=target_ids,
            actions=b1_unit,
            epsilon=epsilon,
            base_hashes=w1_hashes,
            state_identity=w1_state_identity,
            cpu_rng=w1_cpu_rng,
            cuda_rng=w1_cuda_rng,
            panel_label="w1-b1",
        )
        decision1, action_a_unit = _build_score_mix_action(
            synchronous=b1,
            unit_actions=b1_unit,
            scores=w1_refreshed_panel.scores,
            covariance_moments=covariance_moments,
            layer_by_weight=layer_by_weight,
        )

        transported_b0 = transport_fixed_proposal(
            b0,
            descendant_snapshot=b1.snapshot,
            frozen_target_lineage=partial_lineage,
            solver_suffix="w1",
        )
        transported_w0_unit = transport_fixed_actions(
            w0_unit,
            descendant_snapshot=b1.snapshot,
            frozen_target_lineage=partial_lineage,
            solver_suffix="w1",
        )
        w1_fixed_panel = _proposal_panel(
            runtime=runtime,
            request=request,
            contexts=contexts,
            target_ids=target_ids,
            actions=transported_w0_unit,
            epsilon=epsilon,
            base_hashes=w1_hashes,
            state_identity=w1_state_identity,
            cpu_rng=w1_cpu_rng,
            cuda_rng=w1_cuda_rng,
            panel_label="w1-fixed-b0",
        )
        decision_fixed_refreshed, action_b_unit = _build_score_mix_action(
            synchronous=transported_b0,
            unit_actions=transported_w0_unit,
            scores=w1_fixed_panel.scores,
            covariance_moments=covariance_moments,
            layer_by_weight=layer_by_weight,
        )
        action_c_unit = build_score_mix_direction(
            synchronous=transported_b0,
            unit_actions=transported_w0_unit,
            decision=decision0,
            covariance_by_layer=covariance_moments,
            layer_by_weight=layer_by_weight,
        )
        second_proposals = {
            REFRESHED_DIRECTION_REFRESHED_COEFFICIENT: scale_proposal(
                action_a_unit.proposal,
                second_distance,
                solver_suffix="mv2-A-refreshed-direction-refreshed-coefficient",
            ),
            FIXED_DIRECTION_REFRESHED_COEFFICIENT: scale_proposal(
                action_b_unit.proposal,
                second_distance,
                solver_suffix="mv2-B-fixed-direction-refreshed-coefficient",
            ),
            FIXED_DIRECTION_FIXED_COEFFICIENT: scale_proposal(
                action_c_unit.proposal,
                second_distance,
                solver_suffix="mv2-C-fixed-direction-fixed-coefficient",
            ),
        }
        second_energies = validate_matched_c_budget(
            second_proposals,
            expected_c_energy=second_energy,
            covariance_by_layer=covariance_moments,
            layer_by_weight=layer_by_weight,
            exact_action_ids=CONTINUATION_ACTIONS,
        )
        c_cosines = per_layer_c_cosines(
            b0,
            b1,
            covariance_by_layer=covariance_moments,
            layer_by_weight=layer_by_weight,
        )

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("MV-2 feature construction did not restore W0")

    target_identity = {
        "model_id": target_lineage.model_id,
        "context_id": target_lineage.context_id,
        "request_ids": list(target_lineage.request_ids),
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
        "q": MV2_Q,
        "h": MV2_STEP_SIZE,
        "b0": operational_b0,
        "partial_c_energy": partial_energy,
        "second_step_c_energy": second_energy,
        "probe_c_distance": epsilon,
        "target_identity": target_identity,
        "lineage": {
            "origin": target_lineage.to_dict(),
            "h0_sham": h0_lineage.to_dict(),
            "partial_joint": partial_lineage.to_dict(),
        },
        "panels": {
            "w0_b0": w0_panel.to_dict(),
            "h0_sham": h0_verification,
            "w1_b1": w1_refreshed_panel.to_dict(),
            "w1_fixed_b0_rescore": w1_fixed_panel.to_dict(),
            "panel_build_counts": {
                "w0_b0": 1,
                "h0_sham": 0,
                "w1_b1": 1,
                "w1_fixed_b0_rescore": 1,
            },
        },
        "layer_weights": {
            "w0": decision0.layer_weights,
            "w1_refreshed_direction": decision1.layer_weights,
            "w1_fixed_b0_rescore": decision_fixed_refreshed.layer_weights,
        },
        "proposal_direction_hashes": {
            "b0": proposal_direction_hash(b0),
            "h0_b0": proposal_direction_hash(h0_b0),
            "b1": proposal_direction_hash(b1),
            "transported_b0": proposal_direction_hash(transported_b0),
        },
        "w0_w1_c_cosine": c_cosines,
        "compute_plan": {
            "direct_z_compute_count": 1,
            "b0_build_count": 1,
            "h0_refresh_build_count": 1,
            "b1_build_count": 1,
            "fixed_b0_transport_count": 1,
            "w0_ordered_reference_build_count": 1,
            "probe_panel_count": 3,
            "probe_evaluation_count": 3 * 2 * len(PROBE_ACTIONS),
            "operational_branch_count": len(MV2_BRANCH_ORDER),
            "w1_policy_incremental_cost": {
                REFRESHED_DIRECTION_REFRESHED_COEFFICIENT: {
                    "proposal_build_count": 1,
                    "probe_nfe": 2 * len(PROBE_ACTIONS),
                },
                FIXED_DIRECTION_REFRESHED_COEFFICIENT: {
                    "proposal_build_count": 0,
                    "probe_nfe": 2 * len(PROBE_ACTIONS),
                },
                FIXED_DIRECTION_FIXED_COEFFICIENT: {
                    "proposal_build_count": 0,
                    "probe_nfe": 0,
                },
            },
            "nfe_scope": (
                "controlled teacher-forced rewrite forwards only; "
                "EasyEdit internal optimization/activation forwards reported "
                "separately as proposal builds"
            ),
        },
    }
    feature["feature_hash"] = _feature_hash(feature)
    branch_action_hashes = {
        NO_OP_REPLAY: sha256_bytes(b"mv2/no-op-replay"),
        H0_SHAM: h0_action_hash,
        PARTIAL_JOINT: partial_action_hash,
        **{
            action_id: proposal_direction_hash(second_proposals[action_id])
            for action_id in CONTINUATION_ACTIONS
        },
    }
    branch_c_energies = {
        NO_OP_REPLAY: 0.0,
        H0_SHAM: 0.0,
        PARTIAL_JOINT: partial_energy,
        **second_energies,
    }
    action: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": MV2_Q,
        "h": MV2_STEP_SIZE,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(MV2_BRANCH_ORDER),
        "branch_action_hashes": branch_action_hashes,
        "branch_c_energies": branch_c_energies,
        "matched_second_c": True,
        "matched_c_rel_tol": MATCHED_C_REL_TOL,
        "matched_c_abs_tol": MATCHED_C_ABS_TOL,
        "lineage_ids": {
            "origin": target_lineage.lineage_id,
            "h0_sham": h0_lineage.lineage_id,
            "partial": partial_lineage.lineage_id,
        },
        "action_policy": (
            "W0 score_mix partial h=1/2; W1 exact equal-C A/B/C; "
            "all panels and actions built once and reused"
        ),
    }
    action["commitment_hash"] = _feature_hash(action)
    receipt_name, receipt_hash = commit_mv2_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action,
    )

    # Operational outcomes begin only after the exclusive receipt exists.
    def evaluate_frozen_target() -> Any:
        teacher, observed_target_ids = teacher_forced_rewrite_exact(
            runtime,
            request,
            contexts,
        )
        partial_lineage.assert_target_tokens(observed_target_ids)
        return rewrite_metrics(teacher, observed_target_ids)

    after_by_action: dict[str, Any] = {}
    after_by_action[NO_OP_REPLAY] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=b0.snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
    )
    after_by_action[H0_SHAM] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=b0.snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
    )
    after_by_action[PARTIAL_JOINT] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=b0.snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=cpu_rng,
        cuda_rng=cuda_rng,
        first_proposal=partial_proposal,
        descendant_snapshot=b1.snapshot,
    )
    for action_id in CONTINUATION_ACTIONS:
        after_by_action[action_id] = evaluate_temporary_trajectory(
            model=runtime.model,
            origin_snapshot=b0.snapshot,
            evaluator=evaluate_frozen_target,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
            first_proposal=partial_proposal,
            descendant_snapshot=b1.snapshot,
            second_proposal=second_proposals[action_id],
        )
    if tuple(after_by_action) != MV2_BRANCH_ORDER:
        raise MV1Error("MV-2 operational branch order differs from the lock")

    for action_id in MV2_BRANCH_ORDER:
        after = after_by_action[action_id]
        progress = _finite(
            f"{action_id} progress",
            after.utility - baseline.utility,
        )
        outcome = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
            "receipt_sha256": receipt_hash,
            "action_id": action_id,
            "branch_index": MV2_BRANCH_ORDER.index(action_id),
            "progress": progress,
            "c_energy": branch_c_energies[action_id],
            "partial_c_energy": (
                partial_energy
                if action_id in (PARTIAL_JOINT, *CONTINUATION_ACTIONS)
                else 0.0
            ),
            "second_step_c_energy": (
                second_energies[action_id]
                if action_id in CONTINUATION_ACTIONS
                else 0.0
            ),
            "matched_second_c": (
                True if action_id in CONTINUATION_ACTIONS else None
            ),
            "success": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "controlled_nfe": 1,
            **_metric_difference(after, baseline),
        }
        _safe_payload(outcome)
        outcome_writer.write("mv2_refresh_outcome", outcome)

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("MV-2 event did not restore W0/RNG")
    event_wall_seconds = time.perf_counter() - event_started
    analysis_case = {
        "model_alias": runtime.spec.alias,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "pass": True,
        "technical": {
            "exact_panel": True,
            "lineage_exact": True,
            "matched_second_c": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
        },
        "branch_order": list(MV2_BRANCH_ORDER),
        "arms": {
            action_id: {
                "progress": _finite(
                    f"{action_id} analysis progress",
                    after_by_action[action_id].utility - baseline.utility,
                ),
                "c_energy": branch_c_energies[action_id],
                "nfe": 1,
                "success": True,
            }
            for action_id in MV2_BRANCH_ORDER
        },
        "budgets": {"second_step_c_energy": second_energy},
        "lineage": {
            "origin_state_id": origin_source_snapshot.state_id,
            "w1_state_id": w1_source_snapshot.state_id,
            "lineage_id": partial_lineage.lineage_id,
            "target_token_sha256": partial_lineage.target_token_sha256,
            "direct_z_tensor_sha256": partial_lineage.direct_z_tensor_sha256,
            "context_id": partial_lineage.context_id,
        },
        "compute": {
            "controlled_nfe": 1 + 3 * 2 * len(PROBE_ACTIONS) + len(MV2_BRANCH_ORDER),
            "proposal_build_count": 4,
            "probe_panel_count": 3,
            "wall_seconds": event_wall_seconds,
        },
    }
    _safe_payload(analysis_case)
    analysis_case_writer.write("mv2_refresh_analysis_case", analysis_case)
    return {
        "schema_version": MV2_EVENT_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "event_seed": event_seed,
        "target_token_count": int(target_ids.numel()),
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_artifact_size": direct_z.artifact.size,
        "direct_z_compute_count": 1,
        "origin_lineage_id": target_lineage.lineage_id,
        "partial_lineage_id": partial_lineage.lineage_id,
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(MV2_BRANCH_ORDER),
        "analysis_case_count": 1,
        "receipt": {"name": receipt_name, "sha256": receipt_hash},
        "technical": {
            "exact_panel": True,
            "lineage_exact": True,
            "matched_second_c": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
            "h0_identical_state": True,
            "frozen_target_recompute_count": 0,
        },
        "compute": {
            "controlled_nfe": 1 + 3 * 2 * len(PROBE_ACTIONS) + len(MV2_BRANCH_ORDER),
            "proposal_build_count": 4,
            "probe_panel_count": 3,
            "wall_seconds": event_wall_seconds,
        },
        "pass": True,
    }


def _validate_event_result(
    raw: Mapping[str, Any],
    *,
    request: EditRequest,
) -> dict[str, Any]:
    result = dict(raw)
    if set(result) != _EVENT_FIELDS:
        raise ContractError("MV-2 event-result schema differs from the lock")
    if (
        result["schema_version"] != MV2_EVENT_SCHEMA
        or result["case_id"] != request.case_id
        or result["request_id"] != request.request_id
        or result["direct_z_compute_count"] != 1
        or result["feature_count"] != 1
        or result["commitment_count"] != 1
        or result["outcome_count"] != len(MV2_BRANCH_ORDER)
        or result["analysis_case_count"] != 1
        or result["pass"] is not True
    ):
        raise ContractError("MV-2 event result violates the fixed contract")
    for field in (
        "event_seed",
        "target_token_count",
        "direct_z_artifact_size",
    ):
        if (
            isinstance(result[field], bool)
            or not isinstance(result[field], int)
            or result[field] <= 0
        ):
            raise ContractError(f"MV-2 event {field} must be positive")
    for field in (
        "direct_z_artifact_sha256",
        "origin_lineage_id",
        "partial_lineage_id",
    ):
        _full_hash(field, result[field])
    technical = result["technical"]
    required_technical = {
        "exact_panel",
        "lineage_exact",
        "matched_second_c",
        "rollback_exact",
        "firewall_pass",
        "receipt_before_outcome",
        "h0_identical_state",
        "frozen_target_recompute_count",
    }
    if (
        not isinstance(technical, Mapping)
        or set(technical) != required_technical
        or any(
            technical[field] is not True
            for field in required_technical - {"frozen_target_recompute_count"}
        )
        or technical["frozen_target_recompute_count"] != 0
    ):
        raise ContractError("MV-2 technical event gates are incomplete")
    receipt = result["receipt"]
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != {"name", "sha256"}
        or not isinstance(receipt["name"], str)
        or len(receipt["name"]) != 69
        or not receipt["name"].endswith(".json")
    ):
        raise ContractError("MV-2 receipt identity is malformed")
    _full_hash("MV-2 receipt hash", receipt["sha256"])
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
        != 1 + 3 * 2 * len(PROBE_ACTIONS) + len(MV2_BRANCH_ORDER)
        or compute["proposal_build_count"] != 4
        or compute["probe_panel_count"] != 3
        or _finite("MV-2 wall seconds", compute["wall_seconds"]) < 0.0
    ):
        raise ContractError("MV-2 compute accounting differs from the lock")
    return result


def run_mv2_event_loop(
    *,
    requests: Sequence[EditRequest],
    event_runner: Callable[..., Mapping[str, Any]],
    event_writer: SanitizedJsonlWriter,
    event_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Run fail-closed ITD events and stop after the first fatal failure."""

    results: list[dict[str, Any]] = []
    abort_failure_type: str | None = None
    for request in requests:
        try:
            raw = event_runner(request=request, **event_kwargs)
            if not isinstance(raw, Mapping):
                raise ContractError("MV-2 event runner must return a mapping")
            result = _validate_event_result(raw, request=request)
        except Exception as exc:
            # Slurm stderr is an ignored local artifact.  Preserve only the
            # exception type and stack locations needed for technical triage;
            # never serialize ``str(exc)`` because upstream exceptions may
            # contain request text or other evaluation-sensitive material.
            print(
                f"MV2_LOCAL_TRACEBACK_BEGIN type={type(exc).__name__}",
                file=sys.stderr,
            )
            for frame, line_number in traceback.walk_tb(exc.__traceback__):
                print(
                    (
                        f'  File "{frame.f_code.co_filename}", '
                        f"line {line_number}, in {frame.f_code.co_name}"
                    ),
                    file=sys.stderr,
                )
            if isinstance(exc, TensorHashRuntimeError):
                print(
                    (
                        "MV2_LOCAL_TENSOR_HASH_DIAGNOSTIC "
                        f"phase={exc.phase} category={exc.category} "
                        f"dtype={exc.dtype} shape={','.join(map(str, exc.shape))} "
                        f"device={exc.device} numel={exc.numel}"
                    ),
                    file=sys.stderr,
                )
            print("MV2_LOCAL_TRACEBACK_END", file=sys.stderr)
            result = {
                "case_id": request.case_id,
                "request_id": request.request_id,
                "failure_type": type(exc).__name__,
                "failure_class": "fatal_mv2_contract_or_runtime",
                "feature_count": 0,
                "commitment_count": 0,
                "outcome_count": 0,
                "analysis_case_count": 0,
                "rollback_exact": False,
                "pass": False,
            }
            abort_failure_type = type(exc).__name__
        results.append(result)
        event_writer.write("mv2_refresh_case", result)
        if abort_failure_type is not None:
            break
    return results, abort_failure_type


def failure_analysis_case(
    *,
    request: EditRequest,
    model_alias: str,
    context_id: str,
) -> dict[str, Any]:
    """Create an explicit zero-effect ITD row for a failed/not-run event.

    The hashes are deterministic failure sentinels, not claims that model
    states or targets were observed.  Every technical gate is therefore false
    and the single-model analyzer must return ``BLOCK_TECHNICAL_INVALID`` while
    retaining the case in the denominator.
    """

    if model_alias not in MODEL_SPECS:
        raise ContractError("failure analysis row model alias is invalid")
    context_id = _full_hash("failure analysis context ID", context_id)

    def sentinel(label: str) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "schema_version": MV2_EVENT_SCHEMA,
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
            "matched_second_c": False,
            "rollback_exact": False,
            "firewall_pass": False,
            "receipt_before_outcome": False,
        },
        "branch_order": list(MV2_BRANCH_ORDER),
        "arms": {
            action_id: {
                "progress": 0.0,
                "c_energy": 0.0,
                "nfe": 0,
                "success": False,
            }
            for action_id in MV2_BRANCH_ORDER
        },
        "budgets": {"second_step_c_energy": 0.0},
        "lineage": {
            "origin_state_id": sentinel("unobserved-origin-state"),
            "w1_state_id": sentinel("unobserved-w1-state"),
            "lineage_id": sentinel("unobserved-lineage"),
            "target_token_sha256": sentinel("unobserved-target-token"),
            "direct_z_tensor_sha256": sentinel("unobserved-direct-z"),
            "context_id": context_id,
        },
        "compute": {
            "controlled_nfe": 0,
            "proposal_build_count": 0,
            "probe_panel_count": 0,
            "wall_seconds": 0.0,
        },
    }


def run_mv2_refresh(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    event_runner: Callable[..., Mapping[str, Any]] = _run_mv2_event,
) -> dict[str, Any]:
    """Run one fixed model on exact salted ranks [100:112]."""

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
    selection = generate_mv2_selection(root)
    requests = load_counterfact_requests(root, selection.case_ids)
    technical_case_limit = _technical_case_limit(model_alias)
    event_requests = (
        requests
        if technical_case_limit is None
        else requests[:technical_case_limit]
    )
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("MV-2 bridge provenance is outside the fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(MV2_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("MV-2 model loader returned a different fixed spec")
        contexts = _freeze_contexts(bridge, runtime, seed=MV2_RUN_SEED)
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
            "schema_version": MV2_MANIFEST_SCHEMA,
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
                "q": MV2_Q,
                "h": MV2_STEP_SIZE,
                "run_seed": MV2_RUN_SEED,
                "case_count": MV2_CASE_COUNT,
                "bootstrap_seed": MV2_BOOTSTRAP_SEED,
                "bootstrap_resamples": MV2_BOOTSTRAP_RESAMPLES,
                "practical_effect_floor": MV2_PRACTICAL_EFFECT_FLOOR,
                "branch_order": list(MV2_BRANCH_ORDER),
            },
            "target_policy": (
                "direct-z computed once at W0; tensor/target-token/context frozen "
                "and descendant use authorized only by exact lineage"
            ),
            "covariance_policy": (
                "run-start pinned immutable cache; miss/recompute/download blocked"
            ),
            "projector_policy": "preflight identity only; never deserialized",
            "information_firewall": (
                "only canonical request plus allowed rewrite contexts; evaluation "
                "prompts, generations, weights, logits, activations excluded"
            ),
            "artifact_policy": (
                "Git-excluded local tensors; structured streams contain IDs, "
                "hashes, finite scalars, and compact metadata only"
            ),
            "expected_counts": {
                "features": MV2_CASE_COUNT,
                "actions": MV2_CASE_COUNT,
                "outcomes": MV2_CASE_COUNT * len(MV2_BRANCH_ORDER),
                "analysis_cases": MV2_CASE_COUNT,
                "events": MV2_CASE_COUNT,
                "receipts": MV2_CASE_COUNT,
            },
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root,
            runtime=runtime,
            specs=covariance_specs,
        )
        torch.cuda.reset_peak_memory_stats(0)
        with (
            SanitizedJsonlWriter(
                features_path,
                run_id,
                schema_version=MV2_STREAM_SCHEMA,
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path,
                run_id,
                schema_version=MV2_STREAM_SCHEMA,
            ) as action_writer,
            SanitizedJsonlWriter(
                outcomes_path,
                run_id,
                schema_version=MV2_STREAM_SCHEMA,
            ) as outcome_writer,
            SanitizedJsonlWriter(
                analysis_cases_path,
                run_id,
                schema_version=MV2_STREAM_SCHEMA,
            ) as analysis_case_writer,
            SanitizedJsonlWriter(
                events_path,
                run_id,
                schema_version=MV2_STREAM_SCHEMA,
            ) as event_writer,
        ):
            results, abort_failure_type = run_mv2_event_loop(
                requests=event_requests,
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
            while analysis_case_writer.sequence < len(event_requests):
                request = event_requests[analysis_case_writer.sequence]
                analysis_case_writer.write(
                    "mv2_refresh_analysis_case",
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
        planned = len(event_requests)
        attempted = len(results)
        pass_count = sum(bool(result["pass"]) for result in results)
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        exact_counts = bool(
            technical_case_limit is None
            and abort_failure_type is None
            and stream_sequences
            == {
                "features": planned,
                "actions": planned,
                "outcomes": planned * len(MV2_BRANCH_ORDER),
                "analysis_cases": planned,
                "events": planned,
            }
            and receipt_count == planned
        )
        summary = {
            "schema_version": MV2_SUMMARY_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "slurm": slurm,
            "provenance_id": provenance.manifest_id,
            "selection_manifest_id": selection.manifest_id,
            "context_id": contexts.manifest_id,
            "run_status": (
                "technical_diagnostic"
                if technical_case_limit is not None
                else ("aborted" if abort_failure_type is not None else "completed")
            ),
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
                for case_id in tuple(
                    request.case_id for request in event_requests
                )[attempted:]
            ],
            "constants": {
                "layers": list(LAYERS),
                "q": MV2_Q,
                "h": MV2_STEP_SIZE,
                "run_seed": MV2_RUN_SEED,
                "case_count": MV2_CASE_COUNT,
                "bootstrap_seed": MV2_BOOTSTRAP_SEED,
                "bootstrap_resamples": MV2_BOOTSTRAP_RESAMPLES,
                "practical_effect_floor": MV2_PRACTICAL_EFFECT_FLOOR,
                "branch_order": list(MV2_BRANCH_ORDER),
            },
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
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(0)
                ),
                "gpu_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(0)
                ),
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
        prog="ode-edit-mv2-refresh",
        description="Run the fixed MV-2 stale-versus-refreshed diagnostic.",
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
        summary = run_mv2_refresh(
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
