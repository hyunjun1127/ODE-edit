"""Motivation-closing adaptive direction-gate experiment.

One reusable runner executes the same four-hop, outcome-free direction gate
over either native MEMIT proposals or those proposals transformed by the exact
precomputed EasyEdit AlphaEdit projector.  EasyEdit source, covariance files,
CounterFact, and projector bytes are strictly read-only.
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

from .adaptive_path import (
    GATED_PREFIX,
    GATE_RELATIVE_MARGIN,
    build_adaptive_gated_path,
)
from .contracts import ContractError, EditRequest, MemitFactorProposal, canonical_json, sha256_bytes
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage, proposal_direction_hash
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .hooks import TensorHashRuntimeError, assert_tensor_sha256_device_parity
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
    MV1Error,
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
from .projector_adapter import AlphaEditProjectorBank
from .quarter_step_refresh import (
    COEFFICIENT_PREFIX,
    FIXED_PREFIX,
    LAYERS,
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
    NO_OP_REPLAY,
    PROBE_ACTIONS,
    QSTEP_HOP_FRACTION,
    QSTEP_K,
    QSTEP_PROBE_FRACTION,
    REFRESHED_PREFIX,
    _assert_step_energy,
    _build_native_split_path,
    _build_policy_path,
    _build_score_mix_action,
    _commit_action,
    _evaluate_prefix,
    _path_feature,
    _proposal_panel,
    _set_rng,
)
from .trajectory import MATCHED_C_ABS_TOL, MATCHED_C_REL_TOL, evaluate_temporary_trajectory


TRACK_MEMIT = "memit"
TRACK_ALPHAEDIT = "alphaedit_projected"
TRACKS = (TRACK_MEMIT, TRACK_ALPHAEDIT)
AGATE_RUN_SEED = 23
AGATE_CASE_COUNT = 8
AGATE_RANK_START = 124
AGATE_RANK_STOP = 132
AGATE_PRACTICAL_FLOOR = 1e-4
AGATE_BOOTSTRAP_SEED = 20260803
AGATE_BOOTSTRAP_RESAMPLES = 4000

AGATE_JOB_NAMES = MappingProxyType(
    {
        TRACK_MEMIT: "odeedit_agate_memit_pair_v3",
        TRACK_ALPHAEDIT: "odeedit_agate_alpha_pair_v3",
    }
)
AGATE_RUN_IDS = MappingProxyType(
    {
        TRACK_MEMIT: MappingProxyType(
            {
                "llama3-8b-inst": "agate_memit_llama_g0_v3",
                "qwen2.5-7b-inst": "agate_memit_qwen_g0_v3",
            }
        ),
        TRACK_ALPHAEDIT: MappingProxyType(
            {
                "llama3-8b-inst": "agate_alpha_llama_g0_v3",
                "qwen2.5-7b-inst": "agate_alpha_qwen_g0_v3",
            }
        ),
    }
)

GATED_FINAL = "gated_step_4"
REFRESHED_FINAL = "refreshed_step_4"
COEFFICIENT_FINAL = "coefficient_step_4"
FIXED_FINAL = "fixed_step_4"
AGATE_BRANCH_ORDER = (
    NO_OP_REPLAY,
    GATED_FINAL,
    REFRESHED_FINAL,
    COEFFICIENT_FINAL,
    FIXED_FINAL,
    NATIVE_ORDERED_FULL,
    NATIVE_ORDERED_SPLIT4,
)
AGATE_PROBE_PANEL_COUNT = 13
AGATE_CONTROLLED_NFE = 1 + AGATE_PROBE_PANEL_COUNT * 2 * len(PROBE_ACTIONS) + len(AGATE_BRANCH_ORDER)

SELECTION_SCHEMA = "ode-edit-adaptive-gate-selection/v1"
MANIFEST_SCHEMA = "ode-edit-adaptive-gate-manifest/v1"
STREAM_SCHEMA = "ode-edit-adaptive-gate/v1"
SUMMARY_SCHEMA = "ode-edit-adaptive-gate-summary/v1"
EVENT_SCHEMA = "ode-edit-adaptive-gate-event/v1"
RECEIPT_SCHEMA = "ode-edit-adaptive-gate-receipt/v1"


def _full_hash(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _salted_rank(case_id: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            DEFAULT_SELECTION_SEED.encode("utf-8")
            + b"\0"
            + str(case_id).encode("utf-8")
        ).digest(),
        str(case_id),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveGateSelectionManifest:
    seed: str
    source_sha256: str
    source_size: int
    source_row_count: int
    prior_evidence_order_hash: str
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.seed != DEFAULT_SELECTION_SEED
            or len(self.case_ids) != AGATE_CASE_COUNT
            or len(set(self.case_ids)) != AGATE_CASE_COUNT
            or self.source_row_count < AGATE_RANK_STOP
            or self.source_size <= 0
        ):
            raise ContractError("adaptive-gate selection metadata is invalid")
        _full_hash("selection source", self.source_sha256)
        _full_hash("prior evidence order", self.prior_evidence_order_hash)

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
            "schema_version": SELECTION_SCHEMA,
            "seed": self.seed,
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "rank_slice": {
                "start": AGATE_RANK_START,
                "stop": AGATE_RANK_STOP,
                "count": AGATE_CASE_COUNT,
            },
            "prior_evidence_order_hash": self.prior_evidence_order_hash,
            "case_ids": list(self.case_ids),
            "order_hash": self.order_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def select_adaptive_gate_cases(
    all_case_ids: Sequence[str],
    *,
    canonical: CounterFactSelectionManifest,
) -> AdaptiveGateSelectionManifest:
    normalized = tuple(str(case_id) for case_id in all_case_ids)
    if (
        canonical.seed != DEFAULT_SELECTION_SEED
        or len(canonical.ordered_case_ids) != 100
        or len(normalized) != canonical.source_row_count
        or len(set(normalized)) != len(normalized)
    ):
        raise ContractError("adaptive-gate selection requires canonical source")
    ranked = tuple(sorted(normalized, key=_salted_rank))
    if ranked[:100] != canonical.ordered_case_ids:
        raise ContractError("adaptive-gate canonical first100 drifted")
    prior = ranked[:AGATE_RANK_START]
    selected = ranked[AGATE_RANK_START:AGATE_RANK_STOP]
    if set(prior).intersection(selected):
        raise ContractError("adaptive-gate cases overlap prior evidence")
    return AdaptiveGateSelectionManifest(
        seed=DEFAULT_SELECTION_SEED,
        source_sha256=canonical.source_sha256,
        source_size=canonical.source_size,
        source_row_count=canonical.source_row_count,
        prior_evidence_order_hash=sha256_bytes(
            canonical_json(list(prior)).encode("utf-8")
        ),
        case_ids=selected,
    )


def generate_adaptive_gate_selection(easyedit_root: str | Path) -> AdaptiveGateSelectionManifest:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(root, seed=DEFAULT_SELECTION_SEED)
    all_case_ids = scan_counterfact_case_ids(root / "data/counterfact/counterfact.json")
    return select_adaptive_gate_cases(all_case_ids, canonical=canonical)


def _execution_envelope(track: str, model_alias: str, run_id: str) -> None:
    if (
        track not in TRACKS
        or model_alias not in MODEL_SPECS
        or run_id != AGATE_RUN_IDS[track][model_alias]
    ):
        raise MV1Error("adaptive-gate track/model/run differs from precommit")


def _slurm_state(track: str, model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(track, model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial adaptive-gate Slurm identity is forbidden")
    if (
        values["job_name"] != AGATE_JOB_NAMES[track]
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("adaptive-gate Slurm identity differs from fixed envelope")
    return {"under_slurm": True, **values}


def _make_transform(
    bank: AlphaEditProjectorBank | None,
    layer_by_weight: Mapping[str, int],
) -> Callable[[MemitFactorProposal, str], tuple[MemitFactorProposal, Mapping[str, Any]]] | None:
    if bank is None:
        return None

    def transform(
        proposal: MemitFactorProposal,
        label: str,
    ) -> tuple[MemitFactorProposal, Mapping[str, Any]]:
        projected = bank.project_proposal(
            proposal,
            layer_by_weight=layer_by_weight,
            solver_suffix=label,
        )
        return projected.proposal, {
            "projector_manifest_id": projected.projector_manifest_id,
            "right_norm_retention": dict(projected.right_norm_retention),
        }

    return transform


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
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    outcome_writer: SanitizedJsonlWriter,
    analysis_case_writer: SanitizedJsonlWriter,
    track: str,
    projector_bank: AlphaEditProjectorBank | None,
) -> dict[str, Any]:
    event_started = time.perf_counter()
    seed_runtime(_event_seed(AGATE_RUN_SEED, request.case_id))
    layer_by_weight = _layer_by_weight(hparams)
    if tuple(int(layer) for layer in hparams.layers) != LAYERS:
        raise MV1Error("adaptive-gate requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    base_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()

    baseline_teacher, target_ids = teacher_forced_rewrite_exact(runtime, request, contexts)
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("adaptive-gate baseline changed W0/RNG")

    bindings = bridge.load()
    from .quarter_step_refresh import _capture_source_snapshot

    origin_snapshot = _capture_source_snapshot(
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
        raw_b0 = bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
        )
        raw_ordered = bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )

    transform = _make_transform(projector_bank, layer_by_weight)
    w0_transform_metadata: dict[str, Any] = {}
    if transform is None:
        b0, ordered_w0 = raw_b0, raw_ordered
    else:
        b0, sync_metadata = transform(raw_b0, "w0-synchronous")
        ordered_w0, ordered_metadata = transform(raw_ordered, "w0-ordered")
        w0_transform_metadata = {
            "synchronous": dict(sync_metadata),
            "ordered": dict(ordered_metadata),
        }
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

    native_energy = proposal_c_energy(ordered_w0, covariance_moments, layer_by_weight)
    native_distance = math.sqrt(native_energy)
    hop_distance = native_distance * QSTEP_HOP_FRACTION
    hop_energy = native_energy / float(QSTEP_K * QSTEP_K)
    probe_distance = native_distance * QSTEP_PROBE_FRACTION
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (native_energy, native_distance, hop_distance, hop_energy, probe_distance)
    ):
        raise MV1Error("adaptive-gate C-distance budget is invalid")

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
        panel_label=f"agate-{track}-w0",
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
        solver_suffix=f"agate-{track}-common-step-1",
    )
    _assert_step_energy(
        common_step,
        expected=hop_energy,
        covariance_moments=covariance_moments,
        layer_by_weight=layer_by_weight,
    )

    common_kwargs = {
        "runtime": runtime,
        "bridge": bridge,
        "hparams": hparams,
        "contexts": contexts,
        "covariance_specs": covariance_specs,
        "covariance_moments": covariance_moments,
        "request": request,
        "direct_z": direct_z,
        "target_ids": target_ids,
        "origin_snapshot": origin_snapshot,
        "target_lineage": target_lineage,
        "b0": b0,
        "w0_unit": w0_unit,
        "decision0": decision0,
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
        "proposal_transform": transform,
    }
    path_kwargs = {**common_kwargs, "action0": action0}
    refreshed_path = _build_policy_path(policy=REFRESHED_PREFIX, **path_kwargs)
    coefficient_path = _build_policy_path(policy=COEFFICIENT_PREFIX, **path_kwargs)
    fixed_path = _build_policy_path(policy=FIXED_PREFIX, **path_kwargs)
    gated_path = build_adaptive_gated_path(
        **common_kwargs,
        relative_margin=GATE_RELATIVE_MARGIN,
    )
    native_path = _build_native_split_path(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        ordered_w0=ordered_w0,
        origin_snapshot=origin_snapshot,
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
        GATED_PREFIX: gated_path,
        REFRESHED_PREFIX: refreshed_path,
        COEFFICIENT_PREFIX: coefficient_path,
        FIXED_PREFIX: fixed_path,
    }
    if any(len(path.proposals) != QSTEP_K for path in (*paths.values(), native_path)):
        raise ContractError("adaptive-gate path length differs from K=4")
    common_hash = proposal_direction_hash(common_step)
    if (
        any(proposal_direction_hash(path.proposals[0]) != common_hash for path in paths.values())
        or len({path.descendant_snapshots[0].state_id for path in paths.values()}) != 1
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
        raise ContractError("adaptive-gate paths violate common/matched hop contract")
    for label, endpoint in (
        (FIXED_PREFIX, fixed_path.endpoint_energies[-1]),
        (NATIVE_ORDERED_SPLIT4, native_path.endpoint_energies[-1]),
    ):
        if not math.isclose(endpoint, native_energy, rel_tol=2e-4, abs_tol=2e-4):
            raise ContractError(f"{label} endpoint differs from native C budget")
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("adaptive-gate feature construction did not restore W0/RNG")

    feature = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "editor_track": track,
        "k": QSTEP_K,
        "hop_fraction": QSTEP_HOP_FRACTION,
        "native_c_energy": native_energy,
        "native_c_distance": native_distance,
        "per_hop_c_energy": hop_energy,
        "hop_c_distance": hop_distance,
        "probe_c_distance": probe_distance,
        "gate_relative_margin": GATE_RELATIVE_MARGIN,
        "target_identity": {
            "origin_lineage_id": target_lineage.lineage_id,
            "direct_z_tensor_sha256": target_lineage.direct_z_tensor_sha256,
            "direct_z_artifact_sha256": target_lineage.direct_z_artifact_sha256,
            "target_token_sha256": target_lineage.target_token_sha256,
            "direct_z_compute_count": 1,
        },
        "w0_panel": w0_panel.to_dict(),
        "w0_layer_weights": decision0.layer_weights,
        "w0_transform": w0_transform_metadata,
        "paths": {
            **{policy: _path_feature(path) for policy, path in paths.items()},
            "native_split4": _path_feature(native_path),
        },
        "compute_plan": {
            "direct_z_compute_count": 1,
            "proposal_build_count": 8,
            "probe_panel_count": AGATE_PROBE_PANEL_COUNT,
            "probe_evaluation_count": AGATE_PROBE_PANEL_COUNT * 2 * len(PROBE_ACTIONS),
            "operational_branch_count": len(AGATE_BRANCH_ORDER),
            "controlled_nfe": AGATE_CONTROLLED_NFE,
        },
    }
    feature["feature_hash"] = _feature_hash(feature)
    action = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "editor_track": track,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(AGATE_BRANCH_ORDER),
        "path_action_hashes": {
            policy: [proposal_direction_hash(item) for item in path.proposals]
            for policy, path in paths.items()
        },
        "per_hop_c_energy": hop_energy,
        "native_ordered_hash": proposal_direction_hash(ordered_w0),
        "native_split_action_hashes": [
            proposal_direction_hash(item) for item in native_path.proposals
        ],
        "lineage_ids": {
            policy: [lineage.lineage_id for lineage in path.lineages]
            for policy, path in paths.items()
        },
        "native_split_lineage_ids": [
            lineage.lineage_id for lineage in native_path.lineages
        ],
        "gate_choices": [
            panel["selection"] for panel in gated_path.panels[1:]
        ],
        "matched_per_hop_c": True,
        "common_first_hop_exact": True,
        "evaluation_unseen_at_commit": True,
    }
    action["commitment_hash"] = _feature_hash(action)
    receipt_name, receipt_hash = _commit_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action,
        expected_branch_order=AGATE_BRANCH_ORDER,
        feature_event="adaptive_gate_feature",
        action_event="adaptive_gate_action_commitment",
        receipt_schema=RECEIPT_SCHEMA,
    )

    def evaluate_frozen_target() -> Any:
        teacher, observed_ids = teacher_forced_rewrite_exact(runtime, request, contexts)
        target_lineage.assert_target_tokens(observed_ids)
        return rewrite_metrics(teacher, observed_ids)

    after_by_arm: dict[str, Any] = {
        NO_OP_REPLAY: evaluate_temporary_trajectory(
            model=runtime.model,
            origin_snapshot=origin_snapshot,
            evaluator=evaluate_frozen_target,
            cpu_rng=base_cpu_rng,
            cuda_rng=base_cuda_rng,
        )
    }
    for arm_id, path in (
        (GATED_FINAL, gated_path),
        (REFRESHED_FINAL, refreshed_path),
        (COEFFICIENT_FINAL, coefficient_path),
        (FIXED_FINAL, fixed_path),
    ):
        after_by_arm[arm_id] = _evaluate_prefix(
            model=runtime.model,
            origin_snapshot=origin_snapshot,
            proposals=path.proposals,
            descendant_snapshots=path.descendant_snapshots,
            prefix_length=QSTEP_K,
            evaluator=evaluate_frozen_target,
            cpu_rng=base_cpu_rng,
            cuda_rng=base_cuda_rng,
        )
    after_by_arm[NATIVE_ORDERED_FULL] = evaluate_temporary_trajectory(
        model=runtime.model,
        origin_snapshot=origin_snapshot,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
        first_proposal=ordered_w0,
    )
    after_by_arm[NATIVE_ORDERED_SPLIT4] = _evaluate_prefix(
        model=runtime.model,
        origin_snapshot=origin_snapshot,
        proposals=native_path.proposals,
        descendant_snapshots=native_path.descendant_snapshots,
        prefix_length=QSTEP_K,
        evaluator=evaluate_frozen_target,
        cpu_rng=base_cpu_rng,
        cuda_rng=base_cuda_rng,
    )
    if tuple(after_by_arm) != AGATE_BRANCH_ORDER:
        raise ContractError("adaptive-gate outcome branch order drifted")

    endpoint_energy = {
        NO_OP_REPLAY: 0.0,
        GATED_FINAL: gated_path.endpoint_energies[-1],
        REFRESHED_FINAL: refreshed_path.endpoint_energies[-1],
        COEFFICIENT_FINAL: coefficient_path.endpoint_energies[-1],
        FIXED_FINAL: fixed_path.endpoint_energies[-1],
        NATIVE_ORDERED_FULL: native_energy,
        NATIVE_ORDERED_SPLIT4: native_path.endpoint_energies[-1],
    }
    for index, arm_id in enumerate(AGATE_BRANCH_ORDER):
        after = after_by_arm[arm_id]
        outcome = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "editor_track": track,
            "feature_hash": feature["feature_hash"],
            "commitment_hash": action["commitment_hash"],
            "receipt_sha256": receipt_hash,
            "action_id": arm_id,
            "branch_index": index,
            "progress": _finite(f"{arm_id} progress", after.utility - baseline.utility),
            "endpoint_c_energy": endpoint_energy[arm_id],
            "success": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "controlled_nfe": 1,
            **_metric_difference(after, baseline),
        }
        _safe_payload(outcome)
        outcome_writer.write("adaptive_gate_outcome", outcome)

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != base_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("adaptive-gate event did not restore W0/RNG")

    analysis_case = {
        "model_alias": runtime.spec.alias,
        "editor_track": track,
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
            "precomputed_cache_only": True,
            "projector_read_only": True,
        },
        "branch_order": list(AGATE_BRANCH_ORDER),
        "arms": {
            arm_id: {
                "progress": _finite(
                    f"{arm_id} analysis progress",
                    after_by_arm[arm_id].utility - baseline.utility,
                ),
                "endpoint_c_energy": endpoint_energy[arm_id],
                "success": True,
            }
            for arm_id in AGATE_BRANCH_ORDER
        },
        "gate": {
            "relative_margin": GATE_RELATIVE_MARGIN,
            "choices": [panel["selection"] for panel in gated_path.panels[1:]],
            "normalized_predicted_advantages": [
                panel["normalized_predicted_advantage"]
                for panel in gated_path.panels[1:]
            ],
            "refreshed_predicted_scores": [
                panel["refreshed_predicted_score"] for panel in gated_path.panels[1:]
            ],
            "fixed_predicted_scores": [
                panel["fixed_predicted_score"] for panel in gated_path.panels[1:]
            ],
        },
        "geometry": {
            "gated_w0_to_refreshed_c_cosines": list(gated_path.c_cosines),
        },
        "budgets": {
            "native_c_energy": native_energy,
            "native_c_distance": native_distance,
            "hop_c_distance": hop_distance,
            "per_hop_c_energy": hop_energy,
            "probe_c_distance": probe_distance,
        },
        "compute": {
            "controlled_nfe": AGATE_CONTROLLED_NFE,
            "proposal_build_count": 8,
            "probe_panel_count": AGATE_PROBE_PANEL_COUNT,
            "wall_seconds": time.perf_counter() - event_started,
        },
    }
    _safe_payload(analysis_case)
    analysis_case_writer.write("adaptive_gate_analysis_case", analysis_case)
    return {
        "schema_version": EVENT_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "track": track,
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_compute_count": 1,
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(AGATE_BRANCH_ORDER),
        "analysis_case_count": 1,
        "receipt": {"name": receipt_name, "sha256": receipt_hash},
        "technical": dict(analysis_case["technical"]),
        "compute": dict(analysis_case["compute"]),
        "pass": True,
    }


def _validate_event_result(raw: Mapping[str, Any], request: EditRequest, track: str) -> dict[str, Any]:
    result = dict(raw)
    if (
        result.get("schema_version") != EVENT_SCHEMA
        or result.get("case_id") != request.case_id
        or result.get("request_id") != request.request_id
        or result.get("track") != track
        or result.get("feature_count") != 1
        or result.get("commitment_count") != 1
        or result.get("outcome_count") != len(AGATE_BRANCH_ORDER)
        or result.get("analysis_case_count") != 1
        or result.get("direct_z_compute_count") != 1
        or result.get("pass") is not True
    ):
        raise ContractError("adaptive-gate event result differs from contract")
    _full_hash("direct-z artifact", result["direct_z_artifact_sha256"])
    technical = result.get("technical")
    if not isinstance(technical, Mapping) or any(value is not True for value in technical.values()):
        raise ContractError("adaptive-gate technical gate is incomplete")
    compute = result.get("compute")
    if (
        not isinstance(compute, Mapping)
        or compute.get("controlled_nfe") != AGATE_CONTROLLED_NFE
        or compute.get("proposal_build_count") != 8
        or compute.get("probe_panel_count") != AGATE_PROBE_PANEL_COUNT
        or _finite("adaptive-gate wall", compute.get("wall_seconds")) < 0.0
    ):
        raise ContractError("adaptive-gate compute accounting drifted")
    return result


def run_adaptive_gate_refresh(
    *,
    easyedit_root: str | Path,
    track: str,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    started = time.perf_counter()
    _execution_envelope(track, model_alias, run_id)
    slurm = _slurm_state(track, model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_adaptive_gate_selection(root)
    requests = load_counterfact_requests(root, selection.case_ids)
    if len(requests) != AGATE_CASE_COUNT:
        raise ContractError("adaptive-gate request count differs from lock")
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("adaptive-gate bridge provenance is outside fixed manifest")

    projector_bank: AlphaEditProjectorBank | None = None
    try:
        with offline_environment():
            assert_tensor_sha256_device_parity(torch.device("cuda", 0))
            bindings = bridge.load()
            hparams = _load_hparams(root, spec, bindings)
            seed_runtime(AGATE_RUN_SEED)
            runtime = model_loader(model_alias)
            if runtime.spec != spec:
                raise MV1Error("adaptive-gate model loader returned different spec")
            contexts = _freeze_contexts(bridge, runtime, seed=AGATE_RUN_SEED)
            if track == TRACK_ALPHAEDIT:
                projector_bank = AlphaEditProjectorBank.open(root, spec)
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
                "schema_version": MANIFEST_SCHEMA,
                "run_id": run_id,
                "editor_track": track,
                "ode_edit_git": git_state,
                "slurm": slurm,
                "model": runtime.metadata(),
                "hparams_relative_path": spec.hparams_path,
                "selection": selection.to_dict(),
                "selected_case_ids": list(selection.case_ids),
                "contexts": {
                    "manifest_id": contexts.manifest_id,
                    "group_sizes": [len(group) for group in contexts.templates],
                    "raw_templates_persisted": False,
                },
                "provenance_id": provenance.manifest_id,
                "fixed_files": _relative_provenance(provenance, root),
                "projector": (
                    None if projector_bank is None else projector_bank.metadata()
                ),
                "constants": {
                    "layers": list(LAYERS),
                    "k": QSTEP_K,
                    "hop_fraction": QSTEP_HOP_FRACTION,
                    "probe_fraction_of_native_distance": QSTEP_PROBE_FRACTION,
                    "gate_relative_margin": GATE_RELATIVE_MARGIN,
                    "run_seed": AGATE_RUN_SEED,
                    "case_count": AGATE_CASE_COUNT,
                    "bootstrap_seed": AGATE_BOOTSTRAP_SEED,
                    "bootstrap_resamples": AGATE_BOOTSTRAP_RESAMPLES,
                    "practical_effect_floor": AGATE_PRACTICAL_FLOOR,
                    "branch_order": list(AGATE_BRANCH_ORDER),
                },
                "gate_policy": (
                    "at each current gated state, compare refreshed and transported-W0 "
                    "score-mix predicted scores with identical D/64 probes; refresh only "
                    "when normalized advantage exceeds 0.02"
                ),
                "alphaedit_policy": (
                    "for alphaedit_projected, transform every MEMIT low-rank actuator "
                    "as B@P using the pinned EasyEdit projector; mmap/read-only; no "
                    "projector or Wikipedia-stat recomputation"
                ),
                "information_firewall": (
                    "request and approved rewrite contexts only; no evaluation prompt "
                    "or endpoint outcome enters gate decisions"
                ),
                "expected_counts": {
                    "features": AGATE_CASE_COUNT,
                    "actions": AGATE_CASE_COUNT,
                    "outcomes": AGATE_CASE_COUNT * len(AGATE_BRANCH_ORDER),
                    "analysis_cases": AGATE_CASE_COUNT,
                    "events": AGATE_CASE_COUNT,
                    "receipts": AGATE_CASE_COUNT,
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
            results: list[dict[str, Any]] = []
            abort_failure_type: str | None = None
            with (
                SanitizedJsonlWriter(features_path, run_id, schema_version=STREAM_SCHEMA) as feature_writer,
                SanitizedJsonlWriter(actions_path, run_id, schema_version=STREAM_SCHEMA) as action_writer,
                SanitizedJsonlWriter(outcomes_path, run_id, schema_version=STREAM_SCHEMA) as outcome_writer,
                SanitizedJsonlWriter(analysis_cases_path, run_id, schema_version=STREAM_SCHEMA) as analysis_writer,
                SanitizedJsonlWriter(events_path, run_id, schema_version=STREAM_SCHEMA) as event_writer,
            ):
                for request in requests:
                    try:
                        raw = _run_event(
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
                            analysis_case_writer=analysis_writer,
                            track=track,
                            projector_bank=projector_bank,
                        )
                        result = _validate_event_result(raw, request, track)
                    except Exception as exc:
                        print(
                            f"AGATE_LOCAL_TRACEBACK_BEGIN type={type(exc).__name__}",
                            file=sys.stderr,
                        )
                        for frame, line_number in traceback.walk_tb(exc.__traceback__):
                            print(
                                f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                                file=sys.stderr,
                            )
                        if isinstance(exc, TensorHashRuntimeError):
                            print(
                                "AGATE_LOCAL_TENSOR_HASH_DIAGNOSTIC "
                                f"phase={exc.phase} category={exc.category}",
                                file=sys.stderr,
                            )
                        print("AGATE_LOCAL_TRACEBACK_END", file=sys.stderr)
                        result = {
                            "case_id": request.case_id,
                            "request_id": request.request_id,
                            "failure_type": type(exc).__name__,
                            "pass": False,
                        }
                        abort_failure_type = type(exc).__name__
                    results.append(result)
                    event_writer.write("adaptive_gate_case", result)
                    if abort_failure_type is not None:
                        break
                sequences = {
                    "features": feature_writer.sequence,
                    "actions": action_writer.sequence,
                    "outcomes": outcome_writer.sequence,
                    "analysis_cases": analysis_writer.sequence,
                    "events": event_writer.sequence,
                }
            if projector_bank is not None:
                projector_bank.assert_hash_current()
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
                    "outcomes": planned * len(AGATE_BRANCH_ORDER),
                    "analysis_cases": planned,
                    "events": planned,
                }
                and receipt_count == planned
            )
            summary = {
                "schema_version": SUMMARY_SCHEMA,
                "run_id": run_id,
                "model_alias": model_alias,
                "editor_track": track,
                "slurm": slurm,
                "provenance_id": provenance.manifest_id,
                "selection_manifest_id": selection.manifest_id,
                "context_id": contexts.manifest_id,
                "run_status": "aborted" if abort_failure_type else "completed",
                "abort_failure_type": abort_failure_type,
                "planned_case_count": planned,
                "attempted_case_count": attempted,
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
                        "case_id": request.case_id,
                        "status": "attempted" if index < attempted else "not_run_due_to_abort",
                        "pass": bool(results[index].get("pass", False)) if index < attempted else False,
                    }
                    for index, request in enumerate(requests)
                ],
                "constants": manifest["constants"],
                "stream_sequences": sequences,
                "expected_counts_exact": exact_counts,
                "all_rollbacks_exact": bool(
                    abort_failure_type is None
                    and attempted == planned
                    and pass_count == planned
                    and all(
                        bool(result.get("technical", {}).get("rollback_exact", False))
                        for result in results
                    )
                ),
                "covariance_files_loaded": list(loaded_covariances),
                "projector_files_loaded": (
                    [] if projector_bank is None else [projector_bank.path.name]
                ),
                "projector_sha256": (
                    None if projector_bank is None else projector_bank.identity.sha256
                ),
                "direct_z_artifact_count": (
                    len(tuple(direct_z_root.glob("*.pt"))) if direct_z_root.is_dir() else 0
                ),
                "resource": {
                    "wall_seconds": time.perf_counter() - started,
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
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
    finally:
        if projector_bank is not None:
            projector_bank.assert_stat_current()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-adaptive-gate-refresh",
        description="Run the Motivation-closing adaptive direction gate.",
        allow_abbrev=False,
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_adaptive_gate_refresh(
            easyedit_root=args.easyedit_root,
            track=args.track,
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
