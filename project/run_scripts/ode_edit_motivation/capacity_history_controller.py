"""Controller-only runner for the capacity/history Motivation experiment.

Four branches share one request order and one model-independent policy:

* canonical ordered MEMIT;
* cumulative capacity-aware MEMIT;
* canonical-history AlphaEdit;
* cumulative capacity-aware canonical-history AlphaEdit.

The controller never imports or decodes CounterFact evaluation fields.  It
uses only rewrite-authorized prompts, current model state, pinned covariance,
and (for AlphaEdit) the pinned mmap projector plus branch-local key history.
All proposal tensors remain under ignored ``local/`` and are committed before
a separate evaluator process is allowed to load evaluation-only fields.
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
from typing import Any, Callable, Mapping, Sequence

import torch

from .alphaedit_history import AlphaEditHistoryBank
from .alphaedit_proposal_adapter import AlphaEditProposalAdapter
from .alphaedit_reference import AlphaEditSolverConfig, load_alphaedit_solver_config
from .capacity_geometry import compute_w0_denominators
from .capacity_history_analysis import (
    BRANCH_ALPHA_NATIVE,
    BRANCH_ALPHA_QP,
    BRANCH_MEMIT_NATIVE,
    BRANCH_MEMIT_QP,
    BRANCHES,
    CHAIN_LENGTH,
)
from .capacity_qp import (
    build_capacity_layer_terms,
    build_capacity_qp_proposal,
    capacity_state_by_layer,
    solve_capacity_qp,
)
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
    TemporaryExactMemitApplication,
    assert_snapshot_current,
    assert_tensor_sha256_device_parity,
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
from .microseq_artifacts import (
    apply_proposal_in_disposable_process,
    write_proposal_artifact,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
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
    _feature_hash,
    _layer_by_weight,
    _state_identity,
    build_unit_c_actions,
    proposal_c_energy,
    rewrite_metrics,
    teacher_forced_rewrite_exact,
)
from .projector_adapter import AlphaEditProjectorBank
from .quarter_step_refresh import (
    LAYERS,
    QSTEP_PROBE_FRACTION,
    _capture_source_snapshot,
    _commit_action,
    _proposal_panel,
)
from .trajectory import MATCHED_C_ABS_TOL, MATCHED_C_REL_TOL


CONTROLLER_SCHEMA = "ode-edit-capacity-history-controller/v1"
CONTROLLER_STREAM_SCHEMA = "ode-edit-capacity-history-controller-stream/v1"
CONTROLLER_RECEIPT_SCHEMA = "ode-edit-capacity-history-controller-receipt/v1"
CONTROLLER_JOB_NAME = "odeedit_capacity_history_pair_c1_v3"
RUN_SEED = 41
RANK_START = 144
RANK_STOP = 148
POLICY_ID = "capacity-qp-history-k4-native-progress-v2"
MAX_ACCEPTED_ROUNDS = 4
INITIAL_TRUST_FRACTION = 0.25
RETRY_SHRINK = 0.5
MIN_TRUST_RATIO = 0.05
NATIVE_REFERENCE_ABS_TOL = 1e-4
APPLICATION_MODES = {branch: "easyedit_exact" for branch in BRANCHES}
RUN_IDS = {
    BRANCH_MEMIT_NATIVE: {
        "llama3-8b-inst": "caphist_memit_native_llama_c1_v3",
        "qwen2.5-7b-inst": "caphist_memit_native_qwen_c1_v3",
    },
    BRANCH_MEMIT_QP: {
        "llama3-8b-inst": "caphist_memit_qp_llama_c1_v3",
        "qwen2.5-7b-inst": "caphist_memit_qp_qwen_c1_v3",
    },
    BRANCH_ALPHA_NATIVE: {
        "llama3-8b-inst": "caphist_alpha_native_llama_c1_v3",
        "qwen2.5-7b-inst": "caphist_alpha_native_qwen_c1_v3",
    },
    BRANCH_ALPHA_QP: {
        "llama3-8b-inst": "caphist_alpha_qp_llama_c1_v3",
        "qwen2.5-7b-inst": "caphist_alpha_qp_qwen_c1_v3",
    },
}


def sanitized_traceback_frames(exc: BaseException) -> list[dict[str, Any]]:
    """Return code locations without persisting exception text or runtime data."""

    repository_root = Path(__file__).resolve().parents[3]
    frames: list[dict[str, Any]] = []
    for frame, line_number in traceback.walk_tb(exc.__traceback__):
        source = Path(frame.f_code.co_filename)
        try:
            source_label = str(source.resolve().relative_to(repository_root))
        except (OSError, ValueError):
            source_label = source.name
        frames.append(
            {
                "source": source_label,
                "function": frame.f_code.co_name,
                "line": int(line_number),
            }
        )
    return frames


def is_alpha_branch(branch: str) -> bool:
    return branch in {BRANCH_ALPHA_NATIVE, BRANCH_ALPHA_QP}


def is_qp_branch(branch: str) -> bool:
    return branch in {BRANCH_MEMIT_QP, BRANCH_ALPHA_QP}


def policy_parameters() -> dict[str, Any]:
    """Return the exact model-common controller policy."""

    return {
        "policy_id": POLICY_ID,
        "branches": list(BRANCHES),
        "application_modes": dict(APPLICATION_MODES),
        "layers": list(LAYERS),
        "chain_length": CHAIN_LENGTH,
        "seed": RUN_SEED,
        "max_accepted_rounds": MAX_ACCEPTED_ROUNDS,
        "initial_trust_fraction": INITIAL_TRUST_FRACTION,
        "retry_shrink": RETRY_SHRINK,
        "probe_fraction": QSTEP_PROBE_FRACTION,
        "minimum_trust_ratio": MIN_TRUST_RATIO,
        "native_reference_abs_tolerance": NATIVE_REFERENCE_ABS_TOL,
        "progress_request": "remaining-ordered-native-rewrite-utility-gap",
        "terminal": "ordered-native-rewrite-utility-matched-after-accepted-round",
        "first_hit": "diagnostic-only-never-terminal",
        "rejection": "positive-rewrite-gain-and-trust-ratio-else-one-shrunk-retry",
        "retry_failure": "fail-closed-without-scientific-endpoint",
        "nominal_native_distance_budget": 1.0,
        "history_append_policy": "post-accepted-edit-once-history-fixed-within-edit",
        "capacity_normalizer": "fixed-W0-layer-weight-C-energy",
        "evaluation_fields_available": False,
    }


def accept_round(*, rewrite_gain: float, predicted_gain: float) -> tuple[bool, float]:
    """Apply the fixed rewrite-only trust-ratio acceptance rule."""

    actual = float(rewrite_gain)
    predicted = float(predicted_gain)
    if not math.isfinite(actual) or not math.isfinite(predicted) or predicted <= 0.0:
        return False, float("-inf")
    ratio = actual / predicted
    return bool(actual > 0.0 and ratio >= MIN_TRUST_RATIO), ratio


def native_reference_reached(*, utility: float, reference: float) -> bool:
    """Return whether the controller has matched its rewrite-only native endpoint."""

    observed = float(utility)
    target = float(reference)
    if not math.isfinite(observed) or not math.isfinite(target):
        raise ContractError("native rewrite reference must be finite")
    return observed >= target - NATIVE_REFERENCE_ABS_TOL


def requested_native_progress(*, utility: float, reference: float) -> float:
    """Request the entire remaining native utility gap, never a fixed fraction."""

    observed = float(utility)
    target = float(reference)
    if not math.isfinite(observed) or not math.isfinite(target):
        raise ContractError("native rewrite progress inputs must be finite")
    return max(0.0, target - observed)


def capacity_probe_action_ids(actions: Sequence[Any]) -> tuple[str, ...]:
    """Bind the QP probe to its five layer actuators, excluding uniform."""

    observed = tuple(str(action.action_id) for action in actions)
    expected = tuple(f"layer_{layer}" for layer in LAYERS)
    if observed != expected:
        raise ContractError("capacity QP unit-action order differs")
    return expected


@dataclass(frozen=True, slots=True)
class CapacityHistorySelection:
    source_sha256: str
    source_size: int
    source_row_count: int
    case_ids: tuple[str, ...]
    prior_order_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.case_ids) != CHAIN_LENGTH
            or len(set(self.case_ids)) != CHAIN_LENGTH
            or self.source_row_count < RANK_STOP
            or self.source_size <= 0
        ):
            raise ContractError("capacity/history selection metadata is invalid")

    @property
    def order_hash(self) -> str:
        return sha256_bytes(canonical_json(list(self.case_ids)).encode("utf-8"))

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(False)).encode("utf-8"))

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "ode-edit-capacity-history-selection/v1",
            "seed": DEFAULT_SELECTION_SEED,
            "rank_slice": {
                "start": RANK_START,
                "stop": RANK_STOP,
                "count": CHAIN_LENGTH,
            },
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "case_ids": list(self.case_ids),
            "order_hash": self.order_hash,
            "prior_order_hash": self.prior_order_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def _salted_rank(case_id: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            DEFAULT_SELECTION_SEED.encode("utf-8")
            + b"\0"
            + str(case_id).encode("utf-8")
        ).digest(),
        str(case_id),
    )


def select_capacity_history_cases(
    all_case_ids: Sequence[str], *, canonical: CounterFactSelectionManifest
) -> CapacityHistorySelection:
    ranked = tuple(sorted((str(case_id) for case_id in all_case_ids), key=_salted_rank))
    if (
        len(ranked) != canonical.source_row_count
        or len(set(ranked)) != len(ranked)
        or tuple(ranked[:100]) != canonical.ordered_case_ids
    ):
        raise ContractError("capacity/history salted order differs from canonical evidence")
    return CapacityHistorySelection(
        source_sha256=canonical.source_sha256,
        source_size=canonical.source_size,
        source_row_count=canonical.source_row_count,
        case_ids=ranked[RANK_START:RANK_STOP],
        prior_order_hash=sha256_bytes(
            canonical_json(list(ranked[:RANK_START])).encode("utf-8")
        ),
    )


def generate_capacity_history_selection(
    easyedit_root: str | Path,
) -> CapacityHistorySelection:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(root, seed=DEFAULT_SELECTION_SEED)
    all_case_ids = scan_counterfact_case_ids(root / "data/counterfact/counterfact.json")
    return select_capacity_history_cases(all_case_ids, canonical=canonical)


def _execution_envelope(branch: str, model_alias: str, run_id: str) -> None:
    if (
        branch not in BRANCHES
        or model_alias not in MODEL_SPECS
        or run_id != RUN_IDS[branch][model_alias]
    ):
        raise ContractError("capacity/history controller identity differs from precommit")


def _slurm_state(branch: str, model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(branch, model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "role": os.environ.get("ODEEDIT_CAPACITY_PROCESS_ROLE"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise ContractError("partial capacity/history Slurm identity is forbidden")
    if (
        values["job_name"] != CONTROLLER_JOB_NAME
        or values["node"] != "devbox"
        or values["role"] != "controller"
        or not str(values["job_id"]).isdigit()
    ):
        raise ContractError("capacity/history Slurm identity differs from fixed envelope")
    return {"under_slurm": True, **values}


def _proposal_artifact_metadata(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "sha256": _file_sha256(path),
            "tensor_name": path.with_suffix(".pt").name,
            "tensor_sha256": _file_sha256(path.with_suffix(".pt")),
        }
        for path in paths
    ]


def _rewrite_state(
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
) -> tuple[float, bool]:
    teacher, target_ids = teacher_forced_rewrite_exact(runtime, request, contexts)
    metrics = rewrite_metrics(teacher, target_ids)
    return float(metrics.utility), bool(metrics.exact_satisfied)


def _temporary_descendant(
    *,
    runtime: FixedModelRuntime,
    proposal: MemitFactorProposal,
    request: EditRequest,
    contexts: Any,
    hparams: Any,
    weight_names: Sequence[str],
    provenance_id: str,
) -> tuple[SnapshotManifest, float, bool]:
    try:
        with TemporaryExactMemitApplication(runtime.model, proposal):
            utility, first_hit = _rewrite_state(runtime, request, contexts)
            child = _capture_source_snapshot(
                runtime=runtime,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=provenance_id,
            )
            return child, utility, first_hit
    finally:
        assert_snapshot_current(runtime.model, proposal.snapshot)


def _alpha_adapter(
    *,
    bridge: EasyEditBridge,
    config: AlphaEditSolverConfig,
    projector_bank: AlphaEditProjectorBank,
    history_bank: AlphaEditHistoryBank,
    runtime: FixedModelRuntime,
    request: EditRequest,
    hparams: Any,
    contexts: Any,
    direct_z: Any,
    target_ids: torch.Tensor,
    provenance_id: str,
) -> AlphaEditProposalAdapter:
    return AlphaEditProposalAdapter(
        bridge=bridge,
        config=config,
        projector_bank=projector_bank,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        request=request,
        memit_hparams=hparams,
        contexts=contexts,
        model_id=runtime.spec.snapshot_name,
        direct_z=direct_z,
        target_token_ids=target_ids,
        provenance_id=provenance_id,
        history_bank=history_bank,
    )


def _ordered_proposal(
    *,
    branch: str,
    bridge: EasyEditBridge,
    adapter: AlphaEditProposalAdapter | None,
    runtime: FixedModelRuntime,
    request: EditRequest,
    hparams: Any,
    contexts: Any,
    direct_z: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    lineage: FrozenTargetLineage,
) -> MemitFactorProposal:
    if is_alpha_branch(branch):
        if adapter is None:
            raise ContractError("Alpha branch has no proposal adapter")
        with _silence_upstream():
            return adapter.propose_ordered(
                origin_lineage=lineage,
                construction="genuine-p-inside-solve-history",
                solver_suffix="capacity-history-native-reference",
            ).proposal
    with _silence_upstream():
        return bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )


def _synchronous_proposal(
    *,
    branch: str,
    bridge: EasyEditBridge,
    adapter: AlphaEditProposalAdapter | None,
    runtime: FixedModelRuntime,
    request: EditRequest,
    hparams: Any,
    contexts: Any,
    direct_z: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    lineage: FrozenTargetLineage,
    round_index: int,
) -> MemitFactorProposal:
    if is_alpha_branch(branch):
        if adapter is None:
            raise ContractError("Alpha branch has no proposal adapter")
        with _silence_upstream():
            return adapter.propose_synchronous(
                frozen_target_lineage=lineage,
                construction="genuine-p-inside-solve-history",
                solver_suffix=f"capacity-round-{round_index}",
            ).proposal
    with _silence_upstream():
        return bridge.propose_synchronous_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            direct_z,
            covariance_specs,
            model_id=runtime.spec.snapshot_name,
            frozen_target_lineage=lineage,
        )


def _build_edit_action(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    w0_denominators: Mapping[int, float],
    cumulative_factors: list[LowRankFactor],
    direct_z_root: Path,
    proposal_root: Path,
    receipt_root: Path,
    request: EditRequest,
    edit_index: int,
    branch: str,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    history_bank: AlphaEditHistoryBank | None,
    alpha_config: AlphaEditSolverConfig | None,
    projector_bank: AlphaEditProjectorBank | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_runtime(_event_seed(RUN_SEED, request.case_id))
    layer_by_weight = _layer_by_weight(hparams)
    if tuple(int(layer) for layer in hparams.layers) != LAYERS:
        raise ContractError("capacity/history controller requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    origin_hashes = _weight_hashes(runtime.model, weight_names)
    origin_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()
    _, target_ids = teacher_forced_rewrite_exact(runtime, request, contexts)
    if (
        _weight_hashes(runtime.model, weight_names) != origin_hashes
        or _state_identity(runtime) != origin_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise ContractError("rewrite-only tokenization changed controller entry state")

    bindings = bridge.load()
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
    lineage = FrozenTargetLineage.start(
        direct_z=direct_z,
        origin_snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    lineage.assert_authorizes(
        direct_z=direct_z,
        snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    history_before = history_bank.to_dict() if history_bank is not None else None
    adapter: AlphaEditProposalAdapter | None = None
    if is_alpha_branch(branch):
        if history_bank is None or alpha_config is None or projector_bank is None:
            raise ContractError("Alpha branch precomputed/history state is incomplete")
        adapter = _alpha_adapter(
            bridge=bridge,
            config=alpha_config,
            projector_bank=projector_bank,
            history_bank=history_bank,
            runtime=runtime,
            request=request,
            hparams=hparams,
            contexts=contexts,
            direct_z=direct_z,
            target_ids=target_ids,
            provenance_id=bindings.provenance.manifest_id,
        )

    ordered = _ordered_proposal(
        branch=branch,
        bridge=bridge,
        adapter=adapter,
        runtime=runtime,
        request=request,
        hparams=hparams,
        contexts=contexts,
        direct_z=direct_z,
        covariance_specs=covariance_specs,
        lineage=lineage,
    )
    native_energy = proposal_c_energy(ordered, covariance_moments, layer_by_weight)
    native_distance = math.sqrt(native_energy)
    if not math.isfinite(native_distance) or native_distance <= 0.0:
        raise ContractError("capacity/history native C-distance is invalid")

    origin_utility, _origin_first_hit = _rewrite_state(runtime, request, contexts)
    native_child, native_reference_utility, native_reference_first_hit = (
        _temporary_descendant(
            runtime=runtime,
            proposal=ordered,
            request=request,
            contexts=contexts,
            hparams=hparams,
            weight_names=weight_names,
            provenance_id=bindings.provenance.manifest_id,
        )
    )
    if requested_native_progress(
        utility=origin_utility,
        reference=native_reference_utility,
    ) <= NATIVE_REFERENCE_ABS_TOL:
        raise ContractError("ordered native endpoint has no positive rewrite reference gap")

    proposals: list[MemitFactorProposal] = []
    descendants: list[SnapshotManifest] = []
    round_diagnostics: list[dict[str, Any]] = []
    proposal_build_count = 1
    rejected_round_count = 0
    first_hit_reached = False
    accepted_path_distance = 0.0
    endpoint_utility = origin_utility
    matched_native_reference = False

    if not is_qp_branch(branch):
        proposals.append(ordered)
        descendants.append(native_child)
        accepted_path_distance = native_distance
        endpoint_utility = native_reference_utility
        first_hit_reached = native_reference_first_hit
        matched_native_reference = True
        apply_proposal_in_disposable_process(
            runtime.model,
            ordered,
            application_mode=APPLICATION_MODES[branch],
        )
        observed = _capture_source_snapshot(
            runtime=runtime,
            request=request,
            contexts=contexts,
            hparams=hparams,
            weight_names=weight_names,
            provenance_id=bindings.provenance.manifest_id,
        )
        if observed.state_id != native_child.state_id:
            raise ContractError("native permanent endpoint differs from temporary descendant")
        cumulative_factors.extend(ordered.factors)
    else:
        current_snapshot = origin_snapshot
        for round_index in range(1, MAX_ACCEPTED_ROUNDS + 1):
            synchronous = _synchronous_proposal(
                branch=branch,
                bridge=bridge,
                adapter=adapter,
                runtime=runtime,
                request=request,
                hparams=hparams,
                contexts=contexts,
                direct_z=direct_z,
                covariance_specs=covariance_specs,
                lineage=lineage,
                round_index=round_index,
            )
            proposal_build_count += 1
            unit_actions = build_unit_c_actions(
                synchronous, covariance_moments, layer_by_weight
            )[:-1]
            probe_action_ids = capacity_probe_action_ids(unit_actions)
            state_hashes = _weight_hashes(runtime.model, weight_names)
            state_identity = _state_identity(runtime)
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(0).clone()
            panel = _proposal_panel(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                actions=unit_actions,
                epsilon=native_distance * QSTEP_PROBE_FRACTION,
                base_hashes=state_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
                panel_label=f"capacity-t{edit_index}-round-{round_index}",
                expected_action_ids=probe_action_ids,
            )
            before_utility, _ = _rewrite_state(runtime, request, contexts)
            remaining_reference_gain = requested_native_progress(
                utility=before_utility,
                reference=native_reference_utility,
            )
            if remaining_reference_gain <= NATIVE_REFERENCE_ABS_TOL:
                matched_native_reference = True
                endpoint_utility = before_utility
                break
            accepted: tuple[
                MemitFactorProposal,
                SnapshotManifest,
                FrozenTargetLineage,
                float,
                bool,
                float,
                bool,
                dict[str, Any],
            ] | None = None
            for attempt_index, trust_fraction in enumerate(
                (INITIAL_TRUST_FRACTION, INITIAL_TRUST_FRACTION * RETRY_SHRINK),
                start=1,
            ):
                trust_distance = native_distance * trust_fraction
                terms = build_capacity_layer_terms(
                    cumulative_factors=cumulative_factors,
                    unit_actions=unit_actions,
                    slopes=panel.scores,
                    covariance_by_layer=covariance_moments,
                    layer_by_weight=layer_by_weight,
                    w0_denominators=w0_denominators,
                    trust_distance=trust_distance,
                )
                cap_upper = math.fsum(
                    term.slope * term.coefficient_cap for term in terms
                )
                maximum = solve_capacity_qp(
                    terms,
                    requested_progress=cap_upper,
                    trust_distance=trust_distance,
                )
                solution = solve_capacity_qp(
                    terms,
                    requested_progress=remaining_reference_gain,
                    trust_distance=trust_distance,
                )
                if solution.predicted_progress <= 1e-12:
                    raise ContractError("capacity QP has no positive rewrite direction")
                proposal = build_capacity_qp_proposal(
                    synchronous=synchronous,
                    unit_actions=unit_actions,
                    terms=terms,
                    solution=solution,
                    solver_suffix=f"t{edit_index}-r{round_index}-a{attempt_index}",
                )
                if proposal is None:
                    raise ContractError("capacity QP produced an empty action")
                observed_energy = proposal_c_energy(
                    proposal, covariance_moments, layer_by_weight
                )
                if not math.isclose(
                    observed_energy,
                    solution.coefficient_norm * solution.coefficient_norm,
                    rel_tol=MATCHED_C_REL_TOL,
                    abs_tol=MATCHED_C_ABS_TOL,
                ):
                    raise ContractError("capacity QP proposal C-energy differs from coefficients")
                with TemporaryExactMemitApplication(runtime.model, proposal) as application:
                    after_utility, first_hit = _rewrite_state(runtime, request, contexts)
                    child = _capture_source_snapshot(
                        runtime=runtime,
                        request=request,
                        contexts=contexts,
                        hparams=hparams,
                        weight_names=weight_names,
                        provenance_id=bindings.provenance.manifest_id,
                    )
                    rewrite_gain = after_utility - before_utility
                    reference_reached = native_reference_reached(
                        utility=after_utility,
                        reference=native_reference_utility,
                    )
                    accepted_flag, trust_ratio = accept_round(
                        rewrite_gain=rewrite_gain,
                        predicted_gain=solution.predicted_progress,
                    )
                    next_lineage = None
                    if accepted_flag:
                        next_lineage = lineage.derive_adaptive_step(
                            parent_snapshot=current_snapshot,
                            child_snapshot=child,
                            application=application,
                            step_scale=solution.coefficient_norm / native_distance,
                            label=f"capacity_round_{round_index}",
                        )
                diagnostic = {
                    "round_index": round_index,
                    "attempt_index": attempt_index,
                    "trust_fraction": trust_fraction,
                    "utility_before": before_utility,
                    "native_reference_utility": native_reference_utility,
                    "remaining_reference_gain_before": remaining_reference_gain,
                    "maximum_predicted_gain": maximum.constrained_progress,
                    "requested_gain": solution.requested_progress,
                    "predicted_gain": solution.predicted_progress,
                    "rewrite_gain": rewrite_gain,
                    "trust_ratio": trust_ratio,
                    "accepted": accepted_flag,
                    "coefficient_norm": solution.coefficient_norm,
                    "progress_slack": solution.progress_slack,
                    "feasible_without_slack": solution.feasible_without_slack,
                    "coefficients": list(solution.coefficients),
                    "capacity_terms": [term.to_dict() for term in terms],
                    "capacity_before": capacity_state_by_layer(
                        cumulative_factors=cumulative_factors,
                        covariance_by_layer=covariance_moments,
                        layer_by_weight=layer_by_weight,
                        w0_denominators=w0_denominators,
                    ),
                    "proposal_direction_sha256": proposal_direction_hash(proposal),
                    "first_hit_reached": first_hit,
                    "native_reference_reached": reference_reached,
                }
                round_diagnostics.append(diagnostic)
                if accepted_flag:
                    if next_lineage is None:
                        raise ContractError("accepted QP round has no target lineage")
                    accepted = (
                        proposal,
                        child,
                        next_lineage,
                        solution.coefficient_norm,
                        first_hit,
                        after_utility,
                        reference_reached,
                        diagnostic,
                    )
                    break
                rejected_round_count += 1
            if accepted is None:
                raise ContractError("capacity QP rejected both trust attempts")
            (
                proposal,
                child,
                lineage,
                distance,
                first_hit_reached,
                endpoint_utility,
                matched_native_reference,
                _diagnostic,
            ) = accepted
            apply_proposal_in_disposable_process(
                runtime.model,
                proposal,
                application_mode=APPLICATION_MODES[branch],
            )
            observed = _capture_source_snapshot(
                runtime=runtime,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=bindings.provenance.manifest_id,
            )
            if observed.state_id != child.state_id:
                raise ContractError("QP permanent endpoint differs from accepted descendant")
            lineage.assert_authorizes(
                direct_z=direct_z,
                snapshot=observed,
                target_token_ids=target_ids,
            )
            proposals.append(proposal)
            descendants.append(child)
            cumulative_factors.extend(proposal.factors)
            accepted_path_distance += distance
            current_snapshot = child
            if matched_native_reference:
                break

    if not proposals or len(proposals) != len(descendants):
        raise ContractError("capacity/history edit has no accepted endpoint")
    history_append = None
    if adapter is not None:
        history_append = adapter.append_current_post_edit_keys(edit_id=request.case_id)
    history_after = history_bank.to_dict() if history_bank is not None else None
    if history_bank is not None and history_bank.edit_count != edit_index:
        raise ContractError("Alpha history append count differs from edit index")

    action_ids = tuple(
        f"t{edit_index:02d}-step{step_index:02d}"
        for step_index in range(1, len(proposals) + 1)
    )
    artifact_paths = tuple(
        write_proposal_artifact(
            proposal_root,
            action_id=action_id,
            proposal=proposal,
        )
        for action_id, proposal in zip(action_ids, proposals, strict=True)
    )
    feature = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "controller_branch": branch,
        "policy_id": POLICY_ID,
        "application_mode": APPLICATION_MODES[branch],
        "origin_state_identity": origin_state_identity,
        "origin_parameter_hashes": origin_hashes,
        "target_identity": {
            "origin_lineage_id": FrozenTargetLineage.start(
                direct_z=direct_z,
                origin_snapshot=origin_snapshot,
                target_token_ids=target_ids,
            ).lineage_id,
            "direct_z_tensor_sha256": direct_z.tensor_sha256,
            "direct_z_artifact_sha256": direct_z.artifact.sha256,
            "target_token_sha256": FrozenTargetLineage.start(
                direct_z=direct_z,
                origin_snapshot=origin_snapshot,
                target_token_ids=target_ids,
            ).target_token_sha256,
            "direct_z_compute_count": 1,
        },
        "native_c_energy": native_energy,
        "native_c_distance": native_distance,
        "probe_c_distance": native_distance * QSTEP_PROBE_FRACTION,
        "accepted_path_distance": accepted_path_distance,
        "native_reference": {
            "origin_utility": origin_utility,
            "ordered_endpoint_utility": native_reference_utility,
            "ordered_endpoint_first_hit": native_reference_first_hit,
            "controller_endpoint_utility": endpoint_utility,
            "matched": matched_native_reference,
            "abs_tolerance": NATIVE_REFERENCE_ABS_TOL,
            "budget_exhausted": bool(
                is_qp_branch(branch)
                and not matched_native_reference
                and len(proposals) == MAX_ACCEPTED_ROUNDS
            ),
        },
        "proposal_artifacts": _proposal_artifact_metadata(artifact_paths),
        "round_diagnostics": round_diagnostics,
        "history_before": history_before,
        "history_after": history_after,
        "history_append": history_append.to_dict() if history_append is not None else None,
        "controller_only": True,
        "outcome_fields_loaded": False,
    }
    feature["feature_hash"] = _feature_hash(feature)
    action_payload = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "controller_branch": branch,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(action_ids),
        "path_action_hashes": {
            branch: [proposal_direction_hash(proposal) for proposal in proposals]
        },
        "proposal_entry_state_ids": [proposal.entry_snapshot_id for proposal in proposals],
        "proposal_descendant_state_ids": [snapshot.state_id for snapshot in descendants],
        "per_hop_c_energy": [
            proposal_c_energy(proposal, covariance_moments, layer_by_weight)
            for proposal in proposals
        ],
        "application_mode": APPLICATION_MODES[branch],
        "policy_id": POLICY_ID,
        "all_actions_before_outcomes": True,
    }
    action_payload["commitment_hash"] = _feature_hash(action_payload)
    receipt_name, receipt_sha256 = _commit_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action_payload,
        expected_branch_order=action_ids,
        feature_event="capacity_history_controller_feature",
        action_event="capacity_history_controller_action",
        receipt_schema=CONTROLLER_RECEIPT_SCHEMA,
    )
    return {
        "schema_version": CONTROLLER_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "branch": branch,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action_payload["commitment_hash"],
        "receipt_name": receipt_name,
        "receipt_sha256": receipt_sha256,
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_compute_count": 1,
        "proposal_count": len(proposals),
        "proposal_build_count": proposal_build_count,
        "accepted_round_count": len(proposals),
        "rejected_round_count": rejected_round_count,
        "accepted_path_distance": accepted_path_distance,
        "first_hit_reached": first_hit_reached,
        "native_reference_utility": native_reference_utility,
        "endpoint_utility": endpoint_utility,
        "native_reference_reached": matched_native_reference,
        "terminal_state_id": descendants[-1].state_id,
        "history_edit_count": history_bank.edit_count if history_bank is not None else 0,
        "controller_only": True,
        "outcome_fields_loaded": False,
        "wall_seconds": time.perf_counter() - started,
        "pass": True,
    }


def run_capacity_history_controller(
    *,
    easyedit_root: str | Path,
    branch: str,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    started = time.perf_counter()
    _execution_envelope(branch, model_alias, run_id)
    slurm = _slurm_state(branch, model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_capacity_history_selection(root)
    requests = load_counterfact_requests(root, selection.case_ids)
    if len(requests) != CHAIN_LENGTH:
        raise ContractError("capacity/history controller requires exact four requests")
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise ContractError("capacity/history bridge provenance is outside fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise ContractError("capacity/history model loader returned different spec")
        contexts = _freeze_contexts(bridge, runtime, seed=RUN_SEED)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root, runtime=runtime, specs=covariance_specs
        )
        layer_by_weight = _layer_by_weight(hparams)
        w0_denominators = compute_w0_denominators(
            runtime.model,
            covariance_moments,
            layer_by_weight,
            expected_layers=LAYERS,
        )
        history_bank = AlphaEditHistoryBank(LAYERS) if is_alpha_branch(branch) else None
        alpha_config: AlphaEditSolverConfig | None = None
        alpha_reference = None
        projector_bank: AlphaEditProjectorBank | None = None
        if is_alpha_branch(branch):
            alpha_config, alpha_reference = load_alphaedit_solver_config(
                root, model_alias, memit_hparams=hparams
            )
            projector_bank = AlphaEditProjectorBank.open(root, spec)

        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        direct_z_root = destination / "direct_z"
        proposal_root = destination / "proposal_artifacts"
        receipt_root = destination / "action_receipts"
        direct_z_root.mkdir(mode=0o700)
        proposal_root.mkdir(mode=0o700)
        receipt_root.mkdir(mode=0o700)
        parameters = policy_parameters()
        manifest = {
            "schema_version": CONTROLLER_SCHEMA,
            "run_id": run_id,
            "controller_branch": branch,
            "application_mode": APPLICATION_MODES[branch],
            "policy_id": POLICY_ID,
            "policy_parameters": parameters,
            "policy_parameters_sha256": sha256_bytes(
                canonical_json(parameters).encode("utf-8")
            ),
            "ode_edit_git": git_state,
            "slurm": slurm,
            "model": runtime.metadata(),
            "selection": selection.to_dict(),
            "contexts": {
                "manifest_id": contexts.manifest_id,
                "group_sizes": [len(group) for group in contexts.templates],
                "raw_templates_persisted": False,
            },
            "provenance_id": provenance.manifest_id,
            "fixed_files": _relative_provenance(provenance, root),
            "precomputed": {
                "covariances": list(loaded_covariances),
                "covariance_write_or_recompute": False,
                "projector": projector_bank.metadata() if projector_bank is not None else None,
                "projector_write_or_recompute": False,
                "alpha_reference_manifest_id": (
                    alpha_reference.manifest_id if alpha_reference is not None else None
                ),
                "alpha_solver_config_id": (
                    alpha_config.config_id if alpha_config is not None else None
                ),
            },
            "firewall": {
                "controller_only": True,
                "outcome_fields_loaded": False,
                "separate_evaluator_required": True,
            },
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)

        torch.cuda.reset_peak_memory_stats(0)
        cumulative_factors: list[LowRankFactor] = []
        results: list[dict[str, Any]] = []
        failure_type: str | None = None
        with (
            SanitizedJsonlWriter(
                features_path, run_id, schema_version=CONTROLLER_STREAM_SCHEMA
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path, run_id, schema_version=CONTROLLER_STREAM_SCHEMA
            ) as action_writer,
            SanitizedJsonlWriter(
                events_path, run_id, schema_version=CONTROLLER_STREAM_SCHEMA
            ) as event_writer,
        ):
            for edit_index, request in enumerate(requests, start=1):
                try:
                    result = _build_edit_action(
                        runtime=runtime,
                        bridge=bridge,
                        hparams=hparams,
                        contexts=contexts,
                        covariance_specs=covariance_specs,
                        covariance_moments=covariance_moments,
                        w0_denominators=w0_denominators,
                        cumulative_factors=cumulative_factors,
                        direct_z_root=direct_z_root,
                        proposal_root=proposal_root,
                        receipt_root=receipt_root,
                        request=request,
                        edit_index=edit_index,
                        branch=branch,
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        history_bank=history_bank,
                        alpha_config=alpha_config,
                        projector_bank=projector_bank,
                    )
                except Exception as exc:
                    failure_frames = sanitized_traceback_frames(exc)
                    print(
                        canonical_json(
                            {
                                "event": "capacity_history_controller_failure",
                                "error_type": type(exc).__name__,
                                "message_persisted": False,
                                "traceback_frames": failure_frames,
                            }
                        ),
                        file=sys.stderr,
                    )
                    result = {
                        "case_id": request.case_id,
                        "edit_index": edit_index,
                        "failure_type": type(exc).__name__,
                        "failure_message_persisted": False,
                        "failure_frames": failure_frames,
                        "pass": False,
                    }
                    failure_type = type(exc).__name__
                results.append(result)
                event_writer.write("capacity_history_controller_edit", result)
                if failure_type is not None:
                    break
            sequences = {
                "features": feature_writer.sequence,
                "actions": action_writer.sequence,
                "events": event_writer.sequence,
            }
        if projector_bank is not None:
            projector_bank.assert_hash_current()
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        proposal_count = len(tuple(proposal_root.glob("*.json")))
        exact_counts = bool(
            failure_type is None
            and sequences == {"features": 4, "actions": 4, "events": 4}
            and receipt_count == CHAIN_LENGTH
            and proposal_count == len(tuple(proposal_root.glob("*.pt")))
            and proposal_count == sum(int(result["proposal_count"]) for result in results)
            and len(tuple(direct_z_root.glob("*.pt"))) == CHAIN_LENGTH
        )
        summary = {
            "schema_version": CONTROLLER_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "controller_branch": branch,
            "run_status": "aborted" if failure_type else "completed",
            "failure_type": failure_type,
            "all_pass": bool(
                failure_type is None
                and len(results) == CHAIN_LENGTH
                and all(result.get("pass") is True for result in results)
                and exact_counts
            ),
            "selection_manifest_id": selection.manifest_id,
            "policy_parameters_sha256": manifest["policy_parameters_sha256"],
            "stream_sequences": sequences,
            "expected_counts_exact": exact_counts,
            "receipt_count": receipt_count,
            "proposal_artifact_count": proposal_count,
            "direct_z_artifact_count": len(tuple(direct_z_root.glob("*.pt"))),
            "loaded_covariances": list(loaded_covariances),
            "precomputed_projector_only": projector_bank is not None if is_alpha_branch(branch) else True,
            "history_edit_count": history_bank.edit_count if history_bank is not None else 0,
            "controller_only": True,
            "outcome_fields_loaded": False,
            "disposable_process_exit_required": True,
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "features_sha256": _file_sha256(features_path),
                "actions_sha256": _file_sha256(actions_path),
                "events_sha256": _file_sha256(events_path),
            },
            "git_output_written": False,
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-capacity-history-controller", allow_abbrev=False
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_capacity_history_controller(
            easyedit_root=args.easyedit_root,
            branch=args.branch,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    except Exception as exc:
        failure_frames = sanitized_traceback_frames(exc)
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                    "traceback_frames": failure_frames,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
