"""MV-1 v2 score-mix claim-stress diagnostic.

The runner measures six same-snapshot, unit-C central finite differences but
builds its dynamic direction from the five single-layer slopes only.  Features
and the outcome-free dynamic action are durably committed before any
operational outcome is evaluated.

Structured artifacts contain only case IDs, full request hashes, scalar
metrics, and hashes.  Direct-z remains below the local run directory.  The
runner never writes EasyEdit, never deserializes a projector, and never
recomputes covariance moments.
"""

from __future__ import annotations

import argparse
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
    MemitFactorProposal,
    canonical_json,
)
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
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
    _proposal_with_factors,
    _state_identity,
    assert_exact_action_contract,
    build_unit_c_actions,
    proposal_c_energy,
    rewrite_metrics,
    scale_proposal,
    teacher_forced_rewrite_exact,
)


SCORE_MIX_Q = 1.0 / 256.0
PROBE_RATIO = 0.25
SCORE_MIX_ACTION = "score_mix"
ORDERED_GLOBAL_ALPHA_ACTION = "ordered_global_alpha"
NATIVE_MEMIT_ACTION = "native_memit_full"
NO_OP_ACTION = "no_op_replay"
EXPECTED_OUTCOME_ACTIONS = (
    SCORE_MIX_ACTION,
    ACTION_UNIFORM,
    ORDERED_GLOBAL_ALPHA_ACTION,
    NATIVE_MEMIT_ACTION,
    NO_OP_ACTION,
)
EXPECTED_LAYER_COUNT = 5
SCHEMA_VERSION = "ode-edit-mv1-score-mix/v1"


@dataclass(frozen=True, slots=True)
class SliceEnvelope:
    label: str
    start: int
    count: int
    job_name: str
    llama_run_id: str
    qwen_run_id: str

    def run_id_for(self, model_alias: str) -> str:
        if model_alias == "llama3-8b-inst":
            return self.llama_run_id
        if model_alias == "qwen2.5-7b-inst":
            return self.qwen_run_id
        raise MV1Error(f"model is outside the score-mix envelope: {model_alias}")


D0_ENVELOPE = SliceEnvelope(
    label="d0",
    start=3,
    count=5,
    job_name="odeedit_mv1mix_d0_pair_v1",
    llama_run_id="mv1mix_llama_d0_v1",
    qwen_run_id="mv1mix_qwen_d0_v1",
)
D1_ENVELOPE = SliceEnvelope(
    label="d1",
    start=8,
    count=12,
    job_name="odeedit_mv1mix_d1_pair_v1",
    llama_run_id="mv1mix_llama_d1_v1",
    qwen_run_id="mv1mix_qwen_d1_v1",
)
SLICE_ENVELOPES = {
    (D0_ENVELOPE.start, D0_ENVELOPE.count): D0_ENVELOPE,
    (D1_ENVELOPE.start, D1_ENVELOPE.count): D1_ENVELOPE,
}


@dataclass(frozen=True, slots=True)
class ScoreMixDecision:
    """Outcome-free controller decision in ascending layer order."""

    layers: tuple[int, ...]
    slopes: tuple[float, ...]
    weights: tuple[float, ...]
    predicted_score: float
    controller_branch: str

    def __post_init__(self) -> None:
        if (
            len(self.layers) != EXPECTED_LAYER_COUNT
            or len(self.slopes) != EXPECTED_LAYER_COUNT
            or len(self.weights) != EXPECTED_LAYER_COUNT
            or tuple(sorted(self.layers)) != self.layers
            or len(set(self.layers)) != EXPECTED_LAYER_COUNT
        ):
            raise ContractError("score-mix decision requires five ascending layers")
        values = (*self.slopes, *self.weights, self.predicted_score)
        if any(not math.isfinite(float(value)) for value in values):
            raise ContractError("score-mix decision values must be finite")
        if any(weight < 0 for weight in self.weights):
            raise ContractError("score-mix weights must be non-negative")
        if not math.isclose(
            sum(weight * weight for weight in self.weights),
            1.0,
            rel_tol=2e-7,
            abs_tol=2e-7,
        ):
            raise ContractError("score-mix weights must have unit L2 norm")
        if self.controller_branch not in (
            "positive-relu-l2",
            "all-nonpositive-max-onehot",
        ):
            raise ContractError("unknown score-mix controller branch")

    @property
    def layer_weights(self) -> dict[str, float]:
        return {
            f"layer_{layer}": weight
            for layer, weight in zip(self.layers, self.weights)
        }


def derive_score_mix_decision(
    single_layer_scores: Mapping[str, float],
    *,
    layers: Sequence[int],
) -> ScoreMixDecision:
    """Map five single-layer slopes to the locked dynamic mixture.

    Positive slopes use ``relu(g) / ||relu(g)||_2``.  If all slopes are
    non-positive, the maximum slope receives one-hot weight; exact ties resolve
    to the lower layer because layers are sorted before the first match.
    """

    normalized_layers = tuple(int(layer) for layer in layers)
    if (
        len(normalized_layers) != EXPECTED_LAYER_COUNT
        or len(set(normalized_layers)) != EXPECTED_LAYER_COUNT
        or tuple(sorted(normalized_layers)) != normalized_layers
    ):
        raise ContractError("score-mix requires exactly five ascending layers")
    expected_ids = tuple(f"layer_{layer}" for layer in normalized_layers)
    if set(single_layer_scores) != set(expected_ids):
        raise ContractError("score-mix slopes differ from the five locked layers")
    slopes = tuple(float(single_layer_scores[action_id]) for action_id in expected_ids)
    if any(not math.isfinite(value) for value in slopes):
        raise ContractError("score-mix slopes must be finite")

    positive = tuple(max(value, 0.0) for value in slopes)
    positive_norm = math.sqrt(sum(value * value for value in positive))
    if positive_norm > 0.0:
        weights = tuple(value / positive_norm for value in positive)
        branch = "positive-relu-l2"
    else:
        best = max(slopes)
        selected = next(index for index, value in enumerate(slopes) if value == best)
        weights = tuple(
            1.0 if index == selected else 0.0
            for index in range(EXPECTED_LAYER_COUNT)
        )
        branch = "all-nonpositive-max-onehot"
    predicted = sum(weight * slope for weight, slope in zip(weights, slopes))
    return ScoreMixDecision(
        layers=normalized_layers,
        slopes=slopes,
        weights=weights,
        predicted_score=predicted,
        controller_branch=branch,
    )


def build_score_mix_direction(
    *,
    synchronous: MemitFactorProposal,
    unit_actions: Sequence[ActionDirection],
    decision: ScoreMixDecision,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> ActionDirection:
    """Construct ``V_mix = sum_l w_l V_l`` and remeasure unit C energy."""

    expected_ids = tuple(
        [*(f"layer_{layer}" for layer in decision.layers), ACTION_UNIFORM]
    )
    if tuple(action.action_id for action in unit_actions) != expected_ids:
        raise MV1Error("score-mix received a non-canonical unit action set")
    single_actions = unit_actions[:-1]
    mixed_factors = []
    for action, weight in zip(single_actions, decision.weights):
        if len(action.proposal.factors) != 1:
            raise MV1Error("single-layer unit action contains multiple factors")
        action.proposal.assert_same_entry_snapshot(synchronous)
        if weight > 0.0:
            mixed_factors.append(action.proposal.factors[0].scaled(weight))
    if not mixed_factors:
        raise MV1Error("score-mix constructed an empty factor set")
    proposal = _proposal_with_factors(
        synchronous,
        tuple(mixed_factors),
        solver_suffix="score-mix-unit-c",
    )
    observed_energy = proposal_c_energy(
        proposal,
        covariance_by_layer,
        layer_by_weight,
    )
    if not math.isclose(observed_energy, 1.0, rel_tol=2e-5, abs_tol=2e-5):
        raise MV1Error(
            f"score-mix actual C energy is {observed_energy}, expected 1"
        )
    return ActionDirection(
        action_id=SCORE_MIX_ACTION,
        proposal=proposal,
        c_squared_norm=observed_energy,
    )


_FEATURE_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "native_c_energy",
    "operational_c_distance",
    "probe_c_distance",
    "action_scores",
    "context_action_scores",
    "feature_policy",
    "feature_hash",
}
_ACTION_FIELDS = {
    "case_id",
    "request_id",
    "q",
    "q_label",
    "feature_hash",
    "action_id",
    "layer_weights",
    "predicted_score",
    "actual_unit_c_energy",
    "controller",
    "controller_branch",
    "tie_policy",
    "commitment_hash",
}
_OUTCOME_ONLY_FIELDS = {
    "outcome",
    "progress",
    "context_progress",
    "nll_reduction",
    "context_nll_reduction",
    "exact_margin_min",
    "exact_satisfied",
    "rollback_exact",
    "status",
}

# Boolean firewall attestations describe the absence/order of outcome access;
# they are not outcome payloads.  Keep this allow-list exact and value-locked
# so a misspelled key or a flipped attestation still fails closed.
_OUTCOME_FREE_ATTESTATIONS = {
    "outcome_fields_loaded": False,
    "all_actions_before_outcomes": True,
}


def _assert_outcome_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in _OUTCOME_FREE_ATTESTATIONS:
                if item is not _OUTCOME_FREE_ATTESTATIONS[key]:
                    raise ContractError(
                        f"outcome firewall attestation differs: {raw_key}"
                    )
                continue
            if key in _OUTCOME_ONLY_FIELDS or "outcome" in key:
                raise ContractError(
                    f"outcome field is forbidden before commitment: {raw_key}"
                )
            _assert_outcome_free(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_outcome_free(item)


def commit_score_mix_action(
    *,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
    receipt_root: Path,
    feature: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[str, str]:
    """Write, flush, fsync, then exclusively create one outcome-free receipt."""

    if set(feature) != _FEATURE_FIELDS or set(action) != _ACTION_FIELDS:
        raise ContractError("score-mix feature/action schema differs from the lock")
    _assert_outcome_free(feature)
    _assert_outcome_free(action)
    if (
        feature["case_id"] != action["case_id"]
        or feature["request_id"] != action["request_id"]
        or feature["q"] != SCORE_MIX_Q
        or action["q"] != SCORE_MIX_Q
        or action["action_id"] != SCORE_MIX_ACTION
        or action["feature_hash"] != feature["feature_hash"]
    ):
        raise ContractError("score-mix feature/action identity mismatch")
    feature_source = {
        key: value for key, value in feature.items() if key != "feature_hash"
    }
    action_source = {
        key: value for key, value in action.items() if key != "commitment_hash"
    }
    if (
        _feature_hash(feature_source) != feature["feature_hash"]
        or _feature_hash(action_source) != action["commitment_hash"]
    ):
        raise ContractError("score-mix feature/action hash mismatch")
    _safe_payload(feature)
    _safe_payload(action)

    feature_writer.write("mv1mix_feature", feature)
    action_writer.write("mv1mix_action_commitment", action)
    feature_writer.sync()
    action_writer.sync()
    receipt_payload = {
        "schema_version": "ode-edit-mv1-score-mix-receipt/v1",
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "q": SCORE_MIX_Q,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
        "action_id": SCORE_MIX_ACTION,
        "durability": (
            "feature-and-dynamic-action-write+flush+fsync-before-exclusive-receipt"
        ),
        "outcomes_observed_before_commitment": False,
    }
    _assert_outcome_free(
        {
            key: value
            for key, value in receipt_payload.items()
            if key != "outcomes_observed_before_commitment"
        }
    )
    receipt_identity = {
        "case_id": feature["case_id"],
        "request_id": feature["request_id"],
        "q": SCORE_MIX_Q,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action["commitment_hash"],
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


def _slice_envelope(start: int, count: int) -> SliceEnvelope:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        raise ContractError("score-mix slice start/count must be integers")
    try:
        return SLICE_ENVELOPES[(start, count)]
    except KeyError as exc:
        raise ContractError(
            "score-mix allows only calibration slices start=3,count=5 "
            "or start=8,count=12"
        ) from exc


def select_score_mix_cases(
    selection: CounterFactSelectionManifest,
    *,
    start: int,
    count: int,
) -> tuple[str, ...]:
    """Return only the audited post-pilot D0 or D1 calibration slice."""

    envelope = _slice_envelope(start, count)
    if len(selection.calibration) != 20:
        raise ContractError("score-mix requires the fixed 20-case calibration split")
    selected = selection.calibration[envelope.start : envelope.start + envelope.count]
    if (
        len(selected) != envelope.count
        or set(selected).intersection(selection.calibration[:3])
    ):
        raise ContractError("score-mix slice overlaps the three pilot cases")
    return selected


def _execution_envelope(
    model_alias: str,
    run_id: str,
    start: int,
    count: int,
) -> SliceEnvelope:
    envelope = _slice_envelope(start, count)
    if run_id != envelope.run_id_for(model_alias):
        raise MV1Error("run ID is outside the audited score-mix envelope")
    return envelope


def _mv1mix_slurm_state(
    model_alias: str,
    run_id: str,
    start: int,
    count: int,
) -> dict[str, Any]:
    envelope = _execution_envelope(model_alias, run_id, start, count)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial Slurm identity is forbidden")
    if (
        values["job_name"] != envelope.job_name
        or values["node"] != "devbox"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("Slurm identity is outside the audited score-mix envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
        "slice_label": envelope.label,
    }


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
    if tuple(int(layer) for layer in hparams.layers) != tuple(
        sorted(int(layer) for layer in hparams.layers)
    ) or len(layer_by_weight) != EXPECTED_LAYER_COUNT:
        raise MV1Error("score-mix requires the five locked ascending MEMIT layers")
    layers = tuple(int(layer) for layer in hparams.layers)
    weight_names = tuple(layer_by_weight)
    base_hashes = _weight_hashes(runtime.model, weight_names)
    state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()

    baseline_teacher, target_ids = teacher_forced_rewrite_exact(
        runtime,
        request,
        contexts,
    )
    baseline = rewrite_metrics(baseline_teacher, target_ids)
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("baseline evaluation changed the frozen runtime state")

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
    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise RollbackError("proposal construction changed the frozen runtime state")

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
    expected_factor_names = tuple(layer_by_weight)
    expected_action_ids = tuple(
        [*(f"layer_{layer}" for layer in layers), ACTION_UNIFORM]
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
    distance = math.sqrt(SCORE_MIX_Q * native_energy)
    epsilon = PROBE_RATIO * distance
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise MV1Error("score-mix probe distance must be finite and positive")

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    branch_rng_hash = rng_state_hash()
    probe_cache: dict[tuple[str, float], Any] = {}

    def evaluate_probe(
        action: ActionDirection,
        signed_distance: float,
    ) -> Any:
        key = (action.action_id, round(signed_distance, 15))
        if key not in probe_cache:
            probe_cache[key] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=scale_proposal(
                    action.proposal,
                    signed_distance,
                    solver_suffix=f"score-mix-probe-{signed_distance:.12g}",
                ),
                exact_application=False,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return probe_cache[key]

    scores: dict[str, float] = {}
    context_scores: dict[str, list[float]] = {}
    for action in unit_actions:
        plus = evaluate_probe(action, epsilon)
        minus = evaluate_probe(action, -epsilon)
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
    if tuple(scores) != expected_action_ids:
        raise MV1Error("central-FD scores differ from the six locked directions")

    decision = derive_score_mix_decision(
        {action_id: scores[action_id] for action_id in expected_action_ids[:-1]},
        layers=layers,
    )
    mix_action = build_score_mix_direction(
        synchronous=synchronous,
        unit_actions=unit_actions,
        decision=decision,
        covariance_by_layer=covariance_moments,
        layer_by_weight=layer_by_weight,
    )
    q_label = "q_1_256"
    feature_payload: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "native_c_energy": native_energy,
        "operational_c_distance": distance,
        "probe_c_distance": epsilon,
        "action_scores": scores,
        "context_action_scores": context_scores,
        "feature_policy": (
            "same-snapshot/six-unit-c/central-fd/epsilon=d_over_4/"
            "temperature-1-mean-context-token"
        ),
    }
    feature_payload["feature_hash"] = _feature_hash(feature_payload)
    action_payload: dict[str, Any] = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "feature_hash": feature_payload["feature_hash"],
        "action_id": SCORE_MIX_ACTION,
        "layer_weights": decision.layer_weights,
        "predicted_score": decision.predicted_score,
        "actual_unit_c_energy": mix_action.c_squared_norm,
        "controller": "five-single-slopes/relu-l2-else-max-onehot",
        "controller_branch": decision.controller_branch,
        "tie_policy": "exact-tie-lower-layer",
    }
    action_payload["commitment_hash"] = _feature_hash(action_payload)
    receipt_name, receipt_sha256 = commit_score_mix_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature_payload,
        action=action_payload,
    )

    # No operational branch is reachable before the durable receipt above.
    operational_cache: dict[str, Any] = {}

    def evaluate_operational(
        action_id: str,
        proposal: MemitFactorProposal | None,
        *,
        exact_application: bool,
    ) -> Any:
        if action_id not in operational_cache:
            operational_cache[action_id] = _evaluate_branch(
                runtime=runtime,
                request=request,
                contexts=contexts,
                target_ids=target_ids,
                proposal=proposal,
                exact_application=exact_application,
                base_hashes=base_hashes,
                state_identity=state_identity,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
            )
        return operational_cache[action_id]

    uniform_action = unit_actions[-1]
    score_mix_proposal = scale_proposal(
        mix_action.proposal,
        distance,
        solver_suffix="score-mix-operational",
    )
    uniform_proposal = scale_proposal(
        uniform_action.proposal,
        distance,
        solver_suffix="uniform-operational",
    )
    ordered_global_alpha = scale_proposal(
        ordered,
        math.sqrt(SCORE_MIX_Q),
        solver_suffix="ordered-global-alpha-q-1-256",
    )
    # The unit directions and native energy were already remeasured above.
    # Scaling a low-rank factor by ``a`` scales its C energy by ``a**2``;
    # avoid repeating large covariance matvecs in the operational phase.
    remeasured_energies = {
        SCORE_MIX_ACTION: mix_action.c_squared_norm * distance * distance,
        ACTION_UNIFORM: uniform_action.c_squared_norm * distance * distance,
        ORDERED_GLOBAL_ALPHA_ACTION: native_energy * SCORE_MIX_Q,
    }
    expected_operational_energy = distance * distance
    if any(
        not math.isclose(
            energy,
            expected_operational_energy,
            rel_tol=3e-5,
            abs_tol=3e-5,
        )
        for energy in remeasured_energies.values()
    ):
        raise MV1Error("equal-C operational proposals failed remeasurement")

    score_mix_after = evaluate_operational(
        SCORE_MIX_ACTION,
        score_mix_proposal,
        exact_application=False,
    )
    uniform_after = evaluate_operational(
        ACTION_UNIFORM,
        uniform_proposal,
        exact_application=False,
    )
    ordered_after = evaluate_operational(
        ORDERED_GLOBAL_ALPHA_ACTION,
        ordered_global_alpha,
        exact_application=False,
    )
    native_after = evaluate_operational(
        NATIVE_MEMIT_ACTION,
        ordered,
        exact_application=True,
    )
    replay = evaluate_operational(
        NO_OP_ACTION,
        None,
        exact_application=False,
    )
    if replay.logits_hash != baseline.logits_hash:
        raise RollbackError("no-op replay logits differ from the frozen baseline")

    common = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "q": SCORE_MIX_Q,
        "q_label": q_label,
        "feature_hash": feature_payload["feature_hash"],
        "commitment_hash": action_payload["commitment_hash"],
        "status": "completed",
        "rollback_exact": True,
    }
    outcomes = (
        {
            **common,
            "action_id": SCORE_MIX_ACTION,
            "selected_by_controller": True,
            "c_distance": distance,
            "c_energy": remeasured_energies[SCORE_MIX_ACTION],
            "budget_validation": "unit-c-remeasured-then-scaled/equal-c",
            **_metric_difference(score_mix_after, baseline),
        },
        {
            **common,
            "action_id": ACTION_UNIFORM,
            "selected_by_controller": False,
            "c_distance": distance,
            "c_energy": remeasured_energies[ACTION_UNIFORM],
            "budget_validation": "unit-c-remeasured-then-scaled/equal-c",
            **_metric_difference(uniform_after, baseline),
        },
        {
            **common,
            "action_id": ORDERED_GLOBAL_ALPHA_ACTION,
            "selected_by_controller": False,
            "c_distance": distance,
            "c_energy": remeasured_energies[ORDERED_GLOBAL_ALPHA_ACTION],
            "budget_validation": (
                "native-c-remeasured-then-global-alpha-squared/equal-c"
            ),
            **_metric_difference(ordered_after, baseline),
        },
        {
            **common,
            "action_id": NATIVE_MEMIT_ACTION,
            "selected_by_controller": False,
            "c_distance": math.sqrt(native_energy),
            "c_energy": native_energy,
            "budget_validation": "reference-native-full",
            **_metric_difference(native_after, baseline),
        },
        {
            **common,
            "action_id": NO_OP_ACTION,
            "selected_by_controller": False,
            "c_distance": 0.0,
            "c_energy": 0.0,
            "budget_validation": "reference-no-op",
            "logits_hash_equal": True,
            **_metric_difference(replay, baseline),
        },
    )
    if tuple(item["action_id"] for item in outcomes) != EXPECTED_OUTCOME_ACTIONS:
        raise MV1Error("runtime outcome arms differ from the exact score-mix set")
    for payload in outcomes:
        outcome_writer.write("mv1mix_outcome", payload)

    if (
        _weight_hashes(runtime.model, weight_names) != base_hashes
        or _state_identity(runtime) != state_identity
        or rng_state_hash() != branch_rng_hash
    ):
        raise RollbackError("score-mix event did not restore the frozen state")
    return {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "event_seed": event_seed,
        "target_token_count": baseline.target_token_count,
        "native_c_energy": native_energy,
        "q": SCORE_MIX_Q,
        "probe_direction_count": len(scores),
        "feature_count": 1,
        "commitment_count": 1,
        "outcome_count": len(outcomes),
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_artifact_size": direct_z.artifact.size,
        "action_receipt": {
            "name": receipt_name,
            "sha256": receipt_sha256,
        },
        "rollback_exact": True,
        "pass": True,
    }


def run_mv1_score_mix(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    start: int = D0_ENVELOPE.start,
    case_count: int = D0_ENVELOPE.count,
    seed: int = 17,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    """Run one model on exactly the audited D0 or D1 claim-stress slice."""

    started = time.perf_counter()
    envelope = _execution_envelope(model_alias, run_id, start, case_count)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    slurm_state = _mv1mix_slurm_state(
        model_alias,
        run_id,
        start,
        case_count,
    )
    if not slurm_state["under_slurm"] and model_loader is load_fixed_model:
        raise MV1Error(
            "non-Slurm score-mix execution is allowed only with an injected test loader"
        )
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_counterfact_selection(root, seed=selection_seed)
    case_ids = select_score_mix_cases(
        selection,
        start=start,
        count=case_count,
    )
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
                "schema_version": "ode-edit-mv1-score-mix-manifest/v1",
                "run_id": run_id,
                "ode_edit_git": git_state,
                "slurm": slurm_state,
                "model": runtime.metadata(),
                "hparams_relative_path": spec.hparams_path,
                "selection": selection.to_dict(),
                "selected_split": "calibration",
                "selected_slice": {
                    "label": envelope.label,
                    "start": envelope.start,
                    "count": envelope.count,
                    "pilot_case_count_excluded": 3,
                },
                "selected_case_ids": list(case_ids),
                "selected_request_ids": [
                    request.request_id for request in requests
                ],
                "contexts": {
                    "manifest_id": contexts.manifest_id,
                    "source": contexts.source,
                    "group_sizes": [
                        len(group) for group in contexts.templates
                    ],
                    "raw_templates_persisted": False,
                },
                "provenance_id": provenance.manifest_id,
                "fixed_files": _relative_provenance(provenance, root),
                "q": SCORE_MIX_Q,
                "q_label": "q_1_256",
                "probe_ratio": PROBE_RATIO,
                "probe_action_set": [
                    *(f"layer_{layer}" for layer in spec.layers),
                    ACTION_UNIFORM,
                ],
                "controller_input_set": [
                    *(f"layer_{layer}" for layer in spec.layers)
                ],
                "outcome_action_set": list(EXPECTED_OUTCOME_ACTIONS),
                "decision_policy": (
                    "outcome-free five-single relu(g)/L2; "
                    "all-nonpositive max-onehot; exact tie lower layer"
                ),
                "action_receipt_policy": (
                    "feature+dynamic-action JSONL write/flush/fsync then "
                    "exclusive receipt before outcome evaluation"
                ),
                "utility_policy": (
                    "temperature=1 mean-context-token smooth target margin"
                ),
                "teacher_suffix_policy": (
                    "prefix+target single tokenization with exact target suffix"
                ),
                "projector_policy": (
                    "full sha256-and-size preflight only; never deserialized"
                ),
                "covariance_policy": (
                    "verified-read-only; recompute-and-download-blocked"
                ),
                "direct_z_policy": (
                    "computed-once-and-frozen below this local run"
                ),
                "artifact_firewall": (
                    "structured JSON contains case IDs/full request hashes/"
                    "scalars/hashes only; prompts, targets, logits, weights, "
                    "generations, and activations are forbidden; "
                    "activation-derived direct_z remains local-only"
                ),
                "expected_counts": {
                    "features": envelope.count,
                    "commitments": envelope.count,
                    "outcomes": envelope.count * len(EXPECTED_OUTCOME_ACTIONS),
                    "receipts": envelope.count,
                },
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
                schema_version=SCHEMA_VERSION,
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path,
                run_id,
                schema_version=SCHEMA_VERSION,
            ) as action_writer,
            SanitizedJsonlWriter(
                outcomes_path,
                run_id,
                schema_version=SCHEMA_VERSION,
            ) as outcome_writer,
            SanitizedJsonlWriter(
                events_path,
                run_id,
                schema_version=SCHEMA_VERSION,
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
                        "failure_class": (
                            "fatal_score_mix_contract_or_runtime"
                        ),
                        "rollback_exact": False,
                        "pass": False,
                    }
                    abort_failure_type = type(exc).__name__
                event_results.append(result)
                event_writer.write("mv1mix_case", result)
                if abort_failure_type is not None:
                    break

        elapsed = time.perf_counter() - started
        planned = len(requests)
        attempted = len(event_results)
        pass_count = sum(bool(item["pass"]) for item in event_results)
        feature_count = sum(
            int(item.get("feature_count", 0)) for item in event_results
        )
        commitment_count = sum(
            int(item.get("commitment_count", 0)) for item in event_results
        )
        outcome_count = sum(
            int(item.get("outcome_count", 0)) for item in event_results
        )
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        exact_counts = bool(
            feature_count == planned
            and commitment_count == planned
            and outcome_count == planned * len(EXPECTED_OUTCOME_ACTIONS)
            and receipt_count == planned
        )
        summary = {
            "schema_version": "ode-edit-mv1-score-mix-summary/v1",
            "run_id": run_id,
            "model_alias": model_alias,
            "slice": {
                "label": envelope.label,
                "start": envelope.start,
                "count": envelope.count,
            },
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
            "q": SCORE_MIX_Q,
            "probe_direction_count_per_case": 6,
            "feature_count": feature_count,
            "commitment_count": commitment_count,
            "outcome_count": outcome_count,
            "action_receipt_count": receipt_count,
            "expected_counts_exact": exact_counts,
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
            "resource": {
                "wall_seconds": elapsed,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-mv1-score-mix",
        description="Outcome-free score-mix MV-1 claim-stress diagnostic",
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--start-index",
        dest="start",
        type=int,
        choices=(3, 8),
        default=3,
    )
    parser.add_argument("--cases", type=int, choices=(5, 12), default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_mv1_score_mix(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
            start=args.start,
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
