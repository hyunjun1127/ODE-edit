"""Reusable trajectory primitives for the MV-2 refresh diagnostic.

The helpers here are deliberately outcome-agnostic.  They transport a fixed
W0 factor basis to a verified descendant, build one central-difference panel,
check immutable-covariance C budgets, and execute nested temporary branches
with exact rollback/RNG restoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)
from .diagnostic_math import c_cosine_similarity
from .frozen_target_lineage import (
    FrozenTargetLineage,
    proposal_direction_hash,
)
from .hooks import TemporaryLowRankApplication, assert_snapshot_current
from .mv1_calibration import ActionDirection, proposal_c_energy


MATCHED_C_REL_TOL = 3e-5
MATCHED_C_ABS_TOL = 3e-5


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


def action_direction_hash(action: ActionDirection) -> str:
    energy = _finite("action C energy", action.c_squared_norm)
    if energy <= 0.0:
        raise ContractError("action C energy must be positive")
    return sha256_bytes(
        canonical_json(
            {
                "action_id": action.action_id,
                "proposal_direction_hash": proposal_direction_hash(action.proposal),
                "c_squared_norm": energy,
            }
        ).encode("utf-8")
    )


def transport_fixed_proposal(
    proposal: MemitFactorProposal,
    *,
    descendant_snapshot: SnapshotManifest,
    frozen_target_lineage: FrozenTargetLineage,
    solver_suffix: str,
) -> MemitFactorProposal:
    """Reuse W0 tensor directions at W1 without claiming they were refreshed.

    Only expected parameter hashes and the guarded snapshot are rebound.  The
    factor tensor bytes must remain exactly identical.
    """

    if (
        not isinstance(proposal, MemitFactorProposal)
        or not proposal.is_synchronous
        or not isinstance(descendant_snapshot, SnapshotManifest)
    ):
        raise ContractError("fixed-basis transport requires synchronous proposals")
    if (
        type(frozen_target_lineage) is not FrozenTargetLineage
        or len(frozen_target_lineage.hops) != 1
        or frozen_target_lineage.hops[0].label != "partial_joint"
        or frozen_target_lineage.hops[0].step_scale != 0.5
    ):
        raise ContractError("fixed-basis transport requires a verified partial lineage")
    frozen_target_lineage.assert_terminal_state(descendant_snapshot)
    if (
        not isinstance(solver_suffix, str)
        or not solver_suffix
        or any(ord(character) < 32 for character in solver_suffix)
    ):
        raise ContractError("fixed-basis solver suffix is invalid")
    if (
        proposal.snapshot.model_id != descendant_snapshot.model_id
        or proposal.snapshot.context_id != descendant_snapshot.context_id
        or proposal.snapshot.request_ids != descendant_snapshot.request_ids
        or proposal.snapshot.hparams_sha256 != descendant_snapshot.hparams_sha256
        or proposal.snapshot.state_id
        != frozen_target_lineage.origin_state_id
        or {
            record.name: record.sha256
            for record in proposal.snapshot.parameters
        }
        != dict(frozen_target_lineage.origin_parameter_hashes)
    ):
        raise ContractError(
            "fixed-basis proposal is not bound to the lineage-origin W0"
        )
    source_names = tuple(factor.weight_name for factor in proposal.factors)
    source_snapshot_names = tuple(
        record.name for record in proposal.snapshot.parameters
    )
    descendant_names = tuple(record.name for record in descendant_snapshot.parameters)
    if (
        set(source_snapshot_names) != set(descendant_names)
        or not set(source_names).issubset(descendant_names)
    ):
        raise ContractError("fixed-basis transport parameter set mismatch")

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
            sorted(
                {
                    *descendant_snapshot.provenance_ids,
                    frozen_target_lineage.lineage_id,
                }
            )
        ),
    )
    transported = MemitFactorProposal(
        snapshot=rebound_snapshot,
        factors=factors,
        semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
        solver_name=f"{proposal.solver_name}/fixed-b0-transport/{solver_suffix}",
        residual_denominator=proposal.residual_denominator,
    )
    if proposal_direction_hash(transported) != before_hash:
        raise ContractError("fixed-basis transport changed factor tensor bytes")
    return transported


def transport_fixed_actions(
    actions: Sequence[ActionDirection],
    *,
    descendant_snapshot: SnapshotManifest,
    frozen_target_lineage: FrozenTargetLineage,
    solver_suffix: str,
) -> tuple[ActionDirection, ...]:
    """Transport one already-built W0 action set exactly once."""

    normalized = tuple(actions)
    if not normalized or len({action.action_id for action in normalized}) != len(
        normalized
    ):
        raise ContractError("fixed action transport requires unique non-empty actions")
    transported = tuple(
        ActionDirection(
            action_id=action.action_id,
            proposal=transport_fixed_proposal(
                action.proposal,
                descendant_snapshot=descendant_snapshot,
                frozen_target_lineage=frozen_target_lineage,
                solver_suffix=f"{solver_suffix}-{action.action_id}",
            ),
            c_squared_norm=_finite(
                f"transported {action.action_id} C energy",
                action.c_squared_norm,
            ),
        )
        for action in normalized
    )
    if tuple(action.action_id for action in transported) != tuple(
        action.action_id for action in normalized
    ):
        raise ContractError("fixed action transport changed action order")
    return transported


@dataclass(frozen=True, slots=True)
class CentralProbePanel:
    action_ids: tuple[str, ...]
    scores: Mapping[str, float]
    context_scores: Mapping[str, tuple[float, ...]]
    epsilon: float
    evaluation_count: int

    def __post_init__(self) -> None:
        epsilon = _finite("probe epsilon", self.epsilon)
        if epsilon <= 0.0:
            raise ContractError("probe epsilon must be positive")
        if (
            not self.action_ids
            or len(set(self.action_ids)) != len(self.action_ids)
            or set(self.scores) != set(self.action_ids)
            or set(self.context_scores) != set(self.action_ids)
            or self.evaluation_count != 2 * len(self.action_ids)
        ):
            raise ContractError("central-probe panel is incomplete")
        context_count: int | None = None
        for action_id in self.action_ids:
            _finite(f"probe score {action_id}", self.scores[action_id])
            values = tuple(self.context_scores[action_id])
            if not values:
                raise ContractError("probe context-score vector must not be empty")
            if context_count is None:
                context_count = len(values)
            if len(values) != context_count:
                raise ContractError("probe context-score lengths differ")
            for value in values:
                _finite(f"probe context score {action_id}", value)
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def panel_hash(self) -> str:
        return sha256_bytes(
            canonical_json(self.to_dict()).encode("utf-8")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "scores": dict(self.scores),
            "context_scores": {
                action_id: list(self.context_scores[action_id])
                for action_id in self.action_ids
            },
            "epsilon": self.epsilon,
            "evaluation_count": self.evaluation_count,
        }


def build_central_probe_panel(
    actions: Sequence[ActionDirection],
    *,
    epsilon: float,
    evaluate: Callable[[ActionDirection, float], Any],
    expected_action_ids: Sequence[str],
) -> CentralProbePanel:
    """Evaluate each signed probe exactly once and return scalar derivatives."""

    normalized = tuple(actions)
    expected = tuple(expected_action_ids)
    if tuple(action.action_id for action in normalized) != expected:
        raise ContractError("central-probe action order differs from the precommit")
    distance = _finite("probe epsilon", epsilon)
    if distance <= 0.0:
        raise ContractError("probe epsilon must be positive")
    scores: dict[str, float] = {}
    context_scores: dict[str, tuple[float, ...]] = {}
    evaluation_count = 0
    for action in normalized:
        minus = evaluate(action, -distance)
        evaluation_count += 1
        plus = evaluate(action, distance)
        evaluation_count += 1
        try:
            minus_utility = _finite("minus utility", minus.utility)
            plus_utility = _finite("plus utility", plus.utility)
            minus_context = tuple(minus.context_utility)
            plus_context = tuple(plus.context_utility)
        except (AttributeError, TypeError) as exc:
            raise ContractError("probe evaluator returned an invalid metric object") from exc
        if not minus_context or len(minus_context) != len(plus_context):
            raise ContractError("probe context utilities are not paired")
        scores[action.action_id] = (
            plus_utility - minus_utility
        ) / (2.0 * distance)
        context_scores[action.action_id] = tuple(
            (_finite("plus context utility", right) - _finite(
                "minus context utility", left
            ))
            / (2.0 * distance)
            for left, right in zip(minus_context, plus_context)
        )
    return CentralProbePanel(
        action_ids=expected,
        scores=scores,
        context_scores=context_scores,
        epsilon=distance,
        evaluation_count=evaluation_count,
    )


def per_layer_c_cosines(
    before: MemitFactorProposal,
    after: MemitFactorProposal,
    *,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
) -> dict[str, float]:
    """Compare W0 and refreshed W1 directions without dense updates."""

    before_by_name = {factor.weight_name: factor for factor in before.factors}
    after_by_name = {factor.weight_name: factor for factor in after.factors}
    if set(before_by_name) != set(after_by_name) or set(before_by_name) != set(
        layer_by_weight
    ):
        raise ContractError("C-cosine proposal/layer sets differ")
    result: dict[str, float] = {}
    for weight_name in sorted(before_by_name):
        layer = layer_by_weight[weight_name]
        if layer not in covariance_by_layer:
            raise ContractError("C-cosine covariance layer is missing")
        metric = covariance_by_layer[layer].to(
            device=after_by_name[weight_name].right.device,
            dtype=after_by_name[weight_name].right.dtype,
        )
        value = float(
            c_cosine_similarity(
                before_by_name[weight_name],
                after_by_name[weight_name],
                metric,
            ).item()
        )
        if not math.isfinite(value) or value < -1.0001 or value > 1.0001:
            raise ContractError("C-cosine is non-finite or outside its numeric range")
        result[f"layer_{layer}"] = max(-1.0, min(1.0, value))
    return result


def validate_matched_c_budget(
    proposals: Mapping[str, MemitFactorProposal],
    *,
    expected_c_energy: float,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    exact_action_ids: Sequence[str],
) -> dict[str, float]:
    """Remeasure every continuation against one immutable covariance cache."""

    expected_actions = tuple(exact_action_ids)
    if tuple(proposals) != expected_actions:
        raise ContractError("matched-C proposal order differs from the precommit")
    expected = _finite("expected continuation C energy", expected_c_energy)
    if expected <= 0.0:
        raise ContractError("expected continuation C energy must be positive")
    observed = {
        action_id: _finite(
            f"{action_id} C energy",
            proposal_c_energy(
                proposals[action_id],
                covariance_by_layer,
                layer_by_weight,
            ),
        )
        for action_id in expected_actions
    }
    if any(
        not math.isclose(
            value,
            expected,
            rel_tol=MATCHED_C_REL_TOL,
            abs_tol=MATCHED_C_ABS_TOL,
        )
        for value in observed.values()
    ):
        raise ContractError("continuation arms failed the exact matched-C gate")
    return observed


def _set_rng_state(cpu_rng: torch.Tensor, cuda_rng: torch.Tensor | None) -> None:
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        if not torch.cuda.is_available():
            raise ContractError("CUDA RNG state was supplied without CUDA")
        torch.cuda.set_rng_state(cuda_rng, device=0)


def evaluate_temporary_trajectory(
    *,
    model: torch.nn.Module,
    origin_snapshot: SnapshotManifest,
    evaluator: Callable[[], Any],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor | None,
    first_proposal: MemitFactorProposal | None = None,
    descendant_snapshot: SnapshotManifest | None = None,
    second_proposal: MemitFactorProposal | None = None,
) -> Any:
    """Evaluate zero, one, or two steps and restore W0 plus branch RNG exactly."""

    if second_proposal is not None and (
        first_proposal is None or descendant_snapshot is None
    ):
        raise ContractError("second trajectory step requires a verified first step")
    assert_snapshot_current(model, origin_snapshot)
    _set_rng_state(cpu_rng, cuda_rng)
    result: Any
    try:
        if first_proposal is None:
            result = evaluator()
        else:
            with TemporaryLowRankApplication(model, first_proposal):
                if descendant_snapshot is not None:
                    assert_snapshot_current(model, descendant_snapshot)
                if second_proposal is None:
                    result = evaluator()
                else:
                    with TemporaryLowRankApplication(model, second_proposal):
                        result = evaluator()
    finally:
        # Temporary applications restore exact weight backups even when the
        # evaluator raises.  Preserve a common RNG starting point for every
        # branch and for the caller after the branch.
        _set_rng_state(cpu_rng, cuda_rng)
        assert_snapshot_current(model, origin_snapshot)
    return result
