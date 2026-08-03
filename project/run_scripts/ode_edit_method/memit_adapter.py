"""Read-only EasyEdit factor adaptation into ODE-Edit method contracts."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    MemitFactorProposal,
    ProposalSemantics as MotivationProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import build_unit_c_actions

from .contracts import (
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from .hooks import FactorDirection, TorchFactorTrial


def native_terminal_batch(
    proposal: MemitFactorProposal,
    layer_by_weight: Mapping[str, int],
) -> ProposalBatch:
    if proposal.semantics is not MotivationProposalSemantics.ORDERED_GAUSS_SEIDEL:
        raise MethodContractError("native MEMIT requires canonical ordered factors")
    items = []
    for factor in proposal.factors:
        try:
            layer = int(layer_by_weight[factor.weight_name])
        except KeyError as exc:
            raise MethodContractError("native layer mapping is incomplete") from exc
        direction = FactorDirection.from_low_rank_factor(layer, factor)
        items.append(
            LayerProposal(
                layer=layer,
                state_id=proposal.entry_snapshot_id,
                direction_id=direction.direction_id,
                payload=direction,
            )
        )
    items.sort(key=lambda item: item.layer)
    return ProposalBatch(
        snapshot_id=proposal.entry_snapshot_id,
        proposals=tuple(items),
        slopes=tuple(0.0 for _ in items),
        semantics=ProposalSemantics.NATIVE_ORDERED_TERMINAL,
    )


def synchronous_unit_batch(
    proposal: MemitFactorProposal,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    slopes_by_layer: Mapping[int, float],
) -> ProposalBatch:
    if not proposal.is_synchronous:
        raise MethodContractError("full/static proposals must be same-snapshot synchronous")
    actions = build_unit_c_actions(proposal, covariance_by_layer, layer_by_weight)
    # The final motivation action is uniform.  Session 02's QP consumes only
    # one independently normalized direction per canonical layer.
    single_layer = actions[:-1]
    items = []
    slopes = []
    for action in single_layer:
        factor = action.proposal.factors[0]
        layer = int(layer_by_weight[factor.weight_name])
        direction = FactorDirection.from_low_rank_factor(layer, factor)
        items.append(
            LayerProposal(
                layer=layer,
                state_id=proposal.entry_snapshot_id,
                direction_id=direction.direction_id,
                payload=direction,
            )
        )
        try:
            slopes.append(max(0.0, float(slopes_by_layer[layer])))
        except KeyError as exc:
            raise MethodContractError("synchronous slope mapping is incomplete") from exc
    order = sorted(range(len(items)), key=lambda index: items[index].layer)
    return ProposalBatch(
        snapshot_id=proposal.entry_snapshot_id,
        proposals=tuple(items[index] for index in order),
        slopes=tuple(slopes[index] for index in order),
        semantics=ProposalSemantics.CURRENT_SAME_SNAPSHOT,
    )


def coordinate_batch(batch: ProposalBatch, layer: int) -> ProposalBatch:
    if batch.semantics is not ProposalSemantics.CURRENT_SAME_SNAPSHOT:
        raise MethodContractError("coordinate extraction requires a current synchronous batch")
    for proposal, slope in zip(batch.proposals, batch.slopes, strict=True):
        if proposal.layer == layer:
            return ProposalBatch(
                snapshot_id=batch.snapshot_id,
                proposals=(proposal,),
                slopes=(slope,),
                semantics=ProposalSemantics.CURRENT_COORDINATE,
            )
    raise MethodContractError(f"coordinate layer {layer} is absent")


def trial_for_batch(
    model: torch.nn.Module,
    batch: ProposalBatch,
    coefficients: Sequence[float],
) -> TorchFactorTrial:
    directions = tuple(proposal.payload for proposal in batch.proposals)
    if any(not isinstance(direction, FactorDirection) for direction in directions):
        raise MethodContractError("MEMIT proposal payload is not a FactorDirection")
    return TorchFactorTrial(
        model,
        directions,
        coefficients,
        native_exact=batch.semantics is ProposalSemantics.NATIVE_ORDERED_TERMINAL,
    )
