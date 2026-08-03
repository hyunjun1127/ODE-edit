"""Read-only EasyEdit factor adaptation into ODE-Edit method contracts."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    MemitFactorProposal,
    ProposalSemantics as MotivationProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import proposal_c_energy

from .contracts import (
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from .functional_trial import QuantizedFullLinearFunctionalTrial
from .hooks import FactorDirection, TorchFactorTrial, apply_accepted_factors


SIMPLE_T_ROW_BLOCK = 64


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
    *,
    unit_c_norm_epsilon: float,
    unit_c_identity_atol: float,
    unit_c_identity_rtol: float,
) -> ProposalBatch:
    if not proposal.is_synchronous:
        raise MethodContractError("full/static proposals must be same-snapshot synchronous")
    tolerances = (
        float(unit_c_norm_epsilon),
        float(unit_c_identity_atol),
        float(unit_c_identity_rtol),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise MethodContractError("unit-C tolerances must be finite and positive")
    items = []
    slopes = []
    for factor in proposal.factors:
        single = replace(
            proposal,
            factors=(factor,),
            solver_name=f"{proposal.solver_name}/ode-edit-energy-probe",
        )
        energy = float(proposal_c_energy(single, covariance_by_layer, layer_by_weight))
        if not math.isfinite(energy) or energy <= unit_c_norm_epsilon:
            raise MethodContractError("synchronous direction has near-zero C norm")
        unit = factor.scaled(1.0 / math.sqrt(energy))
        unit_proposal = replace(
            proposal,
            factors=(unit,),
            solver_name=f"{proposal.solver_name}/ode-edit-unit-c",
        )
        unit_energy = float(
            proposal_c_energy(unit_proposal, covariance_by_layer, layer_by_weight)
        )
        if not math.isclose(
            unit_energy,
            1.0,
            abs_tol=unit_c_identity_atol,
            rel_tol=unit_c_identity_rtol,
        ):
            raise MethodContractError("synchronous direction is not unit C-normalized")
        layer = int(layer_by_weight[factor.weight_name])
        direction = FactorDirection.from_low_rank_factor(layer, unit)
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
    """Dense mutable trial retained for reference/rollback unit tests only."""

    directions = tuple(proposal.payload for proposal in batch.proposals)
    if any(not isinstance(direction, FactorDirection) for direction in directions):
        raise MethodContractError("MEMIT proposal payload is not a FactorDirection")
    return TorchFactorTrial(
        model,
        directions,
        coefficients,
        native_exact=batch.semantics is ProposalSemantics.NATIVE_ORDERED_TERMINAL,
    )


def functional_trial_for_batch(
    model: torch.nn.Module,
    batch: ProposalBatch,
    coefficients: Sequence[float],
) -> QuantizedFullLinearFunctionalTrial:
    """Construct the common full-linear simple-T adaptive finite trial."""

    directions = tuple(proposal.payload for proposal in batch.proposals)
    if any(not isinstance(direction, FactorDirection) for direction in directions):
        raise MethodContractError("MEMIT proposal payload is not a FactorDirection")
    if batch.semantics not in {
        ProposalSemantics.CURRENT_SAME_SNAPSHOT,
        ProposalSemantics.ENTRY_FROZEN_REBOUND,
        ProposalSemantics.CURRENT_COORDINATE,
    }:
        raise MethodContractError(
            "simple-T production trial accepts adaptive proposal semantics only"
        )
    return QuantizedFullLinearFunctionalTrial(
        model,
        directions,
        coefficients,
        row_block=SIMPLE_T_ROW_BLOCK,
    )


def commit_batch(
    model: torch.nn.Module,
    batch: ProposalBatch,
    coefficients: Sequence[float],
) -> tuple[float, ...]:
    """Persist the exact coefficient vector only after a trial is accepted."""

    directions = tuple(proposal.payload for proposal in batch.proposals)
    if any(not isinstance(direction, FactorDirection) for direction in directions):
        raise MethodContractError("MEMIT proposal payload is not a FactorDirection")
    return apply_accepted_factors(
        model,
        directions,
        coefficients,
        native_exact=batch.semantics is ProposalSemantics.NATIVE_ORDERED_TERMINAL,
    )
