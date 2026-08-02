"""Reusable W0-normalized covariance geometry for ODE-Edit controllers.

The helpers are controller-safe: they inspect only model weights and the
already-loaded editor covariance matrices.  They do not import or decode any
CounterFact evaluation field and never write the upstream EasyEdit cache.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .contracts import ContractError
from .hooks import resolve_parameter


WEIGHT_ENERGY_ROW_BLOCK = 64


class CapacityGeometryError(ContractError):
    """The fixed weight/covariance geometry is invalid."""


def weight_c_energy(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    *,
    row_block: int = WEIGHT_ENERGY_ROW_BLOCK,
) -> float:
    """Return ``tr(W C W.T)`` using bounded accelerator row blocks."""

    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(covariance, torch.Tensor)
        or weight.ndim != 2
        or covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or weight.shape[1] != covariance.shape[0]
        or isinstance(row_block, bool)
        or not isinstance(row_block, int)
        or row_block <= 0
    ):
        raise CapacityGeometryError("weight/covariance geometry differs")
    device = weight.device
    metric = covariance.detach().to(device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, int(weight.shape[0]), row_block):
            block = weight[start : start + row_block].float()
            applied = block @ metric
            total += torch.sum(block.double() * applied.double())
            del block, applied
    result = float(total.detach().cpu())
    del metric, total
    if not math.isfinite(result) or result <= 0.0:
        raise CapacityGeometryError("W0 covariance energy must be positive")
    return result


def compute_w0_denominators(
    model: torch.nn.Module,
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    *,
    expected_layers: Sequence[int],
) -> dict[int, float]:
    """Compute one immutable capacity denominator for every editable layer."""

    expected = tuple(int(layer) for layer in expected_layers)
    if not expected or len(set(expected)) != len(expected):
        raise CapacityGeometryError("expected layers must be unique")
    denominators: dict[int, float] = {}
    for weight_name, raw_layer in layer_by_weight.items():
        layer = int(raw_layer)
        if layer in denominators or layer not in covariance_by_layer:
            raise CapacityGeometryError("layer/covariance mapping is not one-to-one")
        denominators[layer] = weight_c_energy(
            resolve_parameter(model, weight_name),
            covariance_by_layer[layer],
        )
    if set(denominators) != set(expected):
        raise CapacityGeometryError("W0 denominator layers differ from the lock")
    return {layer: denominators[layer] for layer in expected}


__all__ = [
    "CapacityGeometryError",
    "WEIGHT_ENERGY_ROW_BLOCK",
    "compute_w0_denominators",
    "weight_c_energy",
]
