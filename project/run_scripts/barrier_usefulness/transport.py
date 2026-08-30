"""Compact fixed endpoint/candidate tensor transport across ctrl/gate stages."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import ScientificBoundary


def factor_dense(matrix: torch.Tensor, *, tolerance: float, max_rank: int = 8) -> dict[str, Any]:
    """Deterministic pivoted low-rank transport, never used to choose science."""
    source = matrix.detach().float()
    residual = source.clone()
    outputs = []
    inputs = []
    denominator = source.norm().clamp_min(torch.finfo(torch.float32).tiny)
    relative = float(residual.norm().div(denominator).item())
    for _ in range(max_rank):
        row_norms = torch.linalg.vector_norm(residual, dim=1)
        pivot = int(row_norms.argmax().item())
        input_factor = residual[pivot].clone()
        norm2 = torch.dot(input_factor, input_factor)
        if float(norm2.item()) == 0.0:
            break
        output_factor = residual @ input_factor / norm2
        outputs.append(output_factor.detach().cpu())
        inputs.append(input_factor.detach().cpu())
        residual.sub_(torch.outer(output_factor, input_factor))
        relative = float(residual.norm().div(denominator).item())
        if relative <= tolerance:
            break
    if relative > tolerance:
        raise ScientificBoundary(f"endpoint transport rank/tolerance failure: {relative}")
    return {
        "outputs": outputs, "inputs": inputs, "rank": len(outputs),
        "relative_residual": relative, "shape": list(matrix.shape), "dtype": "float32",
    }


def reconstruct(factors: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    shape = tuple(int(value) for value in factors["shape"])
    result = torch.zeros(shape, device=device, dtype=torch.float32)
    for output, input_factor in zip(factors["outputs"], factors["inputs"], strict=True):
        result.add_(torch.outer(output.to(device), input_factor.to(device)))
    return result
