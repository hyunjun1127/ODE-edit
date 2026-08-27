"""Categorical Fisher--Gauss--Newton pullback construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .errors import BGODEScientificBoundary


@dataclass(frozen=True, slots=True)
class FisherPullback:
    matrix: torch.Tensor
    symmetry_residual: float
    minimum_eigenvalue: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.matrix, torch.Tensor)
            or self.matrix.dtype != torch.float32
            or self.matrix.ndim != 2
            or self.matrix.shape[0] != self.matrix.shape[1]
            or self.matrix.requires_grad
            or not bool(torch.isfinite(self.matrix).all())
        ):
            raise BGODEScientificBoundary("Fisher pullback matrix is invalid")
        if not math.isfinite(self.symmetry_residual) or not math.isfinite(self.minimum_eigenvalue):
            raise BGODEScientificBoundary("Fisher diagnostics must be finite")
        tolerance = 64.0 * torch.finfo(torch.float32).eps * max(1.0, float(torch.linalg.matrix_norm(self.matrix)))
        if self.symmetry_residual > tolerance or self.minimum_eigenvalue < -tolerance:
            raise BGODEScientificBoundary("Fisher pullback is not symmetric PSD")


def categorical_fisher_pullback(
    probabilities: torch.Tensor,
    scores: torch.Tensor,
) -> FisherPullback:
    if (
        not isinstance(probabilities, torch.Tensor)
        or probabilities.dtype != torch.float32
        or probabilities.ndim != 1
        or probabilities.requires_grad
        or not bool(torch.isfinite(probabilities).all())
        or not bool((probabilities >= 0).all())
    ):
        raise BGODEScientificBoundary("event probabilities must be finite FP32")
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or scores.shape[0] != probabilities.shape[0]
        or scores.requires_grad
        or not bool(torch.isfinite(scores).all())
    ):
        raise BGODEScientificBoundary("event scores must be finite FP32 [event,actuator]")
    matrix_raw = scores.transpose(0, 1) @ (probabilities[:, None] * scores)
    matrix = (0.5 * (matrix_raw + matrix_raw.transpose(0, 1))).detach().contiguous()
    symmetry = float(torch.linalg.matrix_norm(matrix - matrix.transpose(0, 1)).cpu().item())
    minimum = float(torch.linalg.eigvalsh(matrix).amin().cpu().item())
    return FisherPullback(
        matrix=matrix,
        symmetry_residual=symmetry,
        minimum_eigenvalue=minimum,
    )


__all__ = ["FisherPullback", "categorical_fisher_pullback"]
