"""Peak-bounded construction of the native dense MEMIT system.

The mathematical system and solve are intentionally identical to native
MEMIT.  The only change is allocation order: one owned dense tensor is copied
from the verified covariance source, then scaled and rank-updated in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from project.run_scripts.ode_edit_motivation.diagnostic_math import (
    _validate_memit_inputs,
)


@dataclass(frozen=True, slots=True)
class TransientDenseMemitSolver:
    """Solve ``(lambda C + K K^T) X = K`` with one owned dense system.

    ``covariance`` remains a guarded, read-only source.  ``system`` is the
    sole explicit dense construction allocation and is not retained after the
    call.  ``torch.linalg.solve`` remains the canonical native solver.
    """

    name: str = "transient-dense-native-memit"

    def adjusted_keys(
        self,
        covariance: torch.Tensor,
        keys: torch.Tensor,
        covariance_weight: float,
    ) -> torch.Tensor:
        _validate_memit_inputs(covariance, keys, covariance_weight)
        system = covariance.detach().to(
            device=keys.device,
            dtype=keys.dtype,
            copy=True,
        )
        system.mul_(covariance_weight)
        system.addmm_(keys, keys.transpose(0, 1))
        return torch.linalg.solve(system, keys)
