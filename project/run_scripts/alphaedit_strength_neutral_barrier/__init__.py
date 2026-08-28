"""Strength-neutral, target-excluded barrier extension for AlphaEdit.

The official ``N=1`` AlphaEdit path remains an explicit bypass.  The modules
in this package are used only for split or barrier-guided ``N>=2`` writers.
"""

from .writer import (
    BarrierArm,
    BarrierWriterConfig,
    apply_strength_neutral_barrier_to_model,
)

__all__ = [
    "BarrierArm",
    "BarrierWriterConfig",
    "apply_strength_neutral_barrier_to_model",
]
