"""Typed writer cadence for target-flow experiments.

The cadence decides only when an already-selected target is materialized.  It
does not own the target field, target clock, writer equation, or cache state.
"""

from __future__ import annotations

from enum import Enum

from .contracts import ODEBFContractError
from .scalable_batched_runtime import P1R23_GRID_COUNT


class WriterCadence(str, Enum):
    KSTEP_EACH_OUTER = "KSTEP_EACH_OUTER"
    FINAL_Z_ONESHOT = "FINAL_Z_ONESHOT"

    def writes_at(self, step_index: int) -> bool:
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < P1R23_GRID_COUNT
        ):
            raise ODEBFContractError("writer cadence step differs")
        return (
            True
            if self is WriterCadence.KSTEP_EACH_OUTER
            else step_index == P1R23_GRID_COUNT - 1
        )

    @property
    def writer_calls_per_block(self) -> int:
        return (
            P1R23_GRID_COUNT
            if self is WriterCadence.KSTEP_EACH_OUTER
            else 1
        )


__all__ = ["WriterCadence"]
