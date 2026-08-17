"""Explicit, opt-in scale contracts for P1R52 sequential execution.

The published P1R52 sequential default remains ten B10 batches.  The B100
contract is selected only by the dedicated runner, so existing callers retain
their exact geometry and result identities.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import BATCH_SIZE, ODEBFContractError


B100X10_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R52-LLAMA-SEQUENTIAL-10XB100-FOUR-CELL-V1"
)


@dataclass(frozen=True, slots=True)
class P1R52SequentialScale:
    scale_id: str
    batch_size: int
    round_count: int
    stream_schema: str
    instruction_id: str

    def __post_init__(self) -> None:
        if (
            self.scale_id not in {"b10x10", "b100x10"}
            or isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
            or isinstance(self.round_count, bool)
            or not isinstance(self.round_count, int)
            or self.round_count != 10
            or not self.stream_schema
            or not self.instruction_id
        ):
            raise ODEBFContractError("P1R52 sequential scale differs")

    @property
    def request_count(self) -> int:
        return self.batch_size * self.round_count

    @property
    def history_counts(self) -> tuple[int, ...]:
        return tuple(index * self.batch_size for index in range(self.round_count))

    @property
    def final_label(self) -> str:
        return f"B{self.request_count}"


P1R52_B10X10_SCALE = P1R52SequentialScale(
    "b10x10",
    BATCH_SIZE,
    10,
    "ode-edit-s05-p1r20-historical-h0-fresh-cf-b100/v1",
    "ODEEDIT-S05-P1R52-LLAMA-SOFT-SEQUENTIAL-HISTORICAL-10XB10-V1",
)

P1R52_B100X10_SCALE = P1R52SequentialScale(
    "b100x10",
    100,
    10,
    "ode-edit-s05-p1r52-sequential-b100x10-stream/v1",
    B100X10_INSTRUCTION_ID,
)


def resolve_p1r52_sequential_scale(scale_id: str | None) -> P1R52SequentialScale:
    if scale_id is None or scale_id == P1R52_B10X10_SCALE.scale_id:
        return P1R52_B10X10_SCALE
    if scale_id == P1R52_B100X10_SCALE.scale_id:
        return P1R52_B100X10_SCALE
    raise ODEBFContractError("P1R52 sequential scale identifier differs")


__all__ = [
    "B100X10_INSTRUCTION_ID",
    "P1R52SequentialScale",
    "P1R52_B10X10_SCALE",
    "P1R52_B100X10_SCALE",
    "resolve_p1r52_sequential_scale",
]
