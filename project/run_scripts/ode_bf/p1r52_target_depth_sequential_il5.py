"""Typed lock for the P1R52 Llama J0 IL5 sequential extension."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ODEBFContractError
from .p1r52_target_depth import P1R52TargetDepth


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-LLAMA-J0-IL5-SEQUENTIAL-10XB100-V1"
METHOD_ID = "P1R52-LLAMA-J0-TARGET-DEPTH-IL5-SEQUENTIAL-10XB100"
ATTEMPT_SUFFIX = "target-depth-il5-postenergy-warn-r1-tech-r2"
ROLE = "r52-soft-sequential-h"
RESULT_NAME = (
    "s05-p1r52-llama-j0-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"
)
JOB_NAME = "odeedit_p1r52_il5_seq_b1000_pew1r2"
RUN_TOKEN = (
    "p1r52-llama-j0-il5-sequential-10xb100-postenergy-warn-r1-tech-r2-v1"
)
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


@dataclass(frozen=True, slots=True)
class P1R52IL5SequentialPolicy:
    depth: P1R52TargetDepth = P1R52TargetDepth.IL5_FULL
    inner_telemetry: bool = True
    batch_entry_evaluator: bool = False
    round_count: int = 10
    batch_size: int = 100
    outer_count: int = 8
    h: float = 0.125
    postsolve_energy_warn_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            self.depth is not P1R52TargetDepth.IL5_FULL
            or self.depth.inner_count != 5
            or not self.inner_telemetry
            or self.batch_entry_evaluator
            or self.round_count != 10
            or self.batch_size != 100
            or self.outer_count != 8
            or self.h != 0.125
            or not self.postsolve_energy_warn_enabled
        ):
            raise ODEBFContractError("P1R52 IL5 sequential policy differs")

    @property
    def role(self) -> str:
        return ROLE

    @property
    def request_count(self) -> int:
        return self.round_count * self.batch_size

    @property
    def expected_inner_rows_per_batch(self) -> int:
        return self.depth.inner_count * self.outer_count

    @property
    def inner_h(self) -> float:
        return self.h

    @property
    def expected_request_inner_rows_per_batch(self) -> int:
        return self.expected_inner_rows_per_batch * self.batch_size


POLICY = P1R52IL5SequentialPolicy()


def validate_runtime_activation(
    *,
    role: str,
    scale_id: str,
    attempt_suffix: str | None,
    depth: P1R52TargetDepth | str | None,
    inner_telemetry: bool,
    batch_entry_evaluator: bool,
    postsolve_energy_warn_enabled: bool,
) -> P1R52IL5SequentialPolicy:
    resolved = (
        depth
        if isinstance(depth, P1R52TargetDepth)
        else P1R52TargetDepth(depth)
        if depth is not None
        else None
    )
    if (
        role != ROLE
        or scale_id != "b100x10"
        or attempt_suffix != ATTEMPT_SUFFIX
        or resolved is not POLICY.depth
        or inner_telemetry is not True
        or batch_entry_evaluator is not False
        or postsolve_energy_warn_enabled is not True
    ):
        raise ODEBFContractError("P1R52 IL5 sequential activation differs")
    return POLICY


__all__ = [
    "ATTEMPT_SUFFIX",
    "INSTRUCTION_ID",
    "JOB_NAME",
    "METHOD_ID",
    "POLICY",
    "P1R52IL5SequentialPolicy",
    "RESULT_NAME",
    "ROLE",
    "RUN_TOKEN",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "validate_runtime_activation",
]
