"""Locked IL8/IL10/IL15 Atomic target-depth extension configuration."""

from __future__ import annotations

from typing import Iterable

from .contracts import ODEBFContractError
from .p1r52_target_depth import (
    P1R52_TARGET_DEPTH_EXTENSION_INSTRUCTION_ID,
    P1R52_TARGET_DEPTH_EXTENSION_METHOD_ID,
    P1R52TargetDepth,
)


INSTRUCTION_ID = P1R52_TARGET_DEPTH_EXTENSION_INSTRUCTION_ID
METHOD_ID = P1R52_TARGET_DEPTH_EXTENSION_METHOD_ID
MODEL = "llama3-8b-inst"
ARM = "soft"
CASE_COUNT = 10
REQUEST_COUNT = 100
OUTER_STEP_COUNT = 8
INNER_H = 1.0 / 8.0
PROJECT_GPU_CAP = 3
ATTEMPT_SUFFIX = "extension-inner-telemetry-r1"
EXTENSION_DEPTHS = (
    P1R52TargetDepth.IL8_FULL,
    P1R52TargetDepth.IL10_FULL,
    P1R52TargetDepth.IL15_FULL,
)


def validate_extension_depths(
    values: Iterable[P1R52TargetDepth | str],
) -> tuple[P1R52TargetDepth, ...]:
    depths = tuple(
        value if isinstance(value, P1R52TargetDepth) else P1R52TargetDepth(value)
        for value in values
    )
    if depths != EXTENSION_DEPTHS:
        raise ODEBFContractError("P1R52 extension depths must be exactly IL8/IL10/IL15")
    return depths


def depth_slug(depth: P1R52TargetDepth | str) -> str:
    policy = depth if isinstance(depth, P1R52TargetDepth) else P1R52TargetDepth(depth)
    if policy not in EXTENSION_DEPTHS:
        raise ODEBFContractError("P1R52 extension depth is outside the locked list")
    return policy.value.lower().replace("-", "")


def extension_result_name(depth: P1R52TargetDepth | str) -> str:
    return (
        "s05-p1r52-target-depth-atomic-b10x10-"
        f"{MODEL}-{ARM}-{depth_slug(depth)}-{ATTEMPT_SUFFIX}-v1"
    )


def extension_cells() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "array_task": task,
            "model": MODEL,
            "arm": ARM,
            "depth": depth.value,
            "inner_count": depth.inner_count,
            "inner_h": INNER_H,
            "case_count": CASE_COUNT,
            "request_attempt_count": REQUEST_COUNT,
            "expected_inner_telemetry_rows_per_case": (
                OUTER_STEP_COUNT * depth.inner_count
            ),
            "expected_outer_telemetry_rows_per_case": OUTER_STEP_COUNT,
            "inner_observation_pass_count_per_accepted_inner": 1,
            "outer_observation_pass_count_per_committed_outer": 2,
            "result_name": extension_result_name(depth),
        }
        for task, depth in enumerate(EXTENSION_DEPTHS)
    )


__all__ = [
    "ARM",
    "ATTEMPT_SUFFIX",
    "CASE_COUNT",
    "EXTENSION_DEPTHS",
    "INNER_H",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "MODEL",
    "OUTER_STEP_COUNT",
    "PROJECT_GPU_CAP",
    "REQUEST_COUNT",
    "depth_slug",
    "extension_cells",
    "extension_result_name",
    "validate_extension_depths",
]
