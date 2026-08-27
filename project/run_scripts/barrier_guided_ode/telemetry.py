"""Raw-free immutable telemetry for the BGODE-R1 CPU core."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import BGODEScientificBoundary, UnsupportedBatchBoundary


@dataclass(frozen=True, slots=True)
class ExecutionBoundaryReceipt:
    """Counters that make the fixed-target/history/leakage boundary explicit."""

    batch_size: int
    z_compute_count: int
    z_recompute_count: int
    euler_node_count: int
    history_append_inside_euler_count: int
    terminal_history_append_count: int
    controller_evaluator_input_count: int
    heldout_decision_influence_count: int
    length_normalization_count: int
    progress_penalty_count: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise BGODEScientificBoundary("execution counters must be non-negative integers")
        if self.batch_size != 1:
            raise UnsupportedBatchBoundary("BGODE-R1 atomic batch size must equal one")
        if self.z_compute_count != 1 or self.z_recompute_count != 0:
            raise BGODEScientificBoundary("BGODE-R1 requires one fixed z* computation")
        if self.history_append_inside_euler_count != 0:
            raise BGODEScientificBoundary("history cannot append inside Euler nodes")
        if self.terminal_history_append_count not in {0, 1}:
            raise BGODEScientificBoundary("terminal history append count must be zero or one")
        forbidden = (
            self.controller_evaluator_input_count,
            self.heldout_decision_influence_count,
            self.length_normalization_count,
            self.progress_penalty_count,
        )
        if any(forbidden):
            raise BGODEScientificBoundary("forbidden controller influence is non-zero")


@dataclass(frozen=True, slots=True)
class RayleighianNumericalReceipt:
    dimension: int
    equality_count: int
    matrix_rank: int
    equality_rank: int
    dtype: str
    matrix_scale: float
    pinv_tolerance: float
    range_residual_g: float
    range_residual_a_max: float
    equality_residual: float
    stationarity_residual: float
    objective: float

    def __post_init__(self) -> None:
        for name in ("dimension", "equality_count", "matrix_rank", "equality_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BGODEScientificBoundary(f"{name} must be positive")
        if self.dtype != "torch.float32":
            raise BGODEScientificBoundary("BGODE-R1 numerical core must be FP32")
        for name in (
            "matrix_scale",
            "pinv_tolerance",
            "range_residual_g",
            "range_residual_a_max",
            "equality_residual",
            "stationarity_residual",
            "objective",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise BGODEScientificBoundary(f"{name} must be finite")
        if self.matrix_scale <= 0.0 or self.pinv_tolerance <= 0.0:
            raise BGODEScientificBoundary("matrix scale/tolerance must be positive")
        if min(
            self.range_residual_g,
            self.range_residual_a_max,
            self.equality_residual,
            self.stationarity_residual,
        ) < 0.0:
            raise BGODEScientificBoundary("residuals must be non-negative")
