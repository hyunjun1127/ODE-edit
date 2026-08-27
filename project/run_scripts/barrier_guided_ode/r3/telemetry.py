"""Typed observation-only G0/G1 telemetry boundaries for BGODE-R3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import R3ScientificBoundary


@dataclass(frozen=True, slots=True)
class R3ExecutionBoundary:
    batch_size: int
    fixed_target_compute_count: int
    fixed_target_recompute_count: int
    model_numeric_dtype: str
    event_numeric_dtype: str
    physical_write_dtype: str
    node_history_append_count: int
    terminal_history_append_count: int
    localizer_call_count: int
    endpoint_correction_count: int
    accepted_step_search_count: int
    alternate_solver_count: int
    controller_evaluator_influence_count: int
    retained_factor_count: int
    scientific_promotion: bool

    def __post_init__(self) -> None:
        expected = {
            "batch_size": 1,
            "fixed_target_compute_count": 1,
            "fixed_target_recompute_count": 0,
            "model_numeric_dtype": "torch.float32",
            "event_numeric_dtype": "torch.float64",
            "physical_write_dtype": "torch.float32",
            "node_history_append_count": 0,
            "localizer_call_count": 0,
            "endpoint_correction_count": 0,
            "accepted_step_search_count": 0,
            "alternate_solver_count": 0,
            "controller_evaluator_influence_count": 0,
            "retained_factor_count": 0,
            "scientific_promotion": False,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise R3ScientificBoundary("R3 execution boundary differs from its science lock")
        if self.terminal_history_append_count not in (0, 1):
            raise R3ScientificBoundary("terminal history append count must be pre-commit0 or accepted1")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["R3ExecutionBoundary"]
