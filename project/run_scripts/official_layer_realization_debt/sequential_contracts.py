"""Immutable deployment lock for cumulative B1-to-B10 observation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import LAYERS


INSTRUCTION_ID = "ODEEDIT-S06-OFFICIAL-LAYER-REALIZATION-DEBT-SEQUENTIAL-B10X10-V1"
NONCE = "ODEEDIT-GH-SH4-OFFICIAL-LAYER-DEBT-SEQUENTIAL-20260901-R1"


@dataclass(frozen=True, slots=True)
class SequentialObservationLock:
    layers: tuple[int, ...] = LAYERS
    scalar_reduction_dtype: str = "float64"
    recurrence_relative_tolerance: float = 1.0e-4
    full_fp32: bool = True
    sequential_batch_count: int = 10
    requests_per_batch: int = 10
    total_requests_per_cell: int = 100
    expected_direct_z_per_cell: int = 100
    expected_direct_z_recompute_per_cell: int = 0
    expected_layer_observation_per_cell: int = 50
    expected_terminal_forward_per_cell: int = 10
    cross_batch_weight_continuity: int = 1
    alphaedit_dynamic_cache_continuity: int = 1
    memit_request_history_cache_count: int = 0
    terminal_w0_restore_count: int = 1
    sample_duplication_count: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)
