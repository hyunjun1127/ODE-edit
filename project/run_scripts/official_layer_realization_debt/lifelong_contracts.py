"""Immutable contract for the Official B100 lifelong observation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import LAYERS


INSTRUCTION_ID = (
    "ODEEDIT-S06-OFFICIAL-LAYER-REALIZATION-DEBT-LIFELONG-B100-V1"
)
CONTRACT_SHA256 = (
    "77dcf513bbfe3c2f7451f3e1387d18347d917b2afdc7688403b48f4100041f42"
)
CONTRACT_BYTES = 15_670
CONTRACT_WC_LINES = 517
CAMPAIGN_ORDER_SEED_INDEX = 1
CHECKPOINT_BATCHES = (0, 10, 15, 20, 30, 50, 75, 100)
COMPACT_REPORT_BATCHES = (10, 20, 50, 100)
MODEL_METHOD_PRIORITY = (
    ("qwen2.5-7b-inst", "alphaedit"),
    ("qwen2.5-7b-inst", "memit"),
    ("llama3-8b-inst", "alphaedit"),
    ("llama3-8b-inst", "memit"),
)


@dataclass(frozen=True, slots=True)
class LifelongLock:
    layers: tuple[int, ...] = LAYERS
    batch_size: int = 100
    batch_count: int = 100
    request_count: int = 10_000
    order_seed_index: int = CAMPAIGN_ORDER_SEED_INDEX
    scalar_reduction_dtype: str = "float64"
    recurrence_relative_tolerance: float = 1.0e-4
    checkpoint_batches: tuple[int, ...] = CHECKPOINT_BATCHES
    compact_report_batches: tuple[int, ...] = COMPACT_REPORT_BATCHES
    sustained_violation_batch_count: int = 3
    observer_layer_calls_per_batch: int = 5
    observer_terminal_calls_per_batch: int = 1
    compute_z_per_request: int = 1
    accepted_cache_append_per_request: int = 1
    sample_duplication_count: int = 0
    full_fp32: bool = True
    gpu_cap: int = 2
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "CAMPAIGN_ORDER_SEED_INDEX",
    "CHECKPOINT_BATCHES",
    "COMPACT_REPORT_BATCHES",
    "CONTRACT_BYTES",
    "CONTRACT_SHA256",
    "CONTRACT_WC_LINES",
    "INSTRUCTION_ID",
    "LifelongLock",
    "MODEL_METHOD_PRIORITY",
]
