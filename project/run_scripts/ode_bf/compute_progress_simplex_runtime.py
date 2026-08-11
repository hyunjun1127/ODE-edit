"""Compute-aware Progress-Simplex contracts shared by Atomic and Historical.

The Atomic path uses an empty H sketch.  The transaction API is nevertheless
implemented here so the later Historical phase can commit a fixed-rank sketch
exactly once after a successful weight transaction without replaying prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash


COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-BF-COMPUTE-A1"
)
COMPUTE_PROGRESS_SIMPLEX_METHOD_ID = (
    "P1R23_COMPUTE_AWARE_PROGRESS_SIMPLEX_STRUCTURAL_PH_V1"
)
COMPUTE_TOKEN_BUDGET = 7_200
COMPUTE_FIELD_MODEL_FORWARD_CEILING = 24
ROTATING_CONTEXT_SCHEDULE = ((0, 3), (1, 4), (2, 5))


def rotating_context_ordinals(step_index: int) -> tuple[int, int]:
    if isinstance(step_index, bool) or step_index < 0:
        raise ODEBFContractError("compute-aware rotating step differs")
    return ROTATING_CONTEXT_SCHEDULE[step_index % len(ROTATING_CONTEXT_SCHEDULE)]


def validate_field_forward_count(
    *, target_forwards: int, slope_forwards: int, capture_forwards: int
) -> int:
    values = (target_forwards, slope_forwards, capture_forwards)
    if any(isinstance(item, bool) or item <= 0 for item in values):
        raise ODEBFContractError("compute-aware field forward count differs")
    total = sum(values)
    if total > COMPUTE_FIELD_MODEL_FORWARD_CEILING:
        raise ODEBFContractError("compute-aware field forward ceiling exceeded")
    return total


def empty_historical_sketch_receipt(layer_order: Sequence[int]) -> dict[str, Any]:
    layers = tuple(int(item) for item in layer_order)
    if not layers or len(set(layers)) != len(layers):
        raise ODEBFContractError("compute-aware H layer order differs")
    payload = {
        "schema": "ode-edit-s05-p1r23-fixed-rank-h-sketch/v1",
        "layer_order": list(layers),
        "rank": "NOT_FROZEN_ATOMIC_PHASE",
        "history_item_count": 0,
        "commit_count": 0,
        "rollback_update_count": 0,
        "decision_influence_count": 0,
        "raw_history_replay_count": 0,
        "rows_sha256": {str(layer): canonical_hash([]) for layer in layers},
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


@dataclass(slots=True)
class FixedRankHistoricalSketch:
    """Deterministic truncated-SVD sufficient statistic with staged commit."""

    rank: int
    width_by_layer: Mapping[int, int]
    _rows: dict[int, np.ndarray] = field(init=False, repr=False)
    _pending: dict[int, np.ndarray] | None = field(default=None, init=False, repr=False)
    _pending_item_count: int = field(default=0, init=False, repr=False)
    commit_count: int = 0
    history_item_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or self.rank <= 0 or not self.width_by_layer:
            raise ODEBFContractError("fixed-rank H geometry differs")
        checked = {int(layer): int(width) for layer, width in self.width_by_layer.items()}
        if any(width <= 0 for width in checked.values()):
            raise ODEBFContractError("fixed-rank H width differs")
        self.width_by_layer = checked
        self._rows = {
            layer: np.zeros((0, width), dtype=np.float64)
            for layer, width in sorted(checked.items())
        }

    def stage(
        self,
        projected_keys: Mapping[int, np.ndarray],
        *,
        item_count: int | None = None,
    ) -> str:
        if self._pending is not None or set(projected_keys) != set(self._rows):
            raise ODEBFStateError("fixed-rank H stage differs")
        pending: dict[int, np.ndarray] = {}
        batch_sizes: set[int] = set()
        for layer, current in self._rows.items():
            incoming = np.asarray(projected_keys[layer], dtype=np.float64)
            if incoming.ndim != 2 or incoming.shape[1] != current.shape[1] or not np.isfinite(incoming).all():
                raise ODEBFContractError("fixed-rank H projected key differs")
            batch_sizes.add(int(incoming.shape[0]))
            joined = np.concatenate((current, incoming), axis=0)
            if joined.shape[0] > self.rank:
                _u, singular, vt = np.linalg.svd(joined, full_matrices=False)
                keep = min(self.rank, len(singular))
                joined = singular[:keep, None] * vt[:keep, :]
            pending[layer] = joined
        if not batch_sizes or min(batch_sizes) <= 0:
            raise ODEBFContractError("fixed-rank H batch count differs")
        if item_count is not None:
            if (
                isinstance(item_count, bool)
                or item_count <= 0
                or batch_sizes != {int(item_count)}
            ):
                raise ODEBFContractError("fixed-rank H item count differs")
            pending_item_count = int(item_count)
        else:
            # Backward-compatible generic sketch use may stage a different
            # number of basis rows per layer.  Sequential request accounting
            # always supplies the explicit common logical item count above.
            pending_item_count = max(batch_sizes)
        self._pending = pending
        self._pending_item_count = pending_item_count
        return canonical_hash(
            {str(layer): value.tolist() for layer, value in sorted(pending.items())}
        )

    def finalize(self, *, transaction_committed: bool) -> None:
        if self._pending is None:
            raise ODEBFStateError("fixed-rank H has no staged update")
        if transaction_committed:
            self._rows = self._pending
            self.commit_count += 1
            self.history_item_count += self._pending_item_count
        self._pending = None
        self._pending_item_count = 0

    def gram(self, layer: int) -> np.ndarray:
        if self._pending is not None:
            raise ODEBFStateError("fixed-rank H read during transaction")
        rows = self._rows[int(layer)]
        return rows.T @ rows

    def action(
        self, layer: int, residual: np.ndarray, q: np.ndarray
    ) -> np.ndarray:
        """Return ``B_l S_H^T`` without replaying historical prompts."""

        if self._pending is not None:
            raise ODEBFStateError("fixed-rank H read during transaction")
        rows = self._rows[int(layer)]
        left = np.asarray(residual, dtype=np.float64)
        right = np.asarray(q, dtype=np.float64)
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[1] != right.shape[1]
            or right.shape[0] != rows.shape[1]
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            raise ODEBFContractError("fixed-rank H action geometry differs")
        return left @ (right.T @ rows.T)

    def weight_action(self, layer: int, weight_delta: np.ndarray) -> np.ndarray:
        if self._pending is not None:
            raise ODEBFStateError("fixed-rank H read during transaction")
        rows = self._rows[int(layer)]
        value = np.asarray(weight_delta, dtype=np.float64)
        if (
            value.ndim != 2
            or value.shape[1] != rows.shape[1]
            or not np.isfinite(value).all()
        ):
            raise ODEBFContractError("fixed-rank H weight action geometry differs")
        return value @ rows.T

    def raw_free_payload(self) -> dict[str, Any]:
        if self._pending is not None:
            raise ODEBFStateError("fixed-rank H receipt during transaction")
        payload = {
            "schema": "ode-edit-s05-p1r23-fixed-rank-h-sketch/v1",
            "rank": self.rank,
            "commit_count": self.commit_count,
            "history_item_count": self.history_item_count,
            "layers": {
                str(layer): {
                    "row_count": int(rows.shape[0]),
                    "width": int(rows.shape[1]),
                    "frobenius_norm": float(np.linalg.norm(rows)),
                    "rows_sha256": canonical_hash(rows.tolist()),
                }
                for layer, rows in sorted(self._rows.items())
            },
            "raw_history_replay_count": 0,
        }
        if any(not math.isfinite(item["frobenius_norm"]) for item in payload["layers"].values()):
            raise ODEBFContractError("fixed-rank H receipt is non-finite")
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = [
    "COMPUTE_FIELD_MODEL_FORWARD_CEILING",
    "COMPUTE_PROGRESS_SIMPLEX_INSTRUCTION_ID",
    "COMPUTE_PROGRESS_SIMPLEX_METHOD_ID",
    "COMPUTE_TOKEN_BUDGET",
    "FixedRankHistoricalSketch",
    "empty_historical_sketch_receipt",
    "rotating_context_ordinals",
    "validate_field_forward_count",
]
