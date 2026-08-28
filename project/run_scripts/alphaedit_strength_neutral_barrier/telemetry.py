"""Typed, create-once telemetry for the barrier writer."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StateObservationTelemetry:
    per_event_q_kl: List[float]
    aggregate_q_kl: float
    target_token_log_odds: List[float]
    target_token_nll: List[float]
    target_sequence_nll: List[float]
    source_sequence_nll: List[float]
    target_source_margin: List[float]


@dataclass
class NodeTelemetry:
    layer: int
    layer_index: int
    node: int
    steps: int
    barrier_q_kl: float
    target_nll: List[float]
    target_log_odds: List[float]
    native_barrier_rate: float
    feasible_gradient_norm: float
    correction_norm: float
    native_delta_norm: float
    correction_native_ratio: float
    directional_rate: float
    key_residual_max_abs: float
    right_projection_key_residual_max_abs: float
    max_strength_inner_abs: float
    key_gram_rank: int
    key_gram_condition: float
    key_pinv_rtol: float
    constraint_rank: int
    constraint_condition: float
    constraint_pinv_rtol: float
    correction_active: bool
    native_predictor: bool
    pre_step: StateObservationTelemetry
    post_native_counterfactual: StateObservationTelemetry
    post_guided: StateObservationTelemetry
    actual_dk_norm: float
    actual_dk_max_abs: float
    target_strength_constraint_residual_per_event_abs: List[float]
    target_strength_constraint_residual_max_abs: float
    selected_weight_endpoint_sha256: str
    predictor_restore_pass: bool


@dataclass
class LayerTelemetry:
    layer: int
    entry_z_residual_norm: float
    terminal_z_residual_norm: float
    terminal_barrier_q_kl: float
    terminal_target_nll: List[float]
    nodes: List[NodeTelemetry] = field(default_factory=list)


@dataclass
class WriterTelemetry:
    schema: str
    arm: str
    steps: int
    request_count: int
    target_event_count: int
    target_event_identity_sha256: str
    q0_identity_sha256: str
    fixed_z_compute_count: int
    fixed_z_recompute_count: int
    projector_load_count: int
    cache_append_count: int
    locality_controller_influence_count: int
    rephrase_controller_influence_count: int
    retry_count: int
    imputation_count: int
    dtype: str
    layers: List[LayerTelemetry] = field(default_factory=list)
    terminal_status: str = "RUNNING"
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None
    w0_restore_pass: bool = False
    w0_selected_sha256: Optional[str] = None
    temporary_endpoint_selected_sha256: Optional[str] = None
    authoritative_replay_selected_sha256: Optional[str] = None
    authoritative_replay_max_abs_by_weight: Dict[str, float] = field(default_factory=dict)
    authoritative_endpoint_selected_sha256: Optional[str] = None
    authoritative_endpoint_replay_pass: bool = False
    restored_w0_selected_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_create_once(path: str | Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Publish a regular mode-0600 JSON file without following symlinks."""

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = parent
    while True:
        if current.is_symlink():
            raise RuntimeError(f"telemetry parent is a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"create-once telemetry already exists: {path}")
    data = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data).hexdigest()}
