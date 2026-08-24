"""Thin independent-B100 binding for the P1R54 FZ C3 transaction.

Every array role selects one canonical Phase123 B100 slice and delegates the
science transaction to the frozen P1R54 atomic B100 runtime.  The adapter does
not copy sample payloads and owns no field, writer, cache, evaluator, projector,
covariance, or rollback implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_target_timescale import TargetTimescaleCell, schedule_for_cell
from .p1r52_target_timescale_b100 import run_target_timescale_b100
from .p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from .p1r54_fz_c3_sequential import (
    ROLE as SEQUENTIAL_ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    atomic_b1_source_equivalence,
)
from .scalable_batched_runtime import scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R54-FZ-C3-INDEPENDENT-10XB100-V1"
METHOD_ID = "P1R54-FZ-C3-KSTEP-INDEPENDENT-B100-FULL-FP32"
ROLE_PREFIX = "r54-fz-c3-independent-full-fp32-b"
ROLES = tuple(f"{ROLE_PREFIX}{index:02d}" for index in range(1, ROUND_COUNT + 1))
TECHNICAL_ATTEMPT_SUFFIX = "tech-r2"
RESULT_NAMES = {
    role: (
        f"s05-p1r54-fz-c3-independent-b{index:02d}-b100-"
        f"{TECHNICAL_ATTEMPT_SUFFIX}-v1"
    )
    for index, role in enumerate(ROLES, start=1)
}
HELDOUT_K_INDICES = (7,)


class P1R54FZExecutionMode(str, Enum):
    """Typed single-variable comparison boundary for the two FZ arms."""

    SEQUENTIAL_CONTINUITY = "SEQUENTIAL_CONTINUITY"
    INDEPENDENT_BATCH = "INDEPENDENT_BATCH"


@dataclass(frozen=True, slots=True)
class IndependentBatchCell:
    cell: int
    batch_index: int
    role: str
    result_name: str
    execution_mode: P1R54FZExecutionMode = P1R54FZExecutionMode.INDEPENDENT_BATCH
    weight_entry_version: int = 0
    alpha_cache_entry_version: int = 0
    alpha_cache_entry_width: int = 0
    cross_batch_state_consumption_count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.cell, bool)
            or self.cell != self.batch_index - 1
            or not 0 <= self.cell < ROUND_COUNT
            or self.role != ROLES[self.cell]
            or self.result_name != RESULT_NAMES[self.role]
            or self.execution_mode is not P1R54FZExecutionMode.INDEPENDENT_BATCH
            or self.weight_entry_version != 0
            or self.alpha_cache_entry_version != 0
            or self.alpha_cache_entry_width != 0
            or self.cross_batch_state_consumption_count != 0
        ):
            raise ODEBFContractError("P1R54 independent cell contract differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell": self.cell,
            "batch_index": self.batch_index,
            "batch_label": f"B{self.batch_index}",
            "role": self.role,
            "result_name": self.result_name,
            "execution_mode": self.execution_mode.value,
            "weight_entry_version": self.weight_entry_version,
            "alpha_cache_entry_version": self.alpha_cache_entry_version,
            "alpha_cache_entry_width": self.alpha_cache_entry_width,
            "cross_batch_state_consumption_count": (
                self.cross_batch_state_consumption_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def cell_config(cell: int) -> IndependentBatchCell:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < ROUND_COUNT:
        raise ODEBFContractError("P1R54 independent array cell differs")
    role = ROLES[cell]
    return IndependentBatchCell(cell, cell + 1, role, RESULT_NAMES[role])


def role_for_cell(cell: int) -> str:
    return cell_config(cell).role


def cell_for_role(role: str) -> IndependentBatchCell:
    try:
        return cell_config(ROLES.index(role))
    except ValueError as exc:
        raise ODEBFContractError("P1R54 independent role differs") from exc


def expected_result_name(role: str) -> str:
    return cell_for_role(role).result_name


def is_p1r54_fz_independent_role(role: str | None) -> bool:
    return role in ROLES


def bind_independent_batch_view(
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    *,
    batch_index: int,
) -> tuple[Sequence[Mapping[str, Any]], ...]:
    """Return a reference-only view with the selected sealed slice first.

    The reused atomic runtime consumes only element zero.  Remaining references
    keep its pre-existing full-stream shape gate satisfied; no request mapping,
    payload, cache, history, factor, or model state is copied between cells.
    """

    if (
        isinstance(batch_index, bool)
        or not 1 <= batch_index <= ROUND_COUNT
        or len(stream_batches) != ROUND_COUNT
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P1R54 independent stream geometry differs")
    batch_orders = stream.get("batch_ordered_request_digest_v1")
    if not isinstance(batch_orders, list) or len(batch_orders) != ROUND_COUNT:
        raise ODEBFContractError("P1R54 independent batch-order seal differs")
    selected_offset = batch_index - 1
    observed_order = scalable_ordered_request_digest(
        [
            str(item["request_sha256"])
            for item in stream_batches[selected_offset]
        ]
    )
    if observed_order != batch_orders[selected_offset]:
        raise ODEBFStateError("P1R54 independent selected slice order differs")
    order = (selected_offset, *(index for index in range(ROUND_COUNT) if index != selected_offset))
    return tuple(stream_batches[index] for index in order)


def atomic_source_equivalence(batch_index: int) -> Mapping[str, Any]:
    """Outcome-free proof that only canonical slice selection changes."""

    config = cell_config(batch_index - 1)
    atomic = atomic_b1_source_equivalence()
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-independent-atomic-source-equivalence/v1",
        "instruction_id": INSTRUCTION_ID,
        "execution_mode": config.execution_mode.value,
        "canonical_batch_index": batch_index,
        "atomic_transaction": (
            "project.run_scripts.ode_bf.p1r52_target_timescale_b100."
            "run_target_timescale_b100"
        ),
        "atomic_role": atomic["atomic_role"],
        "sequential_role": SEQUENTIAL_ROLE,
        "amplitude_policy_class": atomic["amplitude_policy_class"],
        "amplitude_arm": atomic["amplitude_arm"],
        "formula": atomic["formula"],
        "target_schedule_identity": atomic["target_schedule_identity"],
        "target_dt": atomic["target_dt"],
        "microsteps_per_outer": atomic["microsteps_per_outer"],
        "target_horizon": atomic["target_horizon"],
        "outer_count": atomic["outer_count"],
        "writer_arm": atomic["writer_arm"],
        "heldout_k_indices": list(HELDOUT_K_INDICES),
        "weight_entry_version": 0,
        "alpha_cache_entry_version": 0,
        "alpha_cache_entry_width": 0,
        "cache_append_after_k8_count": 1,
        "terminal_W0_restore_required": True,
        "different_scientific_input_count": 0,
        "cross_batch_state_consumption_count": 0,
        "sample_payload_copy_count": 0,
        "retry_imputation_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def run_p1r54_fz_c3_independent(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    config = cell_for_role(role)
    selected_view = bind_independent_batch_view(
        stream_batches,
        stream,
        batch_index=config.batch_index,
    )
    policy = EnergyFreeLocalZPolicy(
        EnergyFreeLocalZArm.FZ,
        request_count=BATCH_SIZE,
        outer_count=8,
    )
    return run_target_timescale_b100(
        model,
        tokenizer,
        role=role,
        stream_batches=selected_view,
        stream=stream,
        target_schedule_override=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy=policy,
        heldout_k_indices=HELDOUT_K_INDICES,
        experiment_instruction_id=INSTRUCTION_ID,
        experiment_method_id=METHOD_ID,
        terminal_schema="ode-edit-s05-p1r54-fz-c3-independent-cell-terminal/v1",
        manifest_schema="ode-edit-s05-p1r54-fz-c3-independent-cell-manifest/v1",
        terminal_status="P1R54_FZ_C3_INDEPENDENT_CELL_TERMINAL",
        stage_prefix=f"p1r54_fz_c3_independent_b{config.batch_index:02d}",
        amplitude_writer_layer_apply_count_key="writer_layer_apply_count",
        experiment_metadata={
            "execution_mode": config.execution_mode.value,
            "canonical_batch_index": config.batch_index,
            "canonical_batch_label": f"B{config.batch_index}",
            "canonical_batch_order_sha256": str(
                stream["batch_ordered_request_digest_v1"][config.batch_index - 1]
            ),
            "independent_cell_contract": config.raw_free_payload(),
            "atomic_source_equivalence": atomic_source_equivalence(
                config.batch_index
            ),
            "formula": "F_i,k=m_i,k*||z0_i||*d_i,k",
            "writer_policy": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
            "cold_alpha_cache_entry": True,
            "weight_entry_version": 0,
            "alpha_cache_entry_version": 0,
            "cross_batch_W_cache_history_factor_materializer_consumption_count": 0,
            "sample_payload_count": 1,
            "sample_payload_copy_count": 0,
            "retry_count_contract": 0,
            "imputation_count": 0,
            "heldout_decision_influence_count_contract": 0,
            "reference_execution_count": 0,
            "sequential_job_mutation_count": 0,
            "technical_attempt_suffix": TECHNICAL_ATTEMPT_SUFFIX,
        },
        easyedit_root=easyedit_root,
        **kwargs,
    )


__all__ = [
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "IndependentBatchCell",
    "METHOD_ID",
    "P1R54FZExecutionMode",
    "RESULT_NAMES",
    "ROLES",
    "TECHNICAL_ATTEMPT_SUFFIX",
    "atomic_source_equivalence",
    "bind_independent_batch_view",
    "cell_config",
    "cell_for_role",
    "expected_result_name",
    "is_p1r54_fz_independent_role",
    "role_for_cell",
    "run_p1r54_fz_c3_independent",
]
