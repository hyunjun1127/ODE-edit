"""Immutable deployment contract for ORBODE B100x10 sequential execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import TechnicalBoundary
from .preflight import ORDER_ROOT, STREAM_ROOT, canonical_hash


INSTRUCTION_ID = (
    "ODEEDIT-S06-ORDERED-RESPONSE-BARRIER-ODE-SEQUENTIAL-B100X10-SERVER4-V1"
)
NONCE = "ODEEDIT-ORRBODE-SEQUENTIAL-SH4-20260905-R1"
ARM_ORDER = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
DERIVED_ARM = "ORBHit"
ROUND_INDICES = tuple(range(10))


@dataclass(frozen=True, slots=True)
class SequentialRuntimeLock:
    execution_semantics: str = "ARM_LOCAL_B1_TO_B10_CUMULATIVE_W_CACHE_SEQUENTIAL"
    stream_root: str = STREAM_ROOT
    order_root: str = ORDER_ROOT
    batch_size: int = 100
    batch_count: int = 10
    request_count_per_arm: int = 1_000
    T: float = 1.0
    N: int = 4
    h: float = 0.25
    full_fp32: bool = True
    alpha_history: str = "DYNAMIC_APPEND_ON_SUCCESSFUL_TERMINAL_COMMIT"
    memit_covariance: str = "STATIC_COMPUTATION_CACHE"
    arm_entry_state: str = "COMMON_ORIGINAL_W0_AND_COLD_METHOD_STATE"
    inter_arm_state_carry: int = 0
    inter_batch_state_carry: int = 1
    fixed_z_recompute_within_batch: int = 0
    heldout_controller_influence: int = 0
    retry_count: int = 0
    backtracking_count: int = 0
    fallback_count: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["arms"] = list(ARM_ORDER)
        value["derived_arm"] = DERIVED_ARM
        value["round_indices"] = list(ROUND_INDICES)
        value["identity_sha256"] = canonical_hash(value)
        return value


def validate_batch_chain(
    records: Sequence[Mapping[str, Any]], *, family: str
) -> dict[str, Any]:
    """Validate one arm's ten committed B100 transactions."""

    if family not in {"MEMIT", "AlphaEdit"} or len(records) != 10:
        raise TechnicalBoundary("sequential arm chain shape differs")
    weight_links = 0
    method_links = 0
    fixed_z_compute = 0
    fixed_z_recompute = 0
    request_total = 0
    for index, record in enumerate(records):
        if (
            int(record.get("batch_index", -1)) != index
            or int(record.get("request_count", -1)) != 100
            or record.get("status") != "SEQUENTIAL_BATCH_TERMINAL_VALID"
        ):
            raise TechnicalBoundary("sequential batch terminal contract differs")
        request_total += int(record["request_count"])
        fixed_z_compute += int(record.get("fixed_z_compute_count", -1))
        fixed_z_recompute += int(record.get("fixed_z_recompute_count", -1))
        if index:
            previous = records[index - 1]
            if record.get("entry_weight_sha256") != previous.get(
                "committed_weight_sha256"
            ):
                raise TechnicalBoundary("B exit to next entry W identity differs")
            weight_links += 1
            if record.get("entry_method_state_sha256") != previous.get(
                "committed_method_state_sha256"
            ):
                raise TechnicalBoundary("B exit to next method-state identity differs")
            method_links += 1
    if request_total != 1_000 or fixed_z_compute != 1_000 or fixed_z_recompute != 0:
        raise TechnicalBoundary("sequential fixed-z/request accounting differs")
    return {
        "batch_count": 10,
        "request_count": request_total,
        "weight_commit_to_next_entry": {"numerator": weight_links, "denominator": 9},
        "method_state_commit_to_next_entry": {
            "numerator": method_links,
            "denominator": 9,
        },
        "fixed_z_compute_count": fixed_z_compute,
        "fixed_z_recompute_count": fixed_z_recompute,
        "family": family,
        "status": "SEQUENTIAL_CHAIN_PASS",
    }


__all__ = [
    "ARM_ORDER",
    "DERIVED_ARM",
    "INSTRUCTION_ID",
    "NONCE",
    "ROUND_INDICES",
    "SequentialRuntimeLock",
    "validate_batch_chain",
]
