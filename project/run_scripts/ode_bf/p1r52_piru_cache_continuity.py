"""Narrow Alpha-cache continuity policy for P1R52 PIR-U prefixes.

Only the history supplied to the L5--L8 q-only Alpha solve is selected here.
The current batch and all uncommitted prefix keys remain outside the ledger.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_runtime import P1R23_LAYER_ORDER


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-PIRU-ALPHA-CACHE-CONTINUITY-A1-V1"
METHOD_ID = "P1R52-PIRU-ALPHA-CACHE-CONTINUITY-A1"
CONTRACT_SHA256 = "e25a644e2a5b79ed0f660985f3a3e932856a2184274ead128aa41db8ae613b28"


class PIRUCacheContinuityPolicy(str, Enum):
    LEGACY = "PIRU-LEGACY"
    CACHE_COMPLETE = "PIRU-CACHE-COMPLETE"


LEGACY_ROLE = "r52-pir-u-cache-continuity-legacy-structuralh-on"
CACHE_COMPLETE_ROLE = "r52-pir-u-cache-complete-structuralh-on"
ROLES = (LEGACY_ROLE, CACHE_COMPLETE_ROLE)
ATTEMPT_SUFFIX = "pir-u-alpha-cache-continuity-a1-v1"
STAGE_A_ATTEMPT_SUFFIX = "pir-u-alpha-cache-continuity-a1-stage-a-b10-v1"
RESULT_NAMES = {
    LEGACY_ROLE: "s05-p1r52-piru-alpha-cache-continuity-a1-legacy-10xb100-v1",
    CACHE_COMPLETE_ROLE: (
        "s05-p1r52-piru-alpha-cache-continuity-a1-cache-complete-10xb100-v1"
    ),
}
STAGE_A_RESULT_NAMES = {
    CACHE_COMPLETE_ROLE: (
        "s05-p1r52-piru-alpha-cache-continuity-a1-cache-complete-stage-a-b10-v1"
    ),
}


def policy_for_role(role: str) -> PIRUCacheContinuityPolicy | None:
    if role == LEGACY_ROLE:
        return PIRUCacheContinuityPolicy.LEGACY
    if role == CACHE_COMPLETE_ROLE:
        return PIRUCacheContinuityPolicy.CACHE_COMPLETE
    return None


def committed_history_snapshot(
    history_keys_by_layer: Mapping[int, torch.Tensor],
    *,
    expected_width: int,
    ledger_version: int,
    ledger_digest: str,
) -> tuple[dict[int, torch.Tensor], dict[str, object]]:
    """Freeze and receipt the committed-past solve keys before current B100."""

    layers = tuple(int(layer) for layer in P1R23_LAYER_ORDER)
    if (
        set(history_keys_by_layer) != set(layers)
        or expected_width < 0
        or ledger_version < 0
        or not isinstance(ledger_digest, str)
        or len(ledger_digest) != 64
    ):
        raise ODEBFContractError("PIR-U cache-continuity history identity differs")
    frozen: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for layer in layers:
        value = history_keys_by_layer[layer].detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if (
            value.ndim != 2
            or value.shape[1] != expected_width
            or not torch.isfinite(value).all()
        ):
            raise ODEBFContractError(
                "PIR-U cache-continuity committed history geometry differs"
            )
        frozen[layer] = value.clone()
        rows.append(
            {
                "layer": layer,
                "width": int(value.shape[1]),
                "sha256": tensor_sha256(value),
            }
        )
    receipt: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-piru-cache-continuity-history-snapshot/v1",
        "ledger_version": ledger_version,
        "ledger_digest": ledger_digest,
        "committed_history_record_count": expected_width,
        "layers": rows,
        "current_batch_key_inclusion_count": 0,
        "current_uncommitted_prefix_key_inclusion_count": 0,
        "snapshot_before_current_batch": True,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return frozen, receipt


__all__ = [
    "ATTEMPT_SUFFIX",
    "CACHE_COMPLETE_ROLE",
    "CONTRACT_SHA256",
    "INSTRUCTION_ID",
    "LEGACY_ROLE",
    "METHOD_ID",
    "PIRUCacheContinuityPolicy",
    "RESULT_NAMES",
    "STAGE_A_ATTEMPT_SUFFIX",
    "STAGE_A_RESULT_NAMES",
    "ROLES",
    "committed_history_snapshot",
    "policy_for_role",
]
