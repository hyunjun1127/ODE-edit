"""Canonical, schema-bound digest for one ordered joint B10 request batch."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash


ORDERED_REQUEST_DIGEST_SCHEMA = "ode-edit-s04-ode-bf-ordered-request-digest/v1"
SCALABLE_ORDERED_REQUEST_DIGEST_SCHEMA = (
    "ode-edit-s05-p1r23-scalable-ordered-request-digest/v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def ordered_request_digest_v1(
    request_sha256: Sequence[Any],
    *,
    schema_version: str = ORDERED_REQUEST_DIGEST_SCHEMA,
) -> str:
    """Hash an ordered B10 identity vector with unambiguous canonical JSON."""

    if schema_version != ORDERED_REQUEST_DIGEST_SCHEMA:
        raise ODEBFContractError("ordered request digest schema differs")
    values = tuple(request_sha256)
    if len(values) != BATCH_SIZE:
        raise ODEBFContractError("ordered request digest requires exactly ten values")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in values
    ):
        raise ODEBFContractError("ordered request digest contains an invalid SHA-256")
    if len(set(values)) != BATCH_SIZE:
        raise ODEBFContractError("ordered request digest contains a duplicate")
    return canonical_hash(
        {
            "schema_version": ORDERED_REQUEST_DIGEST_SCHEMA,
            "ordered_request_sha256": list(values),
        }
    )


def ordered_request_digest_scalable_v1(
    request_sha256: Sequence[Any],
) -> str:
    """Preserve the legacy B10 identity and bind one atomic B100 vector."""

    values = tuple(request_sha256)
    if len(values) == BATCH_SIZE:
        return ordered_request_digest_v1(values)
    if len(values) != 100:
        raise ODEBFContractError("scalable ordered request digest batch differs")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in values
    ) or len(set(values)) != 100:
        raise ODEBFContractError("scalable ordered request digest identity differs")
    return canonical_hash(
        {
            "schema_version": SCALABLE_ORDERED_REQUEST_DIGEST_SCHEMA,
            "atomic_joint_batch_size": 100,
            "ordered_request_sha256": list(values),
            "sequential_round_count": 0,
        }
    )
