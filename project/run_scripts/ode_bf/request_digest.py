"""Canonical, schema-bound digest for one ordered joint B10 request batch."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash


ORDERED_REQUEST_DIGEST_SCHEMA = "ode-edit-s04-ode-bf-ordered-request-digest/v1"
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
