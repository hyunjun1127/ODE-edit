"""Outcome-free fresh CounterFact B100 seal for the P1R20 H0 sequence.

Only immutable Git blobs from the accepted P1R19 scientific checkpoint and
the pinned CounterFact byte stream participate in selection.  The selected
requests are split into ten ordered B10 rounds and are shared by every model
and method trajectory.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_selection import (
    P1RequestIdentity,
    _load_canonical_request_map,
    _sha256_file,
    load_p1_request_identities,
    scan_prior_tracked_seals,
)
from .request_digest import ordered_request_digest_v1


P1R20_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-HISTORICAL-H0-BG-SEQUENTIAL-P1R20-V1"
)
P1R20_SELECTION_BASE = "657dafd40de50bb396fc9cfaf3862de1d07dd816"
P1R20_SELECTION_BASE_TREE = "028c49948665f9ebfbd2e713b9fd2b220c57bf49"
P1R20_SELECTION_SALT = "ODEEDIT-S05-P1R20-HISTORICAL-H0-FRESH-CF-B100-V1"
P1R20_SCHEMA = "ode-edit-s05-p1r20-historical-h0-fresh-cf-b100/v1"
P1R20_ROUND_COUNT = 10
P1R20_REQUEST_COUNT = BATCH_SIZE * P1R20_ROUND_COUNT


def _rank(identity: P1RequestIdentity) -> tuple[str, str, int]:
    return (
        hashlib.sha256(
            f"{P1R20_SELECTION_SALT}|{identity.request_sha256}".encode("utf-8")
        ).hexdigest(),
        identity.request_sha256,
        identity.case_id,
    )


def build_historical_h0_fresh_seal(
    dataset_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Select an unseen, collision-free B100 before model/outcome access."""

    dataset = Path(dataset_path).resolve(strict=True)
    identities = load_p1_request_identities(dataset)
    prior = scan_prior_tracked_seals(
        repo_root, base_commit=P1R20_SELECTION_BASE
    )
    collision_counts = collections.Counter(
        item.collision_sha256 for item in identities
    )
    duplicate_groups = {
        item for item, count in collision_counts.items() if count > 1
    }
    historical_groups = {
        item.collision_sha256
        for item in identities
        if item.case_id in prior.case_ids
        or item.request_sha256 in prior.request_sha256
    }
    eligible = tuple(
        item
        for item in identities
        if item.case_id not in prior.case_ids
        and item.request_sha256 not in prior.request_sha256
        and item.collision_sha256 not in historical_groups
        and item.collision_sha256 not in duplicate_groups
    )
    ranked = tuple(sorted(eligible, key=_rank))
    if len(ranked) < P1R20_REQUEST_COUNT:
        raise ODEBFContractError("P1R20 fresh eligible population is too small")
    selected = ranked[:P1R20_REQUEST_COUNT]
    requests = [
        {
            "ordinal": ordinal,
            "round_index": ordinal // BATCH_SIZE,
            "round_ordinal": ordinal % BATCH_SIZE,
            "case_id": item.case_id,
            "request_sha256": item.request_sha256,
            "collision_sha256": item.collision_sha256,
            "relation_sha256": item.relation_sha256,
            "rank_sha256": _rank(item)[0],
        }
        for ordinal, item in enumerate(selected)
    ]
    batch_digests = [
        ordered_request_digest_v1(
            [
                item.request_sha256
                for item in selected[
                    round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
                ]
            ]
        )
        for round_index in range(P1R20_ROUND_COUNT)
    ]
    payload: dict[str, Any] = {
        "schema_version": P1R20_SCHEMA,
        "instruction_id": P1R20_INSTRUCTION_ID,
        "status": "SEALED_BEFORE_MODEL_EVALUATOR_OR_OUTCOME_ACCESS",
        "selection_base_commit": P1R20_SELECTION_BASE,
        "selection_base_tree": P1R20_SELECTION_BASE_TREE,
        "salt": P1R20_SELECTION_SALT,
        "rank_formula": "sha256(utf8(salt + '|' + request_sha256))",
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_round_count": P1R20_ROUND_COUNT,
        "logical_edit_count": P1R20_REQUEST_COUNT,
        "source": {
            "relative_contract": "EasyEdit/data/counterfact/counterfact.json",
            "sha256": _sha256_file(dataset),
            "size_bytes": dataset.stat().st_size,
            "row_count": len(identities),
        },
        "prior_selection_scan": {
            "source_count": prior.source_count,
            "sources_digest": prior.sources_digest,
            "case_count": len(prior.case_ids),
            "case_digest": canonical_hash(sorted(prior.case_ids)),
            "request_count": len(prior.request_sha256),
            "request_digest": canonical_hash(sorted(prior.request_sha256)),
        },
        "collision_policy": (
            "EXCLUDE_PRIOR_AND_ENTIRE_DUPLICATE_"
            "NFKC_CASEFOLD_SUBJECT_RELATION_TEMPLATE_GROUPS"
        ),
        "historical_collision_group_count": len(historical_groups),
        "historical_collision_group_digest": canonical_hash(
            sorted(historical_groups)
        ),
        "duplicate_collision_group_count": len(duplicate_groups),
        "duplicate_collision_group_digest": canonical_hash(
            sorted(duplicate_groups)
        ),
        "eligible_count": len(eligible),
        "eligible_digest": canonical_hash(
            [
                [
                    item.case_id,
                    item.request_sha256,
                    item.collision_sha256,
                ]
                for item in eligible
            ]
        ),
        "requests": requests,
        "batch_ordered_request_digest_v1": batch_digests,
        "all_request_order_sha256": canonical_hash(
            [item.request_sha256 for item in selected]
        ),
        "models": {
            alias: {
                "batch_ordered_request_digest_v1": batch_digests,
                "same_split_and_order": True,
            }
            for alias in MODEL_ALIASES
        },
        "future_round_access_before_round_count": 0,
        "model_import_or_access_count": 0,
        "evaluator_import_or_access_count": 0,
        "heldout_field_open_count": 0,
        "terminal_or_outcome_open_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["root_digest"] = canonical_hash(payload)
    return verify_historical_h0_fresh_seal(payload)


def verify_historical_h0_fresh_seal(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("P1R20 fresh seal root differs")
    requests = payload.get("requests")
    batches = payload.get("batch_ordered_request_digest_v1")
    if (
        payload.get("schema_version") != P1R20_SCHEMA
        or payload.get("instruction_id") != P1R20_INSTRUCTION_ID
        or payload.get("status")
        != "SEALED_BEFORE_MODEL_EVALUATOR_OR_OUTCOME_ACCESS"
        or payload.get("selection_base_commit") != P1R20_SELECTION_BASE
        or payload.get("selection_base_tree") != P1R20_SELECTION_BASE_TREE
        or payload.get("salt") != P1R20_SELECTION_SALT
        or payload.get("edit_batch_size") != BATCH_SIZE
        or payload.get("sequential_round_count") != P1R20_ROUND_COUNT
        or payload.get("logical_edit_count") != P1R20_REQUEST_COUNT
        or not isinstance(requests, list)
        or len(requests) != P1R20_REQUEST_COUNT
        or not isinstance(batches, list)
        or len(batches) != P1R20_ROUND_COUNT
        or [item.get("ordinal") for item in requests]
        != list(range(P1R20_REQUEST_COUNT))
        or [item.get("round_index") for item in requests]
        != [item // BATCH_SIZE for item in range(P1R20_REQUEST_COUNT)]
        or [item.get("round_ordinal") for item in requests]
        != [item % BATCH_SIZE for item in range(P1R20_REQUEST_COUNT)]
        or len({item.get("case_id") for item in requests})
        != P1R20_REQUEST_COUNT
        or len({item.get("request_sha256") for item in requests})
        != P1R20_REQUEST_COUNT
        or len({item.get("collision_sha256") for item in requests})
        != P1R20_REQUEST_COUNT
        or payload.get("future_round_access_before_round_count") != 0
        or payload.get("model_import_or_access_count") != 0
        or payload.get("evaluator_import_or_access_count") != 0
        or payload.get("heldout_field_open_count") != 0
        or payload.get("terminal_or_outcome_open_count") != 0
        or payload.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("P1R20 fresh seal contract differs")
    ranks = [
        hashlib.sha256(
            f"{P1R20_SELECTION_SALT}|{item['request_sha256']}".encode("utf-8")
        ).hexdigest()
        for item in requests
    ]
    if [item.get("rank_sha256") for item in requests] != ranks or ranks != sorted(ranks):
        raise ODEBFContractError("P1R20 fresh seal rank order differs")
    expected_batches = [
        ordered_request_digest_v1(
            [
                str(item["request_sha256"])
                for item in requests[
                    round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
                ]
            ]
        )
        for round_index in range(P1R20_ROUND_COUNT)
    ]
    if batches != expected_batches:
        raise ODEBFContractError("P1R20 fresh seal split/order differs")
    models = payload.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_ALIASES) or any(
        item.get("batch_ordered_request_digest_v1") != expected_batches
        or item.get("same_split_and_order") is not True
        for item in models.values()
    ):
        raise ODEBFContractError("P1R20 model split/order differs")
    payload["root_digest"] = root
    return payload


def load_historical_h0_batches(
    dataset_path: str | Path,
    seal: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    value = verify_historical_h0_fresh_seal(seal)
    dataset = Path(dataset_path).resolve(strict=True)
    source = value["source"]
    if (
        dataset.stat().st_size != source["size_bytes"]
        or _sha256_file(dataset) != source["sha256"]
    ):
        raise ODEBFContractError("P1R20 dataset identity differs")
    approved = {
        str(item["request_sha256"]): (
            int(item["case_id"]),
            str(item["collision_sha256"]),
        )
        for item in value["requests"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    batches: list[tuple[dict[str, Any], ...]] = []
    for round_index in range(P1R20_ROUND_COUNT):
        records = value["requests"][
            round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
        ]
        batch = tuple(loaded[str(item["request_sha256"])] for item in records)
        if ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in batch]
        ) != value["batch_ordered_request_digest_v1"][round_index]:
            raise ODEBFContractError("P1R20 loaded batch order differs")
        batches.append(batch)
    return tuple(batches)


__all__ = [
    "P1R20_INSTRUCTION_ID",
    "P1R20_REQUEST_COUNT",
    "P1R20_ROUND_COUNT",
    "P1R20_SCHEMA",
    "P1R20_SELECTION_BASE",
    "P1R20_SELECTION_BASE_TREE",
    "P1R20_SELECTION_SALT",
    "build_historical_h0_fresh_seal",
    "load_historical_h0_batches",
    "verify_historical_h0_fresh_seal",
]
