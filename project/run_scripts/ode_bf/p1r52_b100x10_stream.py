"""Outcome-free deterministic stream seal for P1R52 sequential 10xB100."""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
from .p1_selection import (
    _load_canonical_request_map,
    _sha256_file,
    load_p1_request_identities,
    scan_prior_tracked_seals,
)
from .p1r24_independent_b10x10_selection import (
    P1R20_SELECTION_BASE,
    P1R20_SELECTION_BASE_TREE,
    P1R20_SELECTION_SALT,
)
from .p1r52_sequential_scale import B100X10_INSTRUCTION_ID
from .scalable_batched_runtime import scalable_ordered_request_digest


STREAM_SCHEMA = "ode-edit-s05-p1r52-sequential-b100x10-stream/v1"
INSTRUCTION_ID = B100X10_INSTRUCTION_ID
BATCH_SIZE = 100
ROUND_COUNT = 10
REQUEST_COUNT = BATCH_SIZE * ROUND_COUNT
SEAL_FILE = "p1r52_sequential_b100x10_stream_seal.json"
PREFIX_SEAL_FILE = "p1r24_independent_b10x10_stream_seal.json"


def _rank(request_sha256: str, case_id: int) -> tuple[str, str, int]:
    return (
        hashlib.sha256(
            f"{P1R20_SELECTION_SALT}|{request_sha256}".encode("utf-8")
        ).hexdigest(),
        request_sha256,
        case_id,
    )


def build_p1r52_b100x10_stream(
    dataset_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Seal the first 1,000 identities from the frozen outcome-free rank."""

    dataset = Path(dataset_path).resolve(strict=True)
    repo = Path(repo_root).resolve(strict=True)
    identities = load_p1_request_identities(dataset)
    prior = scan_prior_tracked_seals(repo, base_commit=P1R20_SELECTION_BASE)
    collision_counts = collections.Counter(item.collision_sha256 for item in identities)
    duplicate_groups = {item for item, count in collision_counts.items() if count > 1}
    historical_groups = {
        item.collision_sha256
        for item in identities
        if item.case_id in prior.case_ids or item.request_sha256 in prior.request_sha256
    }
    eligible = tuple(
        item
        for item in identities
        if item.case_id not in prior.case_ids
        and item.request_sha256 not in prior.request_sha256
        and item.collision_sha256 not in historical_groups
        and item.collision_sha256 not in duplicate_groups
    )
    ranked = tuple(sorted(eligible, key=lambda item: _rank(item.request_sha256, item.case_id)))
    if len(ranked) < REQUEST_COUNT:
        raise ODEBFContractError("P1R52 B100x10 eligible population is too small")
    selected = ranked[:REQUEST_COUNT]
    requests = [
        {
            "ordinal": ordinal,
            "round_index": ordinal // BATCH_SIZE,
            "round_ordinal": ordinal % BATCH_SIZE,
            "case_id": item.case_id,
            "request_sha256": item.request_sha256,
            "collision_sha256": item.collision_sha256,
            "relation_sha256": item.relation_sha256,
            "rank_sha256": _rank(item.request_sha256, item.case_id)[0],
        }
        for ordinal, item in enumerate(selected)
    ]
    batch_digests = [
        scalable_ordered_request_digest(
            [
                item.request_sha256
                for item in selected[
                    round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
                ]
            ]
        )
        for round_index in range(ROUND_COUNT)
    ]
    payload: dict[str, Any] = {
        "schema_version": STREAM_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "status": "SEALED_BEFORE_MODEL_EVALUATOR_OR_OUTCOME_ACCESS",
        "selection_base_commit": P1R20_SELECTION_BASE,
        "selection_base_tree": P1R20_SELECTION_BASE_TREE,
        "salt": P1R20_SELECTION_SALT,
        "rank_formula": "sha256(utf8(salt + '|' + request_sha256))",
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_round_count": ROUND_COUNT,
        "logical_edit_count": REQUEST_COUNT,
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
        "historical_collision_group_digest": canonical_hash(sorted(historical_groups)),
        "duplicate_collision_group_count": len(duplicate_groups),
        "duplicate_collision_group_digest": canonical_hash(sorted(duplicate_groups)),
        "eligible_count": len(eligible),
        "eligible_digest": canonical_hash(
            [
                [item.case_id, item.request_sha256, item.collision_sha256]
                for item in eligible
            ]
        ),
        "requests": requests,
        "batch_ordered_request_digest_v1": batch_digests,
        "all_request_order_sha256": canonical_hash(
            [item.request_sha256 for item in selected]
        ),
        "future_round_access_before_round_count": 0,
        "model_import_or_access_count": 0,
        "evaluator_import_or_access_count": 0,
        "heldout_field_open_count": 0,
        "terminal_or_outcome_open_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["root_digest"] = canonical_hash(payload)
    return verify_p1r52_b100x10_stream(payload)


def verify_p1r52_b100x10_stream(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("P1R52 B100x10 stream root differs")
    requests = payload.get("requests")
    batches = payload.get("batch_ordered_request_digest_v1")
    if (
        payload.get("schema_version") != STREAM_SCHEMA
        or payload.get("instruction_id") != INSTRUCTION_ID
        or payload.get("status") != "SEALED_BEFORE_MODEL_EVALUATOR_OR_OUTCOME_ACCESS"
        or payload.get("selection_base_commit") != P1R20_SELECTION_BASE
        or payload.get("selection_base_tree") != P1R20_SELECTION_BASE_TREE
        or payload.get("salt") != P1R20_SELECTION_SALT
        or payload.get("edit_batch_size") != BATCH_SIZE
        or payload.get("sequential_round_count") != ROUND_COUNT
        or payload.get("logical_edit_count") != REQUEST_COUNT
        or not isinstance(requests, list)
        or len(requests) != REQUEST_COUNT
        or not isinstance(batches, list)
        or len(batches) != ROUND_COUNT
        or [item.get("ordinal") for item in requests] != list(range(REQUEST_COUNT))
        or [item.get("round_index") for item in requests]
        != [index // BATCH_SIZE for index in range(REQUEST_COUNT)]
        or [item.get("round_ordinal") for item in requests]
        != [index % BATCH_SIZE for index in range(REQUEST_COUNT)]
        or len({item.get("case_id") for item in requests}) != REQUEST_COUNT
        or len({item.get("request_sha256") for item in requests}) != REQUEST_COUNT
        or len({item.get("collision_sha256") for item in requests}) != REQUEST_COUNT
        or payload.get("future_round_access_before_round_count") != 0
        or payload.get("model_import_or_access_count") != 0
        or payload.get("evaluator_import_or_access_count") != 0
        or payload.get("heldout_field_open_count") != 0
        or payload.get("terminal_or_outcome_open_count") != 0
        or payload.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("P1R52 B100x10 stream contract differs")
    ranks = [
        _rank(str(item["request_sha256"]), int(item["case_id"]))[0]
        for item in requests
    ]
    if [item.get("rank_sha256") for item in requests] != ranks or ranks != sorted(ranks):
        raise ODEBFContractError("P1R52 B100x10 rank order differs")
    expected_batches = [
        scalable_ordered_request_digest(
            [
                str(item["request_sha256"])
                for item in requests[
                    round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
                ]
            ]
        )
        for round_index in range(ROUND_COUNT)
    ]
    if batches != expected_batches:
        raise ODEBFContractError("P1R52 B100x10 split/order differs")
    if payload.get("all_request_order_sha256") != canonical_hash(
        [str(item["request_sha256"]) for item in requests]
    ):
        raise ODEBFContractError("P1R52 B100x10 full order differs")
    payload["root_digest"] = root
    return payload


def verify_b10_prefix(prefix_seal: Mapping[str, Any], stream: Mapping[str, Any]) -> None:
    prefix = prefix_seal.get("requests")
    selected = stream.get("requests")
    if not isinstance(prefix, list) or not isinstance(selected, list) or len(prefix) != 100:
        raise ODEBFContractError("P1R52 B100x10 prefix source differs")
    if [item.get("request_sha256") for item in selected[:100]] != [
        item.get("request_sha256") for item in prefix
    ]:
        raise ODEBFContractError("P1R52 B100x10 frozen B10 prefix differs")


def load_p1r52_b100x10_batches(
    dataset_path: str | Path,
    seal: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    value = verify_p1r52_b100x10_stream(seal)
    dataset = Path(dataset_path).resolve(strict=True)
    source = value["source"]
    if (
        dataset.stat().st_size != source["size_bytes"]
        or _sha256_file(dataset) != source["sha256"]
    ):
        raise ODEBFContractError("P1R52 B100x10 dataset identity differs")
    approved = {
        str(item["request_sha256"]): (
            int(item["case_id"]),
            str(item["collision_sha256"]),
        )
        for item in value["requests"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    batches: list[tuple[dict[str, Any], ...]] = []
    for round_index in range(ROUND_COUNT):
        records = value["requests"][
            round_index * BATCH_SIZE : (round_index + 1) * BATCH_SIZE
        ]
        batch = tuple(loaded[str(item["request_sha256"])] for item in records)
        if scalable_ordered_request_digest(
            [str(item["request_sha256"]) for item in batch]
        ) != value["batch_ordered_request_digest_v1"][round_index]:
            raise ODEBFContractError("P1R52 B100x10 loaded batch order differs")
        batches.append(batch)
    return tuple(batches)


__all__ = [
    "BATCH_SIZE",
    "INSTRUCTION_ID",
    "PREFIX_SEAL_FILE",
    "REQUEST_COUNT",
    "ROUND_COUNT",
    "SEAL_FILE",
    "STREAM_SCHEMA",
    "build_p1r52_b100x10_stream",
    "load_p1r52_b100x10_batches",
    "verify_b10_prefix",
    "verify_p1r52_b100x10_stream",
]
