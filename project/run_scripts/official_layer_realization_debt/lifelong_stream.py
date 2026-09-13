"""Outcome-free 10k extension of the sealed Phase123 salted-hash stream."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1_selection import (
    _load_canonical_request_map,
    _sha256_file,
    load_p1_request_identities,
    scan_prior_tracked_seals,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import (
    P1R20_SELECTION_BASE,
    P1R20_SELECTION_BASE_TREE,
    P1R20_SELECTION_SALT,
)
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    _rank,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    scalable_ordered_request_digest,
)

from .contracts import DATASET_SHA256, ObservationBoundary
from .lifelong_contracts import CAMPAIGN_ORDER_SEED_INDEX, INSTRUCTION_ID, LifelongLock


SCHEMA = "odeedit.s06.official-layer-realization-debt.lifelong-stream.v1"
TRAINING_COUNT = LifelongLock().request_count
SENTINEL_COUNT = LifelongLock().batch_size
TOTAL_RANKED_COUNT = TRAINING_COUNT + SENTINEL_COUNT


def _request_row(item: Any, ordinal: int, role: str) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "batch_index": ordinal // LifelongLock().batch_size + 1,
        "batch_ordinal": ordinal % LifelongLock().batch_size,
        "case_id": int(item.case_id),
        "request_sha256": str(item.request_sha256),
        "collision_sha256": str(item.collision_sha256),
        "relation_sha256": str(item.relation_sha256),
        "rank_sha256": _rank(item.request_sha256, item.case_id)[0],
        "role": role,
    }


def build_lifelong_stream(dataset_path: Path, repo_root: Path) -> dict[str, Any]:
    """Build one preregistered 10k prefix plus a disjoint sentinel B100."""

    dataset = dataset_path.resolve(strict=True)
    repo = repo_root.resolve(strict=True)
    identities = load_p1_request_identities(dataset)
    prior = scan_prior_tracked_seals(repo, base_commit=P1R20_SELECTION_BASE)
    collision_counts = collections.Counter(item.collision_sha256 for item in identities)
    duplicate_groups = {key for key, count in collision_counts.items() if count > 1}
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
    ranked = tuple(
        sorted(eligible, key=lambda item: _rank(item.request_sha256, item.case_id))
    )
    if len(ranked) < TOTAL_RANKED_COUNT:
        raise ObservationBoundary("lifelong eligible stream is shorter than 10k+sentinel")
    training_items = ranked[:TRAINING_COUNT]
    sentinel_items = ranked[TRAINING_COUNT:TOTAL_RANKED_COUNT]
    training = [
        _request_row(item, ordinal, "LIFELONG_TRAINING")
        for ordinal, item in enumerate(training_items)
    ]
    sentinel = [
        _request_row(item, ordinal, "CHECKPOINT_SENTINEL_NO_COMMIT")
        for ordinal, item in enumerate(sentinel_items)
    ]
    batch_digests = [
        scalable_ordered_request_digest(
            [
                item.request_sha256
                for item in training_items[
                    start : start + LifelongLock().batch_size
                ]
            ]
        )
        for start in range(0, TRAINING_COUNT, LifelongLock().batch_size)
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "status": "SEALED_BEFORE_MODEL_OR_OUTCOME_ACCESS",
        "selection_base_commit": P1R20_SELECTION_BASE,
        "selection_base_tree": P1R20_SELECTION_BASE_TREE,
        "salt": P1R20_SELECTION_SALT,
        "order_seed_index": CAMPAIGN_ORDER_SEED_INDEX,
        "order_seed_count": 1,
        "rank_formula": "sha256(utf8(salt + '|' + request_sha256))",
        "source": {
            "path_role": "EasyEdit/data/counterfact/counterfact.json",
            "sha256": _sha256_file(dataset),
            "bytes": dataset.stat().st_size,
            "row_count": len(identities),
        },
        "collision_policy": (
            "EXCLUDE_PRIOR_AND_ENTIRE_DUPLICATE_"
            "NFKC_CASEFOLD_SUBJECT_RELATION_TEMPLATE_GROUPS"
        ),
        "prior_selection_scan": {
            "source_count": prior.source_count,
            "sources_digest": prior.sources_digest,
            "case_count": len(prior.case_ids),
            "request_count": len(prior.request_sha256),
        },
        "eligible_count": len(eligible),
        "eligible_digest": canonical_hash(
            [
                [item.case_id, item.request_sha256, item.collision_sha256]
                for item in eligible
            ]
        ),
        "batch_size": LifelongLock().batch_size,
        "batch_count": LifelongLock().batch_count,
        "training_request_count": TRAINING_COUNT,
        "sentinel_request_count": SENTINEL_COUNT,
        "training": training,
        "sentinel": sentinel,
        "batch_ordered_request_digest_v1": batch_digests,
        "training_order_sha256": canonical_hash(
            [item.request_sha256 for item in training_items]
        ),
        "sentinel_order_sha256": canonical_hash(
            [item.request_sha256 for item in sentinel_items]
        ),
        "retention_sampling": {
            "earliest_count": 32,
            "recent_count": 32,
            "hash_stratified_count": 64,
            "hash_rank_formula": "sha256('retention|' + request_sha256)",
            "checkpoint_result_influence_count": 0,
        },
        "future_request_content_access_count": 0,
        "model_or_evaluator_access_count": 0,
        "sample_duplication_count": 0,
        "scientific_promotion": False,
    }
    payload["root_digest"] = canonical_hash(payload)
    return verify_lifelong_stream(payload)


def verify_lifelong_stream(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    training = payload.get("training")
    sentinel = payload.get("sentinel")
    batches = payload.get("batch_ordered_request_digest_v1")
    if root != canonical_hash(payload):
        raise ObservationBoundary("lifelong stream root differs")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("instruction_id") != INSTRUCTION_ID
        or payload.get("status") != "SEALED_BEFORE_MODEL_OR_OUTCOME_ACCESS"
        or payload.get("selection_base_commit") != P1R20_SELECTION_BASE
        or payload.get("selection_base_tree") != P1R20_SELECTION_BASE_TREE
        or payload.get("salt") != P1R20_SELECTION_SALT
        or payload.get("order_seed_index") != CAMPAIGN_ORDER_SEED_INDEX
        or payload.get("order_seed_count") != 1
        or payload.get("source", {}).get("sha256") != DATASET_SHA256
        or payload.get("batch_size") != LifelongLock().batch_size
        or payload.get("batch_count") != LifelongLock().batch_count
        or payload.get("training_request_count") != TRAINING_COUNT
        or payload.get("sentinel_request_count") != SENTINEL_COUNT
        or not isinstance(training, list)
        or len(training) != TRAINING_COUNT
        or not isinstance(sentinel, list)
        or len(sentinel) != SENTINEL_COUNT
        or not isinstance(batches, list)
        or len(batches) != LifelongLock().batch_count
        or payload.get("future_request_content_access_count") != 0
        or payload.get("model_or_evaluator_access_count") != 0
        or payload.get("sample_duplication_count") != 0
        or payload.get("scientific_promotion") is not False
    ):
        raise ObservationBoundary("lifelong stream contract differs")
    if [row.get("ordinal") for row in training] != list(range(TRAINING_COUNT)):
        raise ObservationBoundary("lifelong training ordinal differs")
    if len({row.get("request_sha256") for row in [*training, *sentinel]}) != TOTAL_RANKED_COUNT:
        raise ObservationBoundary("lifelong training/sentinel overlap differs")
    ranks = [str(row["rank_sha256"]) for row in [*training, *sentinel]]
    if ranks != sorted(ranks):
        raise ObservationBoundary("lifelong salted rank order differs")
    expected_batches = [
        scalable_ordered_request_digest(
            [str(row["request_sha256"]) for row in training[start : start + 100]]
        )
        for start in range(0, TRAINING_COUNT, 100)
    ]
    if batches != expected_batches:
        raise ObservationBoundary("lifelong B100 partition differs")
    if payload.get("training_order_sha256") != canonical_hash(
        [str(row["request_sha256"]) for row in training]
    ) or payload.get("sentinel_order_sha256") != canonical_hash(
        [str(row["request_sha256"]) for row in sentinel]
    ):
        raise ObservationBoundary("lifelong order root differs")
    payload["root_digest"] = root
    return payload


def verify_existing_1k_prefix(
    lifelong: Mapping[str, Any], existing_seal_path: Path
) -> dict[str, Any]:
    existing = verify_p1r52_b100x10_stream(
        json.loads(existing_seal_path.read_text(encoding="utf-8"))
    )
    expected = [str(row["request_sha256"]) for row in existing["requests"]]
    observed = [
        str(row["request_sha256"])
        for row in verify_lifelong_stream(lifelong)["training"][: len(expected)]
    ]
    if observed != expected:
        raise ObservationBoundary("lifelong stream does not extend sealed Phase123 1k")
    return {
        "status": "EXACT_1K_PREFIX_PASS",
        "request_count": len(expected),
        "existing_root": existing["root_digest"],
        "existing_order": existing["all_request_order_sha256"],
        "observed_prefix_order": canonical_hash(observed),
    }


def load_lifelong_batches(
    dataset_path: Path, seal: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    value = verify_lifelong_stream(seal)
    dataset = dataset_path.resolve(strict=True)
    if _sha256_file(dataset) != DATASET_SHA256:
        raise ObservationBoundary("lifelong dataset identity differs")
    rows = [*value["training"], *value["sentinel"]]
    approved = {
        str(row["request_sha256"]): (
            int(row["case_id"]),
            str(row["collision_sha256"]),
        )
        for row in rows
    }
    loaded = _load_canonical_request_map(dataset, approved)
    training = tuple(
        loaded[str(row["request_sha256"])] for row in value["training"]
    )
    sentinel = tuple(
        loaded[str(row["request_sha256"])] for row in value["sentinel"]
    )
    return training, sentinel


__all__ = [
    "SCHEMA",
    "SENTINEL_COUNT",
    "TRAINING_COUNT",
    "build_lifelong_stream",
    "load_lifelong_batches",
    "verify_existing_1k_prefix",
    "verify_lifelong_stream",
]
