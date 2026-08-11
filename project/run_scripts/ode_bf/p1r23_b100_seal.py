"""Outcome-free B100 superset seal for P1R23.

Only the canonical CounterFact rewrite object is decoded.  Held-out prompts,
model/tokenizer state, evaluator outputs, activations, timings, and results are
never opened by this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from project.run_scripts.ode_alloc.selection import (
    _extract_object_after_key,
    _iter_top_level_objects,
)

from .contracts import ODEBFContractError, canonical_hash
from .p1_selection import (
    _prior_identities_from_json,
    _prior_identities_from_python,
    _selection_blob_paths,
    _sha256_file,
    _tracked_blob,
)


P1R23_B100_SCHEMA = "ode-edit-s05-p1r23-b100-prefix-canonical90-seal/v1"
P1R23_B100_AMENDMENT_ID = (
    "ODEEDIT-S05-ODE-BF-SCALABLE-BATCHED-RUNTIME-P1R23-V1-A1-B100-SEAL"
)
P1R23_B100_BASE_COMMIT = "60fc604e18636499683aebf0f4f24724d8c28981"
P1R23_B10_SEAL_ROOT = "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
P1R23_B10_ORDER = "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
P1R23_PREFIX_COUNT = 10
P1R23_ADDED_COUNT = 90
P1R23_B100_COUNT = P1R23_PREFIX_COUNT + P1R23_ADDED_COUNT
P1R23_COLLISION_NORMALIZATION = (
    "p1r23-nfkc-ws-collapse-trim-casefold-schema-placeholder-v1"
)

_SPACE = re.compile(r"\s+")
_CASE_ID = re.compile(rb'"case_id"\s*:\s*"?(\d+)"?')
_PLACEHOLDER = re.compile(r"\{(?:0)?\}")


def _normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def _normalize_template(value: str) -> str:
    normalized = _normalize_text(value)
    replaced, count = _PLACEHOLDER.subn("{}", normalized)
    if count != 1:
        raise ODEBFContractError("P1R23 CounterFact template placeholder differs")
    return replaced


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ODEBFContractError(f"P1R23 CounterFact {label} is absent")
    return value


@dataclass(frozen=True, slots=True)
class B100CaseIdentity:
    dataset_ordinal: int
    case_id: int
    request_sha256: str
    collision_sha256: str
    relation_sha256: str

    def raw_free_payload(self, *, position: int, role: str) -> dict[str, Any]:
        payload = {
            "position": position,
            "role": role,
            "dataset_ordinal": self.dataset_ordinal,
            "case_id": self.case_id,
            "request_sha256": self.request_sha256,
            "collision_sha256": self.collision_sha256,
            "relation_sha256": self.relation_sha256,
        }
        payload["position_identity_sha256"] = canonical_hash(payload)
        return payload


def _project_identity(blob: bytes, *, dataset_ordinal: int) -> B100CaseIdentity:
    matches = _CASE_ID.findall(blob)
    if len(matches) != 1:
        raise ODEBFContractError("P1R23 CounterFact case identity is ambiguous")
    case_id = int(matches[0])
    try:
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("P1R23 CounterFact rewrite JSON differs") from exc
    if not isinstance(rewrite, Mapping) or set(rewrite) != {
        "prompt",
        "relation_id",
        "subject",
        "target_new",
        "target_true",
    }:
        raise ODEBFContractError("P1R23 CounterFact rewrite schema differs")
    target_new = rewrite["target_new"]
    target_true = rewrite["target_true"]
    if not isinstance(target_new, Mapping) or not isinstance(target_true, Mapping):
        raise ODEBFContractError("P1R23 CounterFact target schema differs")
    prompt = _required_text(rewrite["prompt"], label="prompt")
    relation = _required_text(rewrite["relation_id"], label="relation")
    subject = _required_text(rewrite["subject"], label="subject")
    new_text = _required_text(target_new.get("str"), label="new target")
    old_text = _required_text(target_true.get("str"), label="old target")
    request_payload = {
        "case_id": case_id,
        "prompt": prompt,
        "relation_id": relation,
        "subject": subject,
        "target_new": new_text,
        "target_old": old_text,
    }
    collision_payload = {
        "subject": _normalize_text(subject),
        "relation": _normalize_text(relation),
        "template": _normalize_template(prompt),
    }
    return B100CaseIdentity(
        dataset_ordinal,
        case_id,
        canonical_hash(request_payload),
        canonical_hash(collision_payload),
        canonical_hash({"relation": _normalize_text(relation)}),
    )


def load_b100_case_identities(dataset_path: Path) -> tuple[B100CaseIdentity, ...]:
    dataset = dataset_path.resolve(strict=True)
    identities = tuple(
        _project_identity(blob, dataset_ordinal=ordinal)
        for ordinal, blob in enumerate(_iter_top_level_objects(dataset))
    )
    if not identities:
        raise ODEBFContractError("P1R23 CounterFact source is empty")
    if len({item.case_id for item in identities}) != len(identities):
        raise ODEBFContractError("P1R23 CounterFact case identities repeat")
    if len({item.request_sha256 for item in identities}) != len(identities):
        raise ODEBFContractError("P1R23 CounterFact request identities repeat")
    return identities


def _tracked_role_inventory(
    repo_root: Path,
    *,
    base_commit: str,
) -> tuple[set[int], set[str], list[dict[str, Any]]]:
    repo = repo_root.resolve(strict=True)
    observed = subprocess.run(
        ["git", "rev-parse", base_commit],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != base_commit:
        raise ODEBFContractError("P1R23 exclusion base commit differs")
    case_ids: set[int] = set()
    request_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for relative in _selection_blob_paths(repo, base_commit):
        blob = _tracked_blob(repo, base_commit, relative)
        if relative.endswith(".json"):
            local_cases, local_requests = _prior_identities_from_json(blob)
        else:
            local_cases, local_requests = _prior_identities_from_python(blob, relative)
        if not local_cases and not local_requests:
            continue
        case_ids.update(local_cases)
        request_ids.update(local_requests)
        records.append(
            {
                "path": relative,
                "mode": "100644",
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "case_id_count": len(local_cases),
                "case_id_root": canonical_hash(sorted(local_cases)),
                "request_sha256_count": len(local_requests),
                "request_sha256_root": canonical_hash(sorted(local_requests)),
            }
        )
    return case_ids, request_ids, records


def _load_b10_seal(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    observed = value.pop("root_digest", None)
    if observed != P1R23_B10_SEAL_ROOT or observed != canonical_hash(value):
        raise ODEBFContractError("P1R23 B10 prefix seal differs")
    value["root_digest"] = observed
    requests = value.get("requests")
    if (
        not isinstance(requests, list)
        or len(requests) != P1R23_PREFIX_COUNT
        or value.get("batch_ordered_request_digest_v1") != [P1R23_B10_ORDER]
    ):
        raise ODEBFContractError("P1R23 B10 prefix order differs")
    return value


def build_p1r23_b100_seal(
    dataset_path: Path,
    repo_root: Path,
    b10_seal_path: Path,
    *,
    base_commit: str = P1R23_B100_BASE_COMMIT,
) -> dict[str, Any]:
    """Select exact B10 prefix plus first 90 eligible dataset ordinals."""

    dataset = dataset_path.resolve(strict=True)
    repo = repo_root.resolve(strict=True)
    b10 = _load_b10_seal(b10_seal_path)
    source = b10.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != _sha256_file(dataset)
        or source.get("size_bytes") != dataset.stat().st_size
    ):
        raise ODEBFContractError("P1R23 B100 dataset source differs")
    identities = load_b100_case_identities(dataset)
    by_case = {item.case_id: item for item in identities}
    prefix: list[B100CaseIdentity] = []
    for expected_position, item in enumerate(b10["requests"]):
        case_id = int(item["case_id"])
        observed = by_case.get(case_id)
        if (
            observed is None
            or observed.request_sha256 != str(item["request_sha256"])
            or observed.collision_sha256 != str(item["collision_sha256"])
            or int(item["ordinal"]) != expected_position
        ):
            raise ODEBFContractError("P1R23 B100 prefix identity differs")
        prefix.append(observed)
    prior_cases, prior_requests, role_sources = _tracked_role_inventory(
        repo, base_commit=base_commit
    )
    prefix_cases = {item.case_id for item in prefix}
    prefix_requests = {item.request_sha256 for item in prefix}
    excluded_cases = set(prior_cases) | prefix_cases
    excluded_requests = set(prior_requests) | prefix_requests
    excluded_collisions = {
        item.collision_sha256
        for item in identities
        if item.case_id in excluded_cases
        or item.request_sha256 in excluded_requests
    }
    excluded_collisions.update(item.collision_sha256 for item in prefix)
    eligible = tuple(
        item
        for item in identities
        if item.case_id not in excluded_cases
        and item.request_sha256 not in excluded_requests
        and item.collision_sha256 not in excluded_collisions
    )
    if len(eligible) < P1R23_ADDED_COUNT:
        raise ODEBFContractError("B100_SOURCE_EXHAUSTED")
    added = eligible[:P1R23_ADDED_COUNT]
    ordered = tuple((*prefix, *added))
    if len(ordered) != P1R23_B100_COUNT or len(
        {item.request_sha256 for item in ordered}
    ) != P1R23_B100_COUNT:
        raise ODEBFContractError("P1R23 B100 request inventory differs")
    records = [
        item.raw_free_payload(
            position=position,
            role="P1R19_EXACT_B10_PREFIX" if position < 10 else "CANONICAL_ADDED_90",
        )
        for position, item in enumerate(ordered)
    ]
    exclusion_inventory = {
        "base_commit": base_commit,
        "tracked_role_source_count": len(role_sources),
        "tracked_role_sources": role_sources,
        "tracked_case_id_count": len(prior_cases),
        "tracked_case_id_root": canonical_hash(sorted(prior_cases)),
        "tracked_request_sha256_count": len(prior_requests),
        "tracked_request_sha256_root": canonical_hash(sorted(prior_requests)),
        "excluded_case_id_count": len(excluded_cases),
        "excluded_case_id_root": canonical_hash(sorted(excluded_cases)),
        "excluded_request_sha256_count": len(excluded_requests),
        "excluded_request_sha256_root": canonical_hash(sorted(excluded_requests)),
        "excluded_collision_count": len(excluded_collisions),
        "excluded_collision_root": canonical_hash(sorted(excluded_collisions)),
        "local_result_path_access_count": 0,
        "model_evaluator_outcome_access_count": 0,
    }
    exclusion_inventory["identity_sha256"] = canonical_hash(exclusion_inventory)
    payload: dict[str, Any] = {
        "schema_version": P1R23_B100_SCHEMA,
        "instruction_id": P1R23_B100_AMENDMENT_ID,
        "status": "SEALED_BEFORE_B100_MODEL_OR_OUTCOME_ACCESS",
        "selection_contract": "A_PREFIX_PLUS_CANONICAL_90",
        "source": {
            "relative_contract": source["relative_contract"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
            "row_count": len(identities),
        },
        "normalization": {
            "version": P1R23_COLLISION_NORMALIZATION,
            "unicode": "NFKC",
            "whitespace": "collapse_and_trim",
            "case": "casefold",
            "placeholder": "accepted_schema_single_{}_or_{0}_to_{}",
            "semantic_entity_heuristic_count": 0,
        },
        "prefix": {
            "count": P1R23_PREFIX_COUNT,
            "seal_root": P1R23_B10_SEAL_ROOT,
            "request_order_sha256": P1R23_B10_ORDER,
            "case_order_sha256": canonical_hash([item.case_id for item in prefix]),
            "request_vector_sha256": canonical_hash(
                [item.request_sha256 for item in prefix]
            ),
            "exact_identity_proof": True,
        },
        "added_count": P1R23_ADDED_COUNT,
        "ordered_count": P1R23_B100_COUNT,
        "ordered_cases": records,
        "exclusion_inventory": exclusion_inventory,
        "exclusion_root": exclusion_inventory["identity_sha256"],
        "order_root": canonical_hash(records),
        "source_manifest_root": canonical_hash(role_sources),
        "common_alias_method_selection": True,
        "alias_specific_filter_count": 0,
        "method_specific_filter_count": 0,
        "model_tokenization_or_length_filter_count": 0,
        "model_artifact_evaluator_outcome_access_count": 0,
        "b100_source_exhaustion_policy": "B100_SOURCE_EXHAUSTED",
    }
    payload["seal_root"] = canonical_hash(payload)
    payload["root_digest"] = canonical_hash(payload)
    return payload


def verify_p1r23_b100_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("P1R23 B100 seal root differs")
    if (
        payload.get("schema_version") != P1R23_B100_SCHEMA
        or payload.get("instruction_id") != P1R23_B100_AMENDMENT_ID
        or payload.get("selection_contract") != "A_PREFIX_PLUS_CANONICAL_90"
        or payload.get("ordered_count") != P1R23_B100_COUNT
        or len(payload.get("ordered_cases", ())) != P1R23_B100_COUNT
        or payload.get("model_artifact_evaluator_outcome_access_count") != 0
        or payload.get("alias_specific_filter_count") != 0
        or payload.get("method_specific_filter_count") != 0
    ):
        raise ODEBFContractError("P1R23 B100 seal contract differs")
    payload["root_digest"] = root
    return payload


def load_p1r23_b100_requests(
    dataset_path: Path,
    sealed: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Decode only the ordered B100 rewrite surfaces after seal verification.

    Selection never needs held-out evaluation fields.  This loader therefore
    uses the same bounded ``requested_rewrite`` extraction as the selector and
    independently reprojects every discrete case/request/collision identity.
    """

    value = verify_p1r23_b100_seal(sealed)
    ordered = tuple(value["ordered_cases"])
    by_ordinal = {
        int(item["dataset_ordinal"]): (position, item)
        for position, item in enumerate(ordered)
    }
    if len(by_ordinal) != P1R23_B100_COUNT:
        raise ODEBFContractError("P1R23 B100 dataset ordinals repeat")
    loaded: list[dict[str, Any] | None] = [None] * P1R23_B100_COUNT
    for dataset_ordinal, blob in enumerate(
        _iter_top_level_objects(dataset_path.resolve(strict=True))
    ):
        selected = by_ordinal.get(dataset_ordinal)
        if selected is None:
            continue
        position, record = selected
        identity = _project_identity(blob, dataset_ordinal=dataset_ordinal)
        if (
            identity.case_id != int(record["case_id"])
            or identity.request_sha256 != str(record["request_sha256"])
            or identity.collision_sha256 != str(record["collision_sha256"])
            or identity.relation_sha256 != str(record["relation_sha256"])
        ):
            raise ODEBFContractError("P1R23 B100 sealed request identity differs")
        try:
            rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEBFContractError("P1R23 B100 rewrite JSON differs") from exc
        if not isinstance(rewrite, Mapping):
            raise ODEBFContractError("P1R23 B100 rewrite schema differs")
        target_new = rewrite.get("target_new")
        target_true = rewrite.get("target_true")
        if not isinstance(target_new, Mapping) or not isinstance(target_true, Mapping):
            raise ODEBFContractError("P1R23 B100 target schema differs")
        loaded[position] = {
            "case_id": identity.case_id,
            "prompt": _required_text(rewrite.get("prompt"), label="prompt"),
            "relation_id": _required_text(
                rewrite.get("relation_id"), label="relation"
            ),
            "subject": _required_text(rewrite.get("subject"), label="subject"),
            "target_new": _required_text(target_new.get("str"), label="new target"),
            "target_true": _required_text(target_true.get("str"), label="old target"),
            "request_sha256": identity.request_sha256,
            "collision_sha256": identity.collision_sha256,
            "dataset_ordinal": dataset_ordinal,
            "p1r23_position": position,
        }
    if any(item is None for item in loaded):
        raise ODEBFContractError("P1R23 B100 request load is incomplete")
    requests = tuple(item for item in loaded if item is not None)
    return requests


__all__ = [
    "B100CaseIdentity",
    "P1R23_B100_AMENDMENT_ID",
    "P1R23_B100_BASE_COMMIT",
    "P1R23_B100_SCHEMA",
    "build_p1r23_b100_seal",
    "load_b100_case_identities",
    "load_p1r23_b100_requests",
    "verify_p1r23_b100_seal",
]
