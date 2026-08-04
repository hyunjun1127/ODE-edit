"""Outcome-free seals for the sequential CounterFact B10 P1 stream.

Only tracked seal/manifest blobs from the locked base commit participate in
the prior-selection scan.  Consequently this module never opens local result
roots and rebuilding a seal after it is tracked cannot make the selection
self-referential.
"""

from __future__ import annotations

import ast
import collections
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

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .request_digest import ordered_request_digest_v1


EXPECTED_BASE = "a5b7a60237c85432cbded8487ff04602cf4094e6"
STREAM_SALT = "odebf-s04-p1-seqb10-v1"
P_POPULATION_SALT = "odebf-s04-p1-p-anchor-v1"
STREAM_SCHEMA = "ode-edit-s04-ode-bf-p1-seqb10-seal/v1"
P_POPULATION_SCHEMA = "ode-edit-s04-ode-bf-p1-p-population-seal/v1"
P1_INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-SEQUENTIAL-B10-NATIVE-FLOOR-P1-V1"
P1R2_EXPECTED_BASE = "e753972da50a5d6fa9789ef2e9083c9b0549c3d0"
P1R2_STREAM_SALT = "odebf-s04-p1r2-seqb10-v2"
P1R2_P_POPULATION_SALT = "odebf-s04-p1r2-p-anchor-v2"
P1R2_STREAM_SCHEMA = "ode-edit-s04-ode-bf-p1r2-seqb10-seal/v2"
P1R2_P_POPULATION_SCHEMA = (
    "ode-edit-s04-ode-bf-p1r2-p-population-seal/v2"
)
P1R2_INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-TRUST-RATIO-MEAN-P-P1R2-V1"
STREAM_REQUEST_COUNT = 40
SEQUENTIAL_BATCH_COUNT = 4
P_POPULATION_COUNT = 160
P_SAMPLE_COUNT = 10
P_LINEAGE_LAYOUT = (
    ("controller", 0, 80, 4101),
    ("terminal-confirmation", 80, 120, 4102),
    ("report-only", 120, 160, 4103),
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SPACE = re.compile(r"\s+")
_CASE_ID = re.compile(rb'"case_id"\s*:\s*"?(\d+)"?')


@dataclass(frozen=True, slots=True)
class P1RequestIdentity:
    case_id: int
    request_sha256: str
    collision_sha256: str
    relation_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 0:
            raise ODEBFContractError("P1 case identity is invalid")
        for value in (self.request_sha256, self.collision_sha256, self.relation_sha256):
            if not _HEX64.fullmatch(value):
                raise ODEBFContractError("P1 request identity is not SHA-256")


@dataclass(frozen=True, slots=True)
class PriorSelectionScan:
    case_ids: frozenset[int]
    request_sha256: frozenset[str]
    source_count: int
    sources_digest: str
    base_commit: str


@dataclass(frozen=True, slots=True)
class P1SealPolicy:
    expected_base: str
    stream_salt: str
    p_population_salt: str
    stream_schema: str
    p_population_schema: str
    instruction_id: str
    prior_development_exclusion: bool


P1_SEAL_POLICY = P1SealPolicy(
    EXPECTED_BASE,
    STREAM_SALT,
    P_POPULATION_SALT,
    STREAM_SCHEMA,
    P_POPULATION_SCHEMA,
    P1_INSTRUCTION_ID,
    False,
)
P1R2_SEAL_POLICY = P1SealPolicy(
    P1R2_EXPECTED_BASE,
    P1R2_STREAM_SALT,
    P1R2_P_POPULATION_SALT,
    P1R2_STREAM_SCHEMA,
    P1R2_P_POPULATION_SCHEMA,
    P1R2_INSTRUCTION_ID,
    True,
)


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ODEBFContractError(f"CounterFact {label} is empty")
    return value


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _case_id_from_row(blob: bytes) -> int:
    matches = _CASE_ID.findall(blob)
    if len(matches) != 1:
        raise ODEBFContractError("CounterFact selection case identity is ambiguous")
    return int(matches[0])


def project_p1_request_identity(blob: bytes) -> P1RequestIdentity:
    """Decode only the canonical rewrite object; held-out fields stay unopened."""

    case_id = _case_id_from_row(blob)
    try:
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("CounterFact requested rewrite is invalid JSON") from exc
    required = {"prompt", "relation_id", "subject", "target_new", "target_true"}
    if not isinstance(rewrite, Mapping) or set(rewrite) != required:
        raise ODEBFContractError("CounterFact requested rewrite schema differs")
    if not isinstance(rewrite["target_new"], Mapping) or not isinstance(
        rewrite["target_true"], Mapping
    ):
        raise ODEBFContractError("CounterFact target schema differs")
    prompt = _text(rewrite["prompt"], "prompt")
    relation = _text(rewrite["relation_id"], "relation")
    subject = _text(rewrite["subject"], "subject")
    target_new = _text(rewrite["target_new"].get("str"), "new target")
    target_old = _text(rewrite["target_true"].get("str"), "old target")
    request_payload = {
        "case_id": case_id,
        "prompt": prompt,
        "relation_id": relation,
        "subject": subject,
        "target_new": target_new,
        "target_old": target_old,
    }
    collision_payload = {
        "subject": _normalize(subject),
        "relation": _normalize(relation),
        "template": _normalize(prompt),
    }
    return P1RequestIdentity(
        case_id,
        canonical_hash(request_payload),
        canonical_hash(collision_payload),
        canonical_hash({"relation": _normalize(relation)}),
    )


def load_p1_request_identities(dataset_path: str | Path) -> tuple[P1RequestIdentity, ...]:
    dataset = Path(dataset_path).resolve(strict=True)
    identities = tuple(
        project_p1_request_identity(blob) for blob in _iter_top_level_objects(dataset)
    )
    if not identities or len({item.case_id for item in identities}) != len(identities):
        raise ODEBFContractError("CounterFact P1 case identities repeat")
    if len({item.request_sha256 for item in identities}) != len(identities):
        raise ODEBFContractError("CounterFact P1 request identities repeat")
    return identities


def _selection_blob_paths(repo: Path, base_commit: str) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--name-only", base_commit],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        lowered = relative.casefold()
        if not lowered.endswith((".json", ".py")):
            continue
        if not any(token in lowered for token in ("seal", "manifest", "selection")):
            continue
        paths.append(relative)
    return tuple(sorted(paths))


def _tracked_blob(repo: Path, base_commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{base_commit}:{relative}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _walk_prior_json(value: Any, *, parent_key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            yield normalized, child
            yield from _walk_prior_json(child, parent_key=normalized)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_prior_json(child, parent_key=parent_key)


def _prior_identities_from_json(blob: bytes) -> tuple[set[int], set[str]]:
    try:
        value = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("tracked selection JSON is invalid") from exc
    cases: set[int] = set()
    requests: set[str] = set()
    for key, child in _walk_prior_json(value):
        if key.endswith("case_id"):
            values = (child,)
        elif key.endswith("case_ids") and isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            values = tuple(child)
        else:
            values = ()
        for item in values:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                cases.add(item)
            elif isinstance(item, str) and item.isdigit():
                cases.add(int(item))
        if "request" in key and ("sha" in key or "hash" in key):
            request_values = (
                tuple(child)
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray))
                else (child,)
            )
            requests.update(
                str(item) for item in request_values if isinstance(item, str) and _HEX64.fullmatch(item)
            )
    return cases, requests


def _literal_sequence(node: ast.AST) -> tuple[Any, ...]:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _prior_identities_from_python(blob: bytes, relative: str) -> tuple[set[int], set[str]]:
    try:
        tree = ast.parse(blob.decode("utf-8"), filename=relative)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ODEBFContractError("tracked selection Python source is invalid") from exc
    cases: set[int] = set()
    requests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id.casefold() for target in targets if isinstance(target, ast.Name)]
        values = _literal_sequence(node.value)
        if any(name.endswith(("case_id", "case_ids")) for name in names):
            for item in values:
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                    cases.add(item)
                elif isinstance(item, str) and item.isdigit():
                    cases.add(int(item))
        if any("request" in name and ("sha" in name or "hash" in name) for name in names):
            requests.update(
                item for item in values if isinstance(item, str) and _HEX64.fullmatch(item)
            )
    return cases, requests


def scan_prior_tracked_seals(
    repo_root: str | Path,
    *,
    base_commit: str = EXPECTED_BASE,
) -> PriorSelectionScan:
    repo = Path(repo_root).resolve(strict=True)
    observed = subprocess.run(
        ["git", "rev-parse", base_commit],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != base_commit:
        raise ODEBFContractError("P1 prior-selection base commit differs")
    cases: set[int] = set()
    requests: set[str] = set()
    records: list[dict[str, Any]] = []
    for relative in _selection_blob_paths(repo, base_commit):
        blob = _tracked_blob(repo, base_commit, relative)
        if relative.endswith(".json"):
            local_cases, local_requests = _prior_identities_from_json(blob)
        else:
            local_cases, local_requests = _prior_identities_from_python(blob, relative)
        if not local_cases and not local_requests:
            continue
        cases.update(local_cases)
        requests.update(local_requests)
        records.append(
            {
                "path": relative,
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "case_count": len(local_cases),
                "case_digest": canonical_hash(sorted(local_cases)),
                "request_count": len(local_requests),
                "request_digest": canonical_hash(sorted(local_requests)),
            }
        )
    return PriorSelectionScan(
        frozenset(cases),
        frozenset(requests),
        len(records),
        canonical_hash(records),
        base_commit,
    )


def _rank(salt: str, identity: P1RequestIdentity) -> tuple[str, str, int]:
    ranked = hashlib.sha256(
        f"{salt}|{identity.request_sha256}".encode("utf-8")
    ).hexdigest()
    return ranked, identity.request_sha256, identity.case_id


def _eligible(
    identities: Sequence[P1RequestIdentity],
    prior: PriorSelectionScan,
) -> tuple[tuple[P1RequestIdentity, ...], int, str]:
    counts = collections.Counter(item.collision_sha256 for item in identities)
    duplicate_groups = {key for key, count in counts.items() if count > 1}
    eligible = tuple(
        item
        for item in identities
        if item.case_id not in prior.case_ids
        and item.request_sha256 not in prior.request_sha256
        and item.collision_sha256 not in duplicate_groups
    )
    exclusion_payload = {
        "prior_case_count": len(prior.case_ids),
        "prior_request_count": len(prior.request_sha256),
        "duplicate_group_count": len(duplicate_groups),
        "excluded_total": len(identities) - len(eligible),
    }
    return eligible, len(identities) - len(eligible), canonical_hash(exclusion_payload)


def _source_payload(dataset: Path, row_count: int) -> dict[str, Any]:
    return {
        "relative_contract": "EasyEdit/data/counterfact/counterfact.json",
        "sha256": _sha256_file(dataset),
        "size_bytes": dataset.stat().st_size,
        "row_count": row_count,
    }


def _prior_development_exclusion(
    repo: Path,
    base_commit: str,
) -> tuple[dict[str, Any], frozenset[int], frozenset[str], frozenset[str]]:
    relative_paths = (
        "project/run_scripts/ode_bf/locks/p1_seqb10_stream_seal.json",
        "project/run_scripts/ode_bf/locks/p1_p_population_seal.json",
    )
    values: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for relative in relative_paths:
        blob = _tracked_blob(repo, base_commit, relative)
        try:
            value = json.loads(blob)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEBFContractError("P1R2 development seal is invalid") from exc
        root = value.pop("root_digest", None)
        if root != canonical_hash(value):
            raise ODEBFContractError("P1R2 development seal root differs")
        value["root_digest"] = root
        values.append(value)
        records.append(
            {
                "path": relative,
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "root_digest": root,
            }
        )
    stream, population = values
    stream_items = stream.get("requests")
    population_items = population.get("items")
    if (
        not isinstance(stream_items, list)
        or len(stream_items) != STREAM_REQUEST_COUNT
        or not isinstance(population_items, list)
        or len(population_items) != P_POPULATION_COUNT
    ):
        raise ODEBFContractError("P1R2 development exclusion inventory differs")
    combined = tuple((*stream_items, *population_items))
    case_ids = frozenset(int(item["case_id"]) for item in combined)
    request_ids = frozenset(str(item["request_sha256"]) for item in combined)
    collision_ids = frozenset(str(item["collision_sha256"]) for item in combined)
    if (
        len(case_ids) != STREAM_REQUEST_COUNT + P_POPULATION_COUNT
        or len(request_ids) != STREAM_REQUEST_COUNT + P_POPULATION_COUNT
        or len(collision_ids) != STREAM_REQUEST_COUNT + P_POPULATION_COUNT
    ):
        raise ODEBFContractError("P1R2 development roles are not fully distinct")
    payload = {
        "source_commit": base_commit,
        "tracked_source_count": len(records),
        "tracked_sources_digest": canonical_hash(records),
        "prior_stream_count": STREAM_REQUEST_COUNT,
        "prior_population_count": P_POPULATION_COUNT,
        "combined_case_digest": canonical_hash(sorted(case_ids)),
        "combined_request_digest": canonical_hash(sorted(request_ids)),
        "combined_collision_digest": canonical_hash(sorted(collision_ids)),
        "local_result_paths_read": 0,
        "development_outcomes_read": 0,
    }
    return payload, case_ids, request_ids, collision_ids


def _build_p1_seals(
    dataset_path: str | Path,
    repo_root: str | Path,
    *,
    policy: P1SealPolicy,
    base_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = Path(dataset_path).resolve(strict=True)
    repo = Path(repo_root).resolve(strict=True)
    if base_commit != policy.expected_base:
        raise ODEBFContractError("P1 selection policy/base differs")
    identities = load_p1_request_identities(dataset)
    prior = scan_prior_tracked_seals(repo, base_commit=base_commit)
    development_payload: dict[str, Any] | None = None
    if policy.prior_development_exclusion:
        (
            development_payload,
            development_cases,
            development_requests,
            development_collisions,
        ) = _prior_development_exclusion(repo, base_commit)
        if not development_cases.issubset(prior.case_ids) or not development_requests.issubset(
            prior.request_sha256
        ):
            raise ODEBFContractError("P1R2 prior scan omits development identities")
    else:
        development_collisions = frozenset()
    eligible, excluded_count, exclusion_digest = _eligible(identities, prior)
    if any(item.collision_sha256 in development_collisions for item in eligible):
        raise ODEBFContractError("P1R2 eligible pool retains a development collision")
    if len(eligible) < STREAM_REQUEST_COUNT + P_POPULATION_COUNT:
        raise ODEBFContractError("insufficient disjoint P1 stream/calibration population")

    stream_selected = tuple(
        sorted(eligible, key=lambda item: _rank(policy.stream_salt, item))[:STREAM_REQUEST_COUNT]
    )
    stream_ids = {item.request_sha256 for item in stream_selected}
    stream_collisions = {item.collision_sha256 for item in stream_selected}
    remaining = tuple(
        item
        for item in eligible
        if item.request_sha256 not in stream_ids
        and item.collision_sha256 not in stream_collisions
    )
    population_selected = tuple(
        sorted(remaining, key=lambda item: _rank(policy.p_population_salt, item))[:P_POPULATION_COUNT]
    )
    if len(population_selected) != P_POPULATION_COUNT:
        raise ODEBFContractError("P1 functional P population is incomplete")
    population_ids = {item.request_sha256 for item in population_selected}
    population_collisions = {item.collision_sha256 for item in population_selected}
    if stream_ids.intersection(population_ids) or stream_collisions.intersection(
        population_collisions
    ):
        raise ODEBFContractError("P1 stream and functional P population collide")

    source = _source_payload(dataset, len(identities))
    prior_payload = {
        "base_commit": prior.base_commit,
        "source_count": prior.source_count,
        "sources_digest": prior.sources_digest,
        "case_id_count": len(prior.case_ids),
        "case_ids_digest": canonical_hash(sorted(prior.case_ids)),
        "request_sha256_count": len(prior.request_sha256),
        "request_sha256_digest": canonical_hash(sorted(prior.request_sha256)),
        "local_result_paths_read": 0,
    }
    stream_records: list[dict[str, Any]] = []
    batch_digests: list[str] = []
    for ordinal, item in enumerate(stream_selected):
        stream_records.append(
            {
                "ordinal": ordinal,
                "sequential_batch": ordinal // BATCH_SIZE,
                "ordinal_in_batch": ordinal % BATCH_SIZE,
                "case_id": item.case_id,
                "request_sha256": item.request_sha256,
                "collision_sha256": item.collision_sha256,
            }
        )
    for batch_index in range(SEQUENTIAL_BATCH_COUNT):
        start = batch_index * BATCH_SIZE
        ordered = [
            item["request_sha256"] for item in stream_records[start : start + BATCH_SIZE]
        ]
        batch_digests.append(ordered_request_digest_v1(ordered))
    stream: dict[str, Any] = {
        "schema_version": policy.stream_schema,
        "instruction_id": policy.instruction_id,
        "status": "sealed-before-model-action",
        "salt": policy.stream_salt,
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": SEQUENTIAL_BATCH_COUNT,
        "logical_edit_count": STREAM_REQUEST_COUNT,
        "source": source,
        "rank_formula": "sha256(utf8(salt + '|' + canonical_request_sha256))",
        "request_hash_fields": [
            "case_id",
            "prompt",
            "relation_id",
            "subject",
            "target_new",
            "target_old",
        ],
        "duplicate_policy": (
            "exclude-entire-nfkc-casefold-subject-relation-template-groups"
        ),
        "prior_selection_scan": prior_payload,
        "excluded_count": excluded_count,
        "exclusion_digest": exclusion_digest,
        "requests": stream_records,
        "batch_ordered_request_digest_v1": batch_digests,
        "models": {
            alias: {"same_stream": True, "batch_digests": batch_digests}
            for alias in MODEL_ALIASES
        },
    }
    if development_payload is not None:
        stream["development_exclusion"] = development_payload
    stream["root_digest"] = canonical_hash(stream)

    population_records = [
        {
            "ordinal": ordinal,
            "case_id": item.case_id,
            "request_sha256": item.request_sha256,
            "collision_sha256": item.collision_sha256,
            "stratum": f"relation-hash-quartile-{int(item.relation_sha256[:8], 16) % 4}",
        }
        for ordinal, item in enumerate(population_selected)
    ]
    population_identity = canonical_hash(population_records)
    lineages = {
        name: {
            "seed": seed,
            "sample_count": P_SAMPLE_COUNT,
            "pool_start": start,
            "pool_end": end,
            "allowed_item_sha256": [
                item["request_sha256"] for item in population_records[start:end]
            ],
        }
        for name, start, end, seed in P_LINEAGE_LAYOUT
    }
    population: dict[str, Any] = {
        "schema_version": policy.p_population_schema,
        "instruction_id": policy.instruction_id,
        "status": "sealed-before-model-action",
        "salt": policy.p_population_salt,
        "source": source,
        "source_class": "counterfact-canonical-calibration",
        "population_count": P_POPULATION_COUNT,
        "population_sha256": population_identity,
        "sample_count": P_SAMPLE_COUNT,
        "lineage_separation": "controller-terminal-report-disjoint",
        "schedule_indices": [
            "sequential_batch",
            "correction_cycle",
            "waypoint",
            "replay_batch_id",
        ],
        "arm_identity_in_schedule": False,
        "model_identity_in_schedule": False,
        "stream_root_digest": stream["root_digest"],
        "prior_selection_scan": prior_payload,
        "items": population_records,
        "lineages": lineages,
        "heldout_paraphrase_items": 0,
        "heldout_neighborhood_items": 0,
        "generation_items": 0,
    }
    if development_payload is not None:
        population["development_exclusion"] = development_payload
    population["root_digest"] = canonical_hash(population)
    return stream, population


def build_p1_seals(
    dataset_path: str | Path,
    repo_root: str | Path,
    *,
    base_commit: str = EXPECTED_BASE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_p1_seals(
        dataset_path,
        repo_root,
        policy=P1_SEAL_POLICY,
        base_commit=base_commit,
    )


def build_p1r2_seals(
    dataset_path: str | Path,
    repo_root: str | Path,
    *,
    base_commit: str = P1R2_EXPECTED_BASE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_p1_seals(
        dataset_path,
        repo_root,
        policy=P1R2_SEAL_POLICY,
        base_commit=base_commit,
    )


def _verify_rooted(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("P1 seal root digest differs")
    payload["root_digest"] = root
    if payload.get("schema_version") != schema:
        raise ODEBFContractError("P1 seal schema differs")
    return payload


def _stream_policy(value: Mapping[str, Any]) -> P1SealPolicy:
    schema = value.get("schema_version")
    for policy in (P1_SEAL_POLICY, P1R2_SEAL_POLICY):
        if schema == policy.stream_schema:
            return policy
    raise ODEBFContractError("P1 stream schema differs")


def _population_policy(value: Mapping[str, Any]) -> P1SealPolicy:
    schema = value.get("schema_version")
    for policy in (P1_SEAL_POLICY, P1R2_SEAL_POLICY):
        if schema == policy.p_population_schema:
            return policy
    raise ODEBFContractError("P1 P-population schema differs")


def verify_p1_stream_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = _stream_policy(value)
    payload = _verify_rooted(value, policy.stream_schema)
    if (
        payload.get("salt") != policy.stream_salt
        or payload.get("instruction_id") != policy.instruction_id
        or payload.get("edit_batch_size") != BATCH_SIZE
        or payload.get("sequential_batch_count") != SEQUENTIAL_BATCH_COUNT
        or payload.get("logical_edit_count") != STREAM_REQUEST_COUNT
    ):
        raise ODEBFContractError("P1 stream policy differs")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != STREAM_REQUEST_COUNT:
        raise ODEBFContractError("P1 stream request count differs")
    if [item.get("ordinal") for item in requests] != list(range(STREAM_REQUEST_COUNT)):
        raise ODEBFContractError("P1 stream ordinal differs")
    case_ids = [item.get("case_id") for item in requests]
    hashes = [item.get("request_sha256") for item in requests]
    collisions = [item.get("collision_sha256") for item in requests]
    if not all(len(set(values)) == STREAM_REQUEST_COUNT for values in (case_ids, hashes, collisions)):
        raise ODEBFContractError("P1 stream is not fully distinct")
    expected_batches = []
    for batch_index in range(SEQUENTIAL_BATCH_COUNT):
        start = batch_index * BATCH_SIZE
        batch = requests[start : start + BATCH_SIZE]
        if [item.get("sequential_batch") for item in batch] != [batch_index] * BATCH_SIZE:
            raise ODEBFContractError("P1 stream batch partition differs")
        if [item.get("ordinal_in_batch") for item in batch] != list(range(BATCH_SIZE)):
            raise ODEBFContractError("P1 stream within-batch order differs")
        expected_batches.append(
            ordered_request_digest_v1([str(item["request_sha256"]) for item in batch])
        )
    if payload.get("batch_ordered_request_digest_v1") != expected_batches:
        raise ODEBFContractError("P1 stream batch digest differs")
    development = payload.get("development_exclusion")
    if policy.prior_development_exclusion:
        if (
            not isinstance(development, Mapping)
            or development.get("prior_stream_count") != STREAM_REQUEST_COUNT
            or development.get("prior_population_count") != P_POPULATION_COUNT
            or development.get("local_result_paths_read") != 0
            or development.get("development_outcomes_read") != 0
        ):
            raise ODEBFContractError("P1R2 development exclusion differs")
    elif development is not None:
        raise ODEBFContractError("P1 V1 unexpectedly has development exclusions")
    return payload


def verify_p1_population_seal(
    value: Mapping[str, Any],
    *,
    stream: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _population_policy(value)
    payload = _verify_rooted(value, policy.p_population_schema)
    stream_value = verify_p1_stream_seal(stream)
    if _stream_policy(stream_value) != policy:
        raise ODEBFContractError("P1 stream/population revisions differ")
    if (
        payload.get("salt") != policy.p_population_salt
        or payload.get("instruction_id") != policy.instruction_id
        or payload.get("population_count") != P_POPULATION_COUNT
        or payload.get("sample_count") != P_SAMPLE_COUNT
        or payload.get("stream_root_digest") != stream_value["root_digest"]
    ):
        raise ODEBFContractError("P1 P-population policy differs")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != P_POPULATION_COUNT:
        raise ODEBFContractError("P1 P-population count differs")
    identities = [item.get("request_sha256") for item in items]
    collisions = [item.get("collision_sha256") for item in items]
    if len(set(identities)) != P_POPULATION_COUNT or len(set(collisions)) != P_POPULATION_COUNT:
        raise ODEBFContractError("P1 P-population is not distinct")
    stream_ids = {item["request_sha256"] for item in stream_value["requests"]}
    stream_collisions = {item["collision_sha256"] for item in stream_value["requests"]}
    if stream_ids.intersection(identities) or stream_collisions.intersection(collisions):
        raise ODEBFContractError("P1 P-population collides with the edit stream")
    if payload.get("population_sha256") != canonical_hash(items):
        raise ODEBFContractError("P1 P-population identity differs")
    lineages = payload.get("lineages")
    if not isinstance(lineages, dict) or set(lineages) != {item[0] for item in P_LINEAGE_LAYOUT}:
        raise ODEBFContractError("P1 P-population lineages differ")
    pools: list[set[str]] = []
    for name, start, end, seed in P_LINEAGE_LAYOUT:
        spec = lineages[name]
        expected = identities[start:end]
        if (
            spec.get("seed") != seed
            or spec.get("sample_count") != P_SAMPLE_COUNT
            or spec.get("pool_start") != start
            or spec.get("pool_end") != end
            or spec.get("allowed_item_sha256") != expected
        ):
            raise ODEBFContractError("P1 P-population lineage contract differs")
        pools.append(set(expected))
    if any(pools[left].intersection(pools[right]) for left in range(3) for right in range(left + 1, 3)):
        raise ODEBFContractError("P1 P-population lineages overlap")
    if payload.get("development_exclusion") != stream_value.get(
        "development_exclusion"
    ):
        raise ODEBFContractError("P1 stream/population development exclusions differ")
    return payload


def _load_canonical_request_map(
    dataset_path: str | Path,
    approved: Mapping[str, tuple[int, str]],
) -> dict[str, dict[str, Any]]:
    dataset = Path(dataset_path).resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    wanted_cases = {case_id for case_id, _ in approved.values()}
    for blob in _iter_top_level_objects(dataset):
        case_id = _case_id_from_row(blob)
        if case_id not in wanted_cases:
            continue
        identity = project_p1_request_identity(blob)
        expected = approved.get(identity.request_sha256)
        if expected != (case_id, identity.collision_sha256):
            raise ODEBFContractError("P1 canonical request identity differs")
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
        result[identity.request_sha256] = {
            "case_id": case_id,
            "prompt": rewrite["prompt"],
            "relation_id": rewrite["relation_id"],
            "subject": rewrite["subject"],
            "target_new": rewrite["target_new"]["str"],
            "target_true": rewrite["target_true"]["str"],
            "request_sha256": identity.request_sha256,
        }
    if set(result) != set(approved):
        raise ODEBFContractError("P1 canonical request load is incomplete")
    return result


def load_p1_stream_batches(
    dataset_path: str | Path,
    stream: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    value = verify_p1_stream_seal(stream)
    dataset = Path(dataset_path).resolve(strict=True)
    source = value["source"]
    if dataset.stat().st_size != source["size_bytes"] or _sha256_file(dataset) != source["sha256"]:
        raise ODEBFContractError("P1 stream dataset identity differs")
    approved = {
        item["request_sha256"]: (int(item["case_id"]), item["collision_sha256"])
        for item in value["requests"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    batches: list[tuple[dict[str, Any], ...]] = []
    for batch_index in range(SEQUENTIAL_BATCH_COUNT):
        start = batch_index * BATCH_SIZE
        records = value["requests"][start : start + BATCH_SIZE]
        batch = tuple(loaded[item["request_sha256"]] for item in records)
        if ordered_request_digest_v1([item["request_sha256"] for item in batch]) != value[
            "batch_ordered_request_digest_v1"
        ][batch_index]:
            raise ODEBFContractError("P1 loaded stream request order differs")
        batches.append(batch)
    return tuple(batches)


def load_p1_population_requests(
    dataset_path: str | Path,
    population: Mapping[str, Any],
    *,
    stream: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    value = verify_p1_population_seal(population, stream=stream)
    dataset = Path(dataset_path).resolve(strict=True)
    source = value["source"]
    if dataset.stat().st_size != source["size_bytes"] or _sha256_file(dataset) != source["sha256"]:
        raise ODEBFContractError("P1 P-population dataset identity differs")
    approved = {
        item["request_sha256"]: (int(item["case_id"]), item["collision_sha256"])
        for item in value["items"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    return tuple(loaded[item["request_sha256"]] for item in value["items"])
