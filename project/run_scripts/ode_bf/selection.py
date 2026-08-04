"""Outcome-free deterministic selection of one genuine P0 joint B10 batch."""

from __future__ import annotations

import ast
import collections
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.ode_alloc.selection import (
    RequestIdentity,
    load_projected_request,
    load_request_identities,
)

from .contracts import (
    BATCH_SIZE,
    MODEL_ALIASES,
    ODEBFContractError,
    canonical_hash,
)


P0_B10_SALT = "odebf-s04-p0-b10-v1"
EXPLICIT_EXCLUSIONS = frozenset(
    {2022, 12498, 20964, 768, 17503, 1534, 14652, 9774}
)
_CASE_ID_SCALAR = re.compile(rb'"[A-Za-z0-9_]*case_id"\s*:\s*"?(\d+)"?')
_CASE_IDS_ARRAY = re.compile(
    rb'"[A-Za-z0-9_]*case_ids"\s*:\s*\[([^\]]*)\]', re.DOTALL
)
_DIGITS = re.compile(rb'"?(\d+)"?')


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _rank(identity: RequestIdentity) -> tuple[str, str, int]:
    return (
        hashlib.sha256(
            P0_B10_SALT.encode("utf-8")
            + b"\0"
            + identity.request_hash.encode("ascii")
        ).hexdigest(),
        identity.request_hash,
        identity.case_id,
    )


def _literal_case_ids_from_python(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not any(name.upper().endswith("CASE_IDS") for name in names):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, (str, int)) and str(item).isdigit():
                    found.add(int(item))
    return found


def scan_foreign_tracked_prior_case_ids(
    repo_root: str | Path,
) -> tuple[set[int], str, int]:
    """Scan prior tracked selections without reading this or foreign SH1 namespaces.

    The reusable ODE-Alloc scanner predates ODE-BF.  Calling it after the
    ODE-BF seal is tracked would read the seal itself and make an outcome-free
    rebuild non-idempotent.  This local scanner preserves its lexical contract
    while excluding both scientific implementation namespaces before opening
    any file.  Foreign prior-session paths are likewise filtered before any
    resolution or read.
    """

    repo = Path(repo_root).resolve(strict=True)
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.json", "*.py"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    found: set[int] = set()
    source_records: list[dict[str, Any]] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        lowered = relative.casefold()
        if (
            "session03" in lowered
            or "session_03" in lowered
            or relative.startswith("project/run_scripts/ode_alloc/")
            or relative.startswith("project/run_scripts/session04_ode_alloc_")
            or relative.startswith("project/run_scripts/ode_bf/")
            or relative.startswith("project/run_scripts/session04_ode_bf_")
        ):
            continue
        path = (repo / relative).resolve(strict=True)
        path.relative_to(repo)
        if path.suffix == ".json":
            blob = path.read_bytes()
            local = {int(value) for value in _CASE_ID_SCALAR.findall(blob)}
            for body in _CASE_IDS_ARRAY.findall(blob):
                local.update(int(value) for value in _DIGITS.findall(body))
        else:
            local = _literal_case_ids_from_python(path)
        if local:
            found.update(local)
            source_records.append(
                {"path": relative, "case_ids_sha256": canonical_hash(sorted(local))}
            )
    return found, canonical_hash(source_records), len(source_records)


def build_p0_b10_seal(
    dataset_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve(strict=True)
    repo = Path(repo_root).resolve(strict=True)
    identities = load_request_identities(dataset)
    prior, prior_sources_digest, prior_source_count = scan_foreign_tracked_prior_case_ids(
        repo
    )
    duplicate_counts = collections.Counter(item.duplicate_key for item in identities)
    duplicate_keys = {key for key, count in duplicate_counts.items() if count > 1}
    used = set(EXPLICIT_EXCLUSIONS).union(prior)
    selected: list[RequestIdentity] = []
    for identity in sorted(identities, key=_rank):
        if identity.case_id in used or identity.duplicate_key in duplicate_keys:
            continue
        selected.append(identity)
        used.add(identity.case_id)
        if len(selected) == BATCH_SIZE:
            break
    if len(selected) != BATCH_SIZE:
        raise ODEBFContractError("insufficient disjoint P0 B10 candidates")
    records = [
        {
            "ordinal": ordinal,
            "case_id": identity.case_id,
            "request_sha256": identity.request_hash,
        }
        for ordinal, identity in enumerate(selected)
    ]
    request_order_sha256 = canonical_hash(records)
    payload: dict[str, Any] = {
        "schema_version": "ode-edit-s04-ode-bf-p0-b10-seal/v1",
        "status": "sealed-before-model-outcome",
        "salt": P0_B10_SALT,
        "edit_batch_size": BATCH_SIZE,
        "source": {
            "relative_contract": "EasyEdit/data/counterfact/counterfact.json",
            "sha256": _file_sha256(dataset),
            "size_bytes": dataset.stat().st_size,
            "row_count": len(identities),
        },
        "request_hash_contract": {
            "encoding": "utf-8-canonical-json-sort-keys-compact-sha256",
            "fields": [
                "case_id",
                "prompt",
                "relation_id",
                "subject",
                "target_new",
                "target_old",
            ],
        },
        "explicit_exclusions": sorted(EXPLICIT_EXCLUSIONS),
        "tracked_prior_collision_scan": {
            "case_id_count": len(prior),
            "case_ids_sha256": canonical_hash(sorted(prior)),
            "source_count": prior_source_count,
            "sources_digest": prior_sources_digest,
            "session03_paths_read": 0,
        },
        "duplicate_policy": "exclude-entire-nfkc-casefold-subject-plus-template-groups",
        "requests": records,
        "models": {
            alias: {
                "request_order_sha256": request_order_sha256,
                "same_joint_batch": True,
            }
            for alias in MODEL_ALIASES
        },
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def verify_p0_b10_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    observed_root = payload.pop("root_digest", None)
    if observed_root != canonical_hash(payload):
        raise ODEBFContractError("P0 B10 seal root digest differs")
    payload["root_digest"] = observed_root
    if payload.get("schema_version") != "ode-edit-s04-ode-bf-p0-b10-seal/v1":
        raise ODEBFContractError("P0 B10 seal schema differs")
    if payload.get("edit_batch_size") != BATCH_SIZE or payload.get("salt") != P0_B10_SALT:
        raise ODEBFContractError("P0 B10 seal policy differs")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != BATCH_SIZE:
        raise ODEBFContractError("P0 B10 seal request count differs")
    if [item.get("ordinal") for item in requests] != list(range(BATCH_SIZE)):
        raise ODEBFContractError("P0 B10 request order differs")
    case_ids = [item.get("case_id") for item in requests]
    hashes = [item.get("request_sha256") for item in requests]
    if len(set(case_ids)) != BATCH_SIZE or len(set(hashes)) != BATCH_SIZE:
        raise ODEBFContractError("P0 B10 seal requests are not distinct")
    if set(case_ids).intersection(EXPLICIT_EXCLUSIONS):
        raise ODEBFContractError("P0 B10 seal includes an explicit exclusion")
    expected_order = canonical_hash(requests)
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_ALIASES):
        raise ODEBFContractError("P0 B10 seal model aliases differ")
    if any(
        model.get("request_order_sha256") != expected_order
        or model.get("same_joint_batch") is not True
        for model in models.values()
    ):
        raise ODEBFContractError("P0 B10 model request orders differ")
    return payload


def load_sealed_joint_requests(
    dataset_path: str | Path,
    seal: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    value = verify_p0_b10_seal(seal)
    dataset = Path(dataset_path).resolve(strict=True)
    source = value["source"]
    if dataset.stat().st_size != source["size_bytes"] or _file_sha256(dataset) != source["sha256"]:
        raise ODEBFContractError("P0 B10 dataset identity differs")
    requests: list[dict[str, Any]] = []
    for locked in value["requests"]:
        projected = load_projected_request(
            dataset,
            case_id=int(locked["case_id"]),
            expected_request_hash=str(locked["request_sha256"]),
        )
        requests.append(
            {
                "case_id": projected["case_id"],
                "prompt": projected["prompt"],
                "relation_id": projected["relation_id"],
                "subject": projected["subject"],
                "target_new": projected["target_new"],
                "target_true": projected["target_old"],
                "request_sha256": locked["request_sha256"],
            }
        )
    return tuple(requests)
