"""Outcome-free deterministic P0/P1/anchor seal-candidate construction."""

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
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .barriers import assert_anchor_disjoint
from .contracts import ODEAllocContractError, canonical_hash, canonical_json


P0_SALT = "odealloc-s04-p0-v1"
P1_SALT = "odealloc-s04-p1-v1"
ANCHOR_SALT = "odealloc-s04-anchor-v1"
EXPLICIT_EXCLUSIONS = frozenset(
    {2022, 12498, 20964, 768, 17503, 1534, 14652, 9774}
)
_CASE_ID = re.compile(rb'"case_id"\s*:\s*"?(\d+)"?')
_CASE_ID_SCALAR = re.compile(rb'"[A-Za-z0-9_]*case_id"\s*:\s*"?(\d+)"?')
_CASE_IDS_ARRAY = re.compile(
    rb'"[A-Za-z0-9_]*case_ids"\s*:\s*\[([^\]]*)\]', re.DOTALL
)
_DIGITS = re.compile(rb'"?(\d+)"?')
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    case_id: int
    request_hash: str
    duplicate_key: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_top_level_objects(path: Path) -> Iterator[bytes]:
    """Yield raw objects from a top-level array without decoding row values."""

    started = False
    finished = False
    depth = 0
    in_string = False
    escaped = False
    current: bytearray | None = None
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            for byte in chunk:
                if not started:
                    if chr(byte).isspace():
                        continue
                    if byte != ord("["):
                        raise ODEAllocContractError("selection source is not a JSON array")
                    started = True
                    continue
                if finished:
                    if not chr(byte).isspace():
                        raise ODEAllocContractError("selection source has trailing bytes")
                    continue
                if current is None:
                    if chr(byte).isspace() or byte == ord(","):
                        continue
                    if byte == ord("]"):
                        finished = True
                        continue
                    if byte != ord("{"):
                        raise ODEAllocContractError("selection row is not an object")
                    current = bytearray((byte,))
                    depth = 1
                    in_string = False
                    escaped = False
                    continue
                current.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue
                if byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        yield bytes(current)
                        current = None
    if not started or not finished or current is not None or in_string:
        raise ODEAllocContractError("selection source ended incompletely")


def _extract_object_after_key(blob: bytes, key: bytes) -> bytes:
    marker = b'"' + key + b'"'
    positions = [match.start() for match in re.finditer(re.escape(marker), blob)]
    if len(positions) != 1:
        raise ODEAllocContractError("request projection key is absent or repeated")
    index = positions[0] + len(marker)
    while index < len(blob) and chr(blob[index]).isspace():
        index += 1
    if index >= len(blob) or blob[index] != ord(":"):
        raise ODEAllocContractError("request projection key has no value")
    index += 1
    while index < len(blob) and chr(blob[index]).isspace():
        index += 1
    if index >= len(blob) or blob[index] != ord("{"):
        raise ODEAllocContractError("request projection value is not an object")
    start = index
    depth = 0
    in_string = False
    escaped = False
    while index < len(blob):
        byte = blob[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            depth += 1
        elif byte == ord("}"):
            depth -= 1
            if depth == 0:
                return blob[start : index + 1]
        index += 1
    raise ODEAllocContractError("request projection object is incomplete")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ODEAllocContractError(f"request {name} is empty")
    return value


def _normalize_duplicate(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def project_request_identity(blob: bytes) -> RequestIdentity:
    """Deserialize only requested_rewrite; other row members remain raw bytes."""

    matches = _CASE_ID.findall(blob)
    if len(matches) != 1:
        raise ODEAllocContractError("selection row case identity is ambiguous")
    case_id = int(matches[0])
    try:
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEAllocContractError("requested rewrite is invalid JSON") from exc
    required = {"prompt", "relation_id", "subject", "target_new", "target_true"}
    if not isinstance(rewrite, Mapping) or set(rewrite) != required:
        raise ODEAllocContractError("requested rewrite schema differs")
    target_new = rewrite["target_new"]
    target_true = rewrite["target_true"]
    if not isinstance(target_new, Mapping) or not isinstance(target_true, Mapping):
        raise ODEAllocContractError("requested rewrite targets are invalid")
    payload = {
        "case_id": case_id,
        "prompt": _text(rewrite["prompt"], "prompt"),
        "relation_id": _text(rewrite["relation_id"], "relation"),
        "subject": _text(rewrite["subject"], "subject"),
        "target_new": _text(target_new.get("str"), "new target"),
        "target_old": _text(target_true.get("str"), "old target"),
    }
    duplicate_key = canonical_hash(
        {
            "subject": _normalize_duplicate(payload["subject"]),
            "template": _normalize_duplicate(payload["prompt"]),
        }
    )
    return RequestIdentity(
        case_id=case_id,
        request_hash=canonical_hash(payload),
        duplicate_key=duplicate_key,
    )


def load_request_identities(dataset_path: str | Path) -> tuple[RequestIdentity, ...]:
    source = Path(dataset_path).resolve(strict=True)
    identities = tuple(project_request_identity(blob) for blob in _iter_top_level_objects(source))
    case_ids = tuple(item.case_id for item in identities)
    if not identities or len(case_ids) != len(set(case_ids)):
        raise ODEAllocContractError("selection dataset case identities repeat")
    return identities


def load_projected_request(
    dataset_path: str | Path,
    *,
    case_id: int,
    expected_request_hash: str,
) -> dict[str, Any]:
    """Load only one approved rewrite surface; evaluation fields stay undecoded."""

    if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 0:
        raise ODEAllocContractError("approved P0 case identity is invalid")
    if not isinstance(expected_request_hash, str) or len(expected_request_hash) != 64:
        raise ODEAllocContractError("approved P0 request hash is invalid")
    source = Path(dataset_path).resolve(strict=True)
    selected: bytes | None = None
    for blob in _iter_top_level_objects(source):
        matches = _CASE_ID.findall(blob)
        if len(matches) == 1 and int(matches[0]) == case_id:
            if selected is not None:
                raise ODEAllocContractError("approved P0 case repeats")
            selected = blob
    if selected is None:
        raise ODEAllocContractError("approved P0 case is absent")
    identity = project_request_identity(selected)
    if identity.request_hash != expected_request_hash:
        raise ODEAllocContractError("approved P0 request hash differs")
    try:
        rewrite = json.loads(_extract_object_after_key(selected, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEAllocContractError("approved P0 rewrite is invalid JSON") from exc
    return {
        "case_id": case_id,
        "prompt": _text(rewrite["prompt"], "prompt"),
        "relation_id": _text(rewrite["relation_id"], "relation"),
        "subject": _text(rewrite["subject"], "subject"),
        "target_new": _text(rewrite["target_new"]["str"], "new target"),
        "target_old": _text(rewrite["target_true"]["str"], "old target"),
    }


def _safe_tracked_paths(repo: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.json", "*.py"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths: list[Path] = []
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
        ):
            continue
        candidate = (repo / relative).resolve(strict=True)
        candidate.relative_to(repo)
        paths.append(candidate)
    return tuple(paths)


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
        except (ValueError, TypeError):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, (str, int)) and str(item).isdigit():
                    found.add(int(item))
    return found


def scan_tracked_prior_case_ids(repo_root: str | Path) -> tuple[set[int], str, int]:
    """Lexically extract identity fields without decoding tracked result values."""

    repo = Path(repo_root).resolve(strict=True)
    found: set[int] = set()
    source_records: list[dict[str, Any]] = []
    for path in _safe_tracked_paths(repo):
        relative = str(path.relative_to(repo))
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


def _rank(identity: RequestIdentity, salt: str) -> tuple[str, str, int]:
    digest = hashlib.sha256(
        salt.encode("utf-8") + b"\0" + identity.request_hash.encode("ascii")
    ).hexdigest()
    return digest, identity.request_hash, identity.case_id


def _choose(
    identities: Iterable[RequestIdentity],
    *,
    count: int,
    salt: str,
    used_cases: set[int],
    used_duplicates: set[str],
) -> tuple[RequestIdentity, ...]:
    chosen: list[RequestIdentity] = []
    for identity in sorted(identities, key=lambda item: _rank(item, salt)):
        if identity.case_id in used_cases or identity.duplicate_key in used_duplicates:
            continue
        chosen.append(identity)
        used_cases.add(identity.case_id)
        used_duplicates.add(identity.duplicate_key)
        if len(chosen) == count:
            return tuple(chosen)
    raise ODEAllocContractError("insufficient disjoint outcome-free candidates")


def build_seal_candidate(
    dataset_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    source = Path(dataset_path).resolve(strict=True)
    repo = Path(repo_root).resolve(strict=True)
    identities = load_request_identities(source)
    prior, prior_sources_digest, prior_source_count = scan_tracked_prior_case_ids(repo)
    duplicate_counts = collections.Counter(item.duplicate_key for item in identities)
    duplicate_keys = {
        key for key, count in duplicate_counts.items() if count > 1
    }
    duplicate_case_ids = sorted(
        item.case_id for item in identities if item.duplicate_key in duplicate_keys
    )
    eligible_identities = tuple(
        item for item in identities if item.duplicate_key not in duplicate_keys
    )
    used_cases = set(EXPLICIT_EXCLUSIONS).union(prior)
    used_duplicates: set[str] = set()
    p0 = _choose(
        eligible_identities,
        count=1,
        salt=P0_SALT,
        used_cases=used_cases,
        used_duplicates=used_duplicates,
    )
    p1 = _choose(
        eligible_identities,
        count=4,
        salt=P1_SALT,
        used_cases=used_cases,
        used_duplicates=used_duplicates,
    )
    anchors = _choose(
        eligible_identities,
        count=16,
        salt=ANCHOR_SALT,
        used_cases=used_cases,
        used_duplicates=used_duplicates,
    )
    edit_hashes = [item.request_hash for item in (*p0, *p1)]
    anchor_hashes = [item.request_hash for item in anchors]
    assert_anchor_disjoint(edit_hashes, anchor_hashes)

    def records(items: Sequence[RequestIdentity]) -> list[dict[str, Any]]:
        return [
            {"case_id": item.case_id, "request_hash": item.request_hash}
            for item in items
        ]

    payload: dict[str, Any] = {
        "schema_version": "ode-alloc-s04-split-anchor-seal-candidate/v1",
        "status": "pending-gh-approval-no-scientific-execution",
        "source": {
            "relative_contract": "EasyEdit/data/counterfact/counterfact.json",
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
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
                "target_old"
            ],
            "target_old_source": "requested_rewrite.target_true.str"
        },
        "salts": {"p0": P0_SALT, "p1": P1_SALT, "anchor": ANCHOR_SALT},
        "explicit_exclusions": sorted(EXPLICIT_EXCLUSIONS),
        "tracked_prior_collision_scan": {
            "case_id_count": len(prior),
            "case_ids_sha256": canonical_hash(sorted(prior)),
            "source_count": prior_source_count,
            "sources_digest": prior_sources_digest,
            "session03_paths_read": 0,
        },
        "duplicate_policy": {
            "rule": "exclude-entire-nfkc-casefold-subject-plus-template-duplicate-groups",
            "excluded_case_id_count": len(duplicate_case_ids),
            "excluded_case_ids_sha256": canonical_hash(duplicate_case_ids)
        },
        "p0_identity": records(p0),
        "p1_order": records(p1),
        "pretrained_anchor": records(anchors),
        "pretrained_anchor_contract": "theta0-teacher-on-target-old-no-edit-request-use",
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def write_canonical_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def load_and_verify_seal_candidate(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    observed = value.pop("root_digest", None)
    expected = canonical_hash(value)
    value["root_digest"] = observed
    if observed != expected:
        raise ODEAllocContractError("seal-candidate root digest differs")
    pools = (
        value["p0_identity"],
        value["p1_order"],
        value["pretrained_anchor"],
    )
    if tuple(len(pool) for pool in pools) != (1, 4, 16):
        raise ODEAllocContractError("seal-candidate pool sizes differ")
    case_ids = [item["case_id"] for pool in pools for item in pool]
    request_hashes = [item["request_hash"] for pool in pools for item in pool]
    if len(case_ids) != len(set(case_ids)) or len(request_hashes) != len(set(request_hashes)):
        raise ODEAllocContractError("seal-candidate pools are not disjoint")
    if set(case_ids).intersection(EXPLICIT_EXCLUSIONS):
        raise ODEAllocContractError("seal-candidate includes an explicit exclusion")
    return value


def assert_seal_source_current(
    seal_candidate: Mapping[str, Any],
    dataset_path: str | Path,
) -> None:
    source = Path(dataset_path).resolve(strict=True)
    locked = seal_candidate.get("source")
    if not isinstance(locked, Mapping):
        raise ODEAllocContractError("seal-candidate source identity is absent")
    if (
        source.stat().st_size != locked.get("size_bytes")
        or _sha256_file(source) != locked.get("sha256")
    ):
        raise ODEAllocContractError("seal-candidate dataset identity changed")
