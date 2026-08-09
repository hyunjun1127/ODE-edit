"""Model-free fresh CounterFact B10 selection for P1R14.

The selector reads only immutable Git blobs and the pinned CounterFact byte
stream.  Its dependency closure excludes tensor frameworks, the runtime, an
evaluator, and every result root.  A normalized historical-exclusion seal is
built first; the fresh B10 is then derived only from that rooted exclusion
artifact.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .request_digest import ordered_request_digest_v1


R14_SELECTION_BASE = "e6facd2d5dfae12d3c094b51981ad99951174109"
R14_INSTRUCTION_ID = "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1"
R14_SELECTION_SALT = "ODEEDIT-S05-P1R14-FRESH-CF-B10-V1"
R14_P_ANCHOR_SALT = "ODEEDIT-S05-P1R14-FUNCTIONAL-P-ANCHOR-M60-V1"
R14_P_ANCHOR_COUNT = 60
R14_CONTEXT_CONTRACT = {
    "policy": "PINNED_ALPHAEDIT_CONTEXTS_GENERATE_TWICE_V1",
    "seed": 41,
    "group_sizes": [1, 5],
    "generation_prefixes": ["The", "Therefore", "Because", "I", "You"],
    "generation_length": 10,
    "generation_count": 5,
    "source_path": "easyeditor/models/alphaedit/AlphaEdit_main.py",
    "source_sha256": "a3a459abc4e05f4b3ccaf424a8950245c491686e943fdcaa52b296343e7458b1",
}
R14_EVALUATOR_CONTRACT = {
    "policy": "PINNED_COUNTERFACT_PRIMARY_NO_GENERATION_V1",
    "evaluator_source_sha256": "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
    "aggregator_source_sha256": "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0",
    "generation_call_count": 0,
}
R14_EXCLUSION_SCHEMA = "ode-edit-s05-p1r14-historical-exclusion/v1"
R14_FRESH_SEAL_SCHEMA = "ode-edit-s05-p1r14-fresh-cf-b10/v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX64_BYTES = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_CASE_KEY = re.compile(
    rb"(?i)(?:case_ids?|excluded(?:_case)?_ids?|explicit_exclusions?)"
    rb"[^\n\r]{0,4096}"
)
_DIGITS = re.compile(rb"(?<![0-9])([0-9]{1,9})(?![0-9])")
_REQUEST_LINE = re.compile(
    rb"(?i)(?:request[^\n\r]{0,80}(?:sha|hash)|(?:sha|hash)[^\n\r]{0,80}request)"
    rb"[^\n\r]*"
)
_SPACE = re.compile(r"\s+")
_CASE_ID = re.compile(rb'"case_id"\s*:\s*"?(\d+)"?')


@dataclass(frozen=True, slots=True)
class R14RequestIdentity:
    case_id: int
    request_sha256: str
    subject_template_sha256: str


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def _canonical_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ODEBFContractError(f"R14 CounterFact {label} is empty")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _iter_top_level_objects(path: Path) -> Iterator[bytes]:
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
                        raise ODEBFContractError("R14 selection source is not an array")
                    started = True
                    continue
                if finished:
                    if not chr(byte).isspace():
                        raise ODEBFContractError("R14 selection source has trailing bytes")
                    continue
                if current is None:
                    if chr(byte).isspace() or byte == ord(","):
                        continue
                    if byte == ord("]"):
                        finished = True
                        continue
                    if byte != ord("{"):
                        raise ODEBFContractError("R14 selection row is not an object")
                    current = bytearray((byte,))
                    depth = 1
                    continue
                current.append(byte)
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
                        yield bytes(current)
                        current = None
    if not started or not finished or current is not None or in_string:
        raise ODEBFContractError("R14 selection source ended incompletely")


def _extract_object_after_key(blob: bytes, key: bytes) -> bytes:
    marker = b'"' + key + b'"'
    positions = [item.start() for item in re.finditer(re.escape(marker), blob)]
    if len(positions) != 1:
        raise ODEBFContractError("R14 requested rewrite is absent or repeated")
    cursor = positions[0] + len(marker)
    while cursor < len(blob) and chr(blob[cursor]).isspace():
        cursor += 1
    if cursor >= len(blob) or blob[cursor] != ord(":"):
        raise ODEBFContractError("R14 requested rewrite lacks a value")
    cursor += 1
    while cursor < len(blob) and chr(blob[cursor]).isspace():
        cursor += 1
    if cursor >= len(blob) or blob[cursor] != ord("{"):
        raise ODEBFContractError("R14 requested rewrite is not an object")
    start = cursor
    depth = 0
    in_string = False
    escaped = False
    while cursor < len(blob):
        byte = blob[cursor]
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
                return blob[start : cursor + 1]
        cursor += 1
    raise ODEBFContractError("R14 requested rewrite ended incompletely")


def project_r14_request_identity(blob: bytes) -> R14RequestIdentity:
    matches = _CASE_ID.findall(blob)
    if len(matches) != 1:
        raise ODEBFContractError("R14 CounterFact case identity is ambiguous")
    try:
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("R14 requested rewrite is invalid") from exc
    required = {"prompt", "relation_id", "subject", "target_new", "target_true"}
    if not isinstance(rewrite, Mapping) or set(rewrite) != required:
        raise ODEBFContractError("R14 requested rewrite schema differs")
    if not isinstance(rewrite["target_new"], Mapping) or not isinstance(
        rewrite["target_true"], Mapping
    ):
        raise ODEBFContractError("R14 target schema differs")
    case_id = int(matches[0])
    prompt = _canonical_text(rewrite["prompt"], "prompt")
    subject = _canonical_text(rewrite["subject"], "subject")
    payload = {
        "case_id": case_id,
        "prompt": prompt,
        "relation_id": _canonical_text(rewrite["relation_id"], "relation"),
        "subject": subject,
        "target_new": _canonical_text(rewrite["target_new"].get("str"), "new target"),
        "target_old": _canonical_text(rewrite["target_true"].get("str"), "old target"),
    }
    group = canonical_hash(
        {"subject": _normalize(subject), "template": _normalize(prompt)}
    )
    return R14RequestIdentity(case_id, canonical_hash(payload), group)


def load_r14_identities(dataset_path: str | Path) -> tuple[R14RequestIdentity, ...]:
    path = Path(dataset_path).resolve(strict=True)
    values = tuple(project_r14_request_identity(row) for row in _iter_top_level_objects(path))
    if (
        not values
        or len({item.case_id for item in values}) != len(values)
        or len({item.request_sha256 for item in values}) != len(values)
    ):
        raise ODEBFContractError("R14 CounterFact identity population differs")
    return values


def _controller_request(blob: bytes) -> dict[str, Any]:
    identity = project_r14_request_identity(blob)
    try:
        rewrite = json.loads(_extract_object_after_key(blob, b"requested_rewrite"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFContractError("R14 controller rewrite is malformed") from exc
    if not isinstance(rewrite, Mapping) or not isinstance(
        rewrite.get("target_new"), Mapping
    ):
        raise ODEBFContractError("R14 controller rewrite differs")
    return {
        "case_id": identity.case_id,
        "request_sha256": identity.request_sha256,
        "prompt": _canonical_text(rewrite.get("prompt"), "prompt"),
        "subject": _canonical_text(rewrite.get("subject"), "subject"),
        "target_new": _canonical_text(
            rewrite["target_new"].get("str"), "target-new"
        ),
    }


def load_r14_controller_requests(
    dataset_path: str | Path,
    seal: Mapping[str, Any],
    *,
    population: str = "requests",
) -> tuple[dict[str, Any], ...]:
    """Load only rewrite fields in the exact sealed controller order."""

    if population not in ("requests", "functional_p_anchors"):
        raise ODEBFContractError("R14 controller population differs")
    expected = seal.get(population)
    if not isinstance(expected, list):
        raise ODEBFContractError("R14 sealed controller population is absent")
    wanted = {
        int(item["case_id"]): (ordinal, str(item["request_sha256"]))
        for ordinal, item in enumerate(expected)
    }
    if len(wanted) != len(expected):
        raise ODEBFContractError("R14 sealed controller population repeats")
    observed: list[dict[str, Any] | None] = [None] * len(expected)
    for blob in _iter_top_level_objects(Path(dataset_path).resolve(strict=True)):
        matches = _CASE_ID.findall(blob)
        if len(matches) != 1:
            raise ODEBFContractError("R14 controller row identity differs")
        case_id = int(matches[0])
        if case_id not in wanted:
            continue
        ordinal, request_sha256 = wanted[case_id]
        request = _controller_request(blob)
        if (
            request["request_sha256"] != request_sha256
            or observed[ordinal] is not None
        ):
            raise ODEBFContractError("R14 controller request identity differs")
        observed[ordinal] = request
    if any(item is None for item in observed):
        raise ODEBFContractError("R14 controller request is missing")
    return tuple(item for item in observed if item is not None)


def _git_tree_blobs(repo: Path, base_commit: str) -> tuple[dict[str, Any], ...]:
    observed = subprocess.run(
        ["git", "rev-parse", base_commit], cwd=repo, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != base_commit:
        raise ODEBFContractError("R14 selection base commit differs")
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--long", base_commit],
        cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    records: list[dict[str, Any]] = []
    for raw in result.split(b"\0"):
        if not raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        mode, kind, oid, size = metadata.decode("ascii").split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ODEBFContractError("R14 selection tree member differs")
        records.append(
            {
                "path": raw_path.decode("utf-8"),
                "mode": mode,
                "git_blob_oid": oid,
                "size": int(size),
            }
        )
    return tuple(sorted(records, key=lambda item: str(item["path"])))


def _git_blob(repo: Path, oid: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", oid], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _json_identities(value: Any) -> tuple[set[int], set[str]]:
    cases: set[int] = set()
    requests: set[str] = set()

    def walk(item: Any, key: str = "") -> None:
        lowered = key.casefold()
        if lowered.endswith("case_id"):
            candidates = (item,)
        elif lowered.endswith("case_ids") and isinstance(item, list):
            candidates = tuple(item)
        else:
            candidates = ()
        for candidate in candidates:
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                cases.add(candidate)
            elif isinstance(candidate, str) and candidate.isdigit():
                cases.add(int(candidate))
        if "request" in lowered and ("sha" in lowered or "hash" in lowered):
            candidates = tuple(item) if isinstance(item, list) else (item,)
            requests.update(
                candidate for candidate in candidates
                if isinstance(candidate, str) and _HEX64.fullmatch(candidate)
            )
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                walk(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                walk(child, key)

    walk(value)
    return cases, requests


def _python_identities(text: str) -> tuple[set[int], set[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ODEBFContractError("R14 tracked Python source is invalid") from exc
    cases: set[int] = set()
    requests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = tuple(
            target.id.casefold() for target in targets if isinstance(target, ast.Name)
        )
        try:
            literal = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        values = tuple(literal) if isinstance(literal, (list, tuple, set, frozenset)) else (literal,)
        if any("case" in name and ("id" in name or "exclusion" in name) for name in names):
            cases.update(
                item for item in values
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0
            )
        if any("request" in name and ("sha" in name or "hash" in name) for name in names):
            requests.update(
                item for item in values
                if isinstance(item, str) and _HEX64.fullmatch(item)
            )
    return cases, requests


def _text_identities(blob: bytes) -> tuple[set[int], set[str]]:
    cases: set[int] = set()
    requests: set[str] = set()
    for match in _CASE_KEY.finditer(blob):
        cases.update(int(item) for item in _DIGITS.findall(match.group(0)))
    for match in _REQUEST_LINE.finditer(blob):
        requests.update(item.decode("ascii") for item in _HEX64_BYTES.findall(match.group(0)))
    return cases, requests


def build_r14_historical_exclusion(
    repo_root: str | Path,
    *,
    base_commit: str = R14_SELECTION_BASE,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    base_tree = subprocess.run(
        ["git", "rev-parse", f"{base_commit}^{{tree}}"], cwd=repo, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    cases: set[int] = set()
    requests: set[str] = set()
    sources: list[dict[str, Any]] = []
    for metadata in _git_tree_blobs(repo, base_commit):
        blob = _git_blob(repo, str(metadata["git_blob_oid"]))
        if len(blob) != int(metadata["size"]):
            raise ODEBFContractError("R14 tracked source size differs")
        local_cases: set[int] = set()
        local_requests: set[str] = set()
        status = "NO_IDENTITY"
        suffix = Path(str(metadata["path"])).suffix.casefold()
        if b"\0" in blob:
            status = "BINARY_NO_IDENTITY_LEXER"
            lowered = str(metadata["path"]).casefold()
            if any(token in lowered for token in ("seal", "receipt", "report", "manifest", "selection")):
                raise ODEBFContractError("R14 identity-bearing source is binary")
        else:
            text = blob.decode("utf-8", errors="strict")
            if suffix == ".json":
                try:
                    local_cases, local_requests = _json_identities(json.loads(text))
                    status = "JSON_SCANNED"
                except json.JSONDecodeError as exc:
                    lowered = str(metadata["path"]).casefold()
                    if any(
                        token in lowered
                        for token in ("seal", "receipt", "report", "manifest", "selection")
                    ):
                        raise ODEBFContractError(
                            "R14 identity-bearing JSON source is malformed"
                        ) from exc
                    local_cases, local_requests = _text_identities(blob)
                    status = "TEXT_SCANNED_INVALID_JSON"
            elif suffix == ".py":
                local_cases, local_requests = _python_identities(text)
                text_cases, text_requests = _text_identities(blob)
                local_cases.update(text_cases)
                local_requests.update(text_requests)
                status = "PYTHON_AST_AND_TEXT_SCANNED"
            else:
                local_cases, local_requests = _text_identities(blob)
                status = "TEXT_SCANNED"
        cases.update(local_cases)
        requests.update(local_requests)
        sources.append(
            {
                **metadata,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "scan_status": status,
                "case_count": len(local_cases),
                "case_digest": canonical_hash(sorted(local_cases)),
                "request_count": len(local_requests),
                "request_digest": canonical_hash(sorted(local_requests)),
            }
        )
    payload = {
        "schema_version": R14_EXCLUSION_SCHEMA,
        "instruction_id": R14_INSTRUCTION_ID,
        "selection_base_commit": base_commit,
        "selection_base_tree": base_tree,
        "source_blob_count": len(sources),
        "source_manifest_root": canonical_hash(sources),
        "source_manifest": sources,
        "prior_case_ids": sorted(cases),
        "prior_case_id_count": len(cases),
        "prior_case_ids_sha256": canonical_hash(sorted(cases)),
        "prior_request_sha256": sorted(requests),
        "prior_request_sha256_count": len(requests),
        "prior_request_sha256_digest": canonical_hash(sorted(requests)),
        "local_result_payload_open_count": 0,
        "model_import_or_access_count": 0,
        "evaluator_import_or_access_count": 0,
        "heldout_field_open_count": 0,
        "terminal_or_outcome_open_count": 0,
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def verify_r14_historical_exclusion(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("R14 historical exclusion root differs")
    if (
        payload.get("schema_version") != R14_EXCLUSION_SCHEMA
        or payload.get("instruction_id") != R14_INSTRUCTION_ID
        or payload.get("selection_base_commit") != R14_SELECTION_BASE
        or payload.get("source_blob_count") != len(payload.get("source_manifest", ()))
        or payload.get("source_manifest_root") != canonical_hash(payload.get("source_manifest"))
        or payload.get("prior_case_id_count") != len(payload.get("prior_case_ids", ()))
        or payload.get("prior_request_sha256_count") != len(payload.get("prior_request_sha256", ()))
        or any(payload.get(name) != 0 for name in (
            "local_result_payload_open_count", "model_import_or_access_count",
            "evaluator_import_or_access_count", "heldout_field_open_count",
            "terminal_or_outcome_open_count",
        ))
    ):
        raise ODEBFContractError("R14 historical exclusion contract differs")
    payload["root_digest"] = root
    return payload


def build_r14_fresh_seal(
    dataset_path: str | Path,
    exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    history = verify_r14_historical_exclusion(exclusion)
    dataset = Path(dataset_path).resolve(strict=True)
    identities = load_r14_identities(dataset)
    prior_cases = {int(item) for item in history["prior_case_ids"]}
    prior_requests = {str(item) for item in history["prior_request_sha256"]}
    group_counts = Counter(item.subject_template_sha256 for item in identities)
    duplicate_groups = {key for key, count in group_counts.items() if count > 1}
    historical_groups = {
        item.subject_template_sha256 for item in identities
        if item.case_id in prior_cases or item.request_sha256 in prior_requests
    }
    eligible = tuple(
        item for item in identities
        if item.case_id not in prior_cases
        and item.request_sha256 not in prior_requests
        and item.subject_template_sha256 not in historical_groups
        and item.subject_template_sha256 not in duplicate_groups
    )
    ranked = tuple(
        sorted(
            eligible,
            key=lambda item: (
                hashlib.sha256(
                    f"{R14_SELECTION_SALT}|{item.request_sha256}".encode("utf-8")
                ).hexdigest(),
                item.request_sha256,
                item.case_id,
            ),
        )
    )
    if len(ranked) < BATCH_SIZE:
        raise ODEBFContractError("R14 fresh eligible population is too small")
    selected = ranked[:BATCH_SIZE]
    selected_requests = {item.request_sha256 for item in selected}
    p_ranked = tuple(
        sorted(
            (
                item
                for item in eligible
                if item.request_sha256 not in selected_requests
            ),
            key=lambda item: (
                hashlib.sha256(
                    f"{R14_P_ANCHOR_SALT}|{item.request_sha256}".encode("utf-8")
                ).hexdigest(),
                item.request_sha256,
                item.case_id,
            ),
        )
    )
    if len(p_ranked) < R14_P_ANCHOR_COUNT:
        raise ODEBFContractError("R14 functional P anchor population is too small")
    p_selected = p_ranked[:R14_P_ANCHOR_COUNT]
    requests = [
        {
            "ordinal": ordinal,
            "case_id": item.case_id,
            "request_sha256": item.request_sha256,
            "subject_template_sha256": item.subject_template_sha256,
            "rank_sha256": hashlib.sha256(
                f"{R14_SELECTION_SALT}|{item.request_sha256}".encode("utf-8")
            ).hexdigest(),
        }
        for ordinal, item in enumerate(selected)
    ]
    p_anchors = [
        {
            "ordinal": ordinal,
            "case_id": item.case_id,
            "request_sha256": item.request_sha256,
            "subject_template_sha256": item.subject_template_sha256,
            "rank_sha256": hashlib.sha256(
                f"{R14_P_ANCHOR_SALT}|{item.request_sha256}".encode("utf-8")
            ).hexdigest(),
        }
        for ordinal, item in enumerate(p_selected)
    ]
    payload = {
        "schema_version": R14_FRESH_SEAL_SCHEMA,
        "instruction_id": R14_INSTRUCTION_ID,
        "status": "SEALED_BEFORE_MODEL_OR_OUTCOME_ACCESS",
        "selection_base_commit": R14_SELECTION_BASE,
        "historical_exclusion_root": history["root_digest"],
        "benchmark": "counterfact",
        "salt": R14_SELECTION_SALT,
        "rank_formula": "sha256(utf8(salt + '|' + request_sha256))",
        "duplicate_group_definition": "NFKC_WS_CASEFOLD_SUBJECT_PLUS_TEMPLATE",
        "edit_batch_size": BATCH_SIZE,
        "source": {
            "relative_contract": "EasyEdit/data/counterfact/counterfact.json",
            "sha256": _sha256_file(dataset),
            "size_bytes": dataset.stat().st_size,
            "row_count": len(identities),
        },
        "context_contract": R14_CONTEXT_CONTRACT,
        "context_contract_sha256": canonical_hash(R14_CONTEXT_CONTRACT),
        "evaluator_contract": R14_EVALUATOR_CONTRACT,
        "evaluator_contract_sha256": canonical_hash(R14_EVALUATOR_CONTRACT),
        "population_count": len(identities),
        "prior_case_exclusion_count": len(prior_cases),
        "prior_request_exclusion_count": len(prior_requests),
        "historical_group_exclusion_count": len(historical_groups),
        "historical_group_exclusion_digest": canonical_hash(sorted(historical_groups)),
        "duplicate_group_exclusion_count": len(duplicate_groups),
        "duplicate_group_exclusion_digest": canonical_hash(sorted(duplicate_groups)),
        "eligible_count": len(eligible),
        "eligible_digest": canonical_hash(
            [[item.case_id, item.request_sha256, item.subject_template_sha256] for item in eligible]
        ),
        "requests": requests,
        "functional_p_anchor_salt": R14_P_ANCHOR_SALT,
        "functional_p_anchor_count": R14_P_ANCHOR_COUNT,
        "functional_p_anchors": p_anchors,
        "functional_p_anchor_order_digest": canonical_hash(
            [item.request_sha256 for item in p_selected]
        ),
        "edit_anchor_request_intersection_count": 0,
        "batch_ordered_request_digest_v1": ordered_request_digest_v1(
            [item.request_sha256 for item in selected]
        ),
        "prior_case_intersection_count": 0,
        "prior_request_intersection_count": 0,
        "prior_group_intersection_count": 0,
        "duplicate_group_intersection_count": 0,
        "model_import_or_access_count": 0,
        "evaluator_import_or_access_count": 0,
        "heldout_field_open_count": 0,
        "terminal_or_outcome_open_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["root_digest"] = canonical_hash(payload)
    return payload


def verify_r14_fresh_seal(
    value: Mapping[str, Any],
    *,
    exclusion_root: str,
) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("R14 fresh seal root differs")
    requests = payload.get("requests")
    anchors = payload.get("functional_p_anchors")
    if (
        payload.get("schema_version") != R14_FRESH_SEAL_SCHEMA
        or payload.get("instruction_id") != R14_INSTRUCTION_ID
        or payload.get("status") != "SEALED_BEFORE_MODEL_OR_OUTCOME_ACCESS"
        or payload.get("selection_base_commit") != R14_SELECTION_BASE
        or payload.get("historical_exclusion_root") != exclusion_root
        or payload.get("salt") != R14_SELECTION_SALT
        or payload.get("context_contract") != R14_CONTEXT_CONTRACT
        or payload.get("context_contract_sha256")
        != canonical_hash(R14_CONTEXT_CONTRACT)
        or payload.get("evaluator_contract") != R14_EVALUATOR_CONTRACT
        or payload.get("evaluator_contract_sha256")
        != canonical_hash(R14_EVALUATOR_CONTRACT)
        or not isinstance(requests, list)
        or len(requests) != BATCH_SIZE
        or [item.get("ordinal") for item in requests] != list(range(BATCH_SIZE))
        or len({item.get("case_id") for item in requests}) != BATCH_SIZE
        or len({item.get("request_sha256") for item in requests}) != BATCH_SIZE
        or len({item.get("subject_template_sha256") for item in requests}) != BATCH_SIZE
        or payload.get("functional_p_anchor_salt") != R14_P_ANCHOR_SALT
        or payload.get("functional_p_anchor_count") != R14_P_ANCHOR_COUNT
        or not isinstance(anchors, list)
        or len(anchors) != R14_P_ANCHOR_COUNT
        or [item.get("ordinal") for item in anchors]
        != list(range(R14_P_ANCHOR_COUNT))
        or len({item.get("case_id") for item in anchors}) != R14_P_ANCHOR_COUNT
        or len({item.get("request_sha256") for item in anchors})
        != R14_P_ANCHOR_COUNT
        or bool(
            {item.get("request_sha256") for item in requests}
            & {item.get("request_sha256") for item in anchors}
        )
        or payload.get("edit_anchor_request_intersection_count") != 0
        or payload.get("functional_p_anchor_order_digest")
        != canonical_hash([str(item["request_sha256"]) for item in anchors])
        or payload.get("batch_ordered_request_digest_v1")
        != ordered_request_digest_v1([str(item["request_sha256"]) for item in requests])
        or any(payload.get(name) != 0 for name in (
            "prior_case_intersection_count", "prior_request_intersection_count",
            "prior_group_intersection_count", "duplicate_group_intersection_count",
            "model_import_or_access_count", "evaluator_import_or_access_count",
            "heldout_field_open_count", "terminal_or_outcome_open_count",
        ))
        or payload.get("scientific_promotion_authorized") is not False
    ):
        raise ODEBFContractError("R14 fresh seal contract differs")
    expected_ranks = [
        hashlib.sha256(
            f"{R14_SELECTION_SALT}|{item['request_sha256']}".encode("utf-8")
        ).hexdigest()
        for item in requests
    ]
    if [item.get("rank_sha256") for item in requests] != expected_ranks or expected_ranks != sorted(expected_ranks):
        raise ODEBFContractError("R14 fresh seal rank order differs")
    expected_anchor_ranks = [
        hashlib.sha256(
            f"{R14_P_ANCHOR_SALT}|{item['request_sha256']}".encode("utf-8")
        ).hexdigest()
        for item in anchors
    ]
    if (
        [item.get("rank_sha256") for item in anchors] != expected_anchor_ranks
        or expected_anchor_ranks != sorted(expected_anchor_ranks)
    ):
        raise ODEBFContractError("R14 functional P anchor rank order differs")
    payload["root_digest"] = root
    return payload


__all__ = [
    "R14_EXCLUSION_SCHEMA", "R14_FRESH_SEAL_SCHEMA", "R14_INSTRUCTION_ID",
    "R14_CONTEXT_CONTRACT", "R14_EVALUATOR_CONTRACT",
    "R14_P_ANCHOR_COUNT", "R14_P_ANCHOR_SALT", "R14_SELECTION_BASE",
    "R14_SELECTION_SALT", "R14RequestIdentity",
    "build_r14_fresh_seal", "build_r14_historical_exclusion",
    "load_r14_controller_requests", "load_r14_identities",
    "project_r14_request_identity",
    "verify_r14_fresh_seal", "verify_r14_historical_exclusion",
]
