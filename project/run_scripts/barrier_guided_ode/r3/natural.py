"""Outcome-independent canonical natural-topology bindings for BGODE-R3 G1."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from project.run_scripts.barrier_guided_ode.s1_contract import (
    CANONICAL_ORDER_SHA256,
    CANONICAL_SAMPLE_PAYLOAD_SHA256,
    CANONICAL_STREAM_ROOT,
    CANONICAL_STREAM_SHA256,
)
from project.run_scripts.ode_edit_motivation.contracts import EditRequest, canonical_json, sha256_bytes

from .errors import EventSemanticBoundary


G1_NATURAL_MANIFEST = Path(__file__).with_name("locks") / "bgode-r3-g1-natural-topology-manifest.json"
TOPOLOGIES = (
    "unequal-non-prefix",
    "common-prefix-divergence",
    "source-prefix-target",
    "target-prefix-source",
)
MODEL_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "llama3-8b-inst": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "qwen2.5-7b-inst": "a09a35458c702b33eeacc393d103063234e8bc28",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_mode0600(path: Path) -> Path:
    if path.is_symlink():
        raise EventSemanticBoundary(f"symlink input is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise EventSemanticBoundary(f"regular mode0600 input required: {path}")
    return resolved


def _topology(target: Sequence[int], source: Sequence[int]) -> str:
    target_tokens = tuple(int(value) for value in target)
    source_tokens = tuple(int(value) for value in source)
    shared = 0
    for left, right in zip(target_tokens, source_tokens):
        if left != right:
            break
        shared += 1
    if shared == len(source_tokens) < len(target_tokens):
        return "source-prefix-target"
    if shared == len(target_tokens) < len(source_tokens):
        return "target-prefix-source"
    return "common-prefix-divergence" if shared else "unequal-non-prefix"


def _normalized(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventSemanticBoundary("source/target text is empty")
    stripped = value.strip()
    return stripped if stripped.startswith(" ") else f" {stripped}"


@dataclass(frozen=True, slots=True)
class CanonicalRawRecord:
    ordinal: int
    batch_index: int
    batch_ordinal: int
    case_id: str
    request_sha256: str
    raw_sha256: str
    raw_bytes: int
    row: Mapping[str, Any]


def canonical_records(root: str | Path = CANONICAL_STREAM_ROOT) -> tuple[CanonicalRawRecord, ...]:
    source = Path(root).expanduser().resolve(strict=True)
    receipt_path = _regular_mode0600(source / "rooted-receipt.json")
    index_path = _regular_mode0600(source / "samples/ordered-record-index.json")
    payload_path = _regular_mode0600(source / "samples/selected-counterfact-records.bin")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("stream_root_sha256") != CANONICAL_STREAM_SHA256
        or receipt.get("all_request_order_sha256") != CANONICAL_ORDER_SHA256
        or receipt.get("sample_payload_sha256") != CANONICAL_SAMPLE_PAYLOAD_SHA256
        or receipt.get("request_count") != 1000
        or receipt.get("case_count") != 10
        or receipt.get("sample_payload_count") != 1
        or receipt.get("sample_duplication_count") != 0
        or _sha256(payload_path) != CANONICAL_SAMPLE_PAYLOAD_SHA256
    ):
        raise EventSemanticBoundary("canonical stream receipt differs from the R3 lock")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index.get("records") if isinstance(index, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 1000:
        raise EventSemanticBoundary("canonical record index cardinality differs")
    payload = payload_path.read_bytes()
    result: list[CanonicalRawRecord] = []
    for ordinal, record in enumerate(rows):
        if not isinstance(record, Mapping) or record.get("ordinal") != ordinal:
            raise EventSemanticBoundary("canonical record order differs")
        offset = record.get("raw_offset")
        length = record.get("raw_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (offset, length)):
            raise EventSemanticBoundary("canonical byte range is invalid")
        raw = payload[offset : offset + length]
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != record.get("raw_sha256"):
            raise EventSemanticBoundary("canonical raw record identity differs")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventSemanticBoundary("canonical raw record JSON is invalid") from exc
        if str(value.get("case_id")) != str(record.get("case_id")):
            raise EventSemanticBoundary("canonical case identity differs")
        result.append(
            CanonicalRawRecord(
                ordinal=ordinal,
                batch_index=int(record["batch_index"]),
                batch_ordinal=int(record["batch_ordinal"]),
                case_id=str(record["case_id"]),
                request_sha256=str(record["request_sha256"]),
                raw_sha256=str(record["raw_sha256"]),
                raw_bytes=int(record["raw_bytes"]),
                row=MappingProxyType(value),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class NaturalTokenization:
    model_alias: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    target_raw: str
    source_raw: str
    target_normalized: str
    source_normalized: str
    target_token_ids: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    tokenizer_name_or_path: str
    tokenizer_observed_commit: str
    tokenizer_expected_revision: str
    tokenizer_vocabulary_size: int
    topology: str
    identity: str

    def receipt(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "model_alias": self.model_alias,
            "prompt_token_ids": list(self.prompt_token_ids),
            "schema": "ode-edit-bgode-r3-g1-natural-tokenization/v1",
            "source_normalized": self.source_normalized,
            "source_raw": self.source_raw,
            "source_token_ids": list(self.source_token_ids),
            "target_normalized": self.target_normalized,
            "target_raw": self.target_raw,
            "target_token_ids": list(self.target_token_ids),
            "tokenizer_expected_revision": self.tokenizer_expected_revision,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_observed_commit": self.tokenizer_observed_commit,
            "tokenizer_vocabulary_size": self.tokenizer_vocabulary_size,
            "topology": self.topology,
        }


def tokenize_record(tokenizer: Any, record: CanonicalRawRecord, *, model_alias: str) -> NaturalTokenization:
    if model_alias not in MODEL_REVISIONS:
        raise EventSemanticBoundary("unsupported R3 model alias")
    rewrite = record.row.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise EventSemanticBoundary("canonical record lacks requested_rewrite")
    target_value = rewrite.get("target_new")
    source_value = rewrite.get("target_true")
    if not isinstance(target_value, Mapping) or not isinstance(source_value, Mapping):
        raise EventSemanticBoundary("canonical source/target mappings are invalid")
    target_raw = target_value.get("str")
    source_raw = source_value.get("str")
    prompt_template = rewrite.get("prompt")
    subject = rewrite.get("subject")
    if not isinstance(prompt_template, str) or not isinstance(subject, str):
        raise EventSemanticBoundary("canonical prompt/subject is invalid")
    prompt_text = prompt_template.format(subject)
    target = _normalized(target_raw)
    source = _normalized(source_raw)
    prompt_ids = tuple(int(value) for value in tokenizer.encode(prompt_text, add_special_tokens=True))
    target_ids = tuple(int(value) for value in tokenizer.encode(target, add_special_tokens=False))
    source_ids = tuple(int(value) for value in tokenizer.encode(source, add_special_tokens=False))
    if not prompt_ids or not target_ids or not source_ids or target_ids == source_ids:
        raise EventSemanticBoundary("natural prompt/source/target tokenization is empty or equal")
    if tuple(int(value) for value in tokenizer.encode(prompt_text + target, add_special_tokens=True)) != prompt_ids + target_ids:
        raise EventSemanticBoundary("target continuation is not an exact prompt-boundary suffix")
    if tuple(int(value) for value in tokenizer.encode(prompt_text + source, add_special_tokens=True)) != prompt_ids + source_ids:
        raise EventSemanticBoundary("source continuation is not an exact prompt-boundary suffix")
    vocabulary_size = int(len(tokenizer))
    if any(token < 0 or token >= vocabulary_size for token in (*prompt_ids, *target_ids, *source_ids)):
        raise EventSemanticBoundary("natural token lies outside tokenizer vocabulary")
    commit = str(
        getattr(tokenizer, "_commit_hash", "")
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "")
    )
    body = {
        "model_alias": model_alias,
        "prompt_token_ids": list(prompt_ids),
        "source_normalized": source,
        "source_raw": source_raw,
        "source_token_ids": list(source_ids),
        "target_normalized": target,
        "target_raw": target_raw,
        "target_token_ids": list(target_ids),
        "tokenizer_expected_revision": MODEL_REVISIONS[model_alias],
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_observed_commit": commit,
        "tokenizer_vocabulary_size": vocabulary_size,
        "topology": _topology(target_ids, source_ids),
    }
    return NaturalTokenization(
        model_alias=model_alias,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        target_raw=target_raw,
        source_raw=source_raw,
        target_normalized=target,
        source_normalized=source,
        target_token_ids=target_ids,
        source_token_ids=source_ids,
        tokenizer_name_or_path=body["tokenizer_name_or_path"],
        tokenizer_observed_commit=commit,
        tokenizer_expected_revision=MODEL_REVISIONS[model_alias],
        tokenizer_vocabulary_size=vocabulary_size,
        topology=body["topology"],
        identity=sha256_bytes(canonical_json(body).encode("utf-8")),
    )


def natural_case(model_alias: str, topology: str, *, path: Path = G1_NATURAL_MANIFEST) -> Mapping[str, Any]:
    manifest, _ = load_natural_manifest(path)
    matches = [
        row
        for row in manifest["cases"]
        if row["model_alias"] == model_alias and row["topology"] == topology
    ]
    if len(matches) != 1:
        raise EventSemanticBoundary("natural topology case cardinality differs")
    row = matches[0]
    if row["status"] != "AVAILABLE":
        raise EventSemanticBoundary("requested natural topology is unavailable")
    return MappingProxyType(row)


def load_natural_record(case: Mapping[str, Any]) -> tuple[CanonicalRawRecord, EditRequest, str]:
    records = canonical_records()
    ordinal = case.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or ordinal >= len(records):
        raise EventSemanticBoundary("natural case ordinal is invalid")
    record = records[ordinal]
    if (
        record.case_id != str(case.get("case_id"))
        or record.request_sha256 != case.get("request_sha256")
        or record.raw_sha256 != case.get("raw_sha256")
    ):
        raise EventSemanticBoundary("natural case raw identity differs")
    rewrite = record.row["requested_rewrite"]
    edit = EditRequest.from_mapping(
        {
            "case_id": record.case_id,
            "prompt": rewrite["prompt"],
            "subject": rewrite["subject"],
            "target_new": {"str": rewrite["target_new"]["str"]},
        }
    )
    source = str(rewrite["target_true"]["str"]).strip()
    if edit.to_easyedit()["target_new"] != case.get("target_normalized"):
        raise EventSemanticBoundary("event/compute_z/writer normalized target identity differs")
    return record, edit, source


def load_natural_manifest(path: Path = G1_NATURAL_MANIFEST) -> tuple[Mapping[str, Any], str]:
    resolved = _regular_mode0600(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "ode-edit-bgode-r3-g1-natural-topology-manifest/v1":
        raise EventSemanticBoundary("natural topology manifest schema differs")
    body = {key: item for key, item in value.items() if key != "identity"}
    if sha256_bytes(canonical_json(body).encode("utf-8")) != value.get("identity"):
        raise EventSemanticBoundary("natural topology manifest identity differs")
    return MappingProxyType(value), _sha256(resolved)


__all__ = [
    "CanonicalRawRecord",
    "G1_NATURAL_MANIFEST",
    "MODEL_REVISIONS",
    "NaturalTokenization",
    "TOPOLOGIES",
    "canonical_records",
    "load_natural_manifest",
    "load_natural_record",
    "natural_case",
    "tokenize_record",
]
