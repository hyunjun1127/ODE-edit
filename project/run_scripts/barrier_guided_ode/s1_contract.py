"""Sealed BGODE-R1 S1 sample, tokenizer, and arm identities.

The S1 pilot consumes exactly the first request in the canonical Phase123 B1
stream.  This module performs no model action; it only reprojects the sealed
raw bytes into the small typed request used by the existing EasyEdit bridge.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from project.run_scripts.ode_edit_motivation.contracts import (
    EditRequest,
    canonical_json,
    sha256_bytes,
)

from .errors import BGODEScientificBoundary, EventPartitionBoundary


CANONICAL_STREAM_ROOT = Path(
    "/mnt/raid5/janghj/ODE-edit/local/state/phase123-server1-single-canonical-stream-v1"
)
CANONICAL_STREAM_SHA256 = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
CANONICAL_ORDER_SHA256 = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
CANONICAL_SAMPLE_PAYLOAD_SHA256 = (
    "fe7c8b0cb51abf591e0ec560c0009bdcc47f20e8c4efcf6efc3311710a373475"
)
CANONICAL_CONTEXT_SHA256 = "d86e373536a3d78528bbbb8dc631395035032f15e256136370deeee950fd6885"
S1_BOUNDARY_STRING = "<|eot_id|>"
S1_BOUNDARY_TOKEN_ID = 128_009
S1_CONTEXT_SEED = 41
S1_MODEL_ALIAS = "llama3-8b-inst"
S1_MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"


@dataclass(frozen=True, slots=True)
class S1ModelBinding:
    alias: str
    revision: str
    boundary_string: str
    boundary_token_id: int
    run_id: str


S1_MODEL_BINDINGS: Mapping[str, S1ModelBinding] = MappingProxyType({
    S1_MODEL_ALIAS: S1ModelBinding(
        alias=S1_MODEL_ALIAS,
        revision=S1_MODEL_REVISION,
        boundary_string=S1_BOUNDARY_STRING,
        boundary_token_id=S1_BOUNDARY_TOKEN_ID,
        run_id="s05-bgode-r1-s1-llama-b1-request000-six-arm-v1",
    ),
    "qwen2.5-7b-inst": S1ModelBinding(
        alias="qwen2.5-7b-inst",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        boundary_string="<|im_end|>",
        boundary_token_id=151_645,
        run_id="s05-bgode-r1-s1-qwen-b1-request000-six-arm-v1",
    ),
})


class S1SampleSemanticBoundary(EventPartitionBoundary):
    """The outcome-independent first canonical request has invalid semantics."""


def s1_model_binding(alias: str) -> S1ModelBinding:
    try:
        return S1_MODEL_BINDINGS[alias]
    except KeyError as exc:
        raise S1SampleSemanticBoundary(f"unsupported BGODE S1 model alias: {alias}") from exc


class NonpositiveNativeHorizon(BGODEScientificBoundary):
    """Official AlphaEdit does not define a positive finite Claim-A horizon."""


class S1Arm(str, Enum):
    NATIVE_ALPHAEDIT = "official-native-alphaedit-bypass"
    PLAIN_DYNAMIC = "plain-dynamic"
    FISHER_DYNAMIC = "fisher-only-dynamic"
    FULL_DYNAMIC = "full-moving-barrier-dynamic"
    ONE_STEP_FULL = "one-step-full-barrier"
    FROZEN_FIELD_N4 = "frozen-field-n4-split"


S1_ARM_ORDER = tuple(S1Arm)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_mode_0600(path: Path) -> Path:
    if path.is_symlink():
        raise S1SampleSemanticBoundary(f"symlink input is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or (resolved.stat().st_mode & 0o777) != 0o600:
        raise S1SampleSemanticBoundary(f"regular mode0600 input required: {path}")
    return resolved


@dataclass(frozen=True, slots=True)
class SealedS1Sample:
    edit_request: EditRequest
    target_true: str
    raw_row: Mapping[str, Any]
    raw_sha256: str
    raw_bytes: int
    request_sha256: str
    case_id: str
    batch_label: str
    batch_ordinal: int
    request_ordinal: int
    stream_root_sha256: str
    all_request_order_sha256: str
    sample_payload_sha256: str
    sample_payload_path: str

    @property
    def identity(self) -> str:
        return sha256_bytes(canonical_json(self.receipt()).encode("utf-8"))

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-bgode-r1-s1-sealed-first-request/v1",
            "case_id": self.case_id,
            "batch_label": self.batch_label,
            "batch_ordinal": self.batch_ordinal,
            "request_ordinal": self.request_ordinal,
            "raw_sha256": self.raw_sha256,
            "raw_bytes": self.raw_bytes,
            "request_sha256": self.request_sha256,
            "edit_request_id": self.edit_request.request_id,
            "stream_root_sha256": self.stream_root_sha256,
            "all_request_order_sha256": self.all_request_order_sha256,
            "sample_payload_sha256": self.sample_payload_sha256,
            "sample_payload_path": self.sample_payload_path,
            "selection_rule": "B1_FIRST_REQUEST_ORDINAL_ZERO_OUTCOME_INDEPENDENT",
        }


def load_sealed_s1_sample(root: str | Path = CANONICAL_STREAM_ROOT) -> SealedS1Sample:
    source = Path(root).expanduser().resolve(strict=True)
    receipt_path = _regular_mode_0600(source / "rooted-receipt.json")
    index_path = _regular_mode_0600(source / "samples/ordered-record-index.json")
    payload_path = _regular_mode_0600(source / "samples/selected-counterfact-records.bin")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("stream_root_sha256") != CANONICAL_STREAM_SHA256
        or receipt.get("all_request_order_sha256") != CANONICAL_ORDER_SHA256
        or receipt.get("sample_payload_sha256") != CANONICAL_SAMPLE_PAYLOAD_SHA256
        or receipt.get("request_count") != 1000
        or receipt.get("case_count") != 10
        or receipt.get("sample_payload_count") != 1
        or receipt.get("sample_duplication_count") != 0
    ):
        raise S1SampleSemanticBoundary("canonical stream rooted receipt differs from S1 lock")
    if _sha256_file(payload_path) != CANONICAL_SAMPLE_PAYLOAD_SHA256:
        raise S1SampleSemanticBoundary("canonical sample payload bytes changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records") if isinstance(index, Mapping) else None
    if not isinstance(records, list) or len(records) != 1000:
        raise S1SampleSemanticBoundary("canonical ordered record index has wrong cardinality")
    record = records[0]
    required = {
        "batch_index": 0,
        "batch_label": "B1",
        "batch_ordinal": 0,
        "ordinal": 0,
    }
    if not isinstance(record, Mapping) or any(record.get(key) != value for key, value in required.items()):
        raise S1SampleSemanticBoundary("first canonical record is not B1 ordinal zero")
    offset = record.get("raw_offset")
    length = record.get("raw_bytes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (offset, length)):
        raise S1SampleSemanticBoundary("first canonical byte range is invalid")
    with payload_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    raw_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != length or raw_sha != record.get("raw_sha256"):
        raise S1SampleSemanticBoundary("first canonical raw record hash differs")
    try:
        row = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S1SampleSemanticBoundary("first canonical raw record is invalid JSON") from exc
    rewrite = row.get("requested_rewrite") if isinstance(row, Mapping) else None
    if not isinstance(rewrite, Mapping):
        raise S1SampleSemanticBoundary("first canonical record lacks requested_rewrite")
    target_new = rewrite.get("target_new")
    target_true = rewrite.get("target_true")
    if not isinstance(target_new, Mapping) or not isinstance(target_true, Mapping):
        raise S1SampleSemanticBoundary("first canonical source/target fields are invalid")
    edit_request = EditRequest.from_mapping(
        {
            "case_id": row.get("case_id"),
            "prompt": rewrite.get("prompt"),
            "subject": rewrite.get("subject"),
            "target_new": {"str": target_new.get("str")},
        }
    )
    source_text = target_true.get("str")
    if not isinstance(source_text, str) or not source_text.strip():
        raise S1SampleSemanticBoundary("first canonical source string is empty")
    request_sha = record.get("request_sha256")
    if not isinstance(request_sha, str) or len(request_sha) != 64:
        raise S1SampleSemanticBoundary("first canonical request identity is invalid")
    if str(row.get("case_id")) != str(record.get("case_id")):
        raise S1SampleSemanticBoundary("first canonical case identity differs")
    return SealedS1Sample(
        edit_request=edit_request,
        target_true=source_text.strip(),
        raw_row=row,
        raw_sha256=raw_sha,
        raw_bytes=len(raw),
        request_sha256=request_sha,
        case_id=str(row["case_id"]),
        batch_label="B1",
        batch_ordinal=0,
        request_ordinal=0,
        stream_root_sha256=CANONICAL_STREAM_SHA256,
        all_request_order_sha256=CANONICAL_ORDER_SHA256,
        sample_payload_sha256=CANONICAL_SAMPLE_PAYLOAD_SHA256,
        sample_payload_path=str(payload_path),
    )


@dataclass(frozen=True, slots=True)
class S1Tokenization:
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    boundary_string: str
    boundary_token_id: int
    tokenizer_name_or_path: str
    tokenizer_observed_commit: str
    tokenizer_expected_revision: str
    tokenizer_vocab_size: int
    identity: str
    model_alias: str = S1_MODEL_ALIAS

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-bgode-r1-s1-tokenization/v1",
            "model_alias": self.model_alias,
            "prompt_token_ids": list(self.prompt_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "source_token_ids": list(self.source_token_ids),
            "boundary_string": self.boundary_string,
            "boundary_token_id": self.boundary_token_id,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_observed_commit": self.tokenizer_observed_commit,
            "tokenizer_expected_revision": self.tokenizer_expected_revision,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "identity": self.identity,
        }


@dataclass(frozen=True, slots=True)
class S1VocabularyBinding:
    model_alias: str
    tokenizer_vocab_size: int
    config_vocab_size: int
    output_head_vocab_size: int
    event_vocab_size: int
    identity: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-bgode-r1-s1-model-vocabulary/v1",
            "model_alias": self.model_alias,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "config_vocab_size": self.config_vocab_size,
            "output_head_vocab_size": self.output_head_vocab_size,
            "event_vocab_size": self.event_vocab_size,
            "identity": self.identity,
        }


def seal_s1_model_vocabulary(
    model: Any, tokenization: S1Tokenization
) -> S1VocabularyBinding:
    config_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(config_size, bool) or not isinstance(config_size, int) or config_size <= 0:
        raise S1SampleSemanticBoundary("model config vocabulary is invalid")
    output = model.get_output_embeddings()
    shape = getattr(getattr(output, "weight", None), "shape", ())
    if len(shape) != 2:
        raise S1SampleSemanticBoundary("model output vocabulary is unavailable")
    output_size = int(shape[0])
    if output_size != config_size:
        raise S1SampleSemanticBoundary("model config/output vocabulary differs")
    tokenizer_size = int(tokenization.tokenizer_vocab_size)
    if tokenizer_size <= 0 or tokenizer_size > output_size:
        raise S1SampleSemanticBoundary("tokenizer/model vocabulary relationship is invalid")
    sealed_tokens = (
        *tokenization.prompt_token_ids,
        *tokenization.source_token_ids,
        *tokenization.target_token_ids,
        tokenization.boundary_token_id,
    )
    if any(token < 0 or token >= output_size for token in sealed_tokens):
        raise S1SampleSemanticBoundary("sealed S1 token lies outside model output vocabulary")
    payload = {
        "model_alias": tokenization.model_alias,
        "tokenizer_vocab_size": tokenizer_size,
        "config_vocab_size": config_size,
        "output_head_vocab_size": output_size,
        "event_vocab_size": output_size,
    }
    return S1VocabularyBinding(
        **payload,
        identity=sha256_bytes(canonical_json(payload).encode("utf-8")),
    )


def seal_s1_tokenization(
    tokenizer: Any,
    sample: SealedS1Sample,
    *,
    model_alias: str = S1_MODEL_ALIAS,
) -> S1Tokenization:
    binding = s1_model_binding(model_alias)
    if int(tokenizer.convert_tokens_to_ids(binding.boundary_string)) != binding.boundary_token_id:
        raise S1SampleSemanticBoundary("fixed model termination token identity differs")
    vocab_size = int(len(tokenizer))
    if binding.boundary_token_id >= vocab_size:
        raise S1SampleSemanticBoundary("S1 boundary lies outside tokenizer vocabulary")
    prompt_text = sample.edit_request.prompt.format(sample.edit_request.subject)
    target_text = sample.edit_request.target_new
    source_text = sample.target_true
    target_text = target_text if target_text.startswith(" ") else f" {target_text}"
    source_text = source_text if source_text.startswith(" ") else f" {source_text}"
    prompt_ids = tuple(int(v) for v in tokenizer.encode(prompt_text, add_special_tokens=True))
    target_ids = tuple(int(v) for v in tokenizer.encode(target_text, add_special_tokens=False))
    source_ids = tuple(int(v) for v in tokenizer.encode(source_text, add_special_tokens=False))
    if not prompt_ids or not target_ids or not source_ids:
        raise S1SampleSemanticBoundary("S1 prompt/source/target tokenization is empty")
    for text, suffix in ((target_text, target_ids), (source_text, source_ids)):
        combined = tuple(int(v) for v in tokenizer.encode(prompt_text + text, add_special_tokens=True))
        if combined != prompt_ids + suffix:
            raise S1SampleSemanticBoundary("prompt/completion token concatenation is not exact")
    if binding.boundary_token_id in target_ids or binding.boundary_token_id in source_ids:
        raise S1SampleSemanticBoundary("boundary token already occurs inside source/target")
    if target_ids + (binding.boundary_token_id,) == source_ids + (binding.boundary_token_id,):
        raise S1SampleSemanticBoundary("boundary-appended source/target tokenizations are identical")
    commit = str(getattr(tokenizer, "_commit_hash", "") or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", ""))
    payload = {
        "prompt_token_ids": list(prompt_ids),
        "target_token_ids": list(target_ids),
        "source_token_ids": list(source_ids),
        "model_alias": binding.alias,
        "boundary_string": binding.boundary_string,
        "boundary_token_id": binding.boundary_token_id,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_observed_commit": commit,
        "tokenizer_expected_revision": binding.revision,
        "tokenizer_vocab_size": vocab_size,
    }
    identity = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return S1Tokenization(
        model_alias=binding.alias,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        target_token_ids=target_ids,
        source_token_ids=source_ids,
        boundary_string=binding.boundary_string,
        boundary_token_id=binding.boundary_token_id,
        tokenizer_name_or_path=payload["tokenizer_name_or_path"],
        tokenizer_observed_commit=commit,
        tokenizer_expected_revision=binding.revision,
        tokenizer_vocab_size=vocab_size,
        identity=identity,
    )


def validate_s1_horizon(value: float) -> float:
    try:
        horizon = float(value)
    except (TypeError, ValueError) as exc:
        raise NonpositiveNativeHorizon("native horizon is non-numeric") from exc
    if not (horizon > 0.0 and horizon < float("inf")):
        raise NonpositiveNativeHorizon("native Claim-A horizon is nonpositive or nonfinite")
    return horizon


__all__ = [
    "CANONICAL_CONTEXT_SHA256",
    "CANONICAL_ORDER_SHA256",
    "CANONICAL_STREAM_SHA256",
    "S1Arm",
    "S1_ARM_ORDER",
    "S1_MODEL_BINDINGS",
    "S1ModelBinding",
    "S1SampleSemanticBoundary",
    "S1Tokenization",
    "S1VocabularyBinding",
    "SealedS1Sample",
    "load_sealed_s1_sample",
    "seal_s1_tokenization",
    "seal_s1_model_vocabulary",
    "s1_model_binding",
    "validate_s1_horizon",
]
