"""CPU-only, create-once C4-WebRef-v2 document/token builder.

This module has no model, loss, teacher, network, or Slurm entrypoint.  D0--D2
readiness is deliberately distinct from teacher readiness and BG scientific G0.
The CLI accepts *already downloaded* pinned source files.  Full gzip EOF/CRC and
whole-shard priority sampling are mandatory; a prefix cannot produce a seal.

Overlap input JSON is a deliberately narrow input-only interface::

  {"schema_version": 1, "inspected_scopes": ["Wiki128"],
   "unavailable_scopes": ["unserialized mom2 Wikipedia texts"],
   "evaluator_inputs": [{"id": "wiki-0", "scope": "Wiki128", "text": "..."}],
   "diagnostic_inputs": [{"id": "mom2-0", "scope": "mom2", "text": "..."}]}

Unknown fields (including labels, targets, scores and future subjects) are
rejected. Diagnostic inputs never reject a candidate. No token decoding is
ever fed back into the model-input token sequence.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import hashlib
import heapq
import io
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time
import unicodedata
from urllib.parse import urlsplit, urlunsplit
import zipfile
import zlib

import numpy as np


CONTRACT_ID = "C4-WebRef-v2"
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ROLES = ("S64", "Dev128", "Reserve320", "Report256")
COUNTS = (64, 128, 320, 256)
SEED = 20260915
WORD_RE = re.compile(r"[a-z0-9]+")


class BuildError(RuntimeError):
    """Typed data/config failure; no method/data relaxation is attempted."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_text(text):
    return digest(text.encode("utf-8"))


def identity_stat(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        raise BuildError(f"NOT_REGULAR_INPUT: {path}")
    before = identity_stat(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    if before != identity_stat(path):
        raise BuildError(f"INPUT_CHANGED_DURING_HASH: {path}")
    return {"path": str(path.absolute()), "realpath": str(path.resolve()),
            "bytes": before[2], "sha256": h.hexdigest(),
            "stat_dev_inode_size_mtime_ns": list(before)}


def write_json_once(path, value):
    with Path(path).open("xb") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonical_url(url):
    """Contract normalization only; preserve scheme/path case/query/userinfo."""
    if not isinstance(url, str) or not url:
        raise ValueError("empty/non-string URL")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("not a parseable HTTP(S) URL")
    host = parsed.hostname.lower()
    if any(c.isspace() for c in host):
        raise ValueError("whitespace in host")
    try:
        ip = ipaddress.ip_address(host)
        host = ip.compressed
        printed_host = f"[{host}]" if ip.version == 6 else host
    except ValueError:
        host = host.encode("idna").decode("ascii").lower()
        printed_host = host
    port = parsed.port  # rejects invalid/non-integer/out-of-range ports
    if port is not None and not ((parsed.scheme.lower() == "http" and port == 80)
                                 or (parsed.scheme.lower() == "https" and port == 443)):
        printed_host += f":{port}"
    userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    return urlunsplit((parsed.scheme.lower(), userinfo + printed_host,
                       parsed.path or "/", parsed.query, ""))


def shingles(text, n=13):
    words = WORD_RE.findall(text)
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def near_duplicate(a, b, threshold=0.8):
    if not a or not b:
        return False
    # Exact size upper bound, not an approximate/MinHash similarity test.
    if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
        return False
    intersection = len(a.intersection(b))
    return intersection / (len(a) + len(b) - intersection) >= threshold


def priority_hash(source_row_id, salt, raw_text_sha256=None):
    key = f"{CONTRACT_ID}|{SEED}|{salt}|{source_row_id}"
    if raw_text_sha256 is not None:
        key += "|" + raw_text_sha256
    return sha_text(key)


@dataclasses.dataclass
class RankedRow:
    source_row_id: str
    priority: str
    row_index: int
    source_split: str
    row: dict

    @property
    def key(self):
        return (self.priority, self.source_row_id)

    def __lt__(self, other):
        # heap[0] is the largest/worst priority, including the specified ID tie.
        return self.key > other.key


def scan_shard(path, spec, *, retain, corpus):
    """Scan all JSONL lines, retain the lowest full-digest priorities, validate EOF."""
    path = Path(path)
    identity = file_identity(path)
    expected = spec.get("expected_bytes_from_content_range")
    if expected is not None and identity["bytes"] != expected:
        raise BuildError(f"PINNED_SOURCE_SIZE_MISMATCH: {spec['split']}")
    if spec.get("full_file_sha256") and identity["sha256"] != spec["full_file_sha256"]:
        raise BuildError(f"PINNED_SOURCE_SHA_MISMATCH: {spec['split']}")
    with path.open("rb") as handle:
        prefix = handle.read(spec.get("prefix_bytes_verified", 0))
        prefix_sha = digest(prefix)
        if spec.get("prefix_sha256") and prefix_sha != spec["prefix_sha256"]:
            raise BuildError(f"PINNED_SOURCE_PREFIX_MISMATCH: {spec['split']}")
        handle.seek(-8, os.SEEK_END)
        last_crc, last_isize = struct.unpack("<II", handle.read(8))
    before = identity_stat(path)
    if before != tuple(identity["stat_dev_inode_size_mtime_ns"]):
        raise BuildError(f"INPUT_CHANGED_BEFORE_SCAN: {path}")
    heap = []
    counts = collections.Counter()
    crc, uncompressed = 0, 0
    # gzip.GzipFile verifies CRC and ISIZE for each member, including concatenated
    # members. Consuming the iterator to EOF is required even once the heap fills.
    with gzip.open(path, "rb") as handle:
        for row_index, line in enumerate(handle):
            counts["jsonl_rows"] += 1
            crc = zlib.crc32(line, crc)
            uncompressed += len(line)
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                counts["invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                counts["non_object_json"] += 1
                continue
            counts["valid_json_object_rows"] += 1
            source_id = (f"{corpus['dataset']}@{corpus['revision']}|{spec['split']}|"
                         f"{spec['path']}|{row_index}")
            item = RankedRow(source_id, priority_hash(source_id, "document-priority"),
                             row_index, spec["split"], row)
            if len(heap) < retain:
                heapq.heappush(heap, item)
            elif item.key < heap[0].key:
                heapq.heapreplace(heap, item)
    if before != identity_stat(path):
        raise BuildError(f"INPUT_CHANGED_DURING_SCAN: {path}")
    rows = sorted(heap, key=lambda row: row.key)
    identity.update({"repo": corpus["dataset"], "config": corpus["config"],
                     "revision": corpus["revision"], "split": spec["split"],
                     "shard": spec["path"], "pinned_url": spec.get("url"),
                     "full_file_downloaded": True, "prefix_sha256": prefix_sha,
                     "gzip_crc_validation": "PASS_ALL_MEMBERS_TO_EOF",
                     "gzip_crc32_full_decompressed": f"{crc & 0xffffffff:08x}",
                     "gzip_last_member_trailer_crc32": f"{last_crc:08x}",
                     "gzip_last_member_trailer_isize_mod_2pow32": last_isize,
                     "uncompressed_bytes": uncompressed, "row_counts": dict(counts),
                     "retained_rows": len(rows), "retained_limit": retain,
                     "candidate_priority_root": digest(canonical_json(
                         [[row.source_row_id, row.priority] for row in rows])),
                     "scope": "COMPLETE_PINNED_FIRST_SHARD_NOT_WHOLE_CORPUS"})
    return rows, identity


@dataclasses.dataclass
class Fingerprint:
    full: str
    window: str
    full_shingles: frozenset
    window_shingles: frozenset
    canonical_url: str | None = None

    @classmethod
    def make(cls, full, window=None, url=None):
        full = normalized(full)
        window = full if window is None else normalized(window)
        return cls(full, window, shingles(full), shingles(window), url)

    def duplicate_reason(self, other):
        if self.canonical_url is not None and self.canonical_url == other.canonical_url:
            return "canonical_url"
        if self.full == other.full:
            return "normalized_full_document"
        if self.window == other.window:
            return "normalized_selected_window"
        if near_duplicate(self.full_shingles, other.full_shingles):
            return "near_full_document"
        if near_duplicate(self.window_shingles, other.window_shingles):
            return "near_selected_window"
        return None


class OverlapChecker:
    """Input-only fingerprints; excluded evaluator vs diagnostic mom2 separate."""
    def __init__(self, payload=None):
        self.inspected_scopes = []
        self.unavailable_scopes = ["evaluator inputs NOT_PROVIDED",
                                   "mom2 Wikipedia text NOT_PROVIDED"]
        self.evaluator = []
        self.diagnostic = []
        if payload is None:
            return
        allowed = {"schema_version", "inspected_scopes", "unavailable_scopes",
                   "evaluator_inputs", "diagnostic_inputs"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise BuildError("OVERLAP_INPUT_CONTAINS_UNAPPROVED_FIELDS")
        if payload.get("schema_version") != 1:
            raise BuildError("OVERLAP_INPUT_SCHEMA_MISMATCH")
        for key in ("inspected_scopes", "unavailable_scopes"):
            value = payload.get(key, [])
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise BuildError("OVERLAP_SCOPE_NOT_STRING_LIST")
            setattr(self, key, value)
        for key, target in (("evaluator_inputs", self.evaluator),
                            ("diagnostic_inputs", self.diagnostic)):
            rows = payload.get(key, [])
            if not isinstance(rows, list):
                raise BuildError("OVERLAP_ROWS_NOT_LIST")
            seen = set()
            for row in rows:
                if (not isinstance(row, dict) or set(row) - {"id", "scope", "text"}
                        or not {"id", "scope", "text"}.issubset(row)
                        or any(not isinstance(row[k], str) for k in row)
                        or not row["text"].strip()):
                    raise BuildError("OVERLAP_ROW_CONTAINS_NON_INPUT_FIELDS")
                key_id = (row["scope"], row["id"])
                if key_id in seen:
                    raise BuildError("DUPLICATE_OVERLAP_INPUT_ID")
                seen.add(key_id)
                target.append((key_id, Fingerprint.make(row["text"])))

    @staticmethod
    def matches(fp, references):
        result = []
        for identity, known in references:
            # Both document and selected-window fingerprints are compared with
            # each available evaluator input. No label or score is accepted.
            exact = fp.full == known.full or fp.window == known.full
            near = (near_duplicate(fp.full_shingles, known.full_shingles)
                    or near_duplicate(fp.window_shingles, known.full_shingles))
            if exact or near:
                result.append({"scope": identity[0], "input_id": identity[1],
                               "kind": "exact" if exact else "13gram_jaccard"})
        return result


@dataclasses.dataclass
class Document:
    metadata: dict
    input_ids: list
    fingerprint: Fingerprint


def tokenize_candidate(item, tokenizer, contract):
    row = item.row
    required = contract["filters"]["require_fields"]
    if any(k not in row for k in required):
        return None, "missing_required_field"
    if not all(isinstance(row[k], str) for k in required):
        return None, "non_string_required_field"
    if not row["text"].strip():
        return None, "empty_text"
    try:
        url = canonical_url(row["url"])
    except (ValueError, UnicodeError):
        return None, "invalid_http_url"
    # No tokenizer truncation, padding, chat template, EOS, or re-encoding.
    tokens = tokenizer.encode(row["text"], add_special_tokens=False)
    if len(tokens) < 256:
        return None, "exact_token_length_below_256"
    raw_sha = sha_text(row["text"])
    window_hash = priority_hash(item.source_row_id, "window", raw_sha)
    start = int(window_hash, 16) % (len(tokens) - 256 + 1)
    window = tokens[start:start + 256]
    ids = [int(tokenizer.bos_token_id)] + [int(token) for token in window]
    decoded = tokenizer.decode(window, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    fp = Fingerprint.make(row["text"], decoded, url)
    host = urlsplit(url).hostname
    wiki = contract["filters"]["wikipedia_diagnostics"]
    tags = {"domains": [d for d in wiki["tagged_domains"]
                         if host == d or host.endswith("." + d)],
            "mirror_phrases": [phrase for phrase in wiki["case_insensitive_mirror_phrases"]
                               if phrase.casefold() in row["text"].casefold()]}
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    metadata = {"source_row_id": item.source_row_id, "source_split": item.source_split,
                "original_jsonl_row_index": item.row_index,
                "url": row["url"], "canonical_url": url, "timestamp": row["timestamp"],
                "raw_text": row["text"], "raw_text_sha256": raw_sha,
                "normalized_text_sha256": sha_text(fp.full),
                "normalized_full_document_hash": sha_text(fp.full),
                "normalized_selected_window_hash": sha_text(fp.window),
                "decoded_window_for_diagnostics_only": decoded,
                "priority_hash": item.priority,
                "bank_role_hash": priority_hash(item.source_row_id, "bank-role"),
                "exact_token_length": len(tokens), "window_start": start,
                "window_hash": window_hash,
                "window_input_sha256": digest(np.asarray(ids, dtype="<i8").tobytes()),
                "window_input_hash_encoding": "little-endian int64 input_ids[257] raw bytes",
                "natural_window_special_ids": collections.Counter(
                    str(x) for x in window if x in special_ids),
                "full_document_special_ids_count": sum(int(x) in special_ids for x in tokens),
                "wiki_tags": tags}
    return Document(metadata, ids, fp), None


def select_documents(candidates, tokenizer, contract, *, needed, initial_limit,
                     previous=(), overlap=None):
    """Greedy priority selection with at most one already-ranked pool extension."""
    overlap = overlap or OverlapChecker()
    selected = []
    counters = collections.Counter()
    rejection_details = []
    extended = False
    for ordinal, item in enumerate(candidates):
        if len(selected) == needed:
            break
        if ordinal >= initial_limit:
            extended = True
        counters["candidates_inspected"] += 1
        document, reason = tokenize_candidate(item, tokenizer, contract)
        if reason is None:
            hits = overlap.matches(document.fingerprint, overlap.evaluator)
            if hits:
                reason = "available_evaluator_input_overlap"
                rejection_details.append({"source_row_id": item.source_row_id,
                                          "overlap_matches": hits})
        if reason is None:
            for old in (*previous, *selected):
                reason = document.fingerprint.duplicate_reason(old.fingerprint)
                if reason is not None:
                    reason = "selected_duplicate_" + reason
                    break
        if reason is not None:
            counters[reason] += 1
            continue
        document.metadata["candidate_priority_ordinal"] = ordinal
        selected.append(document)
        counters["selected"] += 1
    return selected, {"counts": dict(counters), "extension_used": extended,
                      "initial_limit": initial_limit,
                      "maximum_pool_rows": len(candidates),
                      "required": needed, "actual": len(selected),
                      "evaluator_overlap_rejections": rejection_details}


class PublicSuffixList:
    """Exact PSL wildcard/exception algorithm, including PRIVATE entries.

    PSL changes composition labels only. It is never consulted by selection.
    """
    def __init__(self, text):
        self.rules, self.wildcards, self.exceptions = set(), set(), set()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            exception = line.startswith("!")
            wildcard = line.startswith("*.")
            name = line[1:] if exception else line[2:] if wildcard else line
            name = name.encode("idna").decode("ascii").lower()
            (self.exceptions if exception else self.wildcards if wildcard
             else self.rules).add(name)
        if not self.rules:
            raise BuildError("PUBLIC_SUFFIX_LIST_EMPTY")

    def registrable_domain(self, host):
        host = host.lower().rstrip(".")
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        labels = host.split(".")
        exception_length, rule_length = 0, 1
        for i in range(len(labels)):
            suffix = ".".join(labels[i:])
            if suffix in self.exceptions:
                exception_length = max(exception_length, len(labels) - i - 1)
            if suffix in self.rules:
                rule_length = max(rule_length, len(labels) - i)
            if i > 0 and suffix in self.wildcards:
                rule_length = max(rule_length, len(labels) - i + 1)
        public_length = exception_length or rule_length
        return ".".join(labels[-min(len(labels), public_length + 1):])


def assign_roles(bank, report):
    bank = sorted(bank, key=lambda doc: (doc.metadata["bank_role_hash"],
                                        doc.metadata["source_row_id"]))
    documents = bank + report
    position = 0
    for role, count in zip(ROLES, COUNTS):
        for doc in documents[position:position + count]:
            doc.metadata["split_role"] = role
            doc.metadata["row_ordinal"] = position
            position += 1
    if len(documents) != sum(COUNTS) or position != sum(COUNTS):
        raise BuildError("REFERENCE_ROLE_CARDINALITY_MISMATCH")
    return documents


def all_pairs_audit(documents):
    pairs, collisions = 0, []
    for i, doc in enumerate(documents):
        for j in range(i):
            pairs += 1
            reason = doc.fingerprint.duplicate_reason(documents[j].fingerprint)
            if reason:
                collisions.append({"i": i, "j": j, "reason": reason})
    if collisions:
        raise BuildError("POST_BUILD_DUPLICATE_AUDIT_FAILED: " + str(collisions[:3]))
    return {"method": "exact normalized URL/full/window + exact 13-word-shingle sets",
            "jaccard_threshold": 0.8, "all_pairs_checked": pairs,
            "duplicate_pairs": collisions, "cross_split_pairs_included": True,
            "scope": "selected reference documents/windows, not corpus-wide"}


def tokenizer_identity(path, tokenizer):
    root = Path(path)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json",
             "merges.txt", "config.json")
    members = [file_identity(root / name) for name in names if (root / name).is_file()]
    if not members or tokenizer.bos_token_id is None:
        raise BuildError("TOKENIZER_IDENTITY_OR_BOS_MISSING")
    import transformers
    return {"path": str(root.resolve()), "members": members,
            "member_root": digest(canonical_json([[Path(m["path"]).name,
                                                   m["bytes"], m["sha256"]]
                                                  for m in members])),
            "class": type(tokenizer).__name__, "is_fast": bool(tokenizer.is_fast),
            "vocabulary_length": len(tokenizer), "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "all_special_ids": tokenizer.all_special_ids,
            "add_bos_token_default": getattr(tokenizer, "add_bos_token", None),
            "encoding_add_special_tokens": False, "explicit_bos_tokens": 1,
            "decode_only_for_diagnostics": True, "trust_remote_code": False,
            "local_files_only": True, "transformers": transformers.__version__,
            "model_revision_required": MODEL_REVISION,
            "model_weights_loaded": False,
            "model_identity_binding_status": "TOKENIZER_FILES_PINNED_W0_WEIGHT_BINDING_BY_TEACHER"}


def validate_contract(contract):
    assertions = [
        contract["contract_id"] == CONTRACT_ID, contract["schema_version"] == 2,
        contract["corpus"]["dataset"] == "allenai/c4",
        contract["corpus"]["config"] == "en", contract["corpus"]["revision"] == REVISION,
        contract["sampling"]["seed"] == SEED,
        contract["sampling"]["initial_retained_candidate_rows"] == {"train": 8192, "validation": 4096},
        contract["sampling"]["single_extension_retained_candidate_rows"] == {"train": 16384, "validation": 8192},
        [contract["splits"][role]["documents"] for role in ROLES] == list(COUNTS),
        contract["tokenization"]["model_revision"] == MODEL_REVISION,
        contract["tokenization"]["natural_text_tokens"] == 256,
        contract["tokenization"]["input_tokens"] == 257,
        contract["tokenization"]["score_input_indices_half_open"] == [129, 257],
        contract["tokenization"]["score_logits_indices_half_open"] == [128, 256],
        contract["filters"]["near_duplicate"]["word_ngram"] == 13,
        contract["filters"]["near_duplicate"]["jaccard_threshold"] == 0.8,
        contract["sampling"]["selection_uses_model_losses"] is False,
        contract["tokenization"]["add_special_tokens_in_text_encoding"] is False,
    ]
    specs = contract["corpus"]["source_files"]
    assertions.extend([
        [(s["split"], s["path"], s["expected_bytes_from_content_range"]) for s in specs]
        == [("train", "en/c4-train.00000-of-01024.json.gz", 319308785),
            ("validation", "en/c4-validation.00000-of-00008.json.gz", 40471190)],
        contract["corpus"]["total_compressed_source_bytes"] == 359779975,
    ])
    if not all(assertions):
        raise BuildError("C4_WEBREF_V2_CONTRACT_DRIFT")


def write_npz_once(path, arrays):
    """Fixed ZipInfo timestamps/order; no pickles or wall-clock byte variation."""
    with Path(path).open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.save(payload, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue())
        handle.flush()
        os.fsync(handle.fileno())


def build_reference(*, contract_path, train, validation, readme, psl, tokenizer_path,
                    output, overlap_inputs=None):
    started = time.monotonic()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        contract_identity = file_identity(contract_path)
        contract = json.loads(Path(contract_path).read_text())
        validate_contract(contract)
        readme_identity, psl_identity = file_identity(readme), file_identity(psl)
        if not readme_identity["bytes"]:
            raise BuildError("SOURCE_README_EMPTY")
        expected_psl = contract["sampling"].get("public_suffix_snapshot_sha256")
        if expected_psl and psl_identity["sha256"] != expected_psl:
            raise BuildError("PUBLIC_SUFFIX_SNAPSHOT_MISMATCH")
        suffixes = PublicSuffixList(Path(psl).read_text(encoding="utf-8"))
        overlap_identity = file_identity(overlap_inputs) if overlap_inputs else None
        overlap = OverlapChecker(json.loads(Path(overlap_inputs).read_text())
                                 if overlap_inputs else None)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path),
                                                   local_files_only=True,
                                                   trust_remote_code=False)
        tok_identity = tokenizer_identity(tokenizer_path, tokenizer)
        if len(tokenizer) != 128256:
            raise BuildError("TOKENIZER_VOCABULARY_MISMATCH")
        sources, pools = [], {}
        for path, spec in zip((train, validation), contract["corpus"]["source_files"]):
            pool, identity = scan_shard(path, spec,
                retain=contract["sampling"]["single_extension_retained_candidate_rows"][spec["split"]],
                corpus=contract["corpus"])
            pools[spec["split"]] = pool
            sources.append(identity)
        bank, train_counts = select_documents(pools["train"], tokenizer, contract,
                                              needed=512, initial_limit=8192, overlap=overlap)
        report, report_counts = select_documents(pools["validation"], tokenizer, contract,
                                                  needed=256, initial_limit=4096,
                                                  previous=bank, overlap=overlap)
        counts = {"train": train_counts, "validation": report_counts}
        write_json_once(output / "filter-counts.json", counts)
        if len(bank) != 512 or len(report) != 256:
            raise BuildError("INSUFFICIENT_ELIGIBLE_DOCUMENTS")
        documents = assign_roles(bank, report)
        pairs = all_pairs_audit(documents)
        doc_ids = [doc.metadata["source_row_id"] for doc in documents]
        if len(set(doc_ids)) != 768:
            raise BuildError("SOURCE_ROW_ID_DUPLICATE")
        input_ids = np.asarray([doc.input_ids for doc in documents], dtype="<i8")
        role_names = [doc.metadata["split_role"] for doc in documents]
        if input_ids.shape != (768, 257):
            raise BuildError("TOKEN_SHAPE_MISMATCH")
        arrays = {"input_ids": input_ids, "source_row_ids": np.asarray(doc_ids),
                  "split_roles": np.asarray(role_names),
                  "score_input_indices": np.arange(129, 257, dtype="<i8"),
                  "score_logits_indices": np.arange(128, 256, dtype="<i8")}
        identity_root = digest(canonical_json([
            [doc.metadata["source_row_id"], doc.metadata["split_role"],
             doc.metadata["window_input_sha256"]] for doc in documents]))
        split_map = {"contract_id": CONTRACT_ID, "roles": {},
                     "document_order_sha256": digest(canonical_json(doc_ids)),
                     "token_array_sha256": digest(input_ids.tobytes()),
                     "token_array_hash_encoding": "little-endian int64[768,257] raw bytes",
                     "reference_identity_sha256": identity_root,
                     "role_order": list(ROLES), "total_documents": 768,
                     "initial_teacher_documents": 192,
                     "teacher_generated": False}
        composition = {"public_suffix": psl_identity, "private_domains_included": True,
                       "PSL_selection_dependency": False, "roles": {}}
        diagnostic_matches = []
        for role, count in zip(ROLES, COUNTS):
            indices = [i for i, name in enumerate(role_names) if name == role]
            if len(indices) != count:
                raise BuildError("ROLE_CARDINALITY_MISMATCH")
            split_map["roles"][role] = dict(contract["splits"][role],
                indices=indices, source_row_ids=[doc_ids[i] for i in indices])
            domains = collections.Counter(suffixes.registrable_domain(
                urlsplit(documents[i].metadata["canonical_url"]).hostname) for i in indices)
            lengths = np.asarray([documents[i].metadata["exact_token_length"] for i in indices])
            composition["roles"][role] = {
                "documents": count, "domains": dict(sorted(domains.items())),
                "top10_domains": domains.most_common(10),
                "top10_document_share": sum(v for _, v in domains.most_common(10)) / count,
                "exact_token_length": {"min": int(lengths.min()), "max": int(lengths.max()),
                                       "mean": float(lengths.mean()), "median": float(np.median(lengths))},
                "wiki_domain_tagged_documents": sum(bool(documents[i].metadata["wiki_tags"]["domains"])
                                                     for i in indices),
                "wiki_mirror_phrase_tagged_documents": sum(bool(documents[i].metadata["wiki_tags"]["mirror_phrases"])
                                                            for i in indices)}
        for doc in documents:
            hits = overlap.matches(doc.fingerprint, overlap.diagnostic)
            if hits:
                diagnostic_matches.append({"source_row_id": doc.metadata["source_row_id"],
                                           "matches": hits, "selection_changed": False})
        overlap_report = dict(pairs, input_asset=overlap_identity,
            inspected_scopes=overlap.inspected_scopes, unavailable_scopes=overlap.unavailable_scopes,
            evaluator_input_count=len(overlap.evaluator),
            diagnostic_input_count=len(overlap.diagnostic), diagnostic_matches=diagnostic_matches,
            sampler_receives_answers_scores_future_subjects=False,
            uninspected_assets_overlap_free_claim=False,
            semantic_duplicate_free_claim=False, wikipedia_mirror_free_claim=False,
            evaluator_selected_overlap_count=0)
        with (output / "reference-documents.jsonl").open("xb") as handle:
            for doc in documents:
                handle.write(canonical_json(doc.metadata) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        write_npz_once(output / "reference-tokens.npz", arrays)
        # Read back exact dtype/shape/IDs with no pickle dependency before sealing.
        with np.load(output / "reference-tokens.npz", allow_pickle=False) as reopened:
            if set(reopened.files) != set(arrays) or any(
                    not np.array_equal(reopened[key], value) for key, value in arrays.items()):
                raise BuildError("TOKEN_NPZ_READBACK_MISMATCH")
        write_json_once(output / "splits.json", split_map)
        write_json_once(output / "reference-composition.json", composition)
        write_json_once(output / "overlap-audit.json", overlap_report)
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).resolve().parent,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            head = "NOT_AVAILABLE_SOURCE_SNAPSHOT"
        source_manifest = {
            "contract_id": CONTRACT_ID, "contract": contract_identity,
            "source_files": sources, "source_readme": readme_identity,
            "readme_pinned_url": f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/README.md",
            "tokenizer": tok_identity, "public_suffix": psl_identity,
            "overlap_input": overlap_identity, "builder_git_head": head,
            "builder_file": file_identity(__file__),
            "software_versions": {"python": sys.version, "numpy": np.__version__,
                                  "platform": platform.platform(), "unicode": unicodedata.unidata_version},
            "canonical_url_idna": "Python stdlib IDNA codec", "sample_seed": SEED,
            "reference_identity_sha256": identity_root,
            "selection_uses_model_losses": False, "teacher_or_model_loaded": False,
            "selection_scope": "whole pinned first train/validation shard, not whole C4",
            "maximum_pool_prefetched": True,
            "extension_only_used_after_initial_pool_insufficient": True,
            "special_token_handling": "encode(add_special_tokens=False), exact token slice, explicit BOS; literal special IDs counted, no option switching"}
        for identity in [contract_identity, readme_identity, psl_identity, *sources,
                         *tok_identity["members"], *([overlap_identity] if overlap_identity else [])]:
            if identity_stat(identity["path"]) != tuple(identity["stat_dev_inode_size_mtime_ns"]):
                raise BuildError("INPUT_CHANGED_BEFORE_SEAL: " + identity["path"])
        write_json_once(output / "source-manifest.json", source_manifest)
        # Hash every closed artifact. The last build-status receipt seals the
        # create-once directory; consumers must never accept a partial directory.
        members = [file_identity(path) for path in sorted(output.iterdir()) if path.is_file()]
        member_root = digest(canonical_json([[Path(m["path"]).name, m["bytes"], m["sha256"]]
                                            for m in members]))
        status = {"status": "TEXT_TOKEN_SEALED_TEACHER_PENDING", "contract_id": CONTRACT_ID,
                  "reference_identity_sha256": identity_root, "members": members,
                  "member_root": member_root, "source_full_shards_downloaded": True,
                  "formal_text_set_ready": True, "exact_token_set_ready": True,
                  "teacher_ready": False, "method_technical_checks_ready": False,
                  "scientific_chain_submitted": False, "G0_PASS": False,
                  "output": str(output.resolve()), "cpu_build_elapsed_seconds": time.monotonic() - started,
                  "losses_observed": 0, "model_forwards": 0, "raw_data_in_git": False,
                  "uninspected_overlap_scope": overlap.unavailable_scopes}
        write_json_once(output / "build-status.json", status)
        return status
    except Exception as exc:
        # Existing outputs are never rewritten/deleted, including failed attempts.
        write_json_once(output / "build-failure.json", {
            "status": "BUILD_FAILED", "exception_type": type(exc).__name__,
            "reason": str(exc), "teacher_ready": False, "G0_PASS": False,
            "elapsed_seconds": time.monotonic() - started,
            "partial_outputs_preserved": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "train", "validation", "readme", "psl", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--overlap-inputs")
    args = parser.parse_args(argv)
    result = build_reference(contract_path=args.contract, train=args.train,
        validation=args.validation, readme=args.readme, psl=args.psl,
        tokenizer_path=args.tokenizer, output=args.output, overlap_inputs=args.overlap_inputs)
    print(json.dumps({"status": result["status"], "output": result["output"],
                      "reference_identity_sha256": result["reference_identity_sha256"],
                      "member_root": result["member_root"], "teacher_ready": False,
                      "G0_PASS": False}, sort_keys=True))


if __name__ == "__main__":
    main()
