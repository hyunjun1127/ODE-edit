"""Read exact sealed B1/B10 rows without copying or reordering the payload."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from .contracts import ORDER_SHA256, SAMPLE_PAYLOAD_SHA256, STREAM_ROOT, STREAM_SHA256, TechnicalBoundary
from .hashing import canonical_hash, file_sha256


def _regular(path: Path) -> None:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise TechnicalBoundary(f"sealed stream member is not a regular non-symlink: {path}")


def load_sealed_rows(count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count not in {1, 10}:
        raise TechnicalBoundary("initial experiment permits only B1 or B10")
    receipt_path = STREAM_ROOT / "rooted-receipt.json"
    index_path = STREAM_ROOT / "samples/ordered-record-index.json"
    payload_path = STREAM_ROOT / "samples/selected-counterfact-records.bin"
    for path in (receipt_path, index_path, payload_path):
        _regular(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("stream_root_sha256") != STREAM_SHA256
        or receipt.get("all_request_order_sha256") != ORDER_SHA256
        or receipt.get("sample_payload_sha256") != SAMPLE_PAYLOAD_SHA256
        or receipt.get("request_count") != 1000
        or receipt.get("sample_payload_count") != 1
        or receipt.get("sample_duplication_count") != 0
        or file_sha256(payload_path) != SAMPLE_PAYLOAD_SHA256
    ):
        raise TechnicalBoundary("canonical stream/root/order identity mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records")
    if not isinstance(records, list) or len(records) != 1000:
        raise TechnicalBoundary("ordered stream index cardinality mismatch")
    selected = records[:count]
    rows: list[dict[str, Any]] = []
    member_receipts = []
    with payload_path.open("rb") as handle:
        for ordinal, record in enumerate(selected):
            if record.get("ordinal") != ordinal or record.get("batch_label") != "B1":
                raise TechnicalBoundary("B1 prefix order mismatch")
            offset, length = record.get("raw_offset"), record.get("raw_bytes")
            if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
                raise TechnicalBoundary("invalid sealed byte range")
            handle.seek(offset)
            raw = handle.read(length)
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != length or digest != record.get("raw_sha256"):
                raise TechnicalBoundary("sealed request member hash mismatch")
            row = json.loads(raw.decode("utf-8"))
            if str(row.get("case_id")) != str(record.get("case_id")):
                raise TechnicalBoundary("case identity differs from ordered index")
            rows.append(row)
            member_receipts.append({
                "ordinal": ordinal,
                "case_id": int(row["case_id"]),
                "request_sha256": record["request_sha256"],
                "raw_sha256": digest,
                "raw_bytes": length,
            })
    return rows, {
        "stream_root_sha256": STREAM_SHA256,
        "all_request_order_sha256": ORDER_SHA256,
        "sample_payload_sha256": SAMPLE_PAYLOAD_SHA256,
        "sample_payload_count": 1,
        "sample_duplication_count": 0,
        "selected_count": count,
        "members": member_receipts,
        "selected_order_root": canonical_hash([row["request_sha256"] for row in member_receipts]),
    }


def official_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer = []
    for row in rows:
        rewrite = row["requested_rewrite"]
        answer.append({
            "case_id": str(row["case_id"]),
            "prompt": rewrite["prompt"],
            "subject": rewrite["subject"],
            "target_new": rewrite["target_new"]["str"],
            "target_true": rewrite["target_true"]["str"],
        })
    return answer

