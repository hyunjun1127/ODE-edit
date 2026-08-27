#!/usr/bin/env python3
"""Seal first outcome-independent natural multi-token topology ordinals."""

from __future__ import annotations

import json
import os
from pathlib import Path

from project.run_scripts.barrier_guided_ode.s1_contract import (
    CANONICAL_ORDER_SHA256,
    CANONICAL_SAMPLE_PAYLOAD_SHA256,
    CANONICAL_STREAM_SHA256,
)
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.gpu_runtime import fixed_pretrained_kwargs, offline_environment
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec

from .natural import G1_NATURAL_MANIFEST, MODEL_REVISIONS, TOPOLOGIES, canonical_records, tokenize_record


def _write_once_or_exact(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing natural manifest differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> int:
    from transformers import AutoTokenizer

    records = canonical_records()
    cases: list[dict[str, object]] = []
    tokenizers: list[dict[str, object]] = []
    with offline_environment():
        for alias in MODEL_REVISIONS:
            tokenizer = AutoTokenizer.from_pretrained(
                **fixed_pretrained_kwargs(fixed_model_spec(alias))
            )
            first: dict[str, tuple[object, object]] = {}
            singleton_predecessors: dict[str, int] = {}
            for record in records:
                tokenization = tokenize_record(tokenizer, record, model_alias=alias)
                topology = tokenization.topology
                if topology not in TOPOLOGIES:
                    raise SystemExit(f"unknown natural topology: {topology}")
                if max(len(tokenization.target_token_ids), len(tokenization.source_token_ids)) <= 1:
                    singleton_predecessors.setdefault(topology, record.ordinal)
                    continue
                first.setdefault(topology, (record, tokenization))
            tokenizers.append(
                {
                    "model_alias": alias,
                    "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                    "observed_commit": str(
                        getattr(tokenizer, "_commit_hash", "")
                        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "")
                    ),
                    "expected_revision": MODEL_REVISIONS[alias],
                    "vocabulary_size": int(len(tokenizer)),
                }
            )
            for topology in TOPOLOGIES:
                if topology not in first:
                    cases.append(
                        {
                            "model_alias": alias,
                            "status": "NATURAL_TOPOLOGY_UNAVAILABLE",
                            "topology": topology,
                        }
                    )
                    continue
                record, tokenization = first[topology]
                row = {
                    "batch_index": record.batch_index,
                    "batch_ordinal": record.batch_ordinal,
                    "case_id": record.case_id,
                    "model_alias": alias,
                    "ordinal": record.ordinal,
                    "raw_bytes": record.raw_bytes,
                    "raw_sha256": record.raw_sha256,
                    "request_sha256": record.request_sha256,
                    "selection_rule": "EARLIEST_CANONICAL_ORDINAL_WITH_NATURAL_TOPOLOGY_AND_AT_LEAST_ONE_MULTI_TOKEN_CONTINUATION",
                    "singleton_predecessor_ordinal": singleton_predecessors.get(topology),
                    "source_normalized": tokenization.source_normalized,
                    "source_raw": tokenization.source_raw,
                    "source_token_ids": list(tokenization.source_token_ids),
                    "status": "AVAILABLE",
                    "target_normalized": tokenization.target_normalized,
                    "target_raw": tokenization.target_raw,
                    "target_token_ids": list(tokenization.target_token_ids),
                    "tokenization_identity": tokenization.identity,
                    "topology": topology,
                }
                row["case_identity"] = sha256_bytes(canonical_json(row).encode("utf-8"))
                cases.append(row)
    body = {
        "canonical_order_sha256": CANONICAL_ORDER_SHA256,
        "canonical_sample_payload_sha256": CANONICAL_SAMPLE_PAYLOAD_SHA256,
        "canonical_stream_sha256": CANONICAL_STREAM_SHA256,
        "cases": cases,
        "model_count": len(MODEL_REVISIONS),
        "model_or_evaluator_outcome_access_count": 0,
        "models": list(MODEL_REVISIONS),
        "request_count_scanned": len(records),
        "sample_selection_influence": "SEALED_ORDER_AND_TOKENIZER_ONLY",
        "schema": "ode-edit-bgode-r3-g1-natural-topology-manifest/v1",
        "scientific_promotion": False,
        "termination_token_count": 0,
        "tokenizers": tokenizers,
        "topologies": list(TOPOLOGIES),
    }
    body["identity"] = sha256_bytes(canonical_json(body).encode("utf-8"))
    G1_NATURAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _write_once_or_exact(G1_NATURAL_MANIFEST, (canonical_json(body) + "\n").encode("utf-8"))
    print(
        canonical_json(
            {
                "available": sum(row["status"] == "AVAILABLE" for row in cases),
                "identity": body["identity"],
                "path": str(G1_NATURAL_MANIFEST.resolve()),
                "unavailable": sum(row["status"] != "AVAILABLE" for row in cases),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
