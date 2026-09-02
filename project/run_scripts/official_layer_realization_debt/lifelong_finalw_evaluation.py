"""Batched, position-safe scalar evaluator for frozen lifelong checkpoints.

The module deliberately publishes only hashes and scalar metrics.  Prompts,
token ids, and logits exist transiently in memory and never enter artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from project.run_scripts.fixed_z_nonuniqueness.evaluation import (
    sequence_metrics,
)
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1_selection import _load_canonical_request_map

from .lifelong_finalw_contracts import (
    DATASET_SHA256,
    FinalWeightBoundary,
    ORDER_ROOT,
    STREAM_ROOT,
)


CATEGORIES = (
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
)


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stream_and_rows(
    seal_path: Path, dataset_path: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise FinalWeightBoundary("CounterFact dataset SHA differs")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    root = seal.pop("root_digest", None)
    if root != canonical_hash(seal) or root != STREAM_ROOT:
        raise FinalWeightBoundary("lifelong stream root differs")
    seal["root_digest"] = root
    training = seal.get("training")
    if (
        not isinstance(training, list)
        or len(training) != 10_000
        or [row.get("ordinal") for row in training] != list(range(10_000))
        or seal.get("training_order_sha256") != ORDER_ROOT
        or canonical_hash([str(row["request_sha256"]) for row in training]) != ORDER_ROOT
        or seal.get("sample_duplication_count") != 0
    ):
        raise FinalWeightBoundary("lifelong stream/order geometry differs")
    approved = {
        str(row["request_sha256"]): (
            int(row["case_id"]),
            str(row["collision_sha256"]),
        )
        for row in training
    }
    canonical = _load_canonical_request_map(dataset_path, approved)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    wanted_cases = {int(row["case_id"]) for row in training}
    heldout = {int(row["case_id"]): row for row in raw if int(row["case_id"]) in wanted_cases}
    if set(heldout) != wanted_cases:
        raise FinalWeightBoundary("heldout evaluator row inventory differs")
    records: list[dict[str, Any]] = []
    for stream_row in training:
        request_sha = str(stream_row["request_sha256"])
        request = canonical[request_sha]
        source = heldout[int(stream_row["case_id"])]
        rewrite = source["requested_rewrite"]
        if (
            str(rewrite["prompt"]) != str(request["prompt"])
            or str(rewrite["subject"]) != str(request["subject"])
            or str(rewrite["target_new"]["str"]) != str(request["target_new"])
            or str(rewrite["target_true"]["str"]) != str(request["target_true"])
        ):
            raise FinalWeightBoundary("evaluator/request source binding differs")
        paraphrase = tuple(str(value) for value in source["paraphrase_prompts"])
        locality = tuple(str(value) for value in source["neighborhood_prompts"])
        if len(paraphrase) != 2 or len(locality) != 10:
            raise FinalWeightBoundary("CounterFact evaluator prompt cardinality differs")
        case_id = int(stream_row["case_id"])
        records.append(
            {
                "ordinal": int(stream_row["ordinal"]),
                "batch_index": int(stream_row["batch_index"]),
                "case_id": case_id,
                "request_sha256": request_sha,
                "case_identity_sha256": hashlib.sha256(
                    f"case|{case_id}|{request_sha}".encode("utf-8")
                ).hexdigest(),
                "prompt": str(request["prompt"]).format(str(request["subject"])),
                "subject": str(request["subject"]),
                "prompt_template": str(request["prompt"]),
                "target_new": str(request["target_new"]),
                "target_true": str(request["target_true"]),
                "paraphrase_prompts": paraphrase,
                "locality_prompts": locality,
            }
        )
    return seal, tuple(records)


def age_stratum(ordinal: int, seen_count: int) -> str:
    if seen_count <= 0 or ordinal < 0 or ordinal >= seen_count:
        raise FinalWeightBoundary("invalid edit-age coordinate")
    early_end = math.ceil(0.2 * seen_count)
    recent_begin = math.floor(0.8 * seen_count)
    if ordinal < early_end:
        return "EARLY_FIRST_20PCT"
    if ordinal >= recent_begin:
        return "RECENT_LAST_20PCT"
    return "MIDDLE_60PCT"


def _category_specs(row: Mapping[str, Any]) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    prompt = str(row["prompt"])
    rephrase = tuple(str(value) for value in row["paraphrase_prompts"])
    locality = tuple(str(value) for value in row["locality_prompts"])
    return (
        ("rewrite_target_new", (prompt,), str(row["target_new"])),
        ("rewrite_target_true", (prompt,), str(row["target_true"])),
        ("rephrase_target_new", rephrase, str(row["target_new"])),
        ("rephrase_target_true", rephrase, str(row["target_true"])),
        ("locality_target_true", locality, str(row["target_true"])),
    )


def _evaluate_chunk(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Canonical evaluator identity is per request/category sequence_metrics.
    # Cross-request rebatching changes FP32 kernel shape and is intentionally
    # forbidden even if it would be faster.
    if batch_size != 16:
        raise FinalWeightBoundary("canonical sequence_metrics batch size must remain 16")
    records: list[dict[str, Any]] = []
    forward_batches = 0
    with torch.inference_mode():
        for row in rows:
            metrics: dict[str, Any] = {}
            for category, prompts, target in _category_specs(row):
                value = sequence_metrics(model, tokenizer, list(prompts), target)
                prompt_values = [
                    {
                        "nll": float(value["nll"][index]),
                        "margin": float(value["margin"][index]),
                        "strict": bool(value["strict"][index]),
                        "target_token_count": int(value["target_token_count"]),
                        "token_accuracy": "NOT_RECORDED_EVALUATOR_SCHEMA",
                        "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
                    }
                    for index in range(int(value["prompt_count"]))
                ]
                if not all(
                    math.isfinite(item[key])
                    for item in prompt_values
                    for key in ("nll", "margin")
                ):
                    raise FinalWeightBoundary("nonfinite canonical evaluator metric")
                token_examples = int(value["prompt_count"]) * int(value["target_token_count"])
                forward_batches += math.ceil(token_examples / 16)
                metrics[category] = {
                    "prompt_count": int(value["prompt_count"]),
                    "strict_count": sum(int(item["strict"]) for item in prompt_values),
                    "token_correct_count": "NOT_RECORDED_EVALUATOR_SCHEMA",
                    "target_token_count": token_examples,
                    "prompts": prompt_values,
                }
            record = {
                "ordinal": int(row["ordinal"]),
                "batch_index": int(row["batch_index"]),
                "case_identity_sha256": str(row["case_identity_sha256"]),
                "request_sha256": str(row["request_sha256"]),
                "metrics": metrics,
                "raw_prompt_logit_generation_publish_count": 0,
            }
            record["identity_sha256"] = canonical_hash(record)
            records.append(record)
    return records, {
        "forward_batch_count": forward_batches,
        "token_example_count": sum(
            int(value["target_token_count"])
            for record in records
            for value in record["metrics"].values()
        ),
        "request_count": len(rows),
    }


def evaluate_requests(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    request_chunk_size: int,
) -> Iterator[tuple[dict[str, Any], dict[str, int]]]:
    for start in range(0, len(rows), request_chunk_size):
        part = rows[start : start + request_chunk_size]
        evaluated, counts = _evaluate_chunk(model, tokenizer, part, batch_size=batch_size)
        for record in evaluated:
            yield record, counts
            counts = {key: 0 for key in counts}


def evaluator_parity_gate(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    batch_size: int,
) -> dict[str, Any]:
    if len(rows) != 3:
        raise FinalWeightBoundary("parity gate requires exact three requests")
    observed, counts = _evaluate_chunk(model, tokenizer, rows, batch_size=batch_size)
    max_abs = 0.0
    strict_equal = True
    for row, record in zip(rows, observed, strict=True):
        for category, prompts, target in _category_specs(row):
            expected = sequence_metrics(model, tokenizer, list(prompts), target)
            actual = record["metrics"][category]["prompts"]
            for index, item in enumerate(actual):
                max_abs = max(
                    max_abs,
                    abs(float(item["nll"]) - float(expected["nll"][index])),
                    abs(float(item["margin"]) - float(expected["margin"][index])),
                )
                strict_equal = strict_equal and bool(item["strict"]) == bool(expected["strict"][index])
    reversed_rows = list(reversed(rows))
    reordered, _ = _evaluate_chunk(model, tokenizer, reversed_rows, batch_size=batch_size)
    by_hash = {record["request_sha256"]: record for record in observed}
    reorder_identity = all(
        by_hash[record["request_sha256"]]["metrics"] == record["metrics"]
        for record in reordered
    )
    if max_abs > tolerance or not strict_equal or not reorder_identity:
        raise FinalWeightBoundary(
            f"evaluator parity failed: max_abs={max_abs}, strict={strict_equal}, reorder={reorder_identity}"
        )
    return {
        "status": "PASS",
        "request_count": 3,
        "category_count": len(CATEGORIES),
        "maximum_nll_or_margin_absolute_error": max_abs,
        "strict_identity": strict_equal,
        "batch_reorder_identity": reorder_identity,
        "tolerance": tolerance,
        "counts": counts,
        "raw_prompt_logit_generation_publish_count": 0,
    }


__all__ = [
    "CATEGORIES",
    "age_stratum",
    "evaluate_requests",
    "evaluator_parity_gate",
    "load_stream_and_rows",
    "sha256_file",
]
