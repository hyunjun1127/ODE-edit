"""Read-only availability audit for the lifelong CounterFact v6 correction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_counterfact_contracts import (
    CANONICAL_COUNTERFACT_EVALUATOR,
    CANONICAL_COUNTERFACT_EVALUATOR_SHA256,
    CANONICAL_COUNTERFACT_ROOT,
    CANONICAL_COUNTERFACT_SUMMARIZER,
    CANONICAL_COUNTERFACT_SUMMARIZER_SHA256,
    CounterFactMetricBoundary,
    CounterFactMetricLock,
    FINALW_EVALUATOR_SHA256,
    FINALW_INPUT_LOCK,
    FINALW_INPUT_LOCK_SHA256,
    FINALW_RESULT_ROOT,
    INSTRUCTION_ID,
    P1_EVALUATOR_SHA256,
    PINNED_EASYEDIT_EVALUATOR_SHA256,
    SCHEMA,
    V4_MANIFEST_SHA256,
    V4_RECEIPT_SHA256,
    V4_RELATIVE,
    V4_REPORT_SHA256,
    V5_MANIFEST_SHA256,
    V5_RECEIPT_SHA256,
    V5_RELATIVE,
    V5_REPORT_SHA256,
)
from .lifelong_finalw_contracts import (
    AMENDED_CHECKPOINTS,
    DATASET,
    EASYEDIT_ROOT,
    ORDER_ROOT,
    STREAM_ROOT,
    STREAM_SEAL,
    cell_mapping,
)
from .lifelong_finalw_evaluation import age_stratum, sha256_file


EXPECTED_CATEGORIES = {
    "rewrite_target_new",
    "rewrite_target_true",
    "rephrase_target_new",
    "rephrase_target_true",
    "locality_target_true",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CounterFactMetricBoundary(f"not a regular non-symlink: {path}")
    return info


def _identity(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    expected = payload.pop("identity_sha256", None)
    if expected != canonical_hash(payload):
        raise CounterFactMetricBoundary(f"canonical identity differs: {label}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_member(path: Path, expected: str, label: str) -> dict[str, Any]:
    info = _regular(path)
    digest = sha256_file(path)
    if digest != expected:
        raise CounterFactMetricBoundary(f"{label} SHA differs")
    return {"label": label, "path": str(path), "bytes": info.st_size, "sha256": digest}


def _historical_package(source_root: Path, relative: Path, expected: tuple[str, str, str]) -> dict[str, Any]:
    report_name = (
        "official-layer-debt-lifelong-finalw-full10k-factual-ko.md"
        if relative == V4_RELATIVE
        else "official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md"
    )
    names = (report_name, "analysis-manifest.json", "rooted-analysis-receipt.json")
    members = []
    for name, digest in zip(names, expected, strict=True):
        path = source_root / relative / name
        info = _regular(path)
        actual = sha256_file(path)
        if actual != digest:
            raise CounterFactMetricBoundary(f"immutable historical package SHA differs: {path}")
        members.append({"path": str(path), "bytes": info.st_size, "sha256": actual})
    return {"root": str(relative), "anchors": members}


def _stream_hashes() -> tuple[str, ...]:
    seal = json.loads(STREAM_SEAL.read_text(encoding="utf-8"))
    root = seal.pop("root_digest", None)
    if root != STREAM_ROOT or canonical_hash(seal) != STREAM_ROOT:
        raise CounterFactMetricBoundary("stream root differs")
    rows = seal.get("training")
    if not isinstance(rows, list) or len(rows) != 10_000:
        raise CounterFactMetricBoundary("stream denominator differs")
    hashes = tuple(str(row["request_sha256"]) for row in rows)
    if canonical_hash(list(hashes)) != ORDER_ROOT:
        raise CounterFactMetricBoundary("stream order root differs")
    return hashes


def build(source_root: Path, expected_head: str, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise CounterFactMetricBoundary(f"refusing existing audit receipt: {output}")
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    if head != expected_head or _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise CounterFactMetricBoundary("audit source identity differs")
    sources = [
        _source_member(
            EASYEDIT_ROOT / "easyeditor/evaluate/evaluate.py",
            PINNED_EASYEDIT_EVALUATOR_SHA256,
            "PINNED_EASYEDIT_GENERIC_EVALUATOR",
        ),
        _source_member(
            CANONICAL_COUNTERFACT_EVALUATOR,
            CANONICAL_COUNTERFACT_EVALUATOR_SHA256,
            "CANONICAL_COUNTERFACT_EVALUATOR",
        ),
        _source_member(
            CANONICAL_COUNTERFACT_SUMMARIZER,
            CANONICAL_COUNTERFACT_SUMMARIZER_SHA256,
            "CANONICAL_COUNTERFACT_SUMMARIZER",
        ),
        _source_member(
            source_root / "project/run_scripts/ode_bf/p1_evaluator.py",
            P1_EVALUATOR_SHA256,
            "REPO_COUNTERFACT_PRIMARY_EVALUATOR",
        ),
        _source_member(
            source_root / "project/run_scripts/official_layer_realization_debt/lifelong_finalw_evaluation.py",
            FINALW_EVALUATOR_SHA256,
            "SEALED_V4_FINALW_EVALUATOR",
        ),
    ]
    if _git(CANONICAL_COUNTERFACT_ROOT, "rev-parse", "HEAD") != "6e3ba9a87978a23b6db2676ba7540216bf5d4fd0":
        raise CounterFactMetricBoundary("canonical CounterFact source HEAD differs")
    if sha256_file(FINALW_INPUT_LOCK) != FINALW_INPUT_LOCK_SHA256:
        raise CounterFactMetricBoundary("final-W input lock SHA differs")
    lock = json.loads(FINALW_INPUT_LOCK.read_text(encoding="utf-8"))
    _identity(lock, "final-W input lock")
    stream_hashes = _stream_hashes()
    historical = [
        _historical_package(
            source_root,
            V4_RELATIVE,
            (V4_REPORT_SHA256, V4_MANIFEST_SHA256, V4_RECEIPT_SHA256),
        ),
        _historical_package(
            source_root,
            V5_RELATIVE,
            (V5_REPORT_SHA256, V5_MANIFEST_SHA256, V5_RECEIPT_SHA256),
        ),
    ]
    members: list[dict[str, Any]] = []
    checkpoint_rows = 0
    rewrite_pair_prompts = 0
    rephrase_pair_prompts = 0
    locality_true_prompts = 0
    locality_new_prompts = 0
    strict_semantic_identity_count = 0
    for cell_index in range(4):
        model, method, _ = cell_mapping(cell_index)
        cell_root = FINALW_RESULT_ROOT / f"cell-{cell_index}"
        terminal_path = cell_root / "terminal.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        _identity(terminal, f"cell-{cell_index} terminal")
        if terminal.get("status") != "TERMINAL_PASS" or terminal.get("checkpoint_denominator") != 7:
            raise CounterFactMetricBoundary("final-W cell terminal differs")
        members.append(
            {
                "kind": "CELL_TERMINAL",
                "cell": cell_index,
                "path": str(terminal_path),
                "bytes": terminal_path.stat().st_size,
                "sha256": sha256_file(terminal_path),
            }
        )
        for count in AMENDED_CHECKPOINTS:
            checkpoint = cell_root / f"checkpoint-{count:05d}"
            receipt_path = checkpoint / "evaluation-receipt.json"
            records_path = checkpoint / "request-metrics.jsonl.gz"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _identity(receipt, f"cell-{cell_index}/{count} receipt")
            if (
                receipt.get("model") != model
                or receipt.get("method") != method
                or receipt.get("accepted_edit_count") != count
                or receipt.get("records", {}).get("rows") != count
                or receipt.get("evaluation_type") != "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS"
            ):
                raise CounterFactMetricBoundary("final-W checkpoint receipt binding differs")
            records_sha = sha256_file(records_path)
            if (
                records_sha != receipt["records"]["sha256"]
                or records_path.stat().st_size != receipt["records"]["bytes"]
            ):
                raise CounterFactMetricBoundary("final-W records member differs")
            members.extend(
                [
                    {
                        "kind": "CHECKPOINT_RECEIPT",
                        "cell": cell_index,
                        "accepted_edit_count": count,
                        "path": str(receipt_path),
                        "bytes": receipt_path.stat().st_size,
                        "sha256": sha256_file(receipt_path),
                    },
                    {
                        "kind": "CHECKPOINT_RECORDS",
                        "cell": cell_index,
                        "accepted_edit_count": count,
                        "path": str(records_path),
                        "bytes": records_path.stat().st_size,
                        "sha256": records_sha,
                        "rows": count,
                    },
                ]
            )
            observed_hashes: list[str] = []
            rows = 0
            with gzip.open(records_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    payload = dict(record)
                    identity = payload.pop("identity_sha256", None)
                    derived_age = payload.pop("age_stratum", None)
                    if identity != canonical_hash(payload):
                        raise CounterFactMetricBoundary("sealed record core identity differs")
                    if derived_age != age_stratum(int(record["ordinal"]), count):
                        raise CounterFactMetricBoundary("sealed record age derivation differs")
                    if set(record["metrics"]) != EXPECTED_CATEGORIES:
                        raise CounterFactMetricBoundary("sealed evaluator category inventory differs")
                    rewrite_new = record["metrics"]["rewrite_target_new"]["prompts"]
                    rewrite_true = record["metrics"]["rewrite_target_true"]["prompts"]
                    rephrase_new = record["metrics"]["rephrase_target_new"]["prompts"]
                    rephrase_true = record["metrics"]["rephrase_target_true"]["prompts"]
                    locality_true = record["metrics"]["locality_target_true"]["prompts"]
                    if not (
                        len(rewrite_new) == len(rewrite_true) == 1
                        and len(rephrase_new) == len(rephrase_true) == 2
                        and len(locality_true) == 10
                    ):
                        raise CounterFactMetricBoundary("sealed prompt cardinality differs")
                    for category in (rewrite_new, rewrite_true, rephrase_new, rephrase_true, locality_true):
                        for prompt in category:
                            if bool(prompt["strict"]) != (float(prompt["margin"]) > 0.0):
                                raise CounterFactMetricBoundary("teacher-forced strict semantics differs")
                            strict_semantic_identity_count += 1
                    observed_hashes.append(str(record["request_sha256"]))
                    rewrite_pair_prompts += 1
                    rephrase_pair_prompts += 2
                    locality_true_prompts += 10
                    locality_new_prompts += int("locality_target_new" in record["metrics"])
                    rows += 1
            if rows != count or tuple(observed_hashes) != stream_hashes[:count]:
                raise CounterFactMetricBoundary("sealed request row/order identity differs")
            checkpoint_rows += rows
    if checkpoint_rows != 120_000 or rewrite_pair_prompts != 120_000 or rephrase_pair_prompts != 240_000:
        raise CounterFactMetricBoundary("sealed reusable prompt-pair denominator differs")
    if locality_true_prompts != 1_200_000 or locality_new_prompts != 0:
        raise CounterFactMetricBoundary("locality availability classification differs")
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA}.availability-audit",
        "instruction_id": INSTRUCTION_ID,
        "status": "MISSING_REQUIRES_EVALUATION_ONLY_BACKFILL",
        "source": {"root": str(source_root), "head": head, "tree": tree, "tracked_clean": True},
        "metric_lock": CounterFactMetricLock().payload(),
        "source_audit": {
            "pinned_easyedit_head": _git(EASYEDIT_ROOT, "rev-parse", "HEAD"),
            "pinned_easyedit_tree": _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}"),
            "canonical_counterfact_head": _git(CANONICAL_COUNTERFACT_ROOT, "rev-parse", "HEAD"),
            "canonical_counterfact_tree": _git(CANONICAL_COUNTERFACT_ROOT, "rev-parse", "HEAD^{tree}"),
            "members": sources,
            "rewrite_rephrase_rule": "target_new_mean_nll < target_true_mean_nll",
            "locality_rule": "target_true_mean_nll < target_new_mean_nll",
            "tie_policy": "FAIL",
        },
        "historical_packages": historical,
        "sealed_evaluation": {
            "terminal_cells": 4,
            "checkpoint_receipts": 28,
            "request_state_rows": checkpoint_rows,
            "rewrite_pair_prompt_denominator": rewrite_pair_prompts,
            "rephrase_pair_prompt_denominator": rephrase_pair_prompts,
            "locality_target_true_prompt_denominator": locality_true_prompts,
            "locality_target_new_prompt_denominator": locality_new_prompts,
            "strict_semantic_checks": strict_semantic_identity_count,
            "rewrite_rephrase_prompt_order_pairing": "PASS_SAME_REQUEST_AND_PROMPT_INDEX",
            "rewrite_rephrase_reuse": "PASS_NO_REEVALUATION_REQUIRED",
            "locality_availability": "MISSING_REQUIRES_EVALUATION_ONLY_BACKFILL",
            "record_identity_scope": "CORE_RECORD_EXCLUDES_DERIVED_AGE_STRATUM; FULL_MEMBER_SHA_BINDS_BYTES",
        },
        "stream": {
            "root": STREAM_ROOT,
            "order": ORDER_ROOT,
            "request_count": len(stream_hashes),
            "dataset_path": str(DATASET),
        },
        "input_lock": {
            "path": str(FINALW_INPUT_LOCK),
            "bytes": FINALW_INPUT_LOCK.stat().st_size,
            "sha256": FINALW_INPUT_LOCK_SHA256,
            "identity_sha256": lock["identity_sha256"],
        },
        "raw_members": members,
        "raw_member_root": canonical_hash(members),
        "edit_replay_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    digest = _atomic_json(output, payload)
    return {
        "status": payload["status"],
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": digest,
        "identity_sha256": payload["identity_sha256"],
        "raw_member_root": payload["raw_member_root"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.expected_head, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
