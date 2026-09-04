"""Evaluation-only canonical CounterFact NS backfill at round-0 W0.

This module intentionally performs no edit, target optimization, key capture,
solve, cache/history update, or writer call.  It evaluates both target-new and
target-true continuations on the ten neighborhood prompts of each request in
the sealed first B100 cohort, then applies the canonical strict NS predicate
``target_true_nll < target_new_nll`` (ties fail).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import (
    LocalityPromptTarget,
    evaluate_prompt_targets,
)
from project.run_scripts.ordered_response_barrier_ode.contracts import assert_full_fp32
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.preflight import (
    DATASET_RELATIVE,
    MODEL_BINDINGS,
    ORDER_ROOT,
    STREAM_ROOT,
    STREAM_SEAL_RELATIVE,
    canonical_hash,
    sha256_file,
    validate_source_checkout,
    validate_stream,
)
from project.run_scripts.ordered_response_barrier_ode.runtime import _load_stream


INSTRUCTION_ID = "ODEEDIT-S06-ORRBODE-ROUND0-PREEDIT-CANONICAL-NS-BACKFILL-V1"
SCHEMA = "odeedit.s06.orbode.round0-preedit-canonical-ns.v1"
MODEL_ORDER = ("llama3-8b-inst", "qwen2.5-7b-inst")
REQUEST_COUNT = 100
PROMPTS_PER_REQUEST = 10
PROMPT_PAIR_COUNT = REQUEST_COUNT * PROMPTS_PER_REQUEST
SELECTED_LAYERS = (4, 5, 6, 7, 8)
SELECTED_WEIGHT_TEMPLATE = "model.layers.{}.mlp.down_proj.weight"


class PreEditNSBoundary(RuntimeError):
    """Fail-closed evaluation or provenance boundary."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise PreEditNSBoundary(f"create-once JSON boundary differs: {path}")
    raw = (_canonical_json(dict(payload)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o600:
        raise PreEditNSBoundary(f"create-once JSON type/mode differs: {path}")
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl_gzip_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise PreEditNSBoundary(f"create-once gzip boundary differs: {path}")
    plain = b"".join((_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
        handle.write(plain)
    raw = buffer.getvalue()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o600:
        raise PreEditNSBoundary(f"create-once gzip type/mode differs: {path}")
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise PreEditNSBoundary(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device("cuda:0"))


def _stat(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != PROMPT_PAIR_COUNT or not np.isfinite(array).all():
        raise PreEditNSBoundary("canonical NS distribution is incomplete or nonfinite")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def build_locality_pairs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[LocalityPromptTarget], list[LocalityPromptTarget]]:
    """Build aligned new/true neighborhood prompt pairs without reordering."""

    new_pairs: list[LocalityPromptTarget] = []
    true_pairs: list[LocalityPromptTarget] = []
    if len(records) != REQUEST_COUNT:
        raise PreEditNSBoundary("round0 pre-edit request denominator differs")
    for record in records:
        rewrite = record.get("requested_rewrite")
        prompts = record.get("neighborhood_prompts")
        if not isinstance(rewrite, Mapping) or not isinstance(prompts, list):
            raise PreEditNSBoundary("CounterFact locality input schema differs")
        if len(prompts) != PROMPTS_PER_REQUEST:
            raise PreEditNSBoundary("CounterFact locality prompt denominator differs")
        target_new = rewrite.get("target_new")
        target_true = rewrite.get("target_true")
        if not isinstance(target_new, Mapping) or not isinstance(target_true, Mapping):
            raise PreEditNSBoundary("CounterFact target schema differs")
        case_id = int(record["case_id"])
        for prompt_index, prompt in enumerate(prompts):
            prompt_text = str(prompt)
            new_pairs.append(
                LocalityPromptTarget(case_id, "locality_target_new", prompt_index, prompt_text, str(target_new["str"]))
            )
            true_pairs.append(
                LocalityPromptTarget(case_id, "locality_target_true", prompt_index, prompt_text, str(target_true["str"]))
            )
    if len(new_pairs) != PROMPT_PAIR_COUNT or len(true_pairs) != PROMPT_PAIR_COUNT:
        raise PreEditNSBoundary("canonical NS prompt pair count differs")
    return new_pairs, true_pairs


def reduce_locality_rows(
    new_rows: Sequence[Mapping[str, Any]],
    true_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove raw text and apply the strict canonical locality predicate."""

    if len(new_rows) != PROMPT_PAIR_COUNT or len(true_rows) != PROMPT_PAIR_COUNT:
        raise PreEditNSBoundary("evaluated locality pair denominator differs")
    if len(request_rows) != REQUEST_COUNT:
        raise PreEditNSBoundary("sealed request binding denominator differs")
    request_by_case = {
        int(row["case_id"]): (ordinal, row)
        for ordinal, row in enumerate(request_rows)
    }
    if len(request_by_case) != REQUEST_COUNT:
        raise PreEditNSBoundary("sealed request case identity is not unique")
    reduced: list[dict[str, Any]] = []
    for new, true in zip(new_rows, true_rows, strict=True):
        left = (int(new["case_id"]), int(new["prompt_index"]), str(new["prompt"]))
        right = (int(true["case_id"]), int(true["prompt_index"]), str(true["prompt"]))
        if left != right or new.get("kind") != "locality_target_new" or true.get("kind") != "locality_target_true":
            raise PreEditNSBoundary("locality target-new/target-true alignment differs")
        sealed_binding = request_by_case.get(left[0])
        if sealed_binding is None:
            raise PreEditNSBoundary("locality case is outside sealed round0")
        request_ordinal, sealed = sealed_binding
        new_nll = float(new["nll"])
        true_nll = float(true["nll"])
        if not math.isfinite(new_nll) or not math.isfinite(true_nll):
            raise PreEditNSBoundary("locality NLL is nonfinite")
        payload: dict[str, Any] = {
            "request_ordinal": request_ordinal,
            "case_id": left[0],
            "request_sha256": str(sealed["request_sha256"]),
            "prompt_index": left[1],
            "prompt_sha256": hashlib.sha256(left[2].encode("utf-8")).hexdigest(),
            "target_new_sha256": hashlib.sha256(str(new["target"]).encode("utf-8")).hexdigest(),
            "target_true_sha256": hashlib.sha256(str(true["target"]).encode("utf-8")).hexdigest(),
            "target_new_token_count": len(new["target_token_ids"]),
            "target_true_token_count": len(true["target_token_ids"]),
            "target_new_nll": new_nll,
            "target_true_nll": true_nll,
            "nll_advantage_new_minus_true": new_nll - true_nll,
            "ns_success": int(true_nll < new_nll),
            "nll_tie": int(true_nll == new_nll),
            "target_new_all_tokens_correct": int(bool(new["all_tokens_correct"])),
            "target_true_all_tokens_correct": int(bool(true["all_tokens_correct"])),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        reduced.append(payload)
    keys = [(row["request_ordinal"], row["prompt_index"]) for row in reduced]
    if len(set(keys)) != PROMPT_PAIR_COUNT or keys != sorted(keys):
        raise PreEditNSBoundary("canonical NS row order or uniqueness differs")
    successes = [int(row["ns_success"]) for row in reduced]
    new_values = [float(row["target_new_nll"]) for row in reduced]
    true_values = [float(row["target_true_nll"]) for row in reduced]
    advantages = [float(row["nll_advantage_new_minus_true"]) for row in reduced]
    summary: dict[str, Any] = {
        "definition": "target_true_nll < target_new_nll; ties fail",
        "request_count": REQUEST_COUNT,
        "prompts_per_request": PROMPTS_PER_REQUEST,
        "prompt_pair_count": PROMPT_PAIR_COUNT,
        "ns_numerator": sum(successes),
        "ns_denominator": len(successes),
        "ns_rate": float(sum(successes) / len(successes)),
        "nll_tie_count": sum(int(row["nll_tie"]) for row in reduced),
        "target_new_nll": _stat(new_values),
        "target_true_nll": _stat(true_values),
        "nll_advantage_new_minus_true": _stat(advantages),
        "target_new_all_tokens_correct_count": sum(int(row["target_new_all_tokens_correct"]) for row in reduced),
        "target_true_all_tokens_correct_count": sum(int(row["target_true_all_tokens_correct"]) for row in reduced),
        "row_identity_root": canonical_hash([row["identity_sha256"] for row in reduced]),
    }
    summary["identity_sha256"] = canonical_hash(summary)
    return reduced, summary


def build_plan() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA}.plan",
        "array": "0-1%2",
        "mapping": {"0": MODEL_ORDER[0], "1": MODEL_ORDER[1]},
        "request_count_per_cell": REQUEST_COUNT,
        "locality_prompt_pair_count_per_cell": PROMPT_PAIR_COUNT,
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "round_indices": [0],
        "entry_state": "EXACT_MODEL_W0_NO_EDIT_COLD_STATE",
        "ns_definition": "target_true_nll < target_new_nll; ties fail",
        "model_load_count_per_cell": 1,
        "editor_run_count": 0,
        "compute_z_count": 0,
        "writer_count": 0,
        "key_capture_count": 0,
        "solve_count": 0,
        "cache_history_mutation_count": 0,
        "model_update_count": 0,
        "gpu_per_cell": 1,
        "project_gpu_cap": 2,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _snapshot(hf_hub_cache: Path, alias: str) -> Path:
    spec = MODEL_BINDINGS[alias]
    return (
        hf_hub_cache
        / str(spec["hf_repo"])
        / "snapshots"
        / str(spec["revision"])
    ).resolve(strict=True)


def _model_metadata(model: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            int(parameter.data_ptr()),
            int(parameter._version),
            tuple(parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
        )
        for name, parameter in model.named_parameters()
    )


def _selected(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    named = dict(model.named_parameters())
    result = {
        SELECTED_WEIGHT_TEMPLATE.format(layer): named[SELECTED_WEIGHT_TEMPLATE.format(layer)]
        for layer in SELECTED_LAYERS
    }
    if len({int(value.data_ptr()) for value in result.values()}) != len(result):
        raise PreEditNSBoundary("selected editing weights alias storage")
    return result


def _selected_hashes(values: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(values.items())}


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.output_root.absolute()
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink():
        raise PreEditNSBoundary("pre-edit NS create-once result root differs")
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    failure_path = root / "failure-boundary.json"
    model: torch.nn.Module | None = None
    before_metadata: tuple[tuple[Any, ...], ...] | None = None
    before_hashes: dict[str, str] | None = None
    stage = "RESULT_ROOT_CREATED"
    action_counts = {
        "model_load": 0,
        "gpu_job": 1,
        "editor_run": 0,
        "compute_z": 0,
        "writer": 0,
        "key_capture": 0,
        "solve": 0,
        "cache_history_mutation": 0,
        "model_update": 0,
        "slurm_submit_inside_job": 0,
    }
    try:
        if args.cell_id not in range(len(MODEL_ORDER)):
            raise PreEditNSBoundary("pre-edit NS cell mapping differs")
        alias = MODEL_ORDER[args.cell_id]
        expected_token = f"s06-orbode-round0-preedit-ns-{alias}-v1"
        if args.run_token != expected_token:
            raise PreEditNSBoundary("pre-edit NS run token differs")
        repo_root = args.repo_root.absolute()
        source = validate_source_checkout(repo_root, args.source_head, args.source_tree)
        tracked_binding = _git(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "project/run_scripts/ordered_response_barrier_ode/preedit_ns_backfill.py",
            "project/run_scripts/session06_orbode_preedit_ns_backfill.sbatch",
        )
        if tracked_binding:
            raise PreEditNSBoundary("pre-edit NS execution source has a dirty shadow")
        stream = validate_stream(repo_root, args.easyedit_artifact_root)
        stage = "SOURCE_STREAM_PASS"
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PreEditNSBoundary("pre-edit NS requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_DISABLED": "true",
            }
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = _snapshot(args.hf_hub_cache, alias)
        spec = MODEL_BINDINGS[alias]
        config = (snapshot / "config.json").resolve(strict=True)
        tokenizer_file = (snapshot / "tokenizer.json").resolve(strict=True)
        if sha256_file(config) != spec["config_sha256"] or sha256_file(tokenizer_file) != spec["tokenizer_sha256"]:
            raise PreEditNSBoundary("HF config/tokenizer identity differs")
        _sync()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()
        _sync()
        model_load_seconds = time.perf_counter() - load_started
        action_counts["model_load"] = 1
        dtype_receipt = assert_full_fp32(model)
        if tokenizer.padding_side != "right":
            raise PreEditNSBoundary("pre-edit tokenizer writer-compatible padding differs")
        torch.cuda.reset_peak_memory_stats()
        before_metadata = _model_metadata(model)
        selected = _selected(model)
        before_hashes = _selected_hashes(selected)
        stage = "MODEL_W0_FP32_PASS"

        dataset = (args.easyedit_artifact_root / DATASET_RELATIVE).resolve(strict=True)
        seal, batches, raw_map = _load_stream(repo_root, dataset, (0,), wave="round0")
        batch = tuple(batches[0])
        records = tuple(raw_map[int(item["case_id"])] for item in batch)
        new_pairs, true_pairs = build_locality_pairs(records)
        _sync()
        evaluate_started = time.perf_counter()
        new_rows = evaluate_prompt_targets(model, tokenizer, new_pairs, device=torch.device("cuda:0"), microbatch_size=16)
        true_rows = evaluate_prompt_targets(model, tokenizer, true_pairs, device=torch.device("cuda:0"), microbatch_size=16)
        _sync()
        evaluator_seconds = time.perf_counter() - evaluate_started
        reduced, summary = reduce_locality_rows(new_rows, true_rows, batch)
        stage = "CANONICAL_NS_REDUCED"

        after_metadata = _model_metadata(model)
        after_hashes = _selected_hashes(selected)
        if before_metadata != after_metadata or before_hashes != after_hashes:
            raise PreEditNSBoundary("pre-edit evaluator mutated model pointer/version/bytes")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise PreEditNSBoundary("pre-edit evaluator populated parameter gradients")
        rows_path = root / "preedit-canonical-ns-prompt-pairs.jsonl.gz"
        rows_sha = _write_jsonl_gzip_once(rows_path, reduced)
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "status": "TERMINAL_VALID",
            "cell_id": args.cell_id,
            "model_alias": alias,
            "source": source,
            "source_head": args.source_head,
            "source_tree": args.source_tree,
            "stream_root": STREAM_ROOT,
            "order_root": ORDER_ROOT,
            "sealed_batch_order_digest": seal["batch_ordered_request_digest_v1"][0],
            "request_order_identity": canonical_hash([row["request_sha256"] for row in batch]),
            "round_index": 0,
            "entry_state": "EXACT_MODEL_W0_NO_EDIT_COLD_STATE",
            "summary": summary,
            "rows": {
                "relative_path": rows_path.name,
                "bytes": rows_path.stat().st_size,
                "mode": "0600",
                "sha256": rows_sha,
                "row_count": len(reduced),
                "raw_prompt_or_target_string_count": 0,
            },
            "model": {
                "hf_snapshot": str(snapshot),
                "revision": spec["revision"],
                "config_sha256": spec["config_sha256"],
                "tokenizer_sha256": spec["tokenizer_sha256"],
                "requested_dtype": "torch.float32",
                "loaded_dtype_receipt": dtype_receipt,
                "padding_side": tokenizer.padding_side,
                "autocast_enabled": False,
                "tf32_enabled": False,
                "quantization_count": 0,
                "bf16_fp16_cast_count": 0,
            },
            "integrity": {
                "all_parameter_pointer_version_shape_dtype_device_unchanged": True,
                "selected_weight_bytes_unchanged": True,
                "selected_weight_sha256": before_hashes,
                "parameter_gradient_count": 0,
                "finite_nll_count": 2 * PROMPT_PAIR_COUNT,
                "nonfinite_count": 0,
            },
            "compute": {
                "model_load_seconds": model_load_seconds,
                "evaluator_seconds": evaluator_seconds,
                "job_total_seconds": time.perf_counter() - started,
                "target_new_forward_microbatches": math.ceil(PROMPT_PAIR_COUNT / 16),
                "target_true_forward_microbatches": math.ceil(PROMPT_PAIR_COUNT / 16),
                "backward_count": 0,
                "generation_count": 0,
            },
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "action_counts": action_counts,
            "technical_failure_count": 0,
            "scientific_failure_count": 0,
            "imputation_count": 0,
            "scientific_promotion": False,
        }
        result["identity_sha256"] = canonical_hash(result)
        result_sha = _write_json_once(root / "result.json", result)
        manifest: dict[str, Any] = {
            "schema": f"{SCHEMA}.manifest",
            "result_sha256": result_sha,
            "rows_sha256": rows_sha,
            "members": [
                {"path": "preedit-canonical-ns-prompt-pairs.jsonl.gz", "sha256": rows_sha, "bytes": rows_path.stat().st_size, "mode": "0600"},
                {"path": "result.json", "sha256": result_sha, "bytes": (root / "result.json").stat().st_size, "mode": "0600"},
            ],
            "source_head": args.source_head,
            "source_tree": args.source_tree,
            "stream_root": STREAM_ROOT,
            "order_root": ORDER_ROOT,
            "model_alias": alias,
        }
        manifest["members_root"] = canonical_hash(manifest["members"])
        manifest["identity_sha256"] = canonical_hash(manifest)
        manifest_sha = _write_json_once(root / "manifest.json", manifest)
        receipt: dict[str, Any] = {
            "schema": f"{SCHEMA}.rooted-receipt",
            "status": "TERMINAL_VALID",
            "model_alias": alias,
            "result_sha256": result_sha,
            "manifest_sha256": manifest_sha,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "members_root": manifest["members_root"],
            "summary_identity_sha256": summary["identity_sha256"],
            "w0_pointer_version_bytes_unchanged": True,
            "editing_action_count": 0,
            "model_gpu_action_count": 1,
            "scientific_promotion": False,
        }
        receipt["identity_sha256"] = canonical_hash(receipt)
        _write_json_once(root / "rooted-receipt.json", receipt)
        stage = "TERMINAL_VALID"
        return result
    except BaseException as exc:
        after_guard: dict[str, Any] = {"available": False}
        if model is not None and before_metadata is not None:
            try:
                current_metadata = _model_metadata(model)
                current_hashes = _selected_hashes(_selected(model)) if before_hashes is not None else None
                after_guard = {
                    "available": True,
                    "parameter_metadata_unchanged": current_metadata == before_metadata,
                    "selected_weight_bytes_unchanged": current_hashes == before_hashes,
                }
            except BaseException as guard_exc:
                after_guard = {"available": True, "guard_exception": repr(guard_exc)}
        failure: dict[str, Any] = {
            "schema": f"{SCHEMA}.failure-boundary",
            "status": "TECHNICAL_INVALID",
            "stage": stage,
            "cell_id": args.cell_id,
            "source_head": args.source_head,
            "source_tree": args.source_tree,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_sha256": hashlib.sha256(traceback.format_exc().encode("utf-8")).hexdigest(),
            "w0_guard": after_guard,
            "action_counts": action_counts,
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "threshold_change_count": 0,
        }
        failure["identity_sha256"] = canonical_hash(failure)
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json_once(failure_path, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-plan", allow_abbrev=False)
    run = sub.add_parser("run-cell", allow_abbrev=False)
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--easyedit-artifact-root", type=Path, required=True)
    run.add_argument("--hf-hub-cache", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--source-head", required=True)
    run.add_argument("--source-tree", required=True)
    run.add_argument("--cell-id", type=int, choices=range(len(MODEL_ORDER)), required=True)
    run.add_argument("--run-token", required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-plan":
        print(_canonical_json(build_plan()))
        return 0
    result = run_cell(args)
    print(_canonical_json({"status": result["status"], "model_alias": result["model_alias"], "summary": result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
