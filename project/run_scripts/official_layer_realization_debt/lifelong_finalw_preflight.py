"""Create-once availability amendment and checkpoint identity seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash

from .lifelong_finalw_contracts import (
    ABSENT_EXACT_CHECKPOINTS,
    AMENDED_CHECKPOINTS,
    ARM_ROOTS,
    DATASET,
    DATASET_SHA256,
    EASYEDIT_HEAD,
    EASYEDIT_ROOT,
    EASYEDIT_TREE,
    EVALUATOR_IDENTITY,
    EvaluationLock,
    FinalWeightBoundary,
    INSTRUCTION_ID,
    LAYERS,
    ORDER_ROOT,
    ORIGINAL_REQUESTED_CHECKPOINTS,
    SCHEMA,
    STREAM_ROOT,
    STREAM_SEAL,
    STREAM_SEAL_SHA256,
    V3_MANIFEST_SHA256,
    V3_RECEIPT_SHA256,
    V3_REPORT_ROOT,
    V3_REPORT_SHA256,
)
from .lifelong_finalw_evaluation import load_stream_and_rows, sha256_file


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FinalWeightBoundary(f"not a regular non-symlink: {path}")
    return info


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def tensor_sha256_chunked(tensor: torch.Tensor, chunk_elements: int = 16 * 1024 * 1024) -> str:
    logical = tensor.detach().to(device="cpu").contiguous().view(-1)
    digest = hashlib.sha256()
    for start in range(0, logical.numel(), chunk_elements):
        block = logical[start : start + chunk_elements].view(torch.uint8).numpy().tobytes(order="C")
        digest.update(block)
    return digest.hexdigest()


def _source_identity(root: Path, expected_head: str) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if head != expected_head or dirty:
        raise FinalWeightBoundary("source HEAD/tracked-clean identity differs")
    return {"root": str(root.resolve()), "head": head, "tree": tree, "tracked_clean": True}


def _easyedit_identity() -> dict[str, Any]:
    if (
        _git(EASYEDIT_ROOT, "rev-parse", "HEAD") != EASYEDIT_HEAD
        or _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}") != EASYEDIT_TREE
        or _git(EASYEDIT_ROOT, "status", "--porcelain", "--untracked-files=no")
    ):
        raise FinalWeightBoundary("pinned stock EasyEdit identity differs")
    return {
        "root": str(EASYEDIT_ROOT),
        "head": EASYEDIT_HEAD,
        "tree": EASYEDIT_TREE,
        "tracked_clean": True,
    }


def _v3_identity() -> dict[str, Any]:
    paths = {
        "report": V3_REPORT_ROOT / "official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md",
        "manifest": V3_REPORT_ROOT / "analysis-manifest.json",
        "receipt": V3_REPORT_ROOT / "rooted-analysis-receipt.json",
    }
    expected = {
        "report": V3_REPORT_SHA256,
        "manifest": V3_MANIFEST_SHA256,
        "receipt": V3_RECEIPT_SHA256,
    }
    members = {}
    for key, path in paths.items():
        info = _regular(path)
        digest = sha256_file(path)
        if digest != expected[key]:
            raise FinalWeightBoundary(f"immutable v3 {key} SHA differs")
        members[key] = {"path": str(path), "bytes": info.st_size, "sha256": digest}
    return members


def _checkpoint_record(
    *, model: str, method: str, root: Path, count: int, result_entry: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint_path = root / "checkpoints" / f"checkpoint-{count:05d}.json"
    state_path = root / "checkpoints" / f"state-{count:05d}.pt"
    if Path(result_entry["path"]).resolve() != checkpoint_path.resolve():
        raise FinalWeightBoundary("result/checkpoint path binding differs")
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state_entry = checkpoint.get("state", {})
    if (
        checkpoint_sha != str(result_entry["sha256"])
        or Path(state_entry.get("path", "")).resolve() != state_path.resolve()
        or int(checkpoint.get("accepted_edit_count", -1)) != count
        or int(checkpoint.get("batch_index", -1)) != count // 100
    ):
        raise FinalWeightBoundary("checkpoint/result metadata differs")
    state_info = _regular(state_path)
    state_sha = sha256_file(state_path)
    if state_sha != str(state_entry.get("sha256")) or state_info.st_size != int(state_entry.get("bytes", -1)):
        raise FinalWeightBoundary("checkpoint state file identity differs")
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=False)
    expected_names = {
        f"model.layers.{layer}.mlp.down_proj.weight" for layer in LAYERS
    }
    weights = state.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != expected_names:
        raise FinalWeightBoundary("checkpoint edited-weight inventory differs")
    weight_members: list[dict[str, Any]] = []
    for name in sorted(weights):
        tensor = weights[name]
        if tensor.dtype is not torch.float32 or tensor.ndim != 2:
            raise FinalWeightBoundary("checkpoint edited tensor is not rank-two FP32")
        digest = tensor_sha256_chunked(tensor)
        if digest != state.get("weight_sha256", {}).get(name):
            raise FinalWeightBoundary("checkpoint edited tensor SHA differs")
        weight_members.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest,
                "bytes": tensor.numel() * tensor.element_size(),
            }
        )
    if (
        int(state.get("accepted_edit_count", -1)) != count
        or int(state.get("batch_index", -1)) != count // 100
        or state.get("stream_root") != STREAM_ROOT
        or state.get("order_root") != ORDER_ROOT
        or state.get("evaluator_identity") != EVALUATOR_IDENTITY
    ):
        raise FinalWeightBoundary("checkpoint embedded contract differs")
    cache = state.get("cache", {})
    if method == "alphaedit":
        tensor = cache.get("tensor")
        if (
            cache.get("kind") != "DYNAMIC_ALPHAEDIT_CACHE_C"
            or cache.get("present") is not True
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype is not torch.float32
            or list(tensor.shape) != list(cache.get("shape", ()))
            or tensor.shape[0] != len(LAYERS)
        ):
            raise FinalWeightBoundary("AlphaEdit checkpoint cache binding differs")
        cache_digest = tensor_sha256_chunked(tensor)
        if cache_digest != cache.get("sha256"):
            raise FinalWeightBoundary("AlphaEdit checkpoint cache tensor SHA differs")
        cache_binding = {
            "kind": cache["kind"],
            "present": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": cache_digest,
            "loaded_into_evaluation_runtime": False,
        }
    else:
        entries = cache.get("entries")
        if (
            cache.get("kind") != "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE"
            or cache.get("request_history_width") != 0
            or not isinstance(entries, list)
            or len(entries) != len(LAYERS)
            or any(row[2] != "torch.float32" for row in entries)
        ):
            raise FinalWeightBoundary("MEMIT covariance binding differs")
        cache_binding = {
            "kind": cache["kind"],
            "identity": str(cache.get("identity")),
            "entry_count": len(entries),
            "request_history_width": 0,
            "entries_contract_sha256": canonical_hash(
                [[row[0], row[1], row[2], row[4]] for row in entries]
            ),
            "loaded_into_evaluation_runtime": False,
        }
    source = state.get("source", {})
    answer = {
        "model": model,
        "method": method,
        "accepted_edit_count": count,
        "batch_index": count // 100,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha,
        },
        "state": {"path": str(state_path), "bytes": state_info.st_size, "sha256": state_sha},
        "edited_weights": weight_members,
        "edited_weight_identity_sha256": canonical_hash(weight_members),
        "cache_or_covariance_binding": cache_binding,
        "journal_chain_root": str(state.get("journal_chain_root")),
        "science_source": source,
    }
    del state
    return answer


def build_lock(source_root: Path, expected_head: str) -> dict[str, Any]:
    source = _source_identity(source_root, expected_head)
    easyedit = _easyedit_identity()
    if sha256_file(STREAM_SEAL) != STREAM_SEAL_SHA256:
        raise FinalWeightBoundary("stream seal file SHA differs")
    seal, rows = load_stream_and_rows(STREAM_SEAL, DATASET)
    if len(rows) != 10_000:
        raise FinalWeightBoundary("evaluated stream denominator differs")
    checkpoints: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    for model, method, raw_root in ARM_ROOTS:
        root = Path(raw_root)
        result_path = root / "result.json"
        result_sha = sha256_file(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "TERMINAL_PASS"
            or result.get("model") != model
            or result.get("method") != method
            or result.get("valid_batch_denominator") != 100
            or result.get("valid_request_denominator") != 10_000
            or result.get("nonfinite_count") != 0
            or result.get("rollback_violation_count") != 0
            or result.get("target_recomputation_count") != 0
        ):
            raise FinalWeightBoundary(f"lifelong terminal contract differs: {model}/{method}")
        indexed = {
            int(value["accepted_edit_count"]): value for value in result.get("checkpoints", ())
        }
        missing_requested = []
        for count in ABSENT_EXACT_CHECKPOINTS:
            paths = (
                root / "checkpoints" / f"checkpoint-{count:05d}.json",
                root / "checkpoints" / f"state-{count:05d}.pt",
            )
            if any(path.exists() or path.is_symlink() for path in paths) or count in indexed:
                raise FinalWeightBoundary("declared absent exact checkpoint unexpectedly exists")
            missing_requested.append(count)
        for count in AMENDED_CHECKPOINTS:
            if count not in indexed:
                raise FinalWeightBoundary("amended exact checkpoint is absent")
            checkpoints.append(
                _checkpoint_record(
                    model=model,
                    method=method,
                    root=root,
                    count=count,
                    result_entry=indexed[count],
                )
            )
        arms.append(
            {
                "model": model,
                "method": method,
                "job_id": str(result.get("job_id", result.get("job", "BOUND_BY_V3_MANIFEST"))),
                "raw_root": str(root),
                "result": {
                    "path": str(result_path),
                    "bytes": result_path.stat().st_size,
                    "sha256": result_sha,
                },
                "science_source": result.get("source"),
                "missing_original_requested_checkpoints": missing_requested,
            }
        )
    checkpoints.sort(key=lambda value: (value["model"], value["method"], value["accepted_edit_count"]))
    checkpoint_root = canonical_hash(checkpoints)
    return {
        "schema": f"{SCHEMA}.availability-amendment",
        "instruction_id": INSTRUCTION_ID,
        "status": "EXACT_STORED_CHECKPOINT_SCHEDULE_PASS",
        "amendment": {
            "class": "OUTCOME_BLIND_EXACT_STATE_AVAILABILITY_AMENDMENT",
            "original_requested": list(ORIGINAL_REQUESTED_CHECKPOINTS),
            "exact_stored_schedule": list(AMENDED_CHECKPOINTS),
            "absent_exact_states": list(ABSENT_EXACT_CHECKPOINTS),
            "edit_replay_count": 0,
            "interpolation_count": 0,
            "nearest_substitution_count": 0,
        },
        "source": source,
        "easyedit": easyedit,
        "stream": {
            "seal_path": str(STREAM_SEAL),
            "seal_sha256": STREAM_SEAL_SHA256,
            "root": seal["root_digest"],
            "order": seal["training_order_sha256"],
            "request_count": len(rows),
            "rephrase_prompt_count": sum(len(row["paraphrase_prompts"]) for row in rows),
            "locality_prompt_count": sum(len(row["locality_prompts"]) for row in rows),
            "sample_duplication_count": 0,
            "dataset_path": str(DATASET),
            "dataset_sha256": DATASET_SHA256,
            "evaluator_identity": EVALUATOR_IDENTITY,
        },
        "evaluation_lock": EvaluationLock().payload(),
        "arms": arms,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "checkpoint_member_root": checkpoint_root,
        "immutable_v3": _v3_identity(),
        "raw_outcomes_opened_for_schedule_decision": 0,
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_lock(args.source_root.resolve(), args.expected_head)
    payload["identity_sha256"] = canonical_hash(payload)
    digest = _atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "path": str(args.output), "sha256": digest, "identity_sha256": payload["identity_sha256"], "checkpoint_count": payload["checkpoint_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
