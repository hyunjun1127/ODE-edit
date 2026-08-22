#!/usr/bin/env python3
"""Evaluate the sealed 10xB100 stream once at the common full-FP32 Llama W0."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.alpha_backend import seed_all
from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard
from project.run_scripts.ode_bf.contracts import COMMON_SEED, ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once
from project.run_scripts.ode_bf.p1r36_independent_b10x10_runtime import _hashes
from project.run_scripts.ode_bf.p1r52_b100x10_stream import (
    BATCH_SIZE,
    ROUND_COUNT,
    SEAL_FILE,
    load_p1r52_b100x10_batches,
    verify_p1r52_b100x10_stream,
)
from project.run_scripts.ode_bf.p1r52_joint_pc_independent_fp32_runtime import (
    full_fp32_parameter_inventory,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_phase_a_execution import (
    load_phase_a_fp32_model,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import _evaluate_batch_entry
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import _endpoint_summary


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-C-WRITER-PHASE1-W0-PREEDIT-V1"
RUN_TOKEN = "p1r52-phase1-llama3-w0-preedit-1000-tech-r1-v1"
RESULT_NAME = "s05-p1r52-phase1-llama3-w0-preedit-1000-tech-r1-v1"


def build_plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-phase1-w0-preedit-plan/v1",
        "model_load_count": 1,
        "model_alias": "llama3-8b-inst",
        "batch_count": ROUND_COUNT,
        "batch_size": BATCH_SIZE,
        "request_count": ROUND_COUNT * BATCH_SIZE,
        "physical_W_transition_count": 0,
        "target_generation_count": 0,
        "writer_call_count": 0,
        "router_call_count": 0,
        "pre_edit_evaluator_count": ROUND_COUNT,
        "controller_routing_writer_influence_count": 0,
        "full_fp32_required": True,
        "temporary_gpu_cap_exception_scope": "THIS_SINGLE_W0_PREEDIT_JOB_ONLY",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-token", required=True, choices=(RUN_TOKEN,))
    args = parser.parse_args(argv)
    if args.output_root.parent.resolve(strict=False) != (REPO_ROOT / "local/odebf/results").resolve(strict=False):
        raise ODEBFContractError("W0 pre-edit result parent differs")
    if args.output_root.name != RESULT_NAME or args.output_root.exists() or args.output_root.is_symlink():
        raise ODEBFContractError("W0 pre-edit create-once result differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip()
    if (head, tree) != (args.source_head, args.source_tree):
        raise ODEBFContractError("W0 pre-edit source identity differs")
    args.output_root.mkdir(mode=0o700, parents=True)
    raw_root = args.output_root / "raw"
    raw_root.mkdir(mode=0o700)
    started = time.perf_counter()
    locks = REPO_ROOT / "project/run_scripts/ode_bf/locks"
    guard = ODEBFArtifactGuard(REPO_ROOT, locks / "p0_artifact_lock.json", "llama3-8b-inst", require_held_ode_alloc=False)
    artifact_receipt = guard.preflight()
    stream = verify_p1r52_b100x10_stream(json.loads((locks / SEAL_FILE).read_text(encoding="utf-8")))
    batches = load_p1r52_b100x10_batches(guard.base_guard.dataset, stream)
    seed_all(COMMON_SEED)
    load_started = time.perf_counter()
    model, tokenizer, hparams, _runtime = load_phase_a_fp32_model(
        guard, "llama3-8b-inst", allow_python_patch_compatible=True
    )
    _sync()
    model_load_seconds = time.perf_counter() - load_started
    inventory = full_fp32_parameter_inventory(model)
    named = dict(model.named_parameters())
    touched = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": named[f"{hparams.rewrite_module_tmp.format(layer)}.weight"]
        for layer in hparams.layers
    }
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    entry_versions = {name: int(value._version) for name, value in touched.items()}
    batch_rows = []
    batch_receipt_sha256 = []
    for round_index, requests in enumerate(batches, start=1):
        _sync()
        receipt, evaluator_seconds = _evaluate_batch_entry(
            model,
            tokenizer,
            guard.base_guard.dataset,
            requests,
            alias="llama3-8b-inst",
            role="llama3-common-W0-pre-edit",
            round_index=round_index,
            entry_weight_sha256=entry_hashes,
            expected_batch_size=BATCH_SIZE,
        )
        _sync()
        if (
            _hashes(touched) != entry_hashes
            or any(int(touched[name].data_ptr()) != entry_pointers[name] for name in touched)
            or any(int(touched[name]._version) != entry_versions[name] for name in touched)
        ):
            raise ODEBFContractError("W0 pre-edit evaluator mutated model weights")
        payload = {
            "schema": "ode-edit-s05-p1r52-phase1-w0-preedit-batch/v1",
            "round": round_index,
            "request_order_sha256": stream["batch_ordered_request_digest_v1"][round_index - 1],
            "summary": _endpoint_summary(receipt),
            "scores": receipt,
            "evaluator_seconds": evaluator_seconds,
            "W0_weight_sha256": entry_hashes,
            "physical_W_transition_count": 0,
            "decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        batch_dir = raw_root / "batches" / f"b{round_index:02d}"
        batch_dir.mkdir(mode=0o700, parents=True)
        sha = _atomic_write_once(batch_dir / "pre-edit.json", payload)
        batch_receipt_sha256.append(sha)
        batch_rows.append({"round": round_index, "summary": payload["summary"], "receipt_sha256": sha})
    guard.assert_unchanged()
    terminal = {
        "schema": "ode-edit-s05-p1r52-phase1-w0-preedit-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "TERMINAL_VALID",
        "source_head": head,
        "source_tree": tree,
        "plan": build_plan(),
        "artifact_receipt": asdict(artifact_receipt),
        "stream_root": stream["root_digest"],
        "stream_order": stream["all_request_order_sha256"],
        "completed_batch_count": len(batch_rows),
        "valid_request_count": len(batch_rows) * BATCH_SIZE,
        "batch_rows": batch_rows,
        "batch_receipt_sha256": batch_receipt_sha256,
        "W0_weight_sha256": entry_hashes,
        "W0_unchanged": True,
        "full_fp32_parameter_inventory": inventory,
        "model_load_seconds": model_load_seconds,
        "job_total_seconds": time.perf_counter() - started,
        "target_generation_count": 0,
        "writer_call_count": 0,
        "router_call_count": 0,
        "physical_assignment_count": 0,
        "bf16_fp16_autocast_quantization_count": 0,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(args.output_root / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-phase1-w0-preedit-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "terminal_sha256": terminal_sha,
        "batch_receipt_sha256": batch_receipt_sha256,
        "stream_root": stream["root_digest"],
        "stream_order": stream["all_request_order_sha256"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(args.output_root / "manifest.json", manifest)
    print(json.dumps({"status": "TERMINAL_VALID", "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
