"""Same-B100 Official AlphaEdit/MEMIT native-z target-only diagnostics."""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _action_frozen_cases, _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_accepted_z_observation import (
    OfficialNativeZCapture,
    evaluate_accepted_z_batch,
)
from .p1r52_official_sequential_baselines import load_official_memit_hparams
from .p1r52_sequential_scale import P1R52SequentialScale
from .scalable_batched_runtime import scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-JOINT-PC-NATIVE-Z-B100-DIAGNOSTIC-V1"


def _native_inputs(role: str, hparams: Any) -> tuple[Any, Any, str]:
    from .p1r52_sequential_runtime import (
        JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE,
        JOINT_PC_MEMIT_NATIVE_Z_ROLE,
    )

    if role == JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE:
        from easyeditor.models.alphaedit import AlphaEdit_main as native_module

        return native_module, hparams, "native-alphaedit-sequential-cache-on-corrected"
    if role == JOINT_PC_MEMIT_NATIVE_Z_ROLE:
        from easyeditor.models.memit import memit_main as native_module

        return native_module, load_official_memit_hparams(), "official-memit-sequential"
    raise ODEBFContractError("Joint P+C native-z role differs")


def run_joint_pc_native_z_b100(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    destination: Path,
    raw_root: Path,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    contexts: Sequence[Sequence[str]],
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    job_ledger: ComputeLedger,
    scale: P1R52SequentialScale,
) -> dict[str, Any]:
    if (
        alias != "llama3-8b-inst"
        or scale.batch_size != 100
        or scale.round_count != 10
        or len(stream_batches) != 10
        or any(len(batch) != 100 for batch in stream_batches)
        or stream.get("root_digest") != "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
        or stream.get("all_request_order_sha256")
        != "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
    ):
        raise ODEBFContractError("Joint P+C native-z B100 stream/scope differs")
    entry_hashes = _hashes(touched)
    if entry_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Joint P+C native-z W0 differs")
    entry_contract = _model_w0_contract(touched)
    request_batch = tuple(stream_batches[0])
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in request_batch]
    )
    if request_order != "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6":
        raise ODEBFContractError("Joint P+C native-z B1 order differs")

    native_module, native_hparams, capture_role = _native_inputs(role, hparams)
    native_contexts = native_module.get_context_templates(model, tokenizer)
    context_receipt = {
        "native_context_sha256": canonical_hash(native_contexts),
        "p1r52_context_sha256": canonical_hash(list(contexts)),
        "native_context_used": True,
    }
    target_layer = int(native_hparams.layers[-1])
    capture = OfficialNativeZCapture(
        role=capture_role,
        requests=request_batch,
        hparams=native_hparams,
    )
    original_backward = torch.autograd.backward
    backward_count = 0

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal backward_count
        backward_count += 1
        job_ledger.increment("backward")
        job_ledger.increment("target_backward")
        return original_backward(*args, **kwargs)

    started = time.perf_counter()
    torch.autograd.backward = counted_backward
    try:
        with capture, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for request in request_batch:
                value = native_module.compute_z(
                    model,
                    tokenizer,
                    dict(request),
                    native_hparams,
                    target_layer,
                    native_contexts,
                )
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    raise ODEBFContractError("Joint P+C native-z compute_z return differs")
    finally:
        torch.autograd.backward = original_backward
    target_wall = time.perf_counter() - started
    binding = capture.finalize()
    if _hashes(touched) != entry_hashes or _model_w0_contract(touched) != entry_contract:
        raise ODEBFStateError("Joint P+C native-z target solve mutated W0")

    cases, _ = _action_frozen_cases(
        dataset_path,
        request_batch,
        arm=role,
        selected_snapshot_sha256=canonical_hash(entry_hashes),
        fixed_budget_slots_completed=0,
    )
    observation, evaluation_wall = evaluate_accepted_z_batch(
        model,
        tokenizer,
        request_batch,
        cases,
        role=capture_role,
        round_index=1,
        binding=binding,
        committed_weight_sha256=entry_hashes,
    )
    if _hashes(touched) != entry_hashes or _model_w0_contract(touched) != entry_contract:
        raise ODEBFStateError("Joint P+C native-z evaluator mutated W0")
    observation_path = raw_root / "case-01" / "accepted-z-rephrase-observation.json"
    observation_sha = _atomic_write_once(observation_path, observation)
    terminal = {
        "schema": "ode-edit-s05-p1r52-joint-pc-native-z-b100-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "source_head": source_head,
        "role": role,
        "native_capture_role": capture_role,
        "stream_root": stream["root_digest"],
        "stream_order": stream["all_request_order_sha256"],
        "b1_request_order_sha256": request_order,
        "request_count": 100,
        "accepted_z": binding.raw_free_payload(),
        "accepted_z_scores": observation["scores"],
        "context_receipt": context_receipt,
        "compute_z_call_count": 100,
        "target_backward_count": backward_count,
        "target_wall_seconds": target_wall,
        "evaluation_wall_seconds": evaluation_wall,
        "writer_apply_count": 0,
        "writer_materialization_count": 0,
        "persistent_weight_mutation_count": 0,
        "W0_entry_sha256": entry_hashes,
        "W0_terminal_sha256": _hashes(touched),
        "W0_restored": True,
        "accepted_z_observation_sha256": observation_sha,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-joint-pc-native-z-b100-manifest/v1",
        "source_head": source_head,
        "role": role,
        "terminal_sha256": terminal_sha,
        "observation_sha256": observation_sha,
        "request_count": 100,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_JOINT_PC_NATIVE_Z_B100_TERMINAL",
        "role": role,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = ["INSTRUCTION_ID", "run_joint_pc_native_z_b100"]
