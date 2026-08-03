"""Executable Session 02 P0 technical profiler (submission is external).

This runner has no Slurm submission capability and performs no scientific
evaluation.  It is invoked only by a separately approved scheduler envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

# Keep the tracked executable command valid without relying on an ambient
# PYTHONPATH.  The fixed EasyEdit path remains supplied by the execution
# envelope/sbatch template.
_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    CHECKPOINT_ORIGINAL_DTYPE_POLICY,
    load_fixed_model_checkpoint_original,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.contracts import EditRequest
from project.run_scripts.ode_edit_method.contracts import Arm, canonical_hash
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.easyedit_backend import EasyEditMemitBackend
from project.run_scripts.ode_edit_method.events import (
    build_allowed_contexts,
    build_teacher_batch,
    score_combined_teacher_batches,
    score_teacher_batch,
)
from project.run_scripts.ode_edit_method.hooks import TorchCheckpoint, base_weight_c_energy
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import LOCK_PATH, controller_config, load_lock
from project.run_scripts.ode_edit_method.preflight import (
    assert_static_lock_identities,
    preflight_static_inputs,
    prepare_concrete_environment,
)
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner


EASYEDIT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")
P0_ARMS = (
    Arm.NATIVE_MEMIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.ONE_REFRESH,
    Arm.FULL_ODE_EDIT,
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _jsonl_append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _require_fields(
    value: Mapping[str, Any], required: Sequence[str], *, label: str
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise RuntimeError(f"{label} lacks locked fields: {missing}")


def _strict_ratio(numerator: float, denominator: float, *, label: str) -> float:
    top = float(numerator)
    bottom = float(denominator)
    if not math.isfinite(top) or not math.isfinite(bottom) or top < 0.0 or bottom <= 0.0:
        raise RuntimeError(f"{label} ratio inputs are non-finite or degenerate")
    return top / bottom


def _require_output_root(repo: Path, candidate: Path) -> Path:
    root = candidate.resolve()
    allowed = (repo / "local" / "results").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"output must stay under {allowed}") from exc
    if not root.name.startswith("session02-"):
        raise ValueError("output directory name must start with session02-")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _receipt_payload(ledger: OmegaLedger) -> list[dict[str, Any]]:
    return [
        {
            "edit_id": receipt.edit_id,
            "terminal_net_energy": dict(receipt.terminal_net_energy),
            "increment": dict(receipt.increment),
            "receipt_id": receipt.receipt_id,
        }
        for receipt in ledger.receipts
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session02-compute-aware-p0",
        description="Execute one fixed-model P0 technical profiler locally",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-alias",
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
        required=True,
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise ValueError("P0 EasyEdit root differs from the fixed method runtime")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    lock = load_lock(args.lock)
    proposal_id = lock.pop("proposal_id")
    if lock["execution_boundary"]["gpu_now"] != 0:
        raise RuntimeError("tracked lock must not grant this runner GPU authority")
    if lock["execution_boundary"]["slurm_now"] is not False:
        raise RuntimeError("tracked lock unexpectedly grants Slurm authority")
    model_lock = lock["models"][args.model_alias]
    case_ids = tuple(lock["selection"]["p0_profiler_case_ids"])
    output_root = _require_output_root(repo, args.output_root)
    for filename in (
        "controller_steps.jsonl",
        "compute.jsonl",
        "evaluation.jsonl",
    ):
        (output_root / filename).touch(exist_ok=False)

    seed = int(lock["execution_seed"])
    seed_runtime(seed)
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        easyedit_root,
        model_alias=args.model_alias,
        case_ids=case_ids,
    )
    assert_static_lock_identities(
        easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=lock["selection"],
    )
    observed_request_ids = [
        EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
        ).request_id
        for request in requests
    ]
    expected_request_ids = lock["selection"]["request_ids"][: len(case_ids)]
    if observed_request_ids != expected_request_ids:
        raise RuntimeError("selected rewrite request identities differ from the lock")
    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_wall_seconds = time.perf_counter() - model_load_start
    runtime_metadata = runtime.metadata()
    if runtime_metadata.get("dtype_policy") != CHECKPOINT_ORIGINAL_DTYPE_POLICY:
        raise RuntimeError("method runtime did not use checkpoint-original dtype policy")
    if runtime_metadata.get("checkpoint_original_dtype") != "torch.bfloat16":
        raise RuntimeError("pinned checkpoint original dtype is not bfloat16")
    if runtime_metadata.get("observed_parameter_dtype") != "torch.bfloat16":
        raise RuntimeError("loaded method parameter dtype is not bfloat16")
    _require_fields(
        runtime_metadata,
        lock["artifact_schema"]["manifest_model_required_fields"],
        label="P0 model runtime metadata",
    )
    prepared = prepare_concrete_environment(
        easyedit_root,
        runtime=runtime,
        fixed_artifacts=fixed_artifacts,
        controller_requests=requests,
        expected_model_lock=model_lock,
        seed=seed,
    )
    event_batch_gates = []
    event_backend_lock = lock["event_backend"]
    for request in requests:
        contexts = build_allowed_contexts(request, prepared.contexts.templates)
        new_batch = build_teacher_batch(runtime.tokenizer, contexts, request.target_new)
        old_batch = build_teacher_batch(runtime.tokenizer, contexts, request.target_old)
        new_reference = score_teacher_batch(runtime.model, new_batch)
        old_reference = score_teacher_batch(runtime.model, old_batch)
        combined_new, combined_old = score_combined_teacher_batches(
            runtime.model,
            (new_batch, old_batch),
            differentiable=False,
        )
        reference_new_tensor = torch.tensor(
            new_reference, device=combined_new.device, dtype=combined_new.dtype
        )
        reference_old_tensor = torch.tensor(
            old_reference, device=combined_old.device, dtype=combined_old.dtype
        )
        atol = float(event_backend_lock["p0_two_forward_atol"])
        rtol = float(event_backend_lock["p0_two_forward_rtol"])
        if not (
            torch.allclose(combined_new, reference_new_tensor, atol=atol, rtol=rtol)
            and torch.allclose(combined_old, reference_old_tensor, atol=atol, rtol=rtol)
        ):
            raise RuntimeError("combined event differs from two-forward P0 reference")
        event_batch_gates.append(
            {
                "case_id": request.case_id,
                "new_max_abs": float(
                    torch.max(torch.abs(combined_new - reference_new_tensor)).cpu()
                ),
                "old_max_abs": float(
                    torch.max(torch.abs(combined_old - reference_old_tensor)).cpu()
                ),
                "atol": atol,
                "rtol": rtol,
            }
        )
    config = controller_config(lock)
    weight_by_layer = {
        layer: f"{prepared.hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in prepared.runtime.spec.layers
    }
    denominators = base_weight_c_energy(
        runtime.model,
        prepared.covariance_by_layer,
        weight_by_layer,
        denominator_epsilon=config.load_denominator_epsilon,
    )
    # Arm/repetition isolation is run setup, not an accepted-step checkpoint.
    # Keep this profiler-only baseline on CPU so reported editor peak GPU memory
    # reflects the single edit-entry transaction used by the method itself.
    baseline = TorchCheckpoint.capture(
        runtime.model,
        tuple(weight_by_layer.values()),
        backup_device="cpu",
    )
    setup_wall_seconds = time.perf_counter() - setup_start - model_load_wall_seconds
    if setup_wall_seconds < 0.0:
        raise RuntimeError("run setup timer became negative after model-load exclusion")
    git_head = _git_head(repo)
    timing = lock["timing"]
    warmups = int(timing["p0_warmup_repetitions"])
    recorded = int(timing["p0_recorded_repetitions"])
    total_runs = (warmups + recorded) * len(P0_ARMS) * len(requests)
    recorded_runs = recorded * len(P0_ARMS) * len(requests)
    setup_share = setup_wall_seconds / max(1, recorded_runs)
    manifest = {
        "schema_version": "ode-edit-session02-p0-manifest/v3",
        "status": "RUNNING_TECHNICAL_ONLY",
        "instruction": {
            "instruction_id": lock["instruction_id"],
            "revision_id": lock["revision_id"],
        },
        "instruction_id": lock["instruction_id"],
        "revision_id": lock["revision_id"],
        "git": {"commit": git_head, "proposal_id": proposal_id},
        "model": runtime_metadata,
        "selection": {
            "case_ids": list(case_ids),
            "order_sha256": canonical_hash(list(case_ids)),
        },
        "contexts": {
            "manifest_id": prepared.contexts.manifest_id,
            "tokenization_manifests": list(prepared.tokenization_manifests),
        },
        "hashes": {
            "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
            "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
            "dataset_sha256": lock["selection"]["dataset_sha256"],
            "hparams_sha256": model_lock["hparams_sha256"],
            "canonical_spec_sha256": lock["canonical_spec"]["sha256"],
            "covariance_files": [
                {
                    "relative_path": relative,
                    "sha256": next(
                        record.sha256
                        for record in fixed_artifacts.files
                        if str(Path(record.path).resolve().relative_to(easyedit_root))
                        == relative
                    ),
                    "size": next(
                        record.size
                        for record in fixed_artifacts.files
                        if str(Path(record.path).resolve().relative_to(easyedit_root))
                        == relative
                    ),
                }
                for relative in prepared.covariance_paths
            ],
        },
        "offline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "policy": {
            "controller": config.to_dict(),
            "arms": [arm.value for arm in P0_ARMS],
            "trial_backend": lock["trial_backend"]["selected_common_backend"],
            "dtype_policy": runtime_metadata["dtype_policy"],
            "checkpoint_original_dtype": runtime_metadata[
                "checkpoint_original_dtype"
            ],
            "observed_parameter_dtype": runtime_metadata[
                "observed_parameter_dtype"
            ],
        },
        "artifact_firewall": {
            "raw_prompt_target_context_persisted": False,
            "evaluation_executed": False,
        },
        "setup_wall_seconds": setup_wall_seconds,
        "model_load_wall_seconds_excluded": model_load_wall_seconds,
        "event_batch_identity": event_batch_gates,
        "warmup_repetitions": warmups,
        "recorded_repetitions": recorded,
        "expected_run_count": total_runs,
    }
    _require_fields(
        manifest,
        lock["artifact_schema"]["manifest_required_sections"],
        label="P0 manifest",
    )
    _require_fields(
        manifest["policy"],
        lock["artifact_schema"]["manifest_policy_required_fields"],
        label="P0 manifest policy",
    )
    _json_write(output_root / "manifest.json", manifest)

    diagnostic_by_rep: dict[int, dict[str, float]] = {}
    compute_by_rep: dict[int, dict[str, Mapping[str, Any]]] = {}
    terminal_geometry_fractions: list[dict[str, Any]] = []
    hook_reference_gates: list[dict[str, Any]] = []
    records = 0
    for repetition in range(warmups + recorded):
        is_warmup = repetition < warmups
        for request in requests:
            for arm in P0_ARMS:
                baseline.restore(runtime.model)
                metrics = EditInstrumentation(
                    f"{request.case_id}:{arm.value}:rep-{repetition}",
                    gpu_timing=True,
                    gpu_device="cuda:0",
                )
                if not is_warmup:
                    metrics.add_wall_seconds("context_setup", setup_share)
                cache_root = output_root / "direct_z"
                cache_path = cache_root / (
                    f"{args.model_alias}-{request.case_id}-{arm.value}-"
                    f"rep-{repetition}.pt"
                )
                # Backend construction captures the exact entry snapshot.  It
                # is run-level method setup, so record it instead of hiding a
                # dense state hash outside both editor and amortized timing.
                with metrics.component("context_setup"):
                    backend = EasyEditMemitBackend(
                        runtime=runtime,
                        bridge=prepared.bridge,
                        hparams=prepared.hparams,
                        contexts=prepared.contexts,
                        request=request,
                        covariance_specs=prepared.covariance_specs,
                        covariance_contract=prepared.covariance_contract,
                        covariance_by_layer=prepared.covariance_by_layer,
                        direct_z_cache_root=cache_root,
                        direct_z_cache_path=cache_path,
                        tau=config.tau,
                        unit_c_norm_epsilon=lock["controller"][
                            "unit_c_norm_epsilon"
                        ],
                        unit_c_identity_atol=lock["controller"][
                            "unit_c_identity_atol"
                        ],
                        unit_c_identity_rtol=lock["controller"][
                            "unit_c_identity_rtol"
                        ],
                        instrumentation=metrics,
                        finite_difference_gate_epsilon=(
                            lock["derivative_backend"][
                                "p0_finite_difference_epsilon"
                            ]
                            if is_warmup and arm is Arm.FULL_ODE_EDIT
                            else None
                        ),
                        finite_difference_gate_atol=lock["derivative_backend"][
                            "p0_scalar_gate_atol"
                        ],
                        finite_difference_gate_rtol=lock["derivative_backend"][
                            "p0_scalar_gate_rtol"
                        ],
                    )
                ledger = OmegaLedger(denominators)
                try:
                    metrics.attach_model(runtime.model)
                    try:
                        result = FiveArmRunner(config, ledger).run(
                            arm,
                            edit_id=f"{request.case_id}:{arm.value}",
                            request=request,
                            backend=backend,
                            instrumentation=metrics,
                        )
                    finally:
                        metrics.detach_model()
                except BaseException:
                    baseline.restore(runtime.model)
                    baseline.assert_exact(runtime.model, include_rng=True)
                    raise
                snapshot = metrics.finalize().to_dict()
                baseline.restore(runtime.model)
                baseline.assert_exact(runtime.model, include_rng=True)
                geometry = diagnostic_by_rep.setdefault(repetition, {})
                if result.direct_z_compute_count == 0:
                    geometry["entry_hit"] = 1.0
                exact_native_distance = None
                if arm is Arm.NATIVE_MEMIT and ledger.receipts:
                    exact_native_distance = math.sqrt(
                        sum(
                            value
                            for _layer, value in ledger.receipts[-1].terminal_net_energy
                        )
                    )
                    geometry["D_native"] = exact_native_distance
                if arm is Arm.FULL_ODE_EDIT and result.direct_z_compute_count:
                    geometry["D_sync_entry"] = backend.entry_synchronous_distance()
                if backend.hook_reference_gate is not None:
                    hook_reference_gates.append(
                        {
                            "repetition": repetition,
                            "case_id": request.case_id,
                            "arm": arm.value,
                            "rows": list(backend.hook_reference_gate),
                        }
                    )
                if is_warmup:
                    continue
                recorded_repetition = repetition - warmups
                compute_by_rep.setdefault(recorded_repetition, {})[arm.value] = snapshot
                wall = float(snapshot["controller_wall_seconds"])
                gpu = float(snapshot["controller_gpu_seconds"])
                terminal_geometry_fractions.append(
                    {
                        "repetition": recorded_repetition,
                        "arm": arm.value,
                        "wall_fraction": (
                            snapshot["component_wall_seconds"]["terminal_geometry"] / wall
                            if wall > 0.0
                            else 0.0
                        ),
                        "gpu_fraction": (
                            snapshot["component_gpu_seconds"]["terminal_geometry"] / gpu
                            if gpu > 0.0
                            else 0.0
                        ),
                    }
                )
                common = {
                    "model": args.model_alias,
                    "case_id": request.case_id,
                    "arm": arm.value,
                    "order_sha256": canonical_hash(list(case_ids)),
                    "seed": seed,
                    "repetition": recorded_repetition,
                    "commit": git_head,
                    "hashes": {
                        "proposal_id": proposal_id,
                        "context_manifest_id": prepared.contexts.manifest_id,
                        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
                        "direct_z": backend.direct_z_identity,
                    },
                    "status": result.status,
                }
                controller_record = {
                    "schema_version": "ode-edit-session02-p0-controller-record/v2",
                    **common,
                    "result": result.to_dict(),
                    "event": list(backend.event_history),
                    "Omega": {
                        "state": ledger.state(),
                        "receipts": _receipt_payload(ledger),
                    },
                }
                _require_fields(
                    controller_record,
                    lock["artifact_schema"]["controller_step_required_fields"],
                    label="P0 controller record",
                )
                _jsonl_append(
                    output_root / "controller_steps.jsonl",
                    controller_record,
                )
                compute_record = {
                    **snapshot,
                    **common,
                    "schema_version": "ode-edit-session02-p0-compute-record/v2",
                    "accounting_schema_version": snapshot["schema_version"],
                    **snapshot["counters"],
                    "D_native": exact_native_distance,
                    "D_native_raw_proposal": backend.native_distance,
                    "D_sync_entry": (
                        backend.entry_synchronous_distance()
                        if arm is Arm.FULL_ODE_EDIT
                        and result.direct_z_compute_count
                        else None
                    ),
                }
                _require_fields(
                    compute_record,
                    lock["artifact_schema"]["compute_required_fields"],
                    label="P0 compute record",
                )
                _jsonl_append(
                    output_root / "compute.jsonl",
                    compute_record,
                )
                records += 1

    terminal_integrity_start = time.perf_counter()
    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    terminal_integrity_wall_seconds = time.perf_counter() - terminal_integrity_start

    radius_diagnostics = []
    for repetition, geometry in sorted(diagnostic_by_rep.items()):
        if geometry.get("entry_hit") == 1.0:
            radius_diagnostics.append(
                {"repetition": repetition, "entry_hit": True, "gate": "not-applicable"}
            )
            continue
        if "D_native" not in geometry or "D_sync_entry" not in geometry:
            raise RuntimeError("P0 radius diagnostic is incomplete")
        ratio = _strict_ratio(
            config.h0_fraction * geometry["D_sync_entry"],
            geometry["D_native"],
            label="h0/D_native",
        )
        radius_diagnostics.append(
            {"repetition": repetition, **geometry, "h0_over_D_native": ratio}
        )
    if not hook_reference_gates:
        raise RuntimeError("P0 hook/reference gate was not observed")
    summary = {
        "schema_version": "ode-edit-session02-p0-summary/v2",
        "status": "COMPLETE_TECHNICAL_ONLY_NO_SCIENTIFIC_EVALUATION",
        "model": args.model_alias,
        "case_ids": list(case_ids),
        "record_count": records,
        "radius_diagnostics": radius_diagnostics,
        "p1_radius_gate": {
            "lower_inclusive": 0.125,
            "upper_inclusive": 0.5,
            "pass": all(
                row.get("entry_hit") is not True
                and 0.125 <= row["h0_over_D_native"] <= 0.5
                for row in radius_diagnostics
            ) and bool(radius_diagnostics),
        },
        "full_native_compute_ratios": [
            {
                "repetition": repetition,
                "wall": _strict_ratio(
                    values[Arm.FULL_ODE_EDIT.value]["controller_wall_seconds"],
                    values[Arm.NATIVE_MEMIT.value]["controller_wall_seconds"],
                    label="Full/Native wall",
                ),
                "gpu": _strict_ratio(
                    values[Arm.FULL_ODE_EDIT.value]["controller_gpu_seconds"],
                    values[Arm.NATIVE_MEMIT.value]["controller_gpu_seconds"],
                    label="Full/Native GPU",
                ),
            }
            for repetition, values in sorted(compute_by_rep.items())
        ],
        "terminal_geometry_fractions": terminal_geometry_fractions,
        "hook_reference_gates": hook_reference_gates,
        "event_batch_identity": event_batch_gates,
        "terminal_integrity_wall_seconds": terminal_integrity_wall_seconds,
        "p1_low_rank_terminal_geometry_required": any(
            row["wall_fraction"] > 0.1 or row["gpu_fraction"] > 0.1
            for row in terminal_geometry_fractions
        ),
        "scientific_outcome_count": 0,
    }
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_ONLY"
    manifest["terminal_integrity_wall_seconds"] = terminal_integrity_wall_seconds
    _json_write(output_root / "manifest.json", manifest)
    terminal_files = [
        output_root / "manifest.json",
        output_root / "controller_steps.jsonl",
        output_root / "compute.jsonl",
        output_root / "evaluation.jsonl",
        output_root / "summary.json",
        *sorted((output_root / "direct_z").glob("*.pt")),
    ]
    _json_write(
        output_root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-session02-terminal-manifest/v1",
            "status": "COMPLETE",
            "files": {
                str(path.relative_to(output_root)): {
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in terminal_files
            },
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
