"""Official MEMIT/AlphaEdit B100 lifelong observational runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS, NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.evaluation import padding_safety_gate
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256

from .continuity import (
    verify_cache_continuity,
    verify_observed_batch,
    verify_weight_continuity,
)
from .contracts import (
    DATASET,
    EVALUATOR_IDENTITY,
    LAYERS,
    Method,
    ObservationBoundary,
)
from .lifelong_contracts import CHECKPOINT_BATCHES, INSTRUCTION_ID, LifelongLock
from .lifelong_evaluation import (
    evaluate_panel,
    load_evaluation_rows,
    retention_cohorts,
)
from .lifelong_journal import (
    EditableStateSnapshot,
    extend_hash_chain,
    save_checkpoint,
    write_json_once,
)
from .lifelong_metrics import action_realization_profiles
from .lifelong_probe import (
    capture_layer_deltas,
    public_same_state_payload,
    registered_completion_geometry,
    same_entry_layer_responses,
)
from .lifelong_stream import load_lifelong_batches, verify_lifelong_stream
from .runtime import (
    _endpoint_metrics,
    _git,
    _load_hparams,
    _load_model,
    _method_module,
    _official_requests,
    _restore_w0,
    _run_apply,
    _touched,
    _warm_official_state,
    _w0_identity,
)
from .sequential_runtime import _alpha_cache_entry_snapshot, _restore_alpha_cache


def _public_apply(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _cache_exit_identity(method: Method, payload: Mapping[str, Any]) -> str:
    if method is Method.ALPHAEDIT:
        return str(payload["alphaedit_dynamic_cache_contract"]["exit"]["sha256"])
    return str(payload["covariance_cache"]["exit"]["identity_sha256"])


def _parity_gate(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    requests: Sequence[Mapping[str, Any]],
    touched: Mapping[str, torch.nn.Parameter],
    batch_index: int,
    expected_entry: Mapping[str, str],
    prior_cache_exit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run observer-on as an isolated gate and accept byte-equal observer-off."""

    entry = EditableStateSnapshot.capture(method, touched)
    on_z: list[torch.Tensor] = []
    torch.manual_seed(661_100 + batch_index)
    torch.cuda.manual_seed_all(661_100 + batch_index)
    on_payload, _ = _run_apply(
        model=model,
        tokenizer=tokenizer,
        method=method,
        hparams=hparams,
        requests=requests,
        touched=touched,
        capture_layers=True,
        reset_alpha_cache=batch_index == 1,
        alpha_cache_history_width=(batch_index - 1) * len(requests),
        z_capture_sink=on_z,
    )
    on_endpoint = _endpoint_metrics(model, tokenizer, requests)
    on_weight = dict(on_payload["edited_sha256"])
    on_cache = _cache_exit_identity(method, on_payload)
    entry.restore(touched)

    torch.manual_seed(661_100 + batch_index)
    torch.cuda.manual_seed_all(661_100 + batch_index)
    off_payload, _ = _run_apply(
        model=model,
        tokenizer=tokenizer,
        method=method,
        hparams=hparams,
        requests=requests,
        touched=touched,
        capture_layers=False,
        reset_alpha_cache=batch_index == 1,
        alpha_cache_history_width=(batch_index - 1) * len(requests),
    )
    off_endpoint = _endpoint_metrics(model, tokenizer, requests)
    checks = {
        "observer_on_weight_equals_accepted_off": on_weight
        == dict(off_payload["edited_sha256"]),
        "observer_on_endpoint_equals_accepted_off": canonical_hash(on_endpoint)
        == canonical_hash(off_endpoint),
        "observer_on_cache_equals_accepted_off": on_cache
        == _cache_exit_identity(method, off_payload),
        "observer_z_hash_equals_control": on_payload["layer_realization_observer"][
            "z_sha256"
        ]
        == off_payload["layer_realization_observer"]["z_sha256"],
        "observer_request_order_equals_control": on_payload[
            "request_order_sha256"
        ]
        == off_payload["request_order_sha256"],
        "observer_counter_tuple_equals_control": on_payload["official_call_audit"]
        == off_payload["official_call_audit"],
        "accepted_compute_z_once_per_request": off_payload[
            "layer_realization_observer"
        ]["direct_z_optimizer_compute_count"]
        == len(requests),
        "accepted_recompute_zero": off_payload["layer_realization_observer"][
            "direct_z_shared_replay_count"
        ]
        == 0,
    }
    if not all(checks.values()):
        entry.restore(touched)
        raise ObservationBoundary(f"first B100 observer parity differs: {checks}")
    observed_checks = verify_observed_batch(
        on_payload, method=method, request_count=len(requests)
    )
    weight = verify_weight_continuity(
        off_payload, expected_entry=expected_entry
    )
    cache = verify_cache_continuity(
        method,
        off_payload,
        batch_index=batch_index,
        request_count=len(requests),
        prior_exit_sha256=prior_cache_exit,
    )
    gate = {
        "status": "FIRST_B100_OBSERVER_ON_OFF_PARITY_PASS",
        "checks": checks,
        "observer_checks": observed_checks,
        "gate_control_scientific_denominator": 0,
        "accepted_production_compute_z": len(requests),
        "gate_control_compute_z": len(requests),
        "accepted_state_source": "OBSERVER_OFF_BYTE_EQUAL_REPLAY",
        "accepted_weight": weight,
        "accepted_cache": cache,
    }
    # Science telemetry is taken from the byte-equal observer-on clone, while
    # the current physical state is the accepted observer-off execution.
    result = {
        "apply": on_payload,
        "accepted_apply_identity_sha256": off_payload["identity_sha256"],
        "endpoint": on_endpoint,
        "gate": gate,
    }
    accepted_state = {
        "weight": weight,
        "cache": cache,
        "cache_exit": _cache_exit_identity(method, off_payload),
    }
    entry.release()
    del on_z
    return result, accepted_state


def _checkpoint_probe(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    sentinel: Sequence[Mapping[str, Any]],
    touched: Mapping[str, torch.nn.Parameter],
    accepted_edit_count: int,
) -> dict[str, Any]:
    entry = EditableStateSnapshot.capture(method, touched)
    requests = _official_requests(sentinel)
    captured_z: list[torch.Tensor] = []
    raw_primary: list[dict[str, Any]] = []
    torch.manual_seed(662_000 + accepted_edit_count)
    torch.cuda.manual_seed_all(662_000 + accepted_edit_count)
    primary_payload, primary_originals = _run_apply(
        model=model,
        tokenizer=tokenizer,
        method=method,
        hparams=hparams,
        requests=requests,
        touched=touched,
        capture_layers=True,
        reset_alpha_cache=(method is Method.ALPHAEDIT and accepted_edit_count == 0),
        alpha_cache_history_width=(
            accepted_edit_count if method is Method.ALPHAEDIT else 0
        ),
        z_capture_sink=captured_z,
        raw_capture_sink=raw_primary,
    )
    primary_deltas = capture_layer_deltas(touched, primary_originals)
    primary_same = same_entry_layer_responses(
        model=model,
        tokenizer=tokenizer,
        method=method,
        hparams=hparams,
        requests=requests,
        touched=touched,
        entry=entry,
        deltas=primary_deltas,
    )
    primary_geometry = registered_completion_geometry(
        raw_ordered=raw_primary[0], same_state=primary_same
    )
    primary_public = {
        "history_state": (
            "ACCUMULATED_DYNAMIC_CACHE" if method is Method.ALPHAEDIT else "STATIC_COVARIANCE"
        ),
        "ordered": _public_apply(primary_payload),
        "same_entry": public_same_state_payload(primary_same),
        "completion_geometry": primary_geometry,
    }
    del primary_originals, primary_deltas, primary_same
    entry.restore(touched)

    reset_public: dict[str, Any] | None = None
    if method is Method.ALPHAEDIT:
        raw_reset: list[dict[str, Any]] = []
        torch.manual_seed(662_000 + accepted_edit_count)
        torch.cuda.manual_seed_all(662_000 + accepted_edit_count)
        reset_payload, reset_originals = _run_apply(
            model=model,
            tokenizer=tokenizer,
            method=method,
            hparams=hparams,
            requests=requests,
            touched=touched,
            capture_layers=True,
            reset_alpha_cache=True,
            alpha_cache_history_width=0,
            fixed_z_replay=captured_z,
            raw_capture_sink=raw_reset,
        )
        reset_observer = reset_payload["layer_realization_observer"]
        if (
            reset_observer["direct_z_optimizer_compute_count"] != 0
            or reset_observer["direct_z_shared_replay_count"] != len(requests)
            or reset_observer["z_sha256"]
            != primary_payload["layer_realization_observer"]["z_sha256"]
        ):
            entry.restore(touched)
            raise ObservationBoundary("checkpoint shared-z reset fork differs")
        reset_deltas = capture_layer_deltas(touched, reset_originals)
        reset_same = same_entry_layer_responses(
            model=model,
            tokenizer=tokenizer,
            method=method,
            hparams=hparams,
            requests=requests,
            touched=touched,
            entry=entry,
            deltas=reset_deltas,
        )
        reset_geometry = registered_completion_geometry(
            raw_ordered=raw_reset[0], same_state=reset_same
        )
        reset_public = {
            "history_state": "RESET_DYNAMIC_CACHE",
            "ordered": _public_apply(reset_payload),
            "same_entry": public_same_state_payload(reset_same),
            "completion_geometry": reset_geometry,
        }
        entry.restore(touched)
        del reset_originals, reset_deltas, reset_same, raw_reset

    restore = entry.restore(touched)
    after = _w0_identity(touched)
    if after["sha256"] != entry.weight_identity["sha256"]:
        raise ObservationBoundary("checkpoint probe contaminated W state")
    result = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-checkpoint-probe.v1",
        "accepted_edit_count": accepted_edit_count,
        "sentinel_request_count": len(requests),
        "sentinel_order_sha256": canonical_hash(
            [str(item["request_sha256"]) for item in requests]
        ),
        "direct_z_optimizer_count": len(requests),
        "direct_z_recompute_count": 0,
        "shared_z_fork_replay_count": (
            len(requests) if method is Method.ALPHAEDIT else 0
        ),
        "primary": primary_public,
        "reset_cache_fork": reset_public,
        "probe_contamination": {
            "weight_restore_exact": True,
            "cache_restore_exact": bool(restore["cache"]["exact"]),
            "accepted_history_append_count": 0,
            "decision_influence_count": 0,
        },
    }
    result["identity_sha256"] = canonical_hash(result)
    del raw_primary, captured_z
    entry.release()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _functional_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    current: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    accepted_count: int,
    evaluation_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    current_panel = evaluate_panel(
        model=model,
        tokenizer=tokenizer,
        requests=current,
        evaluation_rows=evaluation_rows,
        include_rephrase=True,
        include_locality=True,
        panel_role="CURRENT_B100_CHECKPOINT",
    )
    cohorts = retention_cohorts(accepted, recent_end=accepted_count)
    retention = {
        name: evaluate_panel(
            model=model,
            tokenizer=tokenizer,
            requests=rows,
            evaluation_rows=evaluation_rows,
            include_rephrase=False,
            include_locality=False,
            panel_role=f"RETENTION_{name.upper()}",
        )
        for name, rows in cohorts.items()
    }
    return {
        "current": current_panel,
        "retention": retention,
        "controller_or_selection_influence_count": 0,
        "sampling_identity": canonical_hash(
            {
                name: [str(item["request_sha256"]) for item in rows]
                for name, rows in cohorts.items()
            }
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.cell_root.exists() or args.cell_root.is_symlink():
        raise ObservationBoundary(f"refusing to reuse lifelong cell root: {args.cell_root}")
    args.cell_root.mkdir(parents=True, mode=0o700)
    (args.cell_root / "journals").mkdir(mode=0o700)
    (args.cell_root / "checkpoints").mkdir(mode=0o700)
    source_head = _git(args.source_root, "rev-parse", "HEAD")
    source_tree = _git(args.source_root, "rev-parse", "HEAD^{tree}")
    if source_head != args.expected_head or _git(
        args.source_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ObservationBoundary("lifelong source HEAD/clean identity differs")

    seal = verify_lifelong_stream(json.loads(args.stream_seal.read_text(encoding="utf-8")))
    training, sentinel = load_lifelong_batches(DATASET, seal)
    if len(training) != 10_000 or len(sentinel) != 100:
        raise ObservationBoundary("lifelong loaded denominator differs")
    batches = tuple(tuple(training[start : start + 100]) for start in range(0, 10_000, 100))
    model, tokenizer, spec, padding_binding = _load_model(args.source_root, args.model)
    method = Method(args.method)
    hparams_path = args.source_root / (
        spec.memit_hparams if method is Method.MEMIT else spec.alpha_hparams
    )
    hparams = _load_hparams(method, hparams_path)
    hparams.device = 0
    if tuple(int(value) for value in hparams.layers) != LAYERS or str(
        hparams.model_name
    ) != spec.statistics_model_dir:
        raise ObservationBoundary("lifelong Official hparams binding differs")
    touched = _touched(model, hparams)
    w0 = _w0_identity(touched)
    if set(w0["dtypes"].values()) != {"torch.float32"}:
        raise ObservationBoundary("lifelong editable weights are not FULL_FP32")
    w0_values = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in touched.items()
    }
    first_three = _official_requests(batches[0][:3])
    prompts = [str(row["prompt"]).format(str(row["subject"])) for row in first_three]
    padding_gate = padding_safety_gate(
        model,
        tokenizer,
        prompts,
        [str(row["target_new"]) for row in first_three],
        hparams.layer_module_tmp.format(LAYERS[-1]),
        NumericalLock(),
        input_module=hparams.rewrite_module_tmp.format(LAYERS[-1]),
        subject_templates=[str(row["prompt"]) for row in first_three],
        subjects=[str(row["subject"]) for row in first_three],
    )
    warm = _warm_official_state(model, tokenizer, method, hparams)
    initial_alpha_cache = _alpha_cache_entry_snapshot(method)
    case_ids = {int(row["case_id"]) for row in [*training, *sentinel]}
    evaluation_rows = load_evaluation_rows(DATASET, case_ids)

    source = {"head": source_head, "tree": source_tree, "tracked_clean": True}
    chain = canonical_hash({"campaign_id": args.campaign_id, "genesis": True})
    checkpoints: list[dict[str, Any]] = []
    journals: list[dict[str, Any]] = []
    previous_commit = dict(w0["sha256"])
    prior_cache_exit: str | None = None
    terminal_w0_restore: dict[str, Any] | None = None
    terminal_cache_restore: dict[str, Any] | None = None
    first_valid: dict[str, Any] | None = None

    def checkpoint(batch_index: int, current: Sequence[Mapping[str, Any]]) -> None:
        nonlocal chain
        accepted = batch_index * 100
        state_path = args.cell_root / "checkpoints" / f"state-{accepted:05d}.pt"
        state = save_checkpoint(
            path=state_path,
            method=method,
            touched=touched,
            batch_index=batch_index,
            accepted_edit_count=accepted,
            stream_root=str(seal["root_digest"]),
            order_root=str(seal["training_order_sha256"]),
            journal_chain_root=chain,
            source=source,
            evaluator_identity=EVALUATOR_IDENTITY,
        )
        probe = _checkpoint_probe(
            model=model,
            tokenizer=tokenizer,
            method=method,
            hparams=hparams,
            sentinel=sentinel,
            touched=touched,
            accepted_edit_count=accepted,
        )
        functional = (
            {
                "status": "T0_CALIBRATION_PRE_EDIT",
                "sentinel_pre_edit": evaluate_panel(
                    model=model,
                    tokenizer=tokenizer,
                    requests=sentinel,
                    evaluation_rows=evaluation_rows,
                    include_rephrase=True,
                    include_locality=True,
                    panel_role="T0_SENTINEL_PRE_EDIT",
                ),
            }
            if batch_index == 0
            else _functional_checkpoint(
                model=model,
                tokenizer=tokenizer,
                current=current,
                accepted=training,
                accepted_count=accepted,
                evaluation_rows=evaluation_rows,
            )
        )
        payload = {
            "schema": "odeedit.s06.layer-realization-debt.lifelong-checkpoint.v1",
            "instruction_id": INSTRUCTION_ID,
            "campaign_id": args.campaign_id,
            "model": args.model,
            "method": method.value,
            "batch_index": batch_index,
            "accepted_edit_count": accepted,
            "state": state,
            "probe": probe,
            "functional": functional,
            "journal_chain_root": chain,
            "scientific_promotion": False,
        }
        path = args.cell_root / "checkpoints" / f"checkpoint-{accepted:05d}.json"
        digest = write_json_once(path, payload)
        chain = extend_hash_chain(chain, digest, 10_000 + batch_index)
        checkpoints.append(
            {
                "accepted_edit_count": accepted,
                "batch_index": batch_index,
                "path": str(path),
                "sha256": digest,
                "state": state,
                "chain_after": chain,
            }
        )

    try:
        checkpoint(0, sentinel)
        for batch_index, rows in enumerate(batches, start=1):
            requests = _official_requests(rows)
            entry = EditableStateSnapshot.capture(method, touched)
            try:
                if batch_index == 1:
                    batch_core, accepted = _parity_gate(
                        model=model,
                        tokenizer=tokenizer,
                        method=method,
                        hparams=hparams,
                        requests=requests,
                        touched=touched,
                        batch_index=batch_index,
                        expected_entry=previous_commit,
                        prior_cache_exit=prior_cache_exit,
                    )
                    apply_payload = batch_core["apply"]
                    endpoint = batch_core["endpoint"]
                    first_valid = batch_core["gate"]
                    weight_continuity = accepted["weight"]
                    cache_continuity = accepted["cache"]
                    prior_cache_exit = accepted["cache_exit"]
                else:
                    torch.manual_seed(661_100 + batch_index)
                    torch.cuda.manual_seed_all(661_100 + batch_index)
                    apply_payload, _ = _run_apply(
                        model=model,
                        tokenizer=tokenizer,
                        method=method,
                        hparams=hparams,
                        requests=requests,
                        touched=touched,
                        capture_layers=True,
                        reset_alpha_cache=False,
                        alpha_cache_history_width=(batch_index - 1) * len(requests),
                    )
                    verify_observed_batch(
                        apply_payload, method=method, request_count=len(requests)
                    )
                    weight_continuity = verify_weight_continuity(
                        apply_payload, expected_entry=previous_commit
                    )
                    cache_continuity = verify_cache_continuity(
                        method,
                        apply_payload,
                        batch_index=batch_index,
                        request_count=len(requests),
                        prior_exit_sha256=prior_cache_exit,
                    )
                    prior_cache_exit = str(cache_continuity["exit_sha256"])
                    endpoint = _endpoint_metrics(model, tokenizer, requests)
                observer = apply_payload["layer_realization_observer"]
                profile = action_realization_profiles(
                    observer["residual_debt"], observer["weight_action"]
                )
                payload = {
                    "schema": "odeedit.s06.layer-realization-debt.lifelong-b100.v1",
                    "instruction_id": INSTRUCTION_ID,
                    "campaign_id": args.campaign_id,
                    "status": "B100_ATOMIC_TERMINAL_VALID",
                    "model": args.model,
                    "method": method.value,
                    "batch_index": batch_index,
                    "accepted_edit_start": (batch_index - 1) * 100,
                    "accepted_edit_end": batch_index * 100,
                    "request_count": len(requests),
                    "request_order_sha256": apply_payload["request_order_sha256"],
                    "apply": apply_payload,
                    "endpoint": endpoint,
                    "action_realization": profile,
                    "weight_continuity": weight_continuity,
                    "cache_continuity": cache_continuity,
                    "first_valid_gate": first_valid if batch_index == 1 else None,
                    "scientific_promotion": False,
                }
                path = args.cell_root / "journals" / f"batch-{batch_index:03d}.json"
                digest = write_json_once(path, payload)
                chain = extend_hash_chain(chain, digest, batch_index)
                journals.append(
                    {
                        "batch_index": batch_index,
                        "path": str(path),
                        "sha256": digest,
                        "request_count": len(requests),
                        "chain_after": chain,
                        "weight_commit_sha256": dict(
                            weight_continuity["commit_sha256"]
                        ),
                        "cache_exit_sha256": prior_cache_exit,
                    }
                )
                previous_commit = dict(weight_continuity["commit_sha256"])
                entry.release()
            except BaseException as exc:
                rollback = entry.restore(touched)
                failure = {
                    "schema": "odeedit.s06.layer-realization-debt.lifelong-b100-failure.v1",
                    "instruction_id": INSTRUCTION_ID,
                    "batch_index": batch_index,
                    "accepted_edit_count_before": (batch_index - 1) * 100,
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                    "rollback": rollback,
                    "partial_batch_accepted_count": 0,
                    "scientific_denominator": 0,
                    "imputation_count": 0,
                }
                write_json_once(
                    args.cell_root / "journals" / f"batch-{batch_index:03d}-failure.json",
                    failure,
                )
                raise
            if batch_index in CHECKPOINT_BATCHES:
                checkpoint(batch_index, rows)

        terminal_commit = dict(previous_commit)
        terminal_w0_restore = _restore_w0(touched, w0_values, w0)
        terminal_cache_restore = _restore_alpha_cache(method, initial_alpha_cache)
    except BaseException:
        # A failing atomic B100 is already restored to its exact entry state.
        # The process still restores the original launch state before exit.
        _restore_w0(touched, w0_values, w0)
        _restore_alpha_cache(method, initial_alpha_cache)
        raise

    result = {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-cell.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "LIFELONG_10K_TERMINAL_VALID",
        "campaign_id": args.campaign_id,
        "model": args.model,
        "method": method.value,
        "source": source,
        "stream": {
            "path": str(args.stream_seal),
            "sha256": hashlib.sha256(args.stream_seal.read_bytes()).hexdigest(),
            "root": seal["root_digest"],
            "order": seal["training_order_sha256"],
            "sentinel_order": seal["sentinel_order_sha256"],
            "sample_duplication_count": 0,
        },
        "model_binding": {
            "revision": spec.model_revision,
            "snapshot": str(spec.model_path),
            "full_fp32": True,
            "padding": padding_binding,
            "padding_gate": padding_gate,
        },
        "warm_state": warm,
        "lock": LifelongLock().payload(),
        "initial_w0": w0,
        "terminal_committed_weight_sha256": terminal_commit,
        "terminal_w0_restore": terminal_w0_restore,
        "terminal_cache_restore": terminal_cache_restore,
        "first_valid_gate": first_valid,
        "valid_batch_denominator": len(journals),
        "valid_request_denominator": sum(row["request_count"] for row in journals),
        "checkpoint_denominator": len(checkpoints),
        "journal_chain_root": chain,
        "journals": journals,
        "checkpoints": checkpoints,
        "wall_seconds": time.perf_counter() - started,
        "nonfinite_count": 0,
        "rollback_violation_count": 0,
        "target_recomputation_count": 0,
        "scientific_promotion": False,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=tuple(value.value for value in Method), required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--cell-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--stream-seal", required=True, type=Path)
    args = parser.parse_args()
    preexisting = args.cell_root.exists() or args.cell_root.is_symlink()
    try:
        result = run(args)
        write_json_once(args.cell_root / "result.json", result)
    except BaseException as exc:
        if not preexisting:
            args.cell_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            failure = {
                "schema": "odeedit.s06.layer-realization-debt.lifelong-cell-failure.v1",
                "instruction_id": INSTRUCTION_ID,
                "model": args.model,
                "method": args.method,
                "status": "TECHNICAL_HOLD",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-20:],
                "valid_batch_denominator": len(
                    list((args.cell_root / "journals").glob("batch-[0-9][0-9][0-9].json"))
                ),
                "partial_failed_batch_denominator": 0,
                "imputation_count": 0,
                "scientific_promotion": False,
            }
            failure["identity_sha256"] = canonical_hash(failure)
            path = args.cell_root / "failure.json"
            if not path.exists() and not path.is_symlink():
                write_json_once(path, failure)
        raise


if __name__ == "__main__":
    main()
