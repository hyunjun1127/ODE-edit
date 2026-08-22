"""P1R52 accepted target with the pinned Official AlphaEdit writer.

This module owns only the accepted-z bridge and Phase-A paired orchestration.
The target/controller remains the existing P1R52 rollout and the alternative
writer remains EasyEdit's pinned ``apply_AlphaEdit_to_model`` entrypoint.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import math
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import (
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
    load_counterfact_cases_after_freeze,
)
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import _model_w0_contract, _run_ode_arm
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r52_accepted_z_observation import evaluate_accepted_z_batch, r52_binding
from .scalable_batched_model import build_scalable_capture_plan, build_scalable_objective_plan
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import P1R23_GRID_COUNT, P1R23_LAYER_ORDER, scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-TARGET-OFFICIAL-ALPHAEDIT-WRITER-A1-V1"
METHOD_ID = "P1R52-TARGET-OFFICIAL-ALPHAEDIT-WRITER-A1"
PHASE_A_ROLE = "r52-target-official-alphaedit-writer-phase-a"
PHASE_A_RESULT_NAME = "s05-p1r52-target-official-alphaedit-writer-phase-a-v1"
PHASE_A_TECH_R1_RESULT_NAME = "s05-p1r52-target-official-alphaedit-writer-phase-a-tech-r1-v1"
PHASE_A_TECH_R2_RESULT_NAME = "s05-p1r52-target-official-alphaedit-writer-phase-a-tech-r2-v1"
PHASE_A_CASE_COUNT = 10
PHASE_B_ROLE = "r52-target-official-alphaedit-writer-phase-b"
PHASE_B_RESULT_NAME = "s05-p1r52-target-official-alphaedit-writer-phase-b-sequential-10xb100-v1"


def _flatten(rows: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in rows for value in row]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ODEBFContractError("writer comparison metric is empty")
    value = float(sum(float(item) for item in values) / len(values))
    if not math.isfinite(value):
        raise ODEBFContractError("writer comparison metric is nonfinite")
    return value


def _endpoint_summary(scores: Mapping[str, Any]) -> dict[str, Any]:
    rewrite = scores["rewrite_success"]
    rephrase = scores["paraphrase_success"]
    locality = scores["legacy_primary"]["metrics"]["locality-preservation"]
    result = {
        "rewrite_target_new_nll_mean": _mean(_flatten(rewrite["target_new_nll_by_request"])),
        "rewrite_target_true_nll_mean": _mean(_flatten(rewrite["target_true_nll_by_request"])),
        "rephrase_target_new_nll_mean": _mean(_flatten(rephrase["target_new_nll_by_request"])),
        "rephrase_target_true_nll_mean": _mean(_flatten(rephrase["target_true_nll_by_request"])),
        "rewrite_success_numerator": int(rewrite["prompt_numerator"]),
        "rewrite_success_denominator": int(rewrite["prompt_denominator"]),
        "rewrite_accuracy_numerator": int(scores["rewrite_acc"]["prompt_numerator"]),
        "rewrite_accuracy_denominator": int(scores["rewrite_acc"]["prompt_denominator"]),
        "rephrase_success_numerator": int(rephrase["prompt_numerator"]),
        "rephrase_success_denominator": int(rephrase["prompt_denominator"]),
        "rephrase_strict_success_numerator": int(rephrase["strict_request_numerator"]),
        "rephrase_strict_success_denominator": int(rephrase["strict_request_denominator"]),
        "rephrase_accuracy_numerator": int(scores["paraphrase_acc"]["prompt_numerator"]),
        "rephrase_accuracy_denominator": int(scores["paraphrase_acc"]["prompt_denominator"]),
        "rephrase_strict_accuracy_numerator": int(scores["paraphrase_acc"]["strict_request_numerator"]),
        "rephrase_strict_accuracy_denominator": int(scores["paraphrase_acc"]["strict_request_denominator"]),
        "locality_numerator": int(locality["numerator"]),
        "locality_denominator": int(locality["denominator"]),
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def _writer_gap(endpoint: Mapping[str, Any], z: Mapping[str, Any]) -> dict[str, float]:
    return {
        "rewrite_W_minus_z_target_new_nll": float(endpoint["rewrite_target_new_nll_mean"] - z["rewrite_target_new_nll_mean"]),
        "rephrase_W_minus_z_target_new_nll": float(endpoint["rephrase_target_new_nll_mean"] - z["rephrase_target_new_nll_mean"]),
    }


def _weight_energy(
    touched: Mapping[str, torch.nn.Parameter],
    entry_values: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    by_weight: dict[str, float] = {}
    for name in sorted(touched):
        delta = touched[name].detach().float() - entry_values[name].to(
            device=touched[name].device, dtype=torch.float32
        )
        by_weight[name] = float(torch.sum(delta.double().square()).item())
    if not all(math.isfinite(value) for value in by_weight.values()):
        raise ODEBFStateError("Official AlphaEdit post-BF16 update energy is nonfinite")
    norms = {name: math.sqrt(max(value, 0.0)) for name, value in by_weight.items()}
    total_norm = sum(norms.values())
    total_energy = sum(by_weight.values())
    return {
        "realized_bf16_step_energy": by_weight,
        "actual_update_frobenius_norm": norms,
        "actual_update_norm_share": {
            name: (value / total_norm if total_norm > 0.0 else 0.0)
            for name, value in norms.items()
        },
        "squared_energy_share": {
            name: (value / total_energy if total_energy > 0.0 else 0.0)
            for name, value in by_weight.items()
        },
        "total_realized_bf16_step_energy": total_energy,
        "nonfinite_count": 0,
    }


@contextmanager
def isolated_alphaedit_module_state() -> Iterator[dict[str, Any]]:
    """Restore Official AlphaEdit module-global P/cache state after one arm."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    names = ("P", "P_loaded", "P_loaded_from", "cache_c", "cache_c_new")
    present = {name: hasattr(alpha_main, name) for name in names}
    values: dict[str, Any] = {}
    for name in names:
        if present[name]:
            value = getattr(alpha_main, name)
            values[name] = value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    receipt: dict[str, Any] = {"restored": False}
    try:
        yield receipt
    finally:
        for name in names:
            if present[name]:
                setattr(alpha_main, name, values[name])
            elif hasattr(alpha_main, name):
                delattr(alpha_main, name)
        receipt["restored"] = True
        receipt["entry_presence"] = present


@contextmanager
def accepted_z_cache_template(
    requests: Sequence[Mapping[str, Any]],
    accepted_z: torch.Tensor,
    hparams: Any,
    *,
    parent: Path,
    accepted_z_source: str = "P1R52_K8_TERMINAL_TARGET",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Expose accepted z through the exact Official AlphaEdit cache interface."""

    if accepted_z.ndim != 2 or accepted_z.shape[1] != len(requests):
        raise ODEBFContractError("P1R52 accepted-z writer bridge shape differs")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="accepted-z-", dir=parent) as temporary:
        template = str(Path(temporary) / "layer_{}_clamp_{}_case_{}.npz")
        file_rows: list[dict[str, Any]] = []
        layer = int(hparams.layers[-1])
        clamp = hparams.clamp_norm_factor
        for index, request in enumerate(requests):
            path = Path(template.format(layer, clamp, request["case_id"]))
            value = accepted_z[:, index].detach().to(device="cpu").contiguous().numpy()
            np.savez(path, v_star=value)
            loaded = np.load(path)["v_star"]
            if not np.array_equal(loaded, value):
                raise ODEBFStateError("accepted-z Official cache byte binding differs")
            file_rows.append(
                {
                    "request_sha256": str(request["request_sha256"]),
                    "case_id": int(request["case_id"]),
                    "v_star_sha256": tensor_sha256(torch.from_numpy(loaded)),
                    "shape": list(loaded.shape),
                    "dtype": str(loaded.dtype),
                }
            )
        if not accepted_z_source:
            raise ODEBFContractError("accepted-z Official cache source differs")
        receipt = {
            "source": accepted_z_source,
            "request_count": len(requests),
            "request_order_sha256": scalable_ordered_request_digest(
                [str(item["request_sha256"]) for item in requests]
            ),
            "accepted_z_sha256": tensor_sha256(accepted_z),
            "accepted_z_shape": list(accepted_z.shape),
            "accepted_z_dtype": str(accepted_z.dtype),
            "native_alphaedit_compute_z_expected_count": 0,
            "files": file_rows,
        }
        receipt["identity_sha256"] = canonical_hash(receipt)
        yield template, receipt


def _evaluate_w(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    freeze: EndpointActionFreeze,
    ledger: ComputeLedger,
) -> dict[str, Any]:
    counter = ModelForwardCounter(model, ledger)
    try:
        receipt = evaluate_counterfact_success_accuracy_batch(
            model,
            tokenizer,
            cases,
            model_alias="llama3-8b-inst",
            freeze=freeze,
            expected_batch_size=len(cases),
        ).raw_free_payload()
    finally:
        counter.close()
    return receipt


def _run_phase_a_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
    case_root: Path,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
) -> dict[str, Any]:
    if len(requests) != 100:
        raise ODEBFContractError("Phase-A paired case is not B100")
    entry_contract = _model_w0_contract(touched)
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if entry_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Phase-A paired W entry differs")
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    if objective_plan.request_order_sha256 != request_order or capture_plan.request_order_sha256 != request_order:
        raise ODEBFContractError("Phase-A target plan order differs")
    outer_population = tuple(population_by_sha256[item] for item in theta0_cache.request_order)
    snapshot = _entry_parameter_snapshot_sha256(model, dict(base_receipt.parameter_sha256))
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=snapshot,
        )
    finally:
        counter.close()
    state = ArmRuntimeState(
        P1Arm.R_BF,
        # The target run is history-off, but the reusable ledger's constructor
        # intentionally reserves four transactions.  This capacity is inert:
        # no records are appended or consumed in Phase A.
        P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=400, batch_size=100),
        ComputeLedger(),
        ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"phase": "A", "case": case_index, "entry": entry_hashes}),
        ),
        dict(base_values),
    )
    rollout = _run_ode_arm(
        model,
        tokenizer,
        requests,
        alias="llama3-8b-inst",
        arm=FixedE8Arm.SOFT,
        allocation="RS",
        capture_plan=capture_plan,
        objective_plan=objective_plan,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        arm_state=state,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        touched=touched,
        base_receipt=base_receipt,
        base_values=base_values,
        raw_root=case_root / "raw" / "target",
        write_once=_atomic_write_once,
        p1r24=True,
        p1r34=True,
        p1r35=True,
        p1r52=True,
    )
    public = rollout["public"]
    if public["accepted_update_count"] != P1R23_GRID_COUNT or public["tau_final"] != 1.0:
        raise ODEBFStateError("Phase-A P1R52 target K8 differs")
    if _model_w0_contract(touched) != entry_contract or _hashes(touched) != entry_hashes:
        raise ODEBFStateError("Phase-A target generation did not restore W entry")

    binding = r52_binding(PHASE_A_ROLE, requests, hparams, rollout["terminal_target"])
    freeze = EndpointActionFreeze(
        arm=f"{PHASE_A_ROLE}-case-{case_index:02d}",
        sequential_batch=case_index - 1,
        request_order_sha256=request_order,
        selected_snapshot_sha256=public["identity_sha256"],
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path,
        requests,
        freeze,
        expected_batch_size=100,
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        z_observation, _ = evaluate_accepted_z_batch(
            model,
            tokenizer,
            requests,
            cases,
            role=PHASE_A_ROLE,
            round_index=case_index,
            binding=binding,
            committed_weight_sha256=entry_hashes,
        )
    finally:
        counter.close()
    z_scores = z_observation["scores"]
    z_summary = _endpoint_summary(z_scores)

    r52_materializer = AcceptedPhysicalStateMaterializer(model, base_values)
    r52_materialization = r52_materializer.materialize(
        rollout["terminal_factors"], transition_index=P1R23_GRID_COUNT
    )
    r52_scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
    r52_summary = _endpoint_summary(r52_scores)
    r52_restore = r52_materializer.restore()
    if _hashes(touched) != entry_hashes:
        raise ODEBFStateError("Phase-A R52 writer restore differs")

    with isolated_alphaedit_module_state() as alpha_state:
        with accepted_z_cache_template(
            requests,
            rollout["terminal_target"],
            hparams,
            parent=case_root / "private",
        ) as (cache_template, bridge_receipt):
            counter = ModelForwardCounter(model, job_ledger)
            try:
                official_payload, originals = run_official_native_apply(
                    model,
                    tokenizer,
                    requests,
                    hparams,
                    touched=touched,
                    reset_cache=True,
                    cache_history_width=0,
                    cache_template=cache_template,
                    expected_native_compute_z_call_count=0,
                    accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                )
            finally:
                counter.close()
            if any(tensor_sha256(originals[name]) != entry_hashes[name] for name in touched):
                raise ODEBFStateError("Phase-A Official writer entry copy differs")
            official_energy = _weight_energy(touched, base_values)
            official_scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
            official_summary = _endpoint_summary(official_scores)
            official_edited_hashes = _hashes(touched)
            official_restore = _restore_exact_w0(
                touched,
                base_values,
                mutation_lock=mutation_lock,
                expected_contract=entry_contract,
            )
    if not alpha_state["restored"]:
        raise ODEBFStateError("Phase-A Official cache/P isolation differs")
    if _hashes(touched) != entry_hashes or any(int(touched[name].data_ptr()) != entry_pointers[name] for name in touched):
        raise ODEBFStateError("Phase-A cross-arm W contamination detected")

    r52_gap = _writer_gap(r52_summary, z_summary)
    official_gap = _writer_gap(official_summary, z_summary)
    payload = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-a-case/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "case_index": case_index,
        "request_count": 100,
        "request_order_sha256": request_order,
        "accepted_z": binding.raw_free_payload(),
        "accepted_z_hash_exact_across_writers": True,
        "writer_entry_weight_hash_exact_across_writers": True,
        "z_direct": {"summary": z_summary, "scores": z_scores, "observation": z_observation},
        "p1r52_writer": {
            "summary": r52_summary,
            "scores": r52_scores,
            "gap": r52_gap,
            "materialization": r52_materialization,
            "restore": r52_restore,
        },
        "official_alphaedit_writer": {
            "summary": official_summary,
            "scores": official_scores,
            "gap": official_gap,
            "apply": official_payload,
            "accepted_z_bridge": bridge_receipt,
            "edited_weight_sha256": official_edited_hashes,
            "update_energy": official_energy,
            "restore": official_restore,
            "alphaedit_static_p_used": True,
            "alphaedit_cache_c_continuity": True,
            "native_alphaedit_compute_z_call_count": 0,
        },
        "barrier_influence_counts": {
            "r52_structural_h": 0,
            "r52_p_barrier": 0,
            "r52_energy_capacity_barrier": 0,
            "pir": 0,
            "pir_u": 0,
            "fpiq": 0,
            "r52_writer_historical_risk": 0,
        },
        "official_rephrase_gap_minus_r52": float(
            official_gap["rephrase_W_minus_z_target_new_nll"]
            - r52_gap["rephrase_W_minus_z_target_new_nll"]
        ),
        "pilot_transfer_improved": bool(
            official_gap["rephrase_W_minus_z_target_new_nll"]
            < r52_gap["rephrase_W_minus_z_target_new_nll"]
        ),
        "materialization_authoritative_count_by_writer": {"p1r52": 1, "official_alphaedit": 1},
        "cross_arm_cache_contamination_count": 0,
        "cross_arm_weight_contamination_count": 0,
        "W0_restored": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    terminal_sha = _atomic_write_once(case_root / "terminal.json", payload)
    return {"terminal_sha256": terminal_sha, **payload}


def run_phase_a(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    **_: Any,
) -> dict[str, Any]:
    if alias != "llama3-8b-inst" or len(stream_batches) != PHASE_A_CASE_COUNT or any(len(batch) != 100 for batch in stream_batches):
        raise ODEBFContractError("Phase-A matrix differs")
    expected_root = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
    expected_order = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
    if stream.get("root_digest") != expected_root or stream.get("all_request_order_sha256") != expected_order:
        raise ODEBFContractError("Phase-A sealed B100x10 stream differs")
    expected_w0 = _model_w0_contract(touched)
    completed: list[dict[str, Any]] = []
    pilot_pass: bool | None = None
    for case_index, requests in enumerate(stream_batches, start=1):
        if case_index > 1 and pilot_pass is not True:
            break
        seed_all(COMMON_SEED)
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        result = _run_phase_a_case(
            model,
            tokenizer,
            requests,
            case_index=case_index,
            case_root=case_root,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            theta0_cache=theta0_cache,
            dataset_path=dataset_path,
            mutation_lock=mutation_lock,
            touched=touched,
            base_receipt=base_receipt,
            base_values=base_values,
            request_microbatch_size=request_microbatch_size,
            job_ledger=job_ledger,
        )
        completed.append(result)
        if case_index == 1:
            pilot_pass = bool(result["pilot_transfer_improved"])
        stages.record(
            f"post_target_official_writer_phase_a_case_{case_index:02d}",
            {
                "case_index": case_index,
                "pilot_transfer_improved": pilot_pass,
                "W0_restored": _model_w0_contract(touched) == expected_w0,
            },
        )
    deltas = [float(item["official_rephrase_gap_minus_r52"]) for item in completed]
    aggregate_improved = len(completed) == PHASE_A_CASE_COUNT and _mean(deltas) < 0.0
    terminal = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-a-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "phase": "A",
        "planned_case_count": PHASE_A_CASE_COUNT,
        "completed_case_count": len(completed),
        "pilot_transfer_improved": pilot_pass,
        "pilot_gate_definition": "OFFICIAL_REPHRASE_W_MINUS_Z_GAP_LT_P1R52_WRITER_GAP",
        "aggregate_official_minus_r52_rephrase_gap_mean": _mean(deltas),
        "phase_a_transfer_improved": aggregate_improved,
        "phase_b_status": "RELEASE" if aggregate_improved else "STOP_NEGATIVE_PHASE_A",
        "case_terminal_sha256": [item["terminal_sha256"] for item in completed],
        "request_attempt_count": len(completed) * 100,
        "native_alphaedit_compute_z_call_count": sum(
            int(item["official_alphaedit_writer"]["apply"]["native_alphaedit_compute_z_call_count"])
            for item in completed
        ),
        "structural_h_p_energy_pir_piru_fpiq_influence_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "W0_restored": _model_w0_contract(touched) == expected_w0,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-a-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "completed_case_count": len(completed),
        "phase_b_status": terminal["phase_b_status"],
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_TARGET_OFFICIAL_ALPHAEDIT_WRITER_PHASE_A_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "phase_b_status": terminal["phase_b_status"],
        "W0_restored": terminal["W0_restored"],
    }


def _entry_snapshot(
    touched: Mapping[str, torch.nn.Parameter], *, round_index: int
) -> ArmWeightSnapshot:
    hashes = _hashes(touched)
    return ArmWeightSnapshot(
        P1Arm.R_BF,
        round_index,
        hashes,
        canonical_hash(
            {"role": PHASE_B_ROLE, "round": round_index, "parameter_sha256": hashes}
        ),
    )


def _entry_values(
    touched: Mapping[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in touched.items()
    }


def run_phase_b(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    **_: Any,
) -> dict[str, Any]:
    """Run the released fresh-W0 10xB100 Official-writer sequential arm."""

    if (
        alias != "llama3-8b-inst"
        or len(stream_batches) != PHASE_A_CASE_COUNT
        or any(len(batch) != 100 for batch in stream_batches)
    ):
        raise ODEBFContractError("Phase-B matrix differs")
    if (
        stream.get("root_digest")
        != "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
        or stream.get("all_request_order_sha256")
        != "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
    ):
        raise ODEBFContractError("Phase-B sealed B100x10 stream differs")

    w0_contract = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if w0_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Phase-B W0 differs")
    batch_rows: list[dict[str, Any]] = []
    cohort_cases: list[tuple[Any, ...]] = []
    prior_cache_exit: str | None = None
    static_p_identity: str | None = None
    alpha_restore_receipt: dict[str, Any] | None = None

    try:
        with isolated_alphaedit_module_state() as alpha_state:
            alpha_restore_receipt = alpha_state
            for round_index, request_batch in enumerate(stream_batches, start=1):
                requests = tuple(request_batch)
                seed_all(COMMON_SEED)
                case_root = raw_root / "batches" / f"b{round_index:02d}"
                case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
                entry_contract = _model_w0_contract(touched)
                entry_hashes = _hashes(touched)
                entry_values = _entry_values(touched)
                entry_receipt = _entry_snapshot(touched, round_index=round_index - 1)
                request_order = scalable_ordered_request_digest(
                    [str(item["request_sha256"]) for item in requests]
                )
                objective_plan = build_scalable_objective_plan(
                    model,
                    tokenizer,
                    requests,
                    contexts=contexts,
                    request_microbatch_size=min(request_microbatch_size, len(requests)),
                    fact_token_strategy=hparams.fact_token,
                )
                capture_plan = build_scalable_capture_plan(
                    tokenizer,
                    requests,
                    contexts=contexts,
                    request_microbatch_size=min(request_microbatch_size, len(requests)),
                    fact_token_strategy=hparams.fact_token,
                )
                if (
                    objective_plan.request_order_sha256 != request_order
                    or capture_plan.request_order_sha256 != request_order
                ):
                    raise ODEBFContractError("Phase-B target plan order differs")
                outer_population = tuple(
                    population_by_sha256[item] for item in theta0_cache.request_order
                )
                snapshot = _entry_parameter_snapshot_sha256(
                    model, dict(entry_receipt.parameter_sha256)
                )
                counter = ModelForwardCounter(model, job_ledger)
                try:
                    outer_entry_p_cache = build_outer_entry_pretrained_cache(
                        model,
                        tokenizer,
                        outer_population,
                        theta0_cache,
                        outer_entry_snapshot_sha256=snapshot,
                    )
                finally:
                    counter.close()
                target_state = ArmRuntimeState(
                    P1Arm.R_BF,
                    P1HistoryLedger(
                        layer_order=P1R23_LAYER_ORDER,
                        maximum_records=400,
                        batch_size=100,
                    ),
                    ComputeLedger(),
                    entry_receipt,
                    entry_values,
                )
                rollout = _run_ode_arm(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    arm=FixedE8Arm.SOFT,
                    allocation="RS",
                    capture_plan=capture_plan,
                    objective_plan=objective_plan,
                    hparams=hparams,
                    projector=projector,
                    contexts=contexts,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    controller_lock=controller_lock,
                    arm_state=target_state,
                    request_by_sha256=request_by_sha256,
                    population_by_sha256=population_by_sha256,
                    schedule=schedule,
                    outer_entry_p_cache=outer_entry_p_cache,
                    theta0_cache=theta0_cache,
                    touched=touched,
                    base_receipt=entry_receipt,
                    base_values=entry_values,
                    raw_root=case_root / "raw" / "target",
                    write_once=_atomic_write_once,
                    p1r24=True,
                    p1r34=True,
                    p1r35=True,
                    p1r52=True,
                )
                public = rollout["public"]
                if (
                    public["accepted_update_count"] != P1R23_GRID_COUNT
                    or public["tau_final"] != 1.0
                    or _model_w0_contract(touched) != entry_contract
                    or _hashes(touched) != entry_hashes
                ):
                    raise ODEBFStateError("Phase-B target K8/entry restore differs")

                binding = r52_binding(
                    PHASE_B_ROLE, requests, hparams, rollout["terminal_target"]
                )
                freeze = EndpointActionFreeze(
                    arm=f"{PHASE_B_ROLE}-b{round_index:02d}",
                    sequential_batch=round_index - 1,
                    request_order_sha256=request_order,
                    selected_snapshot_sha256=public["identity_sha256"],
                    fixed_budget_slots_completed=P1R23_GRID_COUNT,
                )
                cases = tuple(
                    load_counterfact_cases_after_freeze(
                        dataset_path,
                        requests,
                        freeze,
                        expected_batch_size=100,
                    )
                )
                cohort_cases.append(cases)
                counter = ModelForwardCounter(model, job_ledger)
                try:
                    z_observation, _ = evaluate_accepted_z_batch(
                        model,
                        tokenizer,
                        requests,
                        cases,
                        role=PHASE_B_ROLE,
                        round_index=round_index,
                        binding=binding,
                        committed_weight_sha256=entry_hashes,
                    )
                finally:
                    counter.close()
                z_scores = z_observation["scores"]
                z_summary = _endpoint_summary(z_scores)

                with accepted_z_cache_template(
                    requests,
                    rollout["terminal_target"],
                    hparams,
                    parent=case_root / "private",
                ) as (cache_template, bridge_receipt):
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        official_payload, originals = run_official_native_apply(
                            model,
                            tokenizer,
                            requests,
                            hparams,
                            touched=touched,
                            reset_cache=round_index == 1,
                            cache_history_width=(round_index - 1) * 100,
                            cache_template=cache_template,
                            expected_native_compute_z_call_count=0,
                            accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                        )
                    finally:
                        counter.close()
                if any(
                    tensor_sha256(originals[name]) != entry_hashes[name]
                    for name in touched
                ):
                    raise ODEBFStateError("Phase-B Official writer entry copy differs")
                cache = official_payload["alphaedit_dynamic_cache_contract"]
                if (
                    cache["logical_history_width_at_entry"] != (round_index - 1) * 100
                    or cache["logical_history_width_after_append"] != round_index * 100
                    or (round_index > 1 and cache["entry"]["sha256"] != prior_cache_exit)
                    or (round_index > 1 and not cache["solver_consumed_entry_cache"])
                ):
                    raise ODEBFStateError("Phase-B Official cache continuity differs")
                prior_cache_exit = cache["exit"]["sha256"]
                observed_static = canonical_hash(cache["static_projection"])
                if static_p_identity is None:
                    static_p_identity = observed_static
                elif observed_static != static_p_identity:
                    raise ODEBFStateError("Phase-B Official static P identity differs")
                update_energy = _weight_energy(touched, entry_values)
                if _hashes(touched) == entry_hashes:
                    raise ODEBFStateError("Phase-B Official writer produced no transition")
                w_scores = _evaluate_w(
                    model, tokenizer, cases, freeze=freeze, ledger=job_ledger
                )
                w_summary = _endpoint_summary(w_scores)
                batch_payload = {
                    "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-b-batch/v1",
                    "round": round_index,
                    "request_count": 100,
                    "request_order_sha256": request_order,
                    "entry_weight_sha256": entry_hashes,
                    "commit_weight_sha256": _hashes(touched),
                    "target_public_identity_sha256": public["identity_sha256"],
                    "accepted_z": binding.raw_free_payload(),
                    "z_direct": {"summary": z_summary, "scores": z_scores},
                    "immediate_w": {
                        "summary": w_summary,
                        "scores": w_scores,
                        "gap": _writer_gap(w_summary, z_summary),
                    },
                    "official_alphaedit_writer": {
                        "apply": official_payload,
                        "accepted_z_bridge": bridge_receipt,
                        "update_energy": update_energy,
                        "alphaedit_static_p_used": True,
                        "alphaedit_cache_c_continuity": True,
                        "native_alphaedit_compute_z_call_count": 0,
                    },
                    "barrier_influence_counts": {
                        "r52_structural_h": 0,
                        "r52_p_barrier": 0,
                        "r52_energy_capacity_barrier": 0,
                        "pir": 0,
                        "pir_u": 0,
                        "fpiq": 0,
                        "r52_writer_historical_risk": 0,
                    },
                    "materialization_authoritative_count": 1,
                    "retry_backtracking_count": 0,
                    "physical_W_persists_to_next_batch": round_index < 10,
                }
                batch_payload["identity_sha256"] = canonical_hash(batch_payload)
                batch_sha = _atomic_write_once(case_root / "terminal.json", batch_payload)
                batch_rows.append({"terminal_sha256": batch_sha, **batch_payload})
                stages.record(
                    f"post_target_official_writer_phase_b_b{round_index:02d}",
                    {
                        "round": round_index,
                        "cache_width": round_index * 100,
                        "commit_weight_sha256": batch_payload["commit_weight_sha256"],
                    },
                )

            final_hashes = _hashes(touched)
            final_rows: list[dict[str, Any]] = []
            for round_index, cases in enumerate(cohort_cases, start=1):
                freeze = EndpointActionFreeze(
                    arm=f"{PHASE_B_ROLE}-final-b{round_index:02d}",
                    sequential_batch=9,
                    request_order_sha256=batch_rows[round_index - 1][
                        "request_order_sha256"
                    ],
                    selected_snapshot_sha256=canonical_hash(final_hashes),
                    fixed_budget_slots_completed=P1R23_GRID_COUNT,
                )
                scores = _evaluate_w(
                    model, tokenizer, cases, freeze=freeze, ledger=job_ledger
                )
                final_rows.append(
                    {
                        "round": round_index,
                        "summary": _endpoint_summary(scores),
                        "scores": scores,
                    }
                )
            final_payload = {
                "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-b-final-w10/v1",
                "cohort_count": len(final_rows),
                "request_count": len(final_rows) * 100,
                "final_weight_sha256": final_hashes,
                "cohorts": final_rows,
            }
            final_payload["identity_sha256"] = canonical_hash(final_payload)
            final_sha = _atomic_write_once(raw_root / "final-w10.json", final_payload)
    except BaseException:
        _restore_exact_w0(
            touched,
            base_values,
            mutation_lock=mutation_lock,
            expected_contract=w0_contract,
        )
        raise

    restore = _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=w0_contract,
    )
    if (
        alpha_restore_receipt is None
        or not alpha_restore_receipt["restored"]
        or _hashes(touched) != w0_hashes
        or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched)
    ):
        raise ODEBFStateError("Phase-B final W0/cache restore differs")
    terminal = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-b-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "phase": "B",
        "completed_batch_count": len(batch_rows),
        "request_attempt_count": len(batch_rows) * 100,
        "batch_terminal_sha256": [row["terminal_sha256"] for row in batch_rows],
        "final_w10_sha256": final_sha,
        "alphaedit_static_p_identity_sha256": static_p_identity,
        "alphaedit_cache_c_final_sha256": prior_cache_exit,
        "native_alphaedit_compute_z_call_count": 0,
        "structural_h_p_energy_pir_piru_fpiq_influence_count": 0,
        "materialization_authoritative_count": len(batch_rows),
        "batch_entry_evaluator_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "W0_restored": True,
        "restore": restore,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-target-official-alphaedit-writer-phase-b-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "completed_batch_count": len(batch_rows),
        "final_w10_sha256": final_sha,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_TARGET_OFFICIAL_ALPHAEDIT_WRITER_PHASE_B_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = [
    "INSTRUCTION_ID",
    "METHOD_ID",
    "PHASE_A_CASE_COUNT",
    "PHASE_A_RESULT_NAME",
    "PHASE_A_TECH_R1_RESULT_NAME",
    "PHASE_A_TECH_R2_RESULT_NAME",
    "PHASE_A_ROLE",
    "PHASE_B_RESULT_NAME",
    "PHASE_B_ROLE",
    "accepted_z_cache_template",
    "isolated_alphaedit_module_state",
    "run_phase_a",
    "run_phase_b",
]
