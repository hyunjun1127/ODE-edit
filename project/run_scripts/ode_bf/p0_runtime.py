"""Outcome-free model P0: Native AlphaEdit versus exact AlphaEdit-WB on B10."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from .accounting import ComputeLedger
from .alpha_backend import (
    ALPHA_SOLVE_DTYPE,
    ALPHA_SOLVE_REFERENCE,
    FOUR_PATH_ORDER,
    W64_ASSEMBLER_REFERENCE,
    W64_CAST_DTYPE,
    W64_ENDPOINT_DTYPE,
    W64_PRIMARY_PATH,
    W64_PRIMARY_REFERENCE,
    W64_REDUCED_BACKEND,
    W64_REDUCED_DTYPE,
    capture_native_and_wb_joint_endpoint,
    fresh_contexts_twice,
    load_original_bf16,
    seed_all,
)
from .artifacts import ODEBFArtifactGuard, load_rooted_json, sha256_file
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from . import evaluator as evaluator_module
from .evaluator import ModelEvaluationReceipt, evaluate_counterfact_rewrite_batch
from . import functional as functional_module
from .functional import CumulativeBF16FunctionalTrial, tensor_sha256
from .request_digest import ORDERED_REQUEST_DIGEST_SCHEMA, ordered_request_digest_v1
from .selection import load_sealed_joint_requests, verify_p0_b10_seal
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-W64-CANONICAL-RECEIPT-P0-R3-V1"
SCIENTIFIC_LOCK_INSTRUCTION_ID = (
    "ODEEDIT-S04-ODE-BF-V1P1-EXACT-FIRST-HIT-CPU-P0-V1-A3"
)
NUMERICAL_LOCK_SHA256 = "40421f3f8ef0e47df842268bb68b9c9548398e27e0a9afecb125ab1b6f853fa1"
NUMERICAL_LOCK_PREFIX = NUMERICAL_LOCK_SHA256[:8]
SEED = 41


def expected_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P0 result alias is not locked")
    return f"s04-p0-w64-canonical-receipt-r3-{alias}-{NUMERICAL_LOCK_PREFIX}"


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P0 receipt is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("P0 temporary receipt path exists")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


class StageRecorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sequence = 0
        self.last_stage: str | None = None

    def record(self, stage: str, payload: Mapping[str, Any]) -> str:
        self.sequence += 1
        receipt = {
            "schema": "ode-edit-s04-ode-bf-p0-stage/v1",
            "sequence": self.sequence,
            "stage": stage,
            "payload": dict(payload),
        }
        digest = _atomic_write_once(
            self.root / f"stage-{self.sequence:02d}-{stage}.json",
            receipt,
        )
        self.last_stage = stage
        return digest


class ComponentTimer:
    def __init__(self, ledger: ComputeLedger) -> None:
        self.ledger = ledger

    @contextlib.contextmanager
    def measure(self, component: str) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None
        started = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - started
            gpu = 0.0
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize(0)
                gpu = float(start_event.elapsed_time(end_event)) / 1000.0
            self.ledger.add_time(component, wall_seconds=wall, gpu_seconds=gpu)


class ModelForwardCounter:
    def __init__(self, model: torch.nn.Module, ledger: ComputeLedger) -> None:
        self.ledger = ledger

        def before(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            del module
            self.ledger.increment("model_forward")
            input_ids = kwargs.get("input_ids")
            attention = kwargs.get("attention_mask")
            if attention is not None:
                self.ledger.increment("processed_tokens", int(attention.sum().item()))
            elif input_ids is not None:
                self.ledger.increment("processed_tokens", int(input_ids.numel()))
            elif args and isinstance(args[0], torch.Tensor):
                self.ledger.increment("processed_tokens", int(args[0].numel()))

        self.handle = model.register_forward_pre_hook(before, with_kwargs=True)

    def close(self) -> None:
        self.handle.remove()


def _source_freeze(repo_root: Path, source_head: str) -> None:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != source_head:
        raise ODEBFContractError("P0 execution HEAD differs")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if status:
        raise ODEBFContractError("P0 execution source is not clean")


def _evaluation_payload(receipt: ModelEvaluationReceipt) -> dict[str, Any]:
    return {
        # P0 consumes the exact event only as an identity object.  The event
        # vector/count/aggregate is deliberately sealed behind a digest so the
        # technical pair does not disclose or interpret scientific efficacy.
        "batch_success_sha256": canonical_hash(
            receipt.batch_success.raw_free_payload()
        ),
        "adapter": receipt.batch_success.adapter.value,
        "target_full_vocabulary_logits_sha256": receipt.target_full_vocabulary_logits_sha256,
        "request_order_sha256": receipt.request_order_sha256,
        "target_span_lengths": list(receipt.target_span_lengths),
        "model_forward_count": receipt.model_forward_count,
        "processed_token_count": receipt.processed_token_count,
        "generation_call_count": receipt.generation_call_count,
    }


@dataclass(slots=True)
class DiagnosticEvaluation:
    receipt: ModelEvaluationReceipt
    target_logits: tuple[torch.Tensor, ...]


@contextlib.contextmanager
def _capture_virtual_endpoint_hashes() -> Iterator[dict[str, set[str]]]:
    observed: dict[str, set[str]] = {}
    original_assembler = functional_module.assemble_effective_bf16

    def capture(*args: Any, **kwargs: Any) -> Any:
        effective, stats = original_assembler(*args, **kwargs)
        observed.setdefault(stats.weight_name, set()).add(
            stats.effective_bf16_sha256
        )
        return effective, stats

    functional_module.assemble_effective_bf16 = capture
    try:
        yield observed
    finally:
        functional_module.assemble_effective_bf16 = original_assembler


def _evaluate_with_raw_free_capture(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[dict[str, Any]],
    *,
    alias: str,
) -> DiagnosticEvaluation:
    captured: list[torch.Tensor] = []
    call_count = 0
    original_hash = evaluator_module._hash_tensor_sequence

    def capture_hash(tensors: Any) -> str:
        nonlocal call_count
        call_count += 1
        captured.extend(
            tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for tensor in tensors
        )
        return original_hash(tensors)

    evaluator_module._hash_tensor_sequence = capture_hash
    try:
        receipt = evaluate_counterfact_rewrite_batch(
            model,
            tokenizer,
            requests,
            model_alias=alias,
        )
    finally:
        evaluator_module._hash_tensor_sequence = original_hash
    if call_count != 1 or not captured:
        raise ODEBFContractError("canonical evaluator diagnostic logit capture differs")
    return DiagnosticEvaluation(receipt, tuple(captured))


def _finite_summary(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not torch.isfinite(flat).all():
        raise ODEBFContractError("continuous diagnostic summary is empty or non-finite")
    absolute = torch.abs(flat)
    quantiles = torch.quantile(
        absolute,
        torch.tensor((0.50, 0.95, 0.99), dtype=torch.float64),
        interpolation="nearest",
    )
    return {
        "count": int(flat.numel()),
        "signed_min": float(flat.min().item()),
        "signed_max": float(flat.max().item()),
        "signed_mean": float(flat.mean().item()),
        "absolute_mean": float(absolute.mean().item()),
        "absolute_rms": float(torch.sqrt(torch.mean(flat * flat)).item()),
        "absolute_max": float(absolute.max().item()),
        "absolute_p50": float(quantiles[0].item()),
        "absolute_p95": float(quantiles[1].item()),
        "absolute_p99": float(quantiles[2].item()),
    }


def _functional_difference_summary(
    reference: DiagnosticEvaluation,
    candidate: DiagnosticEvaluation,
) -> dict[str, Any]:
    if len(reference.target_logits) != len(candidate.target_logits):
        raise ODEBFContractError("canonical evaluator logit slice count differs")
    logit_differences: list[torch.Tensor] = []
    for reference_tensor, candidate_tensor in zip(
        reference.target_logits,
        candidate.target_logits,
        strict=True,
    ):
        if reference_tensor.shape != candidate_tensor.shape:
            raise ODEBFContractError("canonical evaluator logit slice shape differs")
        logit_differences.append(candidate_tensor - reference_tensor)
    logits = _finite_summary(torch.cat([value.reshape(-1) for value in logit_differences]))
    reference_scores = reference.receipt.counterfact_scores
    candidate_scores = candidate.receipt.counterfact_scores
    if reference_scores is None or candidate_scores is None:
        raise ODEBFContractError("P0 CounterFact score receipt is absent")
    if len(reference_scores) != len(candidate_scores):
        raise ODEBFContractError("P0 CounterFact score count differs")
    new_diff = torch.tensor(
        [
            candidate_score.target_new_nll - reference_score.target_new_nll
            for reference_score, candidate_score in zip(
                reference_scores,
                candidate_scores,
                strict=True,
            )
        ],
        dtype=torch.float64,
    )
    old_diff = torch.tensor(
        [
            candidate_score.target_true_nll - reference_score.target_true_nll
            for reference_score, candidate_score in zip(
                reference_scores,
                candidate_scores,
                strict=True,
            )
        ],
        dtype=torch.float64,
    )
    margin_diff = old_diff - new_diff
    return {
        "target_full_vocabulary_logit_difference": logits,
        "target_new_nll_difference": _finite_summary(new_diff),
        "target_true_nll_difference": _finite_summary(old_diff),
        "success_margin_difference": _finite_summary(margin_diff),
    }


def _technical_decision_payload(
    evaluation: ModelEvaluationReceipt,
    *,
    certificates_pass: bool,
) -> dict[str, Any]:
    value = {
        "benchmark_adapter_event_sha256": canonical_hash(
            evaluation.batch_success.raw_free_payload()
        ),
        "historical_decision": "not-applicable-empty-history-technical-p0",
        "pretrained_decision": "not-accessed-outcome-free-technical-p0",
        "trust_decision": bool(certificates_pass),
        "request_order_sha256": evaluation.request_order_sha256,
    }
    return {**value, "decision_vector_sha256": canonical_hash(value)}


def classify_four_path_diagnostic(
    *,
    all_endpoint_bytes_exact: bool,
    source_inputs_identical: bool,
    all_finite: bool,
    all_certificates_pass: bool,
    benchmark_bits_exact: bool,
    decisions_exact: bool,
    strict_wb_virtual_commit_exact: bool,
    rollback_and_restore_exact: bool,
    boundary_touched: bool,
    parity_established: bool,
) -> str:
    if (
        not source_inputs_identical
        or not all_finite
        or not all_certificates_pass
        or not benchmark_bits_exact
        or not decisions_exact
        or not strict_wb_virtual_commit_exact
        or not rollback_and_restore_exact
    ):
        return "NON_EQUIVALENT"
    if boundary_touched or not parity_established:
        return "NUMERICALLY_AMBIGUOUS"
    if all_endpoint_bytes_exact:
        return "BYTE_EXACT"
    return "NUMERIC_PATH_DIVERGENCE_CANDIDATE"


def classify_w64_technical_candidate(
    *,
    source_inputs_identical: bool,
    required_paths_finite: bool,
    required_certificates_pass: bool,
    n32_w64_benchmark_bits_exact: bool,
    n32_w64_decisions_exact: bool,
    strict_w64_virtual_commit_exact: bool,
    rollback_and_restore_exact: bool,
    boundary_touched: bool,
    request_and_span_parity: bool,
) -> str:
    """Classify only the prospective W64 path; W32 is diagnostic-only."""

    if (
        not source_inputs_identical
        or not required_paths_finite
        or not required_certificates_pass
        or not n32_w64_benchmark_bits_exact
        or not n32_w64_decisions_exact
        or not strict_w64_virtual_commit_exact
        or not rollback_and_restore_exact
        or boundary_touched
        or not request_and_span_parity
    ):
        return "W64_TECHNICAL_HARD_GATE_FAIL"
    return "W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM"


def _comparison_matrix(layer_receipts: Any) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(FOUR_PATH_ORDER):
        matrix[left] = {}
        for right_index, right in enumerate(FOUR_PATH_ORDER):
            canonical_left, canonical_right = (
                (left, right) if left_index <= right_index else (right, left)
            )
            receipts = []
            for layer in layer_receipts:
                match = next(
                    item
                    for item in layer.pairwise_comparisons
                    if item.reference_path == canonical_left
                    and item.candidate_path == canonical_right
                )
                receipts.append(match)
            mismatch_count = sum(item.mismatch_count for item in receipts)
            element_count = sum(item.element_count for item in receipts)
            matrix[left][right] = {
                "byte_exact": mismatch_count == 0,
                "mismatch_count": mismatch_count,
                "element_count": element_count,
                "mismatch_fraction": (
                    0.0 if element_count == 0 else mismatch_count / element_count
                ),
                "ordered_ulp_max": max(item.ordered_ulp_max for item in receipts),
                "max_abs_difference": max(
                    item.max_abs_difference for item in receipts
                ),
            }
    return matrix


def _restore_entry(
    parameters: Mapping[str, torch.nn.Parameter],
    entry: Mapping[str, torch.Tensor],
    *,
    mutation_lock: threading.RLock,
) -> None:
    with mutation_lock, torch.no_grad():
        for name in sorted(parameters):
            candidate = entry[name].to(device=parameters[name].device)
            parameters[name].copy_(candidate)
            del candidate
    if any(tensor_sha256(parameters[name]) != tensor_sha256(entry[name]) for name in parameters):
        raise ODEBFContractError("P0 arm boundary did not restore W0 exactly")


def _fault_rollback_gate(
    parameters: Mapping[str, torch.nn.Parameter],
    candidates: Mapping[str, torch.Tensor],
    entry: Mapping[str, torch.Tensor],
    *,
    mutation_lock: threading.RLock,
    ledger: ComputeLedger,
) -> tuple[int, ...]:
    names = tuple(sorted(parameters))
    points = (0, len(names) // 2, len(names) - 1)
    observed: list[int] = []
    for point in points:
        transaction = AtomicBatchTransaction(
            parameters,
            transaction_id=f"fault-{point}",
            mutation_lock=mutation_lock,
        )
        for name in names:
            transaction.stage(name, candidates[name])
        try:
            transaction.commit(
                post_commit_verify=lambda: True,
                fault_after_writes=point,
            )
        except RuntimeError as exc:
            if "injected joint batch commit fault" not in str(exc):
                raise
        else:
            raise ODEBFContractError("fault-injected batch transaction unexpectedly committed")
        if any(tensor_sha256(parameters[name]) != tensor_sha256(entry[name]) for name in names):
            raise ODEBFContractError("fault-injected all-layer rollback differs from W0")
        ledger.increment("rollback")
        observed.append(point)
    return tuple(observed)


def _account_evaluation(ledger: ComputeLedger, receipt: ModelEvaluationReceipt) -> None:
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)


def run_p0(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
) -> dict[str, Any]:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P0 alias is not locked")
    _source_freeze(repo_root, source_head)
    expected_parent = (repo_root / "local" / "odebf" / "results").resolve(strict=False)
    destination = output_root.resolve(strict=False)
    if destination.parent != expected_parent or destination.name != expected_result_name(alias):
        raise ODEBFContractError("P0 output namespace differs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("P0 result root is create-once")
    destination.mkdir(mode=0o700, parents=True)
    raw_root = destination / "raw"
    raw_root.mkdir(mode=0o700)
    stages = StageRecorder(raw_root / "stages")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not visible
        or "," in visible
        or visible.casefold() in {"none", "nodevfiles"}
    ):
        raise ODEBFContractError("technical P0 requires exactly one Slurm-visible GPU")
    ledger = ComputeLedger()
    timers = ComponentTimer(ledger)
    started = time.time()

    lock_root = repo_root / "project" / "run_scripts" / "ode_bf" / "locks"
    numerical_path = lock_root / "numerical_lock_p0_r3.json"
    artifact_path = lock_root / "p0_artifact_lock.json"
    seal_path = lock_root / "p0_b10_seal.json"
    sampling_path = lock_root / "p0_cpu_sampling_seal.json"
    guard = ODEBFArtifactGuard(repo_root, artifact_path, alias)
    with timers.measure("artifact_preflight"):
        artifact_receipt = guard.preflight()
        if sha256_file(numerical_path) != NUMERICAL_LOCK_SHA256:
            raise ODEBFContractError("P0 numerical lock byte digest differs")
        numerical, _ = load_rooted_json(
            numerical_path,
            expected_schema="ode-edit-s04-ode-bf-p0-numerical-lock/v1",
        )
        if (
            numerical["instruction_id"] != SCIENTIFIC_LOCK_INSTRUCTION_ID
            or numerical["execution_seed"] != SEED
            or numerical["edit_batch_size"] != 10
            or numerical["rollout"]["k_resolution"] != 8
            or numerical["rollout"]["correction_cycles"] != 1
            or numerical["prospective_technical_path"]["name"]
            != W64_PRIMARY_REFERENCE
            or numerical["prospective_technical_path"]["primary_path"]
            != W64_PRIMARY_PATH
            or numerical["prospective_technical_path"]["reduced_backend"]
            != W64_REDUCED_BACKEND
            or numerical["prospective_technical_path"]["assembler"]
            != W64_ASSEMBLER_REFERENCE
            or numerical["prospective_technical_path"]["w32_fallback"] is not False
            or numerical["ordered_request_digest"]["schema_version"]
            != ORDERED_REQUEST_DIGEST_SCHEMA
            or numerical["ordered_request_digest"]["request_count"] != 10
        ):
            raise ODEBFContractError("P0 numerical lock semantics differ")
        seal_value = json.loads(seal_path.read_text(encoding="utf-8"))
        seal = verify_p0_b10_seal(seal_value)
        sampling, _ = load_rooted_json(
            sampling_path,
            expected_schema="ode-edit-s04-ode-bf-p0-cpu-sampling-seal/v1",
        )
        if sampling["p1_decision_eligible"] is not False:
            raise ODEBFContractError("technical CPU sampling seal became P1 eligible")
        requests = load_sealed_joint_requests(guard.base_guard.dataset, seal)
        sealed_request_order_sha256 = ordered_request_digest_v1(
            [request["request_sha256"] for request in requests]
        )
    stages.record(
        "post_artifact_preflight",
        {
            "artifact_lock_sha256": artifact_receipt.lock_sha256,
            "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
            "seal_root_digest": seal["root_digest"],
            "request_count": len(requests),
            "request_digest_schema": ORDERED_REQUEST_DIGEST_SCHEMA,
            "request_order_sha256": sealed_request_order_sha256,
        },
    )

    seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(0)
    with timers.measure("model_load"):
        model, tokenizer, hparams = load_original_bf16(guard)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    stages.record(
        "post_model_load",
        {
            "parameter_count": parameter_count,
            "floating_dtype_count": 1,
            "floating_dtype": "torch.bfloat16",
            "visible_gpu_count": torch.cuda.device_count(),
        },
    )
    forward_counter = ModelForwardCounter(model, ledger)
    mutation_lock = threading.RLock()
    captured = None
    touched: dict[str, torch.nn.Parameter] = {}
    try:
        with timers.measure("context_generation_twice"):
            contexts, context_sha256 = fresh_contexts_twice(
                model,
                tokenizer,
                seed=SEED,
            )
        _atomic_write_once(
            raw_root / "context_templates.json",
            {
                "schema": "ode-edit-s04-ode-bf-p0-raw-contexts/v1",
                "contexts": contexts,
            },
        )
        stages.record(
            "post_context_lock",
            {
                "context_sha256": context_sha256,
                "fresh_generation_count": 2,
            },
        )

        with timers.measure("joint_native_wb_factor_capture"):
            captured = capture_native_and_wb_joint_endpoint(
                model,
                tokenizer,
                requests,
                hparams,
                guard.projector,
                contexts,
                projector_sha256=guard.spec["projector_sha256"],
                mutation_lock=mutation_lock,
                ledger=ledger,
                model_residual_tolerance=float(
                    numerical["woodbury"]["model_residual_tolerance_proposal"]
                ),
            )
        if captured.initialization.request_order_sha256 != sealed_request_order_sha256:
            raise ODEBFContractError("initialization request digest differs from sealed B10")
        stages.record(
            "post_joint_factor_contract",
            {
                "direct_z_count": captured.initialization.direct_z_initializations,
                "direct_z_sha256": list(captured.direct_z_sha256),
                "key_sha256_by_layer": [list(item) for item in captured.key_sha256_by_layer],
                "factor_shapes": [asdict(item) for item in captured.initialization.factors],
                "canonical_dense_solve": [
                    asdict(item) for item in captured.dense_solve_receipts
                ],
                "solve_reference": ALPHA_SOLVE_REFERENCE,
                "solve_dtype": str(ALPHA_SOLVE_DTYPE),
                "request_digest": {
                    "schema_version": ORDERED_REQUEST_DIGEST_SCHEMA,
                    "sha256": captured.initialization.request_order_sha256,
                },
                "prospective_path": {
                    "name": W64_PRIMARY_REFERENCE,
                    "path": W64_PRIMARY_PATH,
                    "input_boundary_dtype": str(ALPHA_SOLVE_DTYPE),
                    "reduced_dtype": str(W64_REDUCED_DTYPE),
                    "cast_dtype": str(W64_CAST_DTYPE),
                    "endpoint_dtype": str(W64_ENDPOINT_DTYPE),
                    "reduced_backend": W64_REDUCED_BACKEND,
                    "assembler": W64_ASSEMBLER_REFERENCE,
                    "solve_device_classes": sorted(
                        {
                            item.device_class
                            for layer in captured.four_path_layer_receipts
                            for item in layer.path_receipts
                        }
                    ),
                    "w32_fallback": False,
                },
                "target_backward_count": captured.target_backward_count,
            },
        )

        touched = {
            name: dict(model.named_parameters())[name]
            for name in sorted(captured.entry_weights)
        }
        construction_matrix = _comparison_matrix(captured.four_path_layer_receipts)
        stages.record(
            "post_four_path_construction",
            {
                "path_order": list(FOUR_PATH_ORDER),
                "layers": [
                    asdict(item) for item in captured.four_path_layer_receipts
                ],
                "comparison_matrix": construction_matrix,
                "receipt_precedes_cross_solver_equality_assert": True,
            },
        )

        with timers.measure("w64_virtual_evaluation"):
            virtual_trial = CumulativeBF16FunctionalTrial(
                model,
                captured.wb_factors,
                row_block=64,
            )
            with _capture_virtual_endpoint_hashes() as virtual_hash_sets:
                with virtual_trial:
                    virtual_evaluation = _evaluate_with_raw_free_capture(
                        model,
                        tokenizer,
                        requests,
                        alias=alias,
                    )
        virtual_receipt = virtual_evaluation.receipt
        virtual_hashes = {
            name: tuple(sorted(values))
            for name, values in sorted(virtual_hash_sets.items())
        }
        ledger.increment("trial")
        _account_evaluation(ledger, virtual_receipt)
        stages.record(
            "post_w64_virtual",
            {
                "path": W64_PRIMARY_PATH,
                "method": W64_PRIMARY_REFERENCE,
                "max_live_effective_weights": virtual_trial.max_live_effective_weights,
                "maximum_fp32_block_elements": virtual_trial.max_fp32_block_elements,
                "replacement_linear_calls": virtual_trial.replacement_linear_calls,
                "effective_weight_sha256": virtual_hashes,
                "evaluation": _evaluation_payload(virtual_receipt),
            },
        )

        path_evaluations: dict[str, DiagnosticEvaluation] = {}
        path_parameter_sha256: dict[str, dict[str, str]] = {}
        path_runtime: dict[str, dict[str, Any]] = {}
        for path in FOUR_PATH_ORDER:
            holder: list[DiagnosticEvaluation] = []
            candidates = captured.path_candidates[path]
            transaction = AtomicBatchTransaction(
                touched,
                transaction_id=f"four-path-{path.casefold()}-b10",
                mutation_lock=mutation_lock,
            )
            for name in sorted(touched):
                transaction.stage(name, candidates[name])

            def verify_path() -> bool:
                evaluation = _evaluate_with_raw_free_capture(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                )
                holder.append(evaluation)
                return all(
                    tensor_sha256(touched[name])
                    == tensor_sha256(candidates[name])
                    for name in touched
                )

            before_counters = dict(ledger.counters)
            component = f"four_path_{path.casefold()}_atomic_commit_and_verify"
            with timers.measure(component):
                commit = transaction.commit(post_commit_verify=verify_path)
            if len(holder) != 1:
                raise ODEBFContractError("four-path evaluator invocation count differs")
            evaluation = holder[0]
            ledger.increment("commit", commit.commit_count)
            _account_evaluation(ledger, evaluation.receipt)
            path_evaluations[path] = evaluation
            path_parameter_sha256[path] = dict(commit.parameter_sha256)
            path_runtime[path] = {
                "commit_count": commit.commit_count,
                "wall_seconds": ledger.component_wall_seconds[component],
                "gpu_seconds": ledger.component_gpu_seconds[component],
                "counter_delta": {
                    name: ledger.counters[name] - before_counters[name]
                    for name in sorted(ledger.counters)
                },
                "allocated_bytes_after": int(torch.cuda.memory_allocated(0)),
                "reserved_bytes_after": int(torch.cuda.memory_reserved(0)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
            _restore_entry(
                touched,
                captured.entry_weights,
                mutation_lock=mutation_lock,
            )

        with timers.measure("fault_injection_rollback"):
            fault_points = _fault_rollback_gate(
                touched,
                captured.wb_candidates,
                captured.entry_weights,
                mutation_lock=mutation_lock,
                ledger=ledger,
            )
        final_w0_restored = all(
            tensor_sha256(touched[name]) == captured.entry_sha256[name]
            for name in touched
        )
        rollback_and_restore_exact = (
            final_w0_restored and fault_points == (0, 2, 4)
        )

        layer_path_receipts = [
            path_receipt
            for layer_receipt in captured.four_path_layer_receipts
            for path_receipt in layer_receipt.path_receipts
        ]
        layer_inventory_exact = (
            len(captured.four_path_layer_receipts) == 5
            and len(captured.initialization.factors) == 5
            and len({item.layer for item in captured.four_path_layer_receipts}) == 5
            and all(
                tuple(item.path for item in layer.path_receipts)
                == FOUR_PATH_ORDER
                for layer in captured.four_path_layer_receipts
            )
        )
        joint_rank_exact = all(
            item.joint_rank == 10 for item in layer_path_receipts
        )
        device_identity_exact = all(
            len({item.device_class for item in layer.path_receipts}) == 1
            and {item.device_class for item in layer.path_receipts} == {"cuda"}
            for layer in captured.four_path_layer_receipts
        )
        source_inputs_identical = (
            layer_inventory_exact
            and joint_rank_exact
            and device_identity_exact
            and captured.initialization.request_order_sha256
            == sealed_request_order_sha256
            and all(
                len({item.source_identity_sha256 for item in layer.path_receipts})
                == 1
                and all(
                    item.receipt.request_order_sha256
                    == captured.initialization.request_order_sha256
                    for item in path_evaluations.values()
                )
                for layer in captured.four_path_layer_receipts
            )
        )
        all_finite = all(item.finite for item in layer_path_receipts)
        required_paths = ("N32", "D32", W64_PRIMARY_PATH)
        required_paths_finite = all(
            item.finite
            for item in layer_path_receipts
            if item.path in required_paths
        )
        reference_event = path_evaluations["N32"].receipt.batch_success.raw_free_payload()
        all_path_benchmark_bits_exact = all(
            item.receipt.batch_success.raw_free_payload() == reference_event
            for item in path_evaluations.values()
        )
        n32_w64_benchmark_bits_exact = (
            path_evaluations[W64_PRIMARY_PATH].receipt.batch_success.raw_free_payload()
            == reference_event
        )
        canonical_request_digest_exact = all(
            item.receipt.request_order_sha256
            == captured.initialization.request_order_sha256
            for item in path_evaluations.values()
        )
        request_and_span_parity = canonical_request_digest_exact and all(
            item.receipt.request_order_sha256
            == path_evaluations["N32"].receipt.request_order_sha256
            and item.receipt.target_span_lengths
            == path_evaluations["N32"].receipt.target_span_lengths
            for item in path_evaluations.values()
        )
        path_certificate_pass = {
            path: all(
                item.certificate_passed
                for item in layer_path_receipts
                if item.path == path
            )
            for path in FOUR_PATH_ORDER
        }
        required_certificates_pass = all(
            path_certificate_pass[path] for path in required_paths
        )
        decisions = {
            path: _technical_decision_payload(
                path_evaluations[path].receipt,
                certificates_pass=path_certificate_pass[path],
            )
            for path in FOUR_PATH_ORDER
        }
        decisions_exact = len(
            {value["decision_vector_sha256"] for value in decisions.values()}
        ) == 1
        n32_w64_decisions_exact = (
            decisions["N32"]["decision_vector_sha256"]
            == decisions[W64_PRIMARY_PATH]["decision_vector_sha256"]
        )
        expected_w64_hashes = {
            name: tensor_sha256(candidate)
            for name, candidate in captured.path_candidates[W64_PRIMARY_PATH].items()
        }
        virtual_w64_parameter_bytes_exact = (
            set(virtual_hashes) == set(expected_w64_hashes)
            and all(
                virtual_hashes[name] == (expected_w64_hashes[name],)
                for name in expected_w64_hashes
            )
        )
        virtual_w64_logits_exact = (
            virtual_receipt.target_full_vocabulary_logits_sha256
            == path_evaluations[W64_PRIMARY_PATH].receipt.target_full_vocabulary_logits_sha256
        )
        virtual_w64_event_exact = (
            virtual_receipt.batch_success.raw_free_payload()
            == path_evaluations[W64_PRIMARY_PATH].receipt.batch_success.raw_free_payload()
        )
        strict_w64_virtual_commit_exact = (
            virtual_w64_parameter_bytes_exact
            and virtual_w64_logits_exact
            and virtual_w64_event_exact
            and all(
                path_parameter_sha256[W64_PRIMARY_PATH][name]
                == expected_w64_hashes[name]
                for name in expected_w64_hashes
            )
        )
        boundary_touched = any(
            score.target_new_nll == score.target_true_nll
            for path in ("N32", W64_PRIMARY_PATH)
            for item in (path_evaluations[path],)
            for score in (item.receipt.counterfact_scores or ())
        )
        all_endpoint_bytes_exact = all(
            item.byte_exact
            for layer in captured.four_path_layer_receipts
            for item in layer.comparisons_to_n32
        )
        classification = classify_w64_technical_candidate(
            source_inputs_identical=source_inputs_identical,
            required_paths_finite=required_paths_finite,
            required_certificates_pass=required_certificates_pass,
            n32_w64_benchmark_bits_exact=n32_w64_benchmark_bits_exact,
            n32_w64_decisions_exact=n32_w64_decisions_exact,
            strict_w64_virtual_commit_exact=strict_w64_virtual_commit_exact,
            rollback_and_restore_exact=rollback_and_restore_exact,
            boundary_touched=boundary_touched,
            request_and_span_parity=request_and_span_parity,
        )
        first_boundary = next(
            (
                {
                    "boundary": f"{left}_vs_{right}",
                    "layer": layer.layer,
                }
                for left, right in (
                    ("N32", "D32"),
                    ("D32", "W32"),
                    ("W32", "W64"),
                )
                for layer in captured.four_path_layer_receipts
                for item in layer.adjacent_comparisons
                if item.reference_path == left
                and item.candidate_path == right
                and not item.byte_exact
            ),
            None,
        )
        functional_differences = {
            path: _functional_difference_summary(
                path_evaluations["N32"],
                path_evaluations[path],
            )
            for path in FOUR_PATH_ORDER
        }
        diagnostic = {
            "schema": "ode-edit-s04-ode-bf-w64-canonical-receipt/v1",
            "instruction_id": INSTRUCTION_ID,
            "alias": alias,
            "classification": classification,
            "promotion_authorized": False,
            "native_equivalence_claim_authorized": False,
            "prospective_path": W64_PRIMARY_PATH,
            "prospective_method": W64_PRIMARY_REFERENCE,
            "w32_policy": "diagnostic-only-no-fallback",
            "path_order": list(FOUR_PATH_ORDER),
            "source_inputs_identical": source_inputs_identical,
            "layer_inventory_exact": layer_inventory_exact,
            "joint_rank_exact": joint_rank_exact,
            "device_identity_exact": device_identity_exact,
            "all_finite": all_finite,
            "required_paths_finite": required_paths_finite,
            "required_certificates_pass": required_certificates_pass,
            "path_certificate_pass": path_certificate_pass,
            "w32_diagnostic_certificate_pass": path_certificate_pass["W32"],
            "all_path_benchmark_bits_exact": all_path_benchmark_bits_exact,
            "n32_w64_benchmark_bits_exact": n32_w64_benchmark_bits_exact,
            "canonical_request_digest_exact": canonical_request_digest_exact,
            "request_and_span_parity": request_and_span_parity,
            "all_path_decision_parity": decisions_exact,
            "n32_w64_decision_parity": n32_w64_decisions_exact,
            "boundary_touched": boundary_touched,
            "strict_w64_virtual_commit": {
                "executed": True,
                "parameter_bytes_exact": virtual_w64_parameter_bytes_exact,
                "logits_exact": virtual_w64_logits_exact,
                "event_exact": virtual_w64_event_exact,
            },
            "precision_contract": {
                "input_boundary_dtype": str(ALPHA_SOLVE_DTYPE),
                "reduced_dtype": str(W64_REDUCED_DTYPE),
                "cast_dtype": str(W64_CAST_DTYPE),
                "endpoint_dtype": str(W64_ENDPOINT_DTYPE),
                "cast_count": 1,
                "reduced_backend": W64_REDUCED_BACKEND,
                "solve_device_classes": sorted(
                    {item.device_class for item in layer_path_receipts}
                ),
                "assembler": W64_ASSEMBLER_REFERENCE,
                "dense_fp64_full_delta_live": 0,
                "w32_fallback_count": 0,
            },
            "request_digest_schema": ORDERED_REQUEST_DIGEST_SCHEMA,
            "request_digest_sha256": captured.initialization.request_order_sha256,
            "rollback_and_restore_exact": rollback_and_restore_exact,
            "comparison_matrix": construction_matrix,
            "first_separating_boundary": first_boundary,
            "layers": [asdict(item) for item in captured.four_path_layer_receipts],
            "functional_differences_to_n32": functional_differences,
            "evaluations": {
                path: _evaluation_payload(item.receipt)
                for path, item in path_evaluations.items()
            },
            "committed_parameter_sha256": path_parameter_sha256,
            "decisions": decisions,
            "path_runtime": path_runtime,
            "fault_after_writes": list(fault_points),
            "scientific_outcome_count": 0,
            "heldout_generalization_access_count": 0,
            "heldout_locality_access_count": 0,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
        }
        diagnostic_sha256 = _atomic_write_once(
            raw_root / "four_path_diagnostic.json",
            diagnostic,
        )
        identity = {
            "classification": classification,
            "cross_solver_parameter_bytes_exact": all_endpoint_bytes_exact,
            "n32_w64_benchmark_bits_exact": n32_w64_benchmark_bits_exact,
            "n32_w64_decision_parity": n32_w64_decisions_exact,
            "canonical_request_digest_exact": canonical_request_digest_exact,
            "virtual_w64_parameter_bytes_exact": virtual_w64_parameter_bytes_exact,
            "virtual_w64_logits_exact": virtual_w64_logits_exact,
            "virtual_w64_event_exact": virtual_w64_event_exact,
            "w32_diagnostic_certificate_pass": path_certificate_pass["W32"],
            "w32_fallback_count": 0,
            "all_layer_rollback_exact": rollback_and_restore_exact,
            "final_w0_restored": final_w0_restored,
        }
        stages.record(
            "post_identity_verdict",
            {
                "identity": identity,
                "diagnostic_sha256": diagnostic_sha256,
                "first_separating_boundary": first_boundary,
                "fault_after_writes": list(fault_points),
                "all_layer_rollback_exact": rollback_and_restore_exact,
                "final_w0_restored": final_w0_restored,
            },
        )
        if classification == "W64_TECHNICAL_HARD_GATE_FAIL":
            raise ODEBFContractError(
                "W64 canonical technical candidate failed closed"
            )

        ledger.counters["effective_bf16_weight_peak_live"] = max(
            ledger.counters["effective_bf16_weight_peak_live"],
            virtual_trial.max_live_effective_weights,
        )
        ledger.observe_memory(
            allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
            maxrss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        )
        guard.assert_unchanged()
        elapsed = time.time() - started
        terminal = {
            "schema": "ode-edit-s04-ode-bf-p0-w64-canonical-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "status": "PASS_W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM",
            "classification": classification,
            "promotion_authorized": False,
            "native_equivalence_claim_authorized": False,
            "alias": alias,
            "source_head": source_head,
            "edit_batch_size": 10,
            "joint_editor_invocations": 1,
            "k_resolution": 8,
            "correction_cycles": 1,
            "model_p0_mode": (
                "b10-N32-reference-W64-mixed64-canonical-receipt-with-W32-diagnostic"
            ),
            "prospective_method": W64_PRIMARY_REFERENCE,
            "prospective_path": W64_PRIMARY_PATH,
            "w32_policy": "diagnostic-only-no-fallback",
            "scientific_lock_instruction_id": SCIENTIFIC_LOCK_INSTRUCTION_ID,
            "alpha_solve": {
                "reference": ALPHA_SOLVE_REFERENCE,
                "dtype": str(ALPHA_SOLVE_DTYPE),
                "receipts": [
                    asdict(item) for item in captured.dense_solve_receipts
                ],
            },
            "fixed_k_controller_contract_cpu_gate": True,
            "precision_contract": diagnostic["precision_contract"],
            "request_digest": {
                "schema_version": ORDERED_REQUEST_DIGEST_SCHEMA,
                "sha256": captured.initialization.request_order_sha256,
            },
            "identity": identity,
            "four_path_diagnostic_sha256": diagnostic_sha256,
            "first_separating_boundary": first_boundary,
            "comparison_matrix": construction_matrix,
            "path_evaluations": {
                path: _evaluation_payload(item.receipt)
                for path, item in path_evaluations.items()
            },
            "committed_parameter_sha256": path_parameter_sha256,
            "functional_differences_to_n32": functional_differences,
            "path_runtime": path_runtime,
            "four_path_layer_receipts": [
                asdict(item) for item in captured.four_path_layer_receipts
            ],
            "initialization": {
                **asdict(captured.initialization),
                "factors": [asdict(item) for item in captured.initialization.factors],
            },
            "woodbury_certificates": [
                {
                    "layer": layer,
                    "method": certificate.method.value,
                    "small_dimension": certificate.small_dimension,
                    "small_condition": certificate.small_condition,
                    "small_residual": certificate.small_residual,
                    "alpha_linear_residual": certificate.alpha_linear_residual,
                    "passed": certificate.passed,
                }
                for layer, certificate in captured.woodbury_certificates
            ],
            "context_sha256": context_sha256,
            "artifact_receipt": asdict(artifact_receipt),
            "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
            "seal_root_digest": seal["root_digest"],
            "sampling_seal_root_digest": sampling["root_digest"],
            "compute": ledger.raw_free_payload(),
            "elapsed_seconds": elapsed,
            "scientific_outcome_count": 0,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
            "heldout_paraphrase_access_count": 0,
            "heldout_locality_access_count": 0,
            "heldout_generation_access_count": 0,
            "benchmark_evaluator_generation_call_count": 0,
            "editor_context_generation_count": 2,
            "retry_count": 0,
        }
        terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
        summary = {
            "schema": "ode-edit-s04-ode-bf-p0-w64-canonical-summary/v1",
            "status": "PASS_W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM",
            "classification": classification,
            "promotion_authorized": False,
            "alias": alias,
            "terminal_sha256": terminal_sha,
            "four_path_diagnostic_sha256": diagnostic_sha256,
            "first_separating_boundary": first_boundary,
            "comparison_matrix": construction_matrix,
            "identity": identity,
            "peak_allocated_bytes": ledger.peak_allocated_bytes,
            "peak_reserved_bytes": ledger.peak_reserved_bytes,
            "host_maxrss_kib": ledger.host_maxrss_kib,
            "elapsed_seconds": elapsed,
            "scientific_outcome_count": 0,
        }
        summary_sha = _atomic_write_once(destination / "summary.json", summary)
        manifest = {
            "schema": "ode-edit-s04-ode-bf-p0-w64-canonical-manifest/v1",
            "status": "PASS_W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM",
            "classification": classification,
            "promotion_authorized": False,
            "alias": alias,
            "source_head": source_head,
            "edit_batch_size": 10,
            "scientific_outcome_count": 0,
            "files": {
                "terminal.json": terminal_sha,
                "summary.json": summary_sha,
            },
        }
        manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
        return {
            "terminal_sha256": terminal_sha,
            "summary_sha256": summary_sha,
            "manifest_sha256": manifest_sha,
            **summary,
        }
    finally:
        if captured is not None and touched:
            _restore_entry(
                touched,
                captured.entry_weights,
                mutation_lock=mutation_lock,
            )
        forward_counter.close()
        try:
            from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

            alpha_main.CONTEXT_TEMPLATES_CACHE = None
        except Exception:
            pass


def _sanitized_failure(
    exc: BaseException,
    recorder: StageRecorder | None,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    allowed_root = repo_root.resolve(strict=False)
    frames: list[dict[str, Any]] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        path = Path(frame.filename).resolve(strict=False)
        try:
            relative = str(path.relative_to(allowed_root))
        except ValueError:
            if "easyeditor/models/alphaedit" not in str(path):
                continue
            relative = "/".join(path.parts[-4:])
        frames.append(
            {
                "file": relative,
                "function": frame.name,
                "line": frame.lineno,
            }
        )
    return {
        "schema": "ode-edit-s04-ode-bf-p0-failure/v1",
        "status": "FAIL",
        "last_completed_stage": None if recorder is None else recorder.last_stage,
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "allowlisted_frames": frames,
        "retry_permitted": False,
    }


def write_failure_once(
    output_root: Path,
    exc: BaseException,
    *,
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    destination = output_root.resolve(strict=False)
    expected_parent = (repo_root / "local" / "odebf" / "results").resolve(
        strict=False
    )
    allowed_names = {expected_result_name(alias) for alias in MODEL_ALIASES}
    if destination.parent != expected_parent or destination.name not in allowed_names:
        raise ODEBFContractError("P0 failure output namespace differs")
    if not destination.exists():
        expected_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.mkdir(mode=0o700)
    if destination.is_symlink() or not destination.is_dir():
        raise ODEBFContractError("P0 failure output root is not a real directory")
    stages_root = destination / "raw" / "stages"
    recorder: StageRecorder | None = None
    if stages_root.is_dir():
        stage_files = sorted(stages_root.glob("stage-*.json"))
        recorder = StageRecorder(stages_root)
        recorder.sequence = len(stage_files)
        if stage_files:
            recorder.last_stage = json.loads(
                stage_files[-1].read_text(encoding="utf-8")
            )["stage"]
    value = _sanitized_failure(exc, recorder, repo_root=repo_root)
    path = destination / "failure.json"
    if path.exists():
        raise FileExistsError("P0 failure receipt is create-once")
    return _atomic_write_once(path, value), value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--alias", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    try:
        result = run_p0(
            repo_root=args.repo_root,
            alias=args.alias,
            output_root=args.output_root,
            source_head=args.source_head,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        failure_sha256, failure = write_failure_once(
            args.output_root,
            exc,
            repo_root=args.repo_root,
        )
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_NO_RETRY",
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure["exception_message_sha256"],
                    "failure_sha256": failure_sha256,
                    "last_completed_stage": failure["last_completed_stage"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
