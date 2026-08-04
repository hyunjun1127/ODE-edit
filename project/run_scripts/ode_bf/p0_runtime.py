"""Outcome-free model P0: Native AlphaEdit versus exact AlphaEdit-WB on B10."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from .accounting import ComputeLedger
from .alpha_backend import (
    capture_native_and_wb_joint_endpoint,
    fresh_contexts_twice,
    load_original_bf16,
    seed_all,
)
from .artifacts import ODEBFArtifactGuard, load_rooted_json, sha256_file
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .evaluator import ModelEvaluationReceipt, evaluate_counterfact_rewrite_batch
from .functional import CumulativeBF16FunctionalTrial, tensor_sha256
from .selection import load_sealed_joint_requests, verify_p0_b10_seal
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-V1P1-EXACT-FIRST-HIT-CPU-P0-V1-A3"
NUMERICAL_LOCK_SHA256 = "23fe5621612f715c52ef70f94a10ae7eaba759b4ac209e0e0f3e09c81ea9feef"
NUMERICAL_LOCK_PREFIX = NUMERICAL_LOCK_SHA256[:8]
SEED = 41


def expected_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P0 result alias is not locked")
    return f"s04-p0-native-wb-b10-{alias}-{NUMERICAL_LOCK_PREFIX}"


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
    numerical_path = lock_root / "numerical_lock_p0.json"
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
            numerical["instruction_id"] != INSTRUCTION_ID
            or numerical["execution_seed"] != SEED
            or numerical["edit_batch_size"] != 10
            or numerical["rollout"]["k_resolution"] != 8
            or numerical["rollout"]["correction_cycles"] != 1
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
    stages.record(
        "post_artifact_preflight",
        {
            "artifact_lock_sha256": artifact_receipt.lock_sha256,
            "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
            "seal_root_digest": seal["root_digest"],
            "request_count": len(requests),
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
        stages.record(
            "post_joint_factor_contract",
            {
                "direct_z_count": captured.initialization.direct_z_initializations,
                "direct_z_sha256": list(captured.direct_z_sha256),
                "key_sha256_by_layer": [list(item) for item in captured.key_sha256_by_layer],
                "factor_shapes": [asdict(item) for item in captured.initialization.factors],
                "target_backward_count": captured.target_backward_count,
            },
        )

        touched = {
            name: dict(model.named_parameters())[name]
            for name in sorted(captured.entry_weights)
        }
        with timers.measure("q0_virtual_evaluation"):
            virtual_trial = CumulativeBF16FunctionalTrial(
                model,
                captured.wb_factors,
                row_block=64,
            )
            with virtual_trial:
                virtual_receipt = evaluate_counterfact_rewrite_batch(
                    model,
                    tokenizer,
                    requests,
                    model_alias=alias,
                )
        ledger.increment("trial")
        _account_evaluation(ledger, virtual_receipt)
        stages.record(
            "post_q0_virtual",
            {
                "max_live_effective_weights": virtual_trial.max_live_effective_weights,
                "maximum_fp32_block_elements": virtual_trial.max_fp32_block_elements,
                "evaluation": _evaluation_payload(virtual_receipt),
            },
        )

        native_holder: list[ModelEvaluationReceipt] = []
        native_transaction = AtomicBatchTransaction(
            touched,
            transaction_id="native-alphaedit-b10",
            mutation_lock=mutation_lock,
        )
        for name in sorted(touched):
            native_transaction.stage(name, captured.native_candidates[name])

        def verify_native() -> bool:
            receipt = evaluate_counterfact_rewrite_batch(
                model,
                tokenizer,
                requests,
                model_alias=alias,
            )
            native_holder.append(receipt)
            return all(
                tensor_sha256(touched[name])
                == tensor_sha256(captured.native_candidates[name])
                for name in touched
            )

        with timers.measure("native_atomic_commit_and_verify"):
            native_commit = native_transaction.commit(post_commit_verify=verify_native)
        native_receipt = native_holder[0]
        ledger.increment("commit", native_commit.commit_count)
        _account_evaluation(ledger, native_receipt)
        native_parameter_sha = dict(native_commit.parameter_sha256)
        _restore_entry(touched, captured.entry_weights, mutation_lock=mutation_lock)
        stages.record(
            "post_native_writer",
            {
                "commit_count": native_commit.commit_count,
                "parameter_sha256": native_parameter_sha,
                "evaluation": _evaluation_payload(native_receipt),
                "w0_restored": True,
            },
        )

        wb_holder: list[ModelEvaluationReceipt] = []
        wb_transaction = AtomicBatchTransaction(
            touched,
            transaction_id="native-alphaedit-wb-b10",
            mutation_lock=mutation_lock,
        )
        for name in sorted(touched):
            wb_transaction.stage(name, captured.wb_candidates[name])

        def verify_wb() -> bool:
            receipt = evaluate_counterfact_rewrite_batch(
                model,
                tokenizer,
                requests,
                model_alias=alias,
            )
            wb_holder.append(receipt)
            return all(
                tensor_sha256(touched[name])
                == tensor_sha256(captured.wb_candidates[name])
                for name in touched
            )

        with timers.measure("wb_atomic_commit_and_verify"):
            wb_commit = wb_transaction.commit(post_commit_verify=verify_wb)
        wb_receipt = wb_holder[0]
        ledger.increment("commit", wb_commit.commit_count)
        _account_evaluation(ledger, wb_receipt)
        wb_parameter_sha = dict(wb_commit.parameter_sha256)
        identity = {
            "native_wb_parameter_bytes_exact": native_parameter_sha == wb_parameter_sha,
            "native_wb_logits_exact": (
                native_receipt.target_full_vocabulary_logits_sha256
                == wb_receipt.target_full_vocabulary_logits_sha256
            ),
            "native_wb_event_exact": (
                native_receipt.batch_success.raw_free_payload()
                == wb_receipt.batch_success.raw_free_payload()
            ),
            "virtual_wb_logits_exact": (
                virtual_receipt.target_full_vocabulary_logits_sha256
                == wb_receipt.target_full_vocabulary_logits_sha256
            ),
            "virtual_wb_event_exact": (
                virtual_receipt.batch_success.raw_free_payload()
                == wb_receipt.batch_success.raw_free_payload()
            ),
        }
        if not all(identity.values()):
            raise ODEBFContractError("Native/WB/virtual technical identity failed")
        _restore_entry(touched, captured.entry_weights, mutation_lock=mutation_lock)

        with timers.measure("fault_injection_rollback"):
            fault_points = _fault_rollback_gate(
                touched,
                captured.wb_candidates,
                captured.entry_weights,
                mutation_lock=mutation_lock,
                ledger=ledger,
            )
        stages.record(
            "post_identity_verdict",
            {
                "identity": identity,
                "fault_after_writes": list(fault_points),
                "all_layer_rollback_exact": True,
                "final_w0_restored": all(
                    tensor_sha256(touched[name]) == captured.entry_sha256[name]
                    for name in touched
                ),
            },
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
            "schema": "ode-edit-s04-ode-bf-p0-terminal/v1",
            "instruction_id": INSTRUCTION_ID,
            "status": "PASS",
            "alias": alias,
            "source_head": source_head,
            "edit_batch_size": 10,
            "joint_editor_invocations": 1,
            "k_resolution": 8,
            "correction_cycles": 1,
            "model_p0_mode": "q0-native-alphaedit-versus-native-alphaedit-wb-identity",
            "fixed_k_controller_contract_cpu_gate": True,
            "identity": identity,
            "native_evaluation": _evaluation_payload(native_receipt),
            "wb_evaluation": _evaluation_payload(wb_receipt),
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
            "heldout_paraphrase_access_count": 0,
            "heldout_locality_access_count": 0,
            "heldout_generation_access_count": 0,
            "benchmark_evaluator_generation_call_count": 0,
            "editor_context_generation_count": 2,
            "retry_count": 0,
        }
        terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
        summary = {
            "schema": "ode-edit-s04-ode-bf-p0-summary/v1",
            "status": "PASS",
            "alias": alias,
            "terminal_sha256": terminal_sha,
            "identity": identity,
            "peak_allocated_bytes": ledger.peak_allocated_bytes,
            "peak_reserved_bytes": ledger.peak_reserved_bytes,
            "host_maxrss_kib": ledger.host_maxrss_kib,
            "elapsed_seconds": elapsed,
            "scientific_outcome_count": 0,
        }
        summary_sha = _atomic_write_once(destination / "summary.json", summary)
        manifest = {
            "schema": "ode-edit-s04-ode-bf-p0-manifest/v1",
            "status": "PASS",
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
