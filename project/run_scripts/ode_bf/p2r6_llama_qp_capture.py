"""One-shot observation-only Llama outer-1 P2R6 QP capture.

The hook runs the selected AR-CAP physical prefix through outer step zero, seals
the exact outer-step-one QP inputs before the selected route solve, and raises a
typed intentional stop.  It never evaluates the terminal scientific endpoint.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p2r5_stage_a_runtime import STREAM_ORDER, STREAM_ROOT, run_p2_atomic_arm_case
from .p2r6_certified_convex_qp import (
    P2R6_QP_EXTERNAL_TOLERANCE,
    P2R6_QP_INTERNAL_TOLERANCE,
)
from .p2r6_pilot_runtime import P2R6_RUNTIME_POLICY
from .p2r6_semantic_region_controller import (
    P2R6_INSTRUCTION_ID,
    P2R6_NUMERICAL_EPSILON,
    _mass_matrix,
    _semantic_region_optimum,
)
from .scalable_batched_runtime import scalable_ordered_request_digest


CAPTURE_INSTRUCTION_ID = (
    "ODEEDIT-S05-P2R6-LLAMA-OUTER1-QP-CAPTURE-REPAIR-PHASE1-V1"
)
CAPTURE_METHOD_ID = "P2R6-LLAMA-AR-CAP-OUTER1-PRE-QP-CAPTURE-V1"
CAPTURE_ALIAS = "llama3-8b-inst"
CAPTURE_CASE_INDEX = 5
CAPTURE_ARM = "AR-CAP"
CAPTURE_OUTER_STEP = 1
FAILED_SCIENTIFIC_SOURCE_HEAD = "1246e5047bdac2e59d5cc0642ebd8d7104c6d066"
FAILED_SCIENTIFIC_SOURCE_TREE = "ed7c236066afe1c8140ac8b306d5be1785cc08c5"
PYTHONHASHSEED = "48"


def expected_capture_result_name(attempt_suffix: str | None = None) -> str:
    suffix = f"-{attempt_suffix}" if attempt_suffix else ""
    return f"s05-p2r6-llama-outer1-qp-capture-llama3-8b-inst-case-05-ar-cap{suffix}-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_numpy_once(path: Path, value: np.ndarray) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P2R6 capture member is create-once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)
    return {
        "member": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "finite": bool(np.all(np.isfinite(value))),
    }


class P2R6LlamaCaptureComplete(ODEBFStateError):
    """Typed intentional stop after the private capsule seal."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.raw_free_receipt = dict(receipt)
        super().__init__("P2R6 Llama outer1 pre-QP capture intentionally complete")


class LlamaOuter1QPCaptureObserver:
    """Seal exact selected-AR QP inputs without changing a route decision."""

    def __init__(
        self,
        *,
        capsule_root: Path,
        source_head: str,
        request_order_sha256: str,
        expected_w0: str,
    ) -> None:
        self.capsule_root = capsule_root
        self.source_head = source_head
        self.request_order_sha256 = request_order_sha256
        self.expected_w0 = expected_w0
        self.invocation_count = 0
        self.capture_count = 0

    def __call__(
        self,
        response: torch.Tensor,
        deficit: torch.Tensor,
        entry_deficit: torch.Tensor,
        calibration: Any,
        quadratics: Any,
        *,
        arm: str,
        outer_step: int,
    ) -> None:
        self.invocation_count += 1
        if arm != CAPTURE_ARM:
            raise ODEBFContractError("P2R6 capture arm differs")
        if outer_step < CAPTURE_OUTER_STEP:
            return
        if outer_step != CAPTURE_OUTER_STEP or self.capture_count != 0:
            raise ODEBFStateError("P2R6 capture invocation budget differs")

        observed = response.detach().to(device="cpu", dtype=torch.float64).numpy()
        deficit_np = deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
        entry_np = entry_deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
        alpha_start, scale, lower, xi, e1_receipt = _semantic_region_optimum(
            observed,
            deficit_np,
            entry_np,
            scale_policy="CURRENT_DEFICIT",
        )
        capacity = (
            quadratics.capacity_gram.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        capacity_cross = (
            quadratics.capacity_cross.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        mass = _mass_matrix(5, BATCH_SIZE)
        alpha_count = int(alpha_start.size)
        linear_matrix = np.concatenate((-np.eye(alpha_count), mass, -observed), axis=0)
        linear_upper = np.concatenate(
            (np.zeros(alpha_count), np.ones(BATCH_SIZE), -lower), axis=0
        )
        raw_slack = linear_upper - linear_matrix @ alpha_start
        old_row_scale = np.maximum(
            1.0,
            np.maximum(np.linalg.norm(linear_matrix, axis=1), np.abs(linear_upper)),
        )
        bidirectional_row_scale = np.maximum(
            P2R6_QP_INTERNAL_TOLERANCE,
            np.maximum(np.linalg.norm(linear_matrix, axis=1), np.abs(linear_upper)),
        )
        ordered_names = [f"alpha_nonnegative[{index}]" for index in range(alpha_count)]
        ordered_names.extend(f"request_mass_upper[{index}]" for index in range(BATCH_SIZE))
        ordered_names.extend(
            f"semantic_response_lower[{index}]" for index in range(BATCH_SIZE)
        )
        active = np.flatnonzero(raw_slack <= P2R6_QP_EXTERNAL_TOLERANCE)
        active_jacobian = linear_matrix[active]
        singular_values = (
            np.linalg.svd(active_jacobian, compute_uv=False)
            if active_jacobian.size
            else np.empty(0, dtype=np.float64)
        )
        active_rank = int(
            np.linalg.matrix_rank(active_jacobian, tol=P2R6_QP_EXTERNAL_TOLERANCE)
        )
        active_condition = (
            float(np.linalg.cond(active_jacobian))
            if active_jacobian.size and active_rank == min(active_jacobian.shape)
            else float("inf")
        )

        arrays = {
            "S": observed,
            "d": deficit_np,
            "b": lower,
            "Q_C": capacity,
            "c_C": capacity_cross,
            "M": mass,
            "alpha_start": alpha_start,
            "s": scale,
            "linear_matrix": linear_matrix,
            "linear_upper": linear_upper,
            "raw_slack": raw_slack,
            "old_row_scale": old_row_scale,
            "bidirectional_row_scale": bidirectional_row_scale,
        }
        if not all(np.all(np.isfinite(item)) for item in arrays.values()):
            raise ODEBFStateError("P2R6 capture array is nonfinite")
        self.capsule_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.capsule_root.chmod(0o700)
        members = {
            name: _write_numpy_once(self.capsule_root / f"{name}.npy", value)
            for name, value in arrays.items()
        }
        capsule = {
            "schema": "ode-edit-s05-p2r6-llama-outer1-private-qp-capsule/v1",
            "instruction_id": CAPTURE_INSTRUCTION_ID,
            "method_id": CAPTURE_METHOD_ID,
            "privacy": "PRIVATE_TECHNICAL_REPLAY_NOT_FOR_PUBLIC_HANDOFF",
            "failed_scientific_source_head": FAILED_SCIENTIFIC_SOURCE_HEAD,
            "failed_scientific_source_tree": FAILED_SCIENTIFIC_SOURCE_TREE,
            "capture_source_head": self.source_head,
            "model_alias": CAPTURE_ALIAS,
            "case_index": CAPTURE_CASE_INDEX,
            "arm": arm,
            "outer_step": outer_step,
            "request_order_sha256": self.request_order_sha256,
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "common_seed": COMMON_SEED,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "NOT_SET"),
            "expected_pythonhashseed": PYTHONHASHSEED,
            "entry_w0_contract_sha256": self.expected_w0,
            "xi": float(xi),
            "e1_receipt": dict(e1_receipt),
            "calibration": calibration.raw_free_payload(),
            "ordered_constraint_names": ordered_names,
            "constraint_order": ["alpha_nonnegative", "request_mass_upper", "semantic_response_lower"],
            "active_constraint_indices": active.tolist(),
            "active_constraint_names": [ordered_names[int(index)] for index in active],
            "active_constraint_count": int(active.size),
            "active_jacobian_rank": active_rank,
            "active_jacobian_nullity": int(alpha_count - active_rank),
            "active_jacobian_condition": (
                active_condition if np.isfinite(active_condition) else "INFINITE"
            ),
            "active_jacobian_singular_values": singular_values.tolist(),
            "raw_max_violation": max(0.0, -float(np.min(raw_slack))),
            "raw_min_slack": float(np.min(raw_slack)),
            "objective_min_eigenvalue": float(np.min(np.linalg.eigvalsh(capacity))),
            "objective_max_eigenvalue": float(np.max(np.linalg.eigvalsh(capacity))),
            "old_row_scale_formula": "max(1,norm2(A_i),abs(rhs_i))",
            "candidate_bidirectional_row_scale_formula": (
                "max(norm2(A_i),abs(rhs_i),P2R6_QP_INTERNAL_TOLERANCE)"
            ),
            "scientific_feasible_set_envelope": 0.0,
            "b_relaxation": 0.0,
            "added_model_forward_count": 0,
            "added_model_backward_count": 0,
            "added_materialization_count": 0,
            "terminal_evaluator_count": 0,
            "scientific_endpoint_count": 0,
            "capture_invocation_budget": 1,
            "members": members,
        }
        capsule["identity_sha256"] = canonical_hash(capsule)
        capsule_sha = _atomic_write_once(self.capsule_root / "capsule.json", capsule)
        self.capture_count += 1
        stop_receipt = {
            "schema": "ode-edit-s05-p2r6-llama-outer1-capture-intentional-stop/v1",
            "instruction_id": CAPTURE_INSTRUCTION_ID,
            "capsule_path": str(self.capsule_root),
            "capsule_sha256": capsule_sha,
            "capsule_identity_sha256": capsule["identity_sha256"],
            "capture_count": self.capture_count,
            "observer_invocation_count": self.invocation_count,
            "outer0_selected_physical_write_completed": True,
            "outer1_selected_qp_solve_count": 0,
            "terminal_evaluator_count": 0,
            "scientific_endpoint_count": 0,
        }
        stop_receipt["identity_sha256"] = canonical_hash(stop_receipt)
        raise P2R6LlamaCaptureComplete(stop_receipt)


def run_p2r6_llama_qp_capture(
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
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
) -> dict[str, Any]:
    if (
        alias != CAPTURE_ALIAS
        or len(stream_batches) != 10
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("P2R6 Llama capture input differs")
    expected_w0 = _model_w0_contract(touched)
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P2R6 Llama capture entry W0 differs")
    requests = stream_batches[CAPTURE_CASE_INDEX - 1]
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    capsule_root = destination / "private-replay" / "llama-case05-ar-cap-outer01"
    observer = LlamaOuter1QPCaptureObserver(
        capsule_root=capsule_root,
        source_head=source_head,
        request_order_sha256=request_order,
        expected_w0=expected_w0,
    )
    policy = replace(
        P2R6_RUNTIME_POLICY,
        instruction_id=CAPTURE_INSTRUCTION_ID,
        method_id=CAPTURE_METHOD_ID,
        receipt_namespace="p2r6-llama-outer1-qp-capture",
        arm_label_prefix="P2R6-CAPTURE",
        allowed_arms=(CAPTURE_ARM,),
        allowed_cases={CAPTURE_ALIAS: (CAPTURE_CASE_INDEX,)},
        shadow_solver=None,
        pre_route_observer=observer,
    )
    seed_all(COMMON_SEED)
    case_root = raw_root / "cases" / "case-05" / "ar-cap"
    try:
        run_p2_atomic_arm_case(
            model,
            tokenizer,
            requests,
            alias=alias,
            arm=CAPTURE_ARM,
            case_index=CAPTURE_CASE_INDEX,
            case_root=case_root,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            dataset_path=dataset_path,
            mutation_lock=mutation_lock,
            touched=touched,
            base_values=base_values,
            expected_w0=expected_w0,
            request_microbatch_size=request_microbatch_size,
            job_ledger=job_ledger,
            runtime_policy=policy,
        )
    except P2R6LlamaCaptureComplete as intentional:
        restore = _restore_exact_w0(
            touched,
            base_values,
            mutation_lock=mutation_lock,
            expected_contract=expected_w0,
        )
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P2R6 capture post-stop W0 differs")
        terminal = {
            "schema": "ode-edit-s05-p2r6-llama-outer1-capture-terminal/v1",
            "instruction_id": CAPTURE_INSTRUCTION_ID,
            "source_head": source_head,
            "failed_scientific_source_head": FAILED_SCIENTIFIC_SOURCE_HEAD,
            "alias": alias,
            "case_index": CAPTURE_CASE_INDEX,
            "arm": CAPTURE_ARM,
            "status": "CAPTURE_COMPLETE_INTENTIONAL_PRE_QP_STOP",
            "capsule_stop_receipt": intentional.raw_free_receipt,
            "capture_complete": observer.capture_count == 1,
            "finite_capsule": True,
            "outer0_writer_materialization_count": 1,
            "outer1_selected_qp_solve_count": 0,
            "terminal_evaluator_count": 0,
            "scientific_endpoint_count": 0,
            "retry_count": 0,
            "W0_restore": restore,
            "W0_pointer_restored_exact": restore["pointer_restored_exact"],
            "W0_byte_restored_exact": restore["byte_restored_exact"],
            "job_compute": job_ledger.raw_free_payload(),
        }
        terminal["identity_sha256"] = canonical_hash(terminal)
        terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
        manifest = {
            "schema": "ode-edit-s05-p2r6-llama-outer1-capture-manifest/v1",
            "source_head": source_head,
            "terminal_sha256": terminal_sha,
            "capsule_sha256": intentional.raw_free_receipt["capsule_sha256"],
            "capsule_identity_sha256": intentional.raw_free_receipt[
                "capsule_identity_sha256"
            ],
            "capture_invocation_count": 1,
            "W0_restored": True,
            "terminal_evaluator_count": 0,
            "scientific_endpoint_count": 0,
        }
        manifest["identity_sha256"] = canonical_hash(manifest)
        manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
        stages.record(
            "p2r6_llama_outer1_qp_capture_complete",
            {
                "terminal_sha256": terminal_sha,
                "manifest_sha256": manifest_sha,
                "W0_restored": True,
                "scientific_endpoint_count": 0,
            },
        )
        return {
            "status": "P2R6_LLAMA_OUTER1_QP_CAPTURE_COMPLETE",
            "terminal_sha256": terminal_sha,
            "manifest_sha256": manifest_sha,
            "capsule_sha256": intentional.raw_free_receipt["capsule_sha256"],
            "W0_restored": True,
            "scientific_endpoint_count": 0,
        }
    raise ODEBFStateError("P2R6 capture reached scientific route unexpectedly")


__all__ = [
    "CAPTURE_ALIAS",
    "CAPTURE_ARM",
    "CAPTURE_CASE_INDEX",
    "CAPTURE_INSTRUCTION_ID",
    "CAPTURE_METHOD_ID",
    "LlamaOuter1QPCaptureObserver",
    "P2R6LlamaCaptureComplete",
    "expected_capture_result_name",
    "run_p2r6_llama_qp_capture",
]
