"""Private one-shot Qwen step-5 replay capture for P2R6 RED-R1.

This module is capture instrumentation only.  It serializes the already-built
model-free routing payload immediately before the failing AR-CAP shadow solve,
then raises a typed intentional stop.  It never selects an action or invokes a
model, writer, materializer, or evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .contracts import COMMON_SEED, ODEBFContractError, canonical_hash
from .p2r5_sdrt_writer import SDRTCalibration, SDRTQuadratics
from .p2r6_semantic_region_controller import (
    P2R6RoutingTechnicalError,
    _mass_matrix,
    _semantic_region_optimum,
    _solve_region_quadratic,
)
from .p2r5_stage_a_runtime import STREAM_ORDER, STREAM_ROOT


CAPTURE_INSTRUCTION_ID = (
    "ODEEDIT-S05-P2R6-SEMANTIC-REGION-CONTROLLER-PILOT-V1-"
    "RED-INTEGRATION-R1-CAPTURE-A1"
)
TECH_R8_SCIENTIFIC_HEAD = "e7c86a5598a7507f4f981068ba1fa8db4df13df9"
PRESERVED_FAILURE_IDENTITY = (
    "4bb0af34b8c835bdd2f8ab22ab68590cee2cf189b6ea2940c04f8e0c1415a6c4"
)
PRESERVED_OBSERVABILITY_IDENTITY = (
    "b102f40884bf73cae20640b659d45a5029462f780210155b0a82c0362a64c036"
)
CAPTURE_ALIAS = "qwen2.5-7b-inst"
CAPTURE_ARM = "AS-CAP"
CAPTURE_SHADOW_ARM = "AR-CAP"
CAPTURE_CASE_INDEX = 1
CAPTURE_OUTER_STEP = 5


@dataclass(frozen=True, slots=True)
class P2R6ReplayCaptureComplete(RuntimeError):
    """Intentional stop after a complete private replay capsule is sealed."""

    capsule_root: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]

    def __str__(self) -> str:
        return "P2R6 replay capture complete; intentional pre-shadow stop"


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError("P2R6 replay capsule directory is a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private_bytes_once(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P2R6 replay capsule member exists")
    _private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _npy_payload(value: np.ndarray) -> tuple[np.ndarray, bytes]:
    canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    stream = io.BytesIO()
    np.save(stream, canonical, allow_pickle=False)
    return canonical, stream.getvalue()


def _seal_capsule(
    capsule_root: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if capsule_root.exists() or capsule_root.is_symlink():
        raise FileExistsError("P2R6 replay capsule root exists")
    _private_directory(capsule_root)
    members: dict[str, dict[str, Any]] = {}
    for name in sorted(arrays):
        canonical, payload = _npy_payload(arrays[name])
        path = capsule_root / f"{name}.npy"
        file_sha = _write_private_bytes_once(path, payload)
        members[name] = {
            "filename": path.name,
            "shape": list(canonical.shape),
            "dtype": canonical.dtype.str,
            "finite": bool(np.all(np.isfinite(canonical))),
            "npy_bytes": len(payload),
            "npy_sha256": file_sha,
            "raw_c_order_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        }
    if not all(bool(item["finite"]) for item in members.values()):
        raise ODEBFContractError("P2R6 replay capsule contains nonfinite values")
    receipt = {
        "schema": "ode-edit-s05-p2r6-red-r1-private-replay-capsule/v1",
        **dict(metadata),
        "members": members,
        "member_count": len(members),
        "capsule_complete": True,
        "capsule_private": True,
        "raw_prompt_target_generation_weight_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    payload = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    receipt_sha = _write_private_bytes_once(capsule_root / "capsule.json", payload)
    return receipt_sha, receipt


def build_capture_hook(
    *,
    instrumentation_source_head: str,
    instrumentation_source_tree: str,
) -> Callable[..., None]:
    """Return the exact one-shot pre-shadow capture hook."""

    def capture(
        response: torch.Tensor,
        deficit: torch.Tensor,
        entry_deficit: torch.Tensor,
        calibration: SDRTCalibration,
        quadratics: SDRTQuadratics,
        *,
        alias: str,
        selected_arm: str,
        case_index: int,
        outer_step: int,
        request_order_sha256: str,
        expected_w0_sha256: str,
        case_root: Path,
    ) -> None:
        del calibration
        if (alias, selected_arm, case_index) != (
            CAPTURE_ALIAS,
            CAPTURE_ARM,
            CAPTURE_CASE_INDEX,
        ):
            raise ODEBFContractError("P2R6 capture selected trajectory differs")
        if outer_step != CAPTURE_OUTER_STEP:
            return
        observed = response.detach().to(device="cpu", dtype=torch.float64).numpy()
        deficit_np = deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
        entry_np = entry_deficit.detach().to(device="cpu", dtype=torch.float64).numpy()
        alpha_start, scale, lower, xi, e1_receipt = _semantic_region_optimum(
            observed,
            deficit_np,
            entry_np,
            scale_policy="CURRENT_DEFICIT",
        )
        capacity = quadratics.capacity_gram.detach().to(device="cpu", dtype=torch.float64).numpy()
        capacity_cross = quadratics.capacity_cross.detach().to(
            device="cpu", dtype=torch.float64
        ).numpy()
        mass = _mass_matrix(5, 10)
        arrays = {
            "S": observed,
            "d": deficit_np,
            "b": lower,
            "Q_C": capacity,
            "c_C": capacity_cross,
            "alpha_start": alpha_start,
            "semantic_scale": scale,
            "mass_matrix": mass,
        }
        capsule_root = case_root / "private-replay" / "qwen-as-outer05-ar-cap"
        metadata = {
            "instruction_id": CAPTURE_INSTRUCTION_ID,
            "scientific_source_head": TECH_R8_SCIENTIFIC_HEAD,
            "instrumentation_source_head": instrumentation_source_head,
            "instrumentation_source_tree": instrumentation_source_tree,
            "preserved_failure_identity": PRESERVED_FAILURE_IDENTITY,
            "preserved_observability_identity": PRESERVED_OBSERVABILITY_IDENTITY,
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "common_seed": COMMON_SEED,
            "model_alias": alias,
            "case_index": case_index,
            "selected_arm": selected_arm,
            "failing_shadow_arm": CAPTURE_SHADOW_ARM,
            "outer_step": outer_step,
            "request_order_sha256": request_order_sha256,
            "W0_sha256": expected_w0_sha256,
            "e1_scale_policy": "CURRENT_DEFICIT",
            "e1_xi": xi,
            "e1_receipt": dict(e1_receipt),
            "quadratics_identity_sha256": quadratics.identity_sha256,
            "constraint_identities": {
                "alpha_nonnegative": True,
                "per_request_mass_upper": 1.0,
                "mass_matrix_raw_sha256": hashlib.sha256(
                    np.ascontiguousarray(mass, dtype=np.dtype("<f8")).tobytes(order="C")
                ).hexdigest(),
                "semantic_region": "S_ALPHA_GE_B",
                "stage": "AR_CAPACITY_IN_SEMANTIC_REGION",
            },
            "instrumentation_model_forward_count": 0,
            "instrumentation_model_backward_count": 0,
            "instrumentation_materialization_count": 0,
            "instrumentation_decision_influence_count": 0,
            "red_amendment_influence_count": 0,
            "terminal_evaluator_access_count": 0,
            "intentional_stop_before_shadow_solve": True,
        }
        receipt_sha, receipt = _seal_capsule(capsule_root, arrays, metadata)
        raise P2R6ReplayCaptureComplete(capsule_root, receipt_sha, receipt)

    return capture


def load_capsule(capsule_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    receipt_path = capsule_root / "capsule.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = dict(receipt)
    identity = observed.pop("identity_sha256", None)
    if identity != canonical_hash(observed):
        raise ODEBFContractError("P2R6 replay capsule receipt root differs")
    arrays: dict[str, np.ndarray] = {}
    for name, member in receipt["members"].items():
        path = capsule_root / member["filename"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != member["npy_sha256"]:
            raise ODEBFContractError("P2R6 replay capsule member SHA differs")
        value = np.load(io.BytesIO(payload), allow_pickle=False)
        if list(value.shape) != member["shape"] or value.dtype.str != member["dtype"]:
            raise ODEBFContractError("P2R6 replay capsule member geometry differs")
        arrays[name] = np.asarray(value, dtype=np.float64)
    return arrays, receipt


def replay_preserved_failure(capsule_root: Path) -> dict[str, Any]:
    """Run the untouched TECH-R8 solver and require the preserved failure identity."""

    arrays, capsule = load_capsule(capsule_root)
    observed_repeats: list[dict[str, Any]] = []
    for _ in range(2):
        try:
            _solve_region_quadratic(
                arrays["alpha_start"],
                arrays["Q_C"],
                arrays["c_C"],
                arrays["S"],
                arrays["b"],
                stage="AR_CAPACITY_IN_SEMANTIC_REGION",
            )
        except P2R6RoutingTechnicalError as exc:
            observed_repeats.append(dict(exc.raw_free_receipt))
        else:
            raise ODEBFContractError("P2R6 preserved failure unexpectedly passed")
    identities = [item.get("identity_sha256") for item in observed_repeats]
    if identities != [PRESERVED_OBSERVABILITY_IDENTITY] * 2:
        raise ODEBFContractError("P2R6 preserved observability identity differs")
    observed = observed_repeats[0]
    verification = {
        "schema": "ode-edit-s05-p2r6-red-r1-replay-verification/v1",
        "instruction_id": CAPTURE_INSTRUCTION_ID,
        "capsule_identity_sha256": capsule["identity_sha256"],
        "preserved_failure_identity": PRESERVED_FAILURE_IDENTITY,
        "expected_observability_identity": PRESERVED_OBSERVABILITY_IDENTITY,
        "observed_observability_identity": observed["identity_sha256"],
        "technical_observability": observed,
        "exact_identity_match": True,
        "deterministic_repeat_count": 2,
        "status": "P2R6_CAPTURE_REPRODUCES_PRESERVED_FAILURE",
    }
    verification["identity_sha256"] = canonical_hash(verification)
    return verification


__all__ = [
    "CAPTURE_INSTRUCTION_ID",
    "P2R6ReplayCaptureComplete",
    "build_capture_hook",
    "load_capsule",
    "replay_preserved_failure",
]
