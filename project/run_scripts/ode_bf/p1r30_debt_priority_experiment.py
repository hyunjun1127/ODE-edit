"""Closed P1R30 stage dispatcher around the shared P1R24 K8 runtime."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .p1_backend import PinnedCovarianceRegistry
from .p1_controller import P1ControllerLock
from .p1_replay import Theta0TeacherCache
from .p1_scalable_batched_experiment import _run_ode_pair
from .p1_state import ArmWeightSnapshot
from .p1r30_debt_priority import P1R30_INSTRUCTION_ID, P1R30_METHOD_ID
from .p1r30_debt_priority_panel import P1R30_ROLES
from .sampling import StatelessReplaySchedule
from .scalable_batched_runtime import scalable_ordered_request_digest


P1R30_SCHEMA = "ode-edit-s05-p1r30-debt-priority-a0-barrier"
P1R30_B1_PREFIX_REQUEST_ORDER = (
    "f52fe9d5e8c8aceac2c45c8c7b20b07339868a8cefecc18d21eba9023c72d4d0"
)
P1R30_B10_REQUEST_ORDER = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)
P1R30_SEAL_ROOT = (
    "3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628"
)


def p1r30_role_dispatch(
    role: str,
) -> tuple[str, FixedE8Arm | None, bool]:
    """Map a closed execution role without changing controller semantics."""
    if role not in P1R30_ROLES:
        raise ODEBFContractError("P1R30 execution role differs")
    allocation = (
        "BG"
        if role in ("P1R30_B10_BG_PAIR", "P1R30_B10_BG_DEBT_SOFT")
        else "RS"
    )
    selected_arm = (
        FixedE8Arm.NEUTRAL
        if role == "P1R30_B10_RS_DEBT_NEUTRAL"
        else FixedE8Arm.SOFT
        if role in (
            "P1R30_B10_RS_DEBT_SOFT",
            "P1R30_B10_BG_DEBT_SOFT",
        )
        else None
    )
    return allocation, selected_arm, role == "P1R30_B10_BG_PAIR"


def run_p1r30_debt_priority(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache | None,
    dataset_path: Path,
    mutation_lock: threading.RLock,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    write_once: Any,
    request_microbatch_size: int,
    numerical_lock: Mapping[str, Any],
    numerical_lock_sha256: str,
    inherited_runtime_lock_sha256: str,
) -> dict[str, Any]:
    del stages, mutation_lock
    allocation, selected_arm, paired = p1r30_role_dispatch(role)
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    technical_smoke = role == "P1R30_B1_RS_REFERENCE_SOFT"
    expected_request_count = 1 if technical_smoke else 10
    if (
        len(requests) != expected_request_count
        or numerical_lock.get("instruction_id") != P1R30_INSTRUCTION_ID
        or numerical_lock.get("method_id") != P1R30_METHOD_ID
        or numerical_lock.get("base_checkpoint")
        != "ce8c6c36348752f1407f7d713d30e6b5c727379b"
        or numerical_lock.get("inputs", {}).get("seal_root") != P1R30_SEAL_ROOT
        or numerical_lock.get("inputs", {}).get("request_order")
        != P1R30_B10_REQUEST_ORDER
        or (
            technical_smoke
            and request_order != P1R30_B1_PREFIX_REQUEST_ORDER
        )
        or (
            not technical_smoke
            and request_order != P1R30_B10_REQUEST_ORDER
        )
    ):
        raise ODEBFContractError("P1R30 inherited atomic input lock differs")
    expected_stream_order = (
        P1R30_B1_PREFIX_REQUEST_ORDER
        if technical_smoke
        else P1R30_B10_REQUEST_ORDER
    )
    if stream["batch_ordered_request_digest_v1"][0] != expected_stream_order:
        raise ODEBFContractError("P1R30 stream order differs")

    preflight = {
        "schema": f"{P1R30_SCHEMA}-execution-preflight/v1",
        "instruction_id": P1R30_INSTRUCTION_ID,
        "method_id": P1R30_METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "role": role,
        "request_count": len(requests),
        "request_order_sha256": request_order,
        "request_microbatch_size": min(request_microbatch_size, len(requests)),
        "target_writer_base": "EXACT_P1R24_A0",
        "allocation": allocation,
        "BG_access_count": 1 if allocation == "BG" else 0,
        "historical_sequential_access_count": 0,
        "p1r27_controller_inheritance_count": 0,
        "p1r29_controller_inheritance_count": 0,
        "debt_total_update_magnitude_influence_count": 0,
        "inverse_slope_operation_count": 0,
        "online_functional_p_model_forward_count": 0,
        "retry_backtracking_count": 0,
        "p1r30_numerical_lock_sha256": numerical_lock_sha256,
        "inherited_p1r23_runtime_lock_sha256": inherited_runtime_lock_sha256,
    }
    preflight["identity_sha256"] = canonical_hash(preflight)
    write_once(raw_root / "execution-preflight.json", preflight)

    return _run_ode_pair(
        model,
        tokenizer,
        alias=alias,
        destination=destination,
        raw_root=raw_root,
        source_head=source_head,
        requests=requests,
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
        touched=touched,
        base_receipt=base_receipt,
        base_values=base_values,
        job_ledger=job_ledger,
        write_once=write_once,
        request_microbatch_size=request_microbatch_size,
        allocation=allocation,
        p1r30=True,
        selected_p1r30_arm=selected_arm,
        technical_smoke=technical_smoke,
        paired_p1r30=paired,
    )


__all__ = ["p1r30_role_dispatch", "run_p1r30_debt_priority"]
