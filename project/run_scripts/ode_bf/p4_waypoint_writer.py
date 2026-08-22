"""P4 K8 waypoint and Official AlphaEdit writer call contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256


P4_OUTER_STEPS = 8
P4_DYNAMIC_ARMS = ("A+", "A±", "Native")


def build_p4_write_waypoint(
    current_terminal: torch.Tensor,
    target_star: torch.Tensor,
    *,
    outer_step_index: int,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if (
        current_terminal.dtype != torch.float32
        or target_star.dtype != torch.float32
        or current_terminal.ndim != 2
        or current_terminal.shape != target_star.shape
        or outer_step_index < 0
        or outer_step_index >= P4_OUTER_STEPS
        or not bool(torch.isfinite(current_terminal).all())
        or not bool(torch.isfinite(target_star).all())
    ):
        raise ODEBFContractError("P4 waypoint geometry differs")
    remaining = P4_OUTER_STEPS - outer_step_index
    coefficient = 1.0 / remaining
    waypoint = (
        current_terminal + coefficient * (target_star - current_terminal)
    ).contiguous()
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-k8-write-waypoint/v1",
        "outer_step_index": outer_step_index,
        "K": P4_OUTER_STEPS,
        "remaining_steps": remaining,
        "lambda_k": coefficient,
        "formula": "y_k+lambda_k*(z_star_k-y_k)",
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "target_star_sha256": tensor_sha256(target_star),
        "write_waypoint_sha256": tensor_sha256(waypoint),
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return waypoint, receipt


def verify_current_state_refresh(
    rows: Sequence[Mapping[str, Any]], *, arm: str
) -> Mapping[str, Any]:
    if arm not in P4_DYNAMIC_ARMS or len(rows) != P4_OUTER_STEPS:
        raise ODEBFContractError("P4 current-state refresh inventory differs")
    expected_native = 8 if arm == "Native" else 0
    for index, row in enumerate(rows):
        if (
            row.get("outer_step_index") != index
            or row.get("target_gradient_refresh_count") != (0 if arm == "Native" else 1)
            or row.get("current_terminal_residual_refresh_count") != 1
            or row.get("current_key_refresh_count") != 1
            or row.get("alpha_solve_refresh_count") != 1
            or row.get("official_writer_apply_count") != 1
            or row.get("native_compute_z_call_count")
            != (1 if arm == "Native" else 0)
        ):
            raise ODEBFContractError("P4 current-state refresh contract differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-current-state-refresh/v1",
        "arm": arm,
        "outer_step_count": P4_OUTER_STEPS,
        "target_gradient_refresh_count": sum(
            int(row["target_gradient_refresh_count"]) for row in rows
        ),
        "current_terminal_residual_refresh_count": P4_OUTER_STEPS,
        "current_key_refresh_count": P4_OUTER_STEPS,
        "alpha_solve_refresh_count": P4_OUTER_STEPS,
        "official_writer_apply_count": P4_OUTER_STEPS,
        "native_compute_z_call_count": expected_native,
        "official_writer_direct_call": "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model",
        "native_compute_z_direct_call": "easyeditor.models.alphaedit.compute_z.compute_z",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def run_p4_official_writer_step(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    arm: str,
    outer_step_index: int,
    touched: Mapping[str, torch.nn.Parameter],
    private_root: Path,
    write_waypoint: torch.Tensor | None,
    reset_cache: bool,
    batch_entry_cache_width: int,
    apply_callable: Callable[..., tuple[dict[str, Any], Mapping[str, torch.Tensor]]] | None = None,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Invoke the pinned Official writer with exact P4 compute-z semantics.

    The surrounding ``P4CacheTransaction`` owns whether the writer-produced
    cache candidate is discarded or installed after the successful K8 commit.
    This function owns only one direct Official apply call and its accepted-z
    bridge for the two causal arms.
    """

    if arm not in P4_DYNAMIC_ARMS or outer_step_index < 0 or outer_step_index >= 8:
        raise ODEBFContractError("P4 Official writer step identity differs")
    if batch_entry_cache_width < 0:
        raise ODEBFContractError("P4 Official writer cache width differs")
    if bool(reset_cache) != (batch_entry_cache_width == 0):
        raise ODEBFContractError("P4 Official writer reset/cache contract differs")
    if apply_callable is None:
        from .scalable_batched_native import run_official_native_apply

        apply_callable = run_official_native_apply

    if arm == "Native":
        if write_waypoint is not None:
            raise ODEBFContractError("P4 Native arm received an accepted waypoint")
        payload, originals = apply_callable(
            model,
            tokenizer,
            requests,
            hparams,
            touched=touched,
            reset_cache=reset_cache,
            cache_history_width=batch_entry_cache_width,
            cache_template=None,
            expected_native_compute_z_call_count=len(requests),
            accepted_z_source="OFFICIAL_NATIVE_COMPUTE_Z_REFERENCE",
        )
        bridge = None
    else:
        if (
            write_waypoint is None
            or write_waypoint.dtype != torch.float32
            or write_waypoint.ndim != 2
            or write_waypoint.shape[1] != len(requests)
            or not bool(torch.isfinite(write_waypoint).all())
        ):
            raise ODEBFContractError("P4 accepted waypoint geometry differs")
        from .p1r52_target_official_alphaedit_writer import accepted_z_cache_template

        source = f"P4_{arm}_K{outer_step_index + 1}_WRITE_WAYPOINT"
        with accepted_z_cache_template(
            requests,
            write_waypoint,
            hparams,
            parent=private_root,
            accepted_z_source=source,
        ) as (template, bridge):
            payload, originals = apply_callable(
                model,
                tokenizer,
                requests,
                hparams,
                touched=touched,
                reset_cache=reset_cache,
                cache_history_width=batch_entry_cache_width,
                cache_template=template,
                expected_native_compute_z_call_count=0,
                accepted_z_source=source,
            )
    wrapped: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-official-alphaedit-writer-step/v1",
        "arm": arm,
        "outer_step_index": outer_step_index,
        "official_apply": payload,
        "accepted_z_bridge": bridge,
        "official_writer_direct_call_count": 1,
        "native_compute_z_expected_count": len(requests) if arm == "Native" else 0,
        "batch_entry_cache_width": batch_entry_cache_width,
        "cache_candidate_install_authority": "P4CacheTransaction_AFTER_SUCCESSFUL_K8_COMMIT",
    }
    wrapped["identity_sha256"] = canonical_hash(wrapped)
    return wrapped, originals


__all__ = [
    "P4_DYNAMIC_ARMS",
    "P4_OUTER_STEPS",
    "build_p4_write_waypoint",
    "run_p4_official_writer_step",
    "verify_current_state_refresh",
]
