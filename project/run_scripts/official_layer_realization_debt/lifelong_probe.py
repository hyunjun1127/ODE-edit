"""Checkpoint-only same-state probes and registered completion geometry."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256

from .contracts import LAYERS, Method, ObservationBoundary
from .lifelong_journal import EditableStateSnapshot
from .observer import _output_component
from .runtime import _method_module, _w0_identity


def capture_layer_deltas(
    touched: Mapping[str, torch.nn.Parameter],
    originals: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(touched) != set(originals) or len(touched) != len(LAYERS):
        raise ObservationBoundary("checkpoint delta inventory differs")
    return {
        name: (
            parameter.detach().to(device="cpu", dtype=torch.float32)
            - originals[name].detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        for name, parameter in touched.items()
    }


def _activation(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    requests: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    module = _method_module(method)
    value = module.get_module_input_output_at_words(
        model,
        tokenizer,
        int(hparams.layers[-1]),
        context_templates=[str(item["prompt"]) for item in requests],
        words=[str(item["subject"]) for item in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )
    output = _output_component(method, value)
    if output.ndim != 2 or output.shape[0] != len(requests) or not torch.isfinite(output).all():
        raise ObservationBoundary("same-state activation geometry differs")
    return output.detach().to(device="cpu", dtype=torch.float64).contiguous()


def same_entry_layer_responses(
    *,
    model: Any,
    tokenizer: Any,
    method: Method,
    hparams: Any,
    requests: Sequence[Mapping[str, Any]],
    touched: Mapping[str, torch.nn.Parameter],
    entry: EditableStateSnapshot,
    deltas: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Apply each already-computed Official layer delta from one entry state."""

    from easyeditor.util.device import copy_to_param

    entry.restore(touched)
    base = _activation(
        model=model,
        tokenizer=tokenizer,
        method=method,
        hparams=hparams,
        requests=requests,
    )
    responses: list[torch.Tensor] = []
    action_squared: list[float] = []
    rows: list[dict[str, Any]] = []
    for layer, (name, parameter) in zip(LAYERS, touched.items(), strict=True):
        entry.restore(touched)
        delta = deltas[name].to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            copy_to_param(
                parameter,
                entry.weights[name].to(device=parameter.device, dtype=parameter.dtype)
                + delta,
            )
        observed = _activation(
            model=model,
            tokenizer=tokenizer,
            method=method,
            hparams=hparams,
            requests=requests,
        )
        response = observed - base
        if not torch.isfinite(response).all():
            raise ObservationBoundary("same-state response is nonfinite")
        flat = deltas[name].to(dtype=torch.float64).reshape(-1)
        action = float(torch.dot(flat, flat).item())
        if not math.isfinite(action) or action <= 0.0:
            raise ObservationBoundary("same-state layer action is nonpositive")
        responses.append(response)
        action_squared.append(action)
        rows.append(
            {
                "layer": layer,
                "delta_sha256": tensor_sha256(deltas[name]),
                "response_sha256": tensor_sha256(response.to(torch.float32)),
                "delta_frobenius_magnitude": math.sqrt(action),
                "delta_frobenius_squared": action,
                "response_frobenius_magnitude": float(
                    torch.linalg.vector_norm(response).item()
                ),
            }
        )
    restore = entry.restore(touched)
    return {
        "schema": "odeedit.s06.layer-realization-debt.same-entry-probe.v1",
        "entry_activation_sha256": tensor_sha256(base.to(torch.float32)),
        "layers": rows,
        "additional_compute_z": 0,
        "additional_key_compute": 0,
        "additional_solve": 0,
        "activation_forward_count": 1 + len(LAYERS),
        "restore": restore,
        "_base": base,
        "_responses": tuple(responses),
        "_action_squared": tuple(action_squared),
    }


def _minimum_action(
    residual: torch.Tensor,
    responses: Sequence[torch.Tensor],
    action_squared: Sequence[float],
) -> dict[str, Any]:
    if not responses:
        norm = torch.linalg.vector_norm(residual)
        return {
            "minimum_expected_completion_action": 0.0 if float(norm) == 0.0 else float("inf"),
            "unreachable_fraction": 0.0 if float(norm) == 0.0 else 1.0,
            "effective_rank": 0,
            "minimum_effective_singular_value": 0.0,
            "coefficient": [],
        }
    r = residual.to(dtype=torch.float64).reshape(-1)
    columns = []
    for response, action in zip(responses, action_squared, strict=True):
        if not math.isfinite(action) or action <= 0.0:
            raise ObservationBoundary("completion metric action differs")
        columns.append(response.to(dtype=torch.float64).reshape(-1) / math.sqrt(action))
    design = torch.stack(columns, dim=1)
    solution = torch.linalg.lstsq(design, r.unsqueeze(1)).solution[:, 0]
    projected = design @ solution
    residual_norm = torch.linalg.vector_norm(r)
    unreachable = torch.linalg.vector_norm(r - projected) / residual_norm
    singular = torch.linalg.svdvals(design)
    tolerance = torch.finfo(torch.float64).eps * max(design.shape) * singular.max()
    active = singular[singular > tolerance]
    value = 0.5 * torch.dot(solution, solution)
    output = {
        "minimum_expected_completion_action": float(value.item()),
        "unreachable_fraction": float(unreachable.item()),
        "effective_rank": int(active.numel()),
        "minimum_effective_singular_value": (
            float(active.min().item()) if active.numel() else 0.0
        ),
        "coefficient": [float(item) for item in solution.tolist()],
    }
    if not all(
        math.isfinite(float(output[key]))
        for key in (
            "minimum_expected_completion_action",
            "unreachable_fraction",
            "minimum_effective_singular_value",
        )
    ):
        raise ObservationBoundary("completion geometry is nonfinite")
    return output


def registered_completion_geometry(
    *,
    raw_ordered: Mapping[str, Any],
    same_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate G=J P_H M^-1 P_H^T J^T in the Official action basis.

    The registered basis contains the five actual Official layer updates.  It
    is an observational low-rank action shell, not a full parameter-space
    controllability claim.
    """

    z = torch.stack(
        [value.to(dtype=torch.float64).reshape(-1) for value in raw_ordered["z"]]
    )
    pre = [value.to(dtype=torch.float64) for value in raw_ordered["pre_layer"]]
    terminal = raw_ordered["terminal"].to(dtype=torch.float64)
    responses = tuple(same_state["_responses"])
    action = tuple(float(value) for value in same_state["_action_squared"])
    if any(value.shape != z.shape for value in [*pre, terminal, *responses]):
        raise ObservationBoundary("completion geometry tensor shape differs")
    a0 = 0.5 * sum(action)
    if not math.isfinite(a0) or a0 <= 0.0:
        raise ObservationBoundary("completion geometry A0 differs")
    residuals = [z - value for value in pre]
    residuals.append(z - terminal)
    waypoints: list[dict[str, Any]] = []
    for waypoint in range(len(LAYERS) + 1):
        if waypoint == len(LAYERS):
            initial_norm = torch.linalg.vector_norm(residuals[0])
            completion = {
                "minimum_expected_completion_action": 0.0,
                "unreachable_fraction": float(
                    (torch.linalg.vector_norm(residuals[waypoint]) / initial_norm).item()
                ),
                "effective_rank": 0,
                "minimum_effective_singular_value": 0.0,
            }
        else:
            completion = _minimum_action(
                residuals[waypoint], responses[waypoint:], action[waypoint:]
            )
        spent = 0.5 * sum(action[:waypoint])
        value = float(completion["minimum_expected_completion_action"])
        h = a0 - spent - value
        actual_tail = 0.5 * sum(action[waypoint:])
        waypoints.append(
            {
                "waypoint": waypoint,
                "remaining_layer_count": len(LAYERS) - waypoint,
                "A0": a0,
                "A_committed": spent,
                "V_to_go": value,
                "Vbar": value / a0,
                "Abar_committed": spent / a0,
                "completion_budget_h": h,
                "completion_budget_h_normalized": h / a0,
                "actual_official_tail_action": actual_tail,
                "unreachable_fraction": completion["unreachable_fraction"],
                "effective_rank": completion["effective_rank"],
                "minimum_effective_singular_value": completion[
                    "minimum_effective_singular_value"
                ],
            }
        )
    primary = waypoints[0]
    return {
        "schema": "odeedit.s06.layer-realization-debt.registered-completion-geometry.v1",
        "authority": "ACTUAL_OFFICIAL_FIVE_LAYER_UPDATE_BASIS",
        "full_parameter_space_claim": False,
        "history_conditioning": "CAPTURED_IN_ACTUAL_OFFICIAL_LAYER_ACTIONS",
        "A0": a0,
        "V_to_go": primary["V_to_go"],
        "Vbar": primary["Vbar"],
        "unreachable_fraction": primary["unreachable_fraction"],
        "minimum_completion_budget_h_normalized": min(
            float(row["completion_budget_h_normalized"]) for row in waypoints
        ),
        "waypoints": waypoints,
        "identity_sha256": canonical_hash(waypoints),
    }


def public_same_state_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


__all__ = [
    "capture_layer_deltas",
    "public_same_state_payload",
    "registered_completion_geometry",
    "same_entry_layer_responses",
]
