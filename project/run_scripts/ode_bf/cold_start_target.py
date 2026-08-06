"""Cold target-only bootstrap and matched full-residual P1R6 runtime.

The cold action path is intentionally independent of Native/direct-z.  It
starts from the canonical W0 lookup activations, moves only a temporary target
state under the sealed target-new teacher-forced objective, and then advances
that target and the virtual BF16 weight state on one shared pseudo-time clock.
Native is opened by :func:`run_cold_diagnostic` only after both cold rollouts
have been frozen.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import W64_ASSEMBLER_REFERENCE
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .functional_p_secant import FunctionalPFieldPolicy
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import (
    AdaptiveTauClock,
    AdaptiveVariant,
    FirstHitRecord,
    FirstHitTracker,
    StepSizeIndependentVelocity,
    adaptive_lock,
    fraction_payload,
)
from .p1_adaptive_runtime import (
    AcceptedSnapshot,
    ActiveField,
    AdaptiveReceiptRecorder,
    FunctionalPDecisionPolicy,
    VariantRollout,
    _append_accepted_snapshot,
    _build_active_field,
    _controller_replay_entry,
    _evaluate_rewrite,
    _factor_map,
    _factor_state,
    _history_actions,
    _history_keys,
    _merge_factors,
    _omega_state,
    _parameter_contract_sha256,
    _postfreeze_stepwise_panel,
    _rng_identity,
    _run_trial,
    _target_new_panel_contrast,
    _terminal_confirm_snapshots,
)
from .p1_backend import (
    FULL_CURRENT_RESIDUAL_DEFINITION,
    FULL_CURRENT_RESIDUAL_DIVISOR,
    P1_DYNAMIC_REFERENCE,
    P1DynamicField,
    PinnedCovarianceRegistry,
    _virtual_context,
    build_p1_dynamic_field,
    signed_progress_gradient,
)
from .p1_controller import (
    AcceptedLayerContribution,
    P1ControllerLock,
    build_p1_routing_problem,
    project_matched_bf_velocity,
    solve_matched_raw_velocity,
)
from .p1_replay import OuterEntryPretrainedCache, Theta0TeacherCache, build_outer_entry_pretrained_cache
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .request_digest import ordered_request_digest_v1
from .routing import PreservationConstraintPolicy, RoutingStatus
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, evaluate_routing_objective


COLD_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COLDSTART-STRUCTP-SOFTP-NOVETO-FULLTAU-P1R6-V1"
)
COLD_SCHEMA_NAMESPACE = "ode-edit-s05-cold-structp-softp-noveto-p1r6"
COLD_PANEL_LABELS = (
    "COLD-FR-A8-NEWNLL-NOSOFT-NOVETO",
    "COLD-FR-A8-NEWNLL-SOFT-NOVETO",
)
COLD_BOOT_DT_MAX = Fraction(1, 8)
COLD_BOOT_DT_MIN = Fraction(1, 128)
COLD_BOOT_RADIUS = Fraction(1, 4)
COLD_BOOT_RETRY_CAP = 4
COLD_BOOT_ACCEPTED_CAP = 8
COLD_BOOT_TRIAL_CAP = 32
COLD_JOINT_TARGET_EPSILON = 1.0e-8
COLD_RHO_ACCEPT = 0.1
COLD_METRIC_TOLERANCE = 1.0e-6
COLD_LAYER_ORDER = (4, 5, 6, 7, 8)
COLD_FIELD_SEMANTIC_SCHEMA = "ode-edit-s05-cold-field-semantic-identity/v1"
COLD_FIELD_SEMANTIC_EXCLUSION_ALLOWLIST = (
    "layers[].covariance_receipt.wall_seconds",
    "field.model_forward_count",
    "field.processed_token_count",
)
COLD_COVARIANCE_RECEIPT_FIELDS = (
    "layer",
    "source_sha256",
    "source_size",
    "matrix_shape",
    "sample_count",
    "q_shape",
    "q_sha256",
    "action_sha256",
    "gram_sha256",
    "finite",
    "wall_seconds",
)
COLD_COVARIANCE_SEMANTIC_FIELDS = (
    "layer",
    "source_sha256",
    "source_size",
    "matrix_shape",
    "sample_count",
    "q_shape",
    "q_sha256",
    "action_sha256",
    "gram_sha256",
    "finite",
)


def _unwrap_layer_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise ODEBFContractError("cold target hook output is not a tensor")
    return value


def _rewrap_layer_output(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


@dataclass(frozen=True, slots=True)
class ColdTargetMetric:
    """Fixed diagonal target metric ``G_i=I/||z_i,base||^2``."""

    request_order_sha256: str
    z_base_sha256: str
    norm_squared: tuple[float, ...]
    identity_sha256: str

    @classmethod
    def from_z_base(
        cls, z_base: torch.Tensor, request_order_sha256: str
    ) -> "ColdTargetMetric":
        if (
            not isinstance(z_base, torch.Tensor)
            or z_base.ndim != 2
            or z_base.shape[1] != BATCH_SIZE
            or not torch.isfinite(z_base).all()
            or len(request_order_sha256) != 64
        ):
            raise ODEBFContractError("cold z-base metric geometry differs")
        values = tuple(
            float(torch.sum(z_base[:, index].to(dtype=torch.float64) ** 2))
            for index in range(BATCH_SIZE)
        )
        if any(not math.isfinite(value) or value <= 1.0e-12 for value in values):
            raise ODEBFContractError("cold z-base metric is degenerate")
        z_sha = tensor_sha256(z_base)
        payload = {
            "schema": "ode-edit-s05-cold-target-metric/v1",
            "request_order_sha256": request_order_sha256,
            "z_base_sha256": z_sha,
            "metric": "per-request-I-over-z-base-norm-squared",
            "metric_evaluation_dtype": "torch.float64",
            "norm_squared": list(values),
            "degenerate_floor": 1.0e-12,
            "native_or_direct_z_dependency_count": 0,
        }
        return cls(
            request_order_sha256,
            z_sha,
            values,
            canonical_hash(payload),
        )

    def distance(self, left: torch.Tensor, right: torch.Tensor) -> tuple[float, ...]:
        if left.shape != right.shape or left.ndim != 2 or left.shape[1] != BATCH_SIZE:
            raise ODEBFContractError("cold target metric distance geometry differs")
        delta = left.to(device="cpu", dtype=torch.float64) - right.to(
            device="cpu", dtype=torch.float64
        )
        result = tuple(
            float(torch.linalg.vector_norm(delta[:, index]) / math.sqrt(self.norm_squared[index]))
            for index in range(BATCH_SIZE)
        )
        if not all(math.isfinite(value) for value in result):
            raise ODEBFContractError("cold target metric distance is non-finite")
        return result

    def unit_descent(self, gradient: torch.Tensor) -> tuple[torch.Tensor, tuple[float, ...]]:
        if (
            gradient.ndim != 2
            or gradient.shape[1] != BATCH_SIZE
            or not torch.isfinite(gradient).all()
        ):
            raise ODEBFContractError("cold target gradient geometry differs")
        source = gradient.detach().to(device="cpu", dtype=torch.float64)
        velocity = torch.empty_like(source)
        gradient_norms: list[float] = []
        for index, norm_squared in enumerate(self.norm_squared):
            column = source[:, index]
            norm = float(torch.linalg.vector_norm(column))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ODEBFContractError("cold target gradient is degenerate")
            gradient_norms.append(norm)
            velocity[:, index] = -math.sqrt(norm_squared) * column / norm
        # The target state is intentionally FP32.  Normalize the actual FP32
        # vector just inside the unit sphere so a later FP32 state addition
        # can satisfy the locked ``delta_tau + 1e-8`` certificate without
        # treating roundoff as target clipping or projection.
        velocity = velocity.to(dtype=torch.float32).contiguous()
        target_norm = 1.0 - 0.75 * COLD_METRIC_TOLERANCE
        observed = self.distance(velocity, torch.zeros_like(velocity))
        for index, value in enumerate(observed):
            if not math.isfinite(value) or value <= 0.0:
                raise ODEBFContractError("cold target velocity is degenerate")
            velocity[:, index].mul_(target_norm / value)
        unit = self.distance(velocity, torch.zeros_like(velocity))
        if any(
            not math.isclose(value, 1.0, rel_tol=COLD_METRIC_TOLERANCE, abs_tol=COLD_METRIC_TOLERANCE)
            for value in unit
        ):
            raise ODEBFContractError("cold target velocity is not unit-G")
        return velocity, tuple(gradient_norms)

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-cold-target-metric/v1",
            "request_order_sha256": self.request_order_sha256,
            "z_base_sha256": self.z_base_sha256,
            "metric": "per-request-I-over-z-base-norm-squared",
            "metric_evaluation_dtype": "torch.float64",
            "norm_squared": list(self.norm_squared),
            "degenerate_floor": 1.0e-12,
            "native_or_direct_z_dependency_count": 0,
            "identity_sha256": self.identity_sha256,
        }


class TargetStateActivationOverlay:
    """Replace one request lookup activation per routed objective forward."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        target_state: torch.Tensor,
        lookup_positions: Sequence[int],
        *,
        contexts_per_request: int = 6,
    ) -> None:
        if (
            target_state.ndim != 2
            or target_state.shape[1] != BATCH_SIZE
            or contexts_per_request != 6
            or len(tuple(lookup_positions)) != BATCH_SIZE * contexts_per_request
        ):
            raise ODEBFContractError("cold activation overlay geometry differs")
        self.model = model
        self.layer_name = layer_name
        self.target_state = target_state
        self.lookup_positions = tuple(int(value) for value in lookup_positions)
        self.contexts_per_request = contexts_per_request
        self.calls = 0
        self._handle: Any = None

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.calls >= len(self.lookup_positions):
            raise ODEBFContractError("cold activation overlay received extra forward")
        activation = _unwrap_layer_output(output)
        request_index = self.calls // self.contexts_per_request
        raw_position = self.lookup_positions[self.calls]
        target = self.target_state[:, request_index].to(
            device=activation.device, dtype=activation.dtype
        )
        patched = activation.clone()
        if activation.ndim != 3 or activation.shape[-1] != target.numel():
            raise ODEBFContractError("cold activation overlay layout differs")
        if activation.shape[0] == 1:
            position = raw_position if raw_position >= 0 else activation.shape[1] + raw_position
            if position < 0 or position >= activation.shape[1]:
                raise ODEBFContractError("cold activation lookup is out of range")
            patched[0, position, :] = target
        elif activation.shape[1] == 1:
            position = raw_position if raw_position >= 0 else activation.shape[0] + raw_position
            if position < 0 or position >= activation.shape[0]:
                raise ODEBFContractError("cold activation lookup is out of range")
            patched[position, 0, :] = target
        else:
            raise ODEBFContractError("cold activation overlay batch layout differs")
        self.calls += 1
        return _rewrap_layer_output(output, patched)

    def __enter__(self) -> "TargetStateActivationOverlay":
        module = self.model.get_submodule(self.layer_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def assert_complete(self) -> None:
        if self._handle is not None or self.calls != len(self.lookup_positions):
            raise ODEBFStateError("cold activation overlay call/cleanup contract differs")


def cold_lookup_positions(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    fact_token_strategy: str,
) -> tuple[int, ...]:
    """Render exactly the target-new objective's request-major 1+5 plan."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    groups = tuple(tuple(group) for group in contexts)
    if tuple(len(group) for group in groups) != (1, 5) or len(requests) != BATCH_SIZE:
        raise ODEBFContractError("cold lookup context geometry differs")
    flattened = tuple(template for group in groups for template in group)
    values: list[int] = []
    for request in requests:
        prompt = str(request["prompt"])
        subject = str(request["subject"])
        for template in flattened:
            surface = template.format(prompt)
            if surface.count("{}") != 1:
                raise ODEBFContractError("cold lookup surface lost subject field")
            values.append(
                int(
                    alpha_main.find_fact_lookup_idx(
                        surface,
                        subject,
                        tokenizer,
                        fact_token_strategy,
                        verbose=False,
                    )
                )
            )
    if len(values) != BATCH_SIZE * 6:
        raise ODEBFContractError("cold lookup plan count differs")
    return tuple(values)


def capture_cold_z_base(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
) -> torch.Tensor:
    """Capture W0 canonical lookup outputs without computing a Native target."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("cold z-base requires one B10")
    before_rng = _rng_identity()
    output = alpha_main.get_module_input_output_at_words(
        model,
        tokenizer,
        int(hparams.layers[-1]),
        context_templates=[str(item["prompt"]) for item in requests],
        words=[str(item["subject"]) for item in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1]
    z_base = output.T.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if z_base.ndim != 2 or z_base.shape[1] != BATCH_SIZE or not torch.isfinite(z_base).all():
        raise ODEBFContractError("cold z-base capture differs")
    if _rng_identity() != before_rng:
        raise ODEBFStateError("cold z-base capture changed RNG")
    return z_base


@dataclass(frozen=True, slots=True)
class ColdObjectiveReceipt:
    value: float
    per_request_values: tuple[float, ...]
    target_state_sha256: str
    target_span_sha256: str
    context_sha256: str
    lookup_plan_sha256: str
    model_forward_count: int
    processed_token_count: int
    backward_count: int
    gradient: torch.Tensor | None
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-cold-target-objective/v1",
            "value": self.value,
            "per_request_values": list(self.per_request_values),
            "target_state_sha256": self.target_state_sha256,
            "target_span_sha256": self.target_span_sha256,
            "context_sha256": self.context_sha256,
            "lookup_plan_sha256": self.lookup_plan_sha256,
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
            "backward_count": self.backward_count,
            "gradient_present": self.gradient is not None,
            "target_old_access_count": 0,
            "native_or_direct_z_access_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_cold_target_objective(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    layer_name: str,
    lookup_positions: Sequence[int],
    target_state: torch.Tensor,
    require_gradient: bool,
) -> ColdObjectiveReceipt:
    """Target-new-only teacher objective under a temporary z-layer overlay."""

    device = next(model.parameters()).device
    flat = target_state.to(device=device, dtype=torch.float32).contiguous().view(-1)
    if require_gradient:
        flat = flat.detach().requires_grad_(True)
    view = flat.view(target_state.shape)
    before_rng = _rng_identity()
    overlay = TargetStateActivationOverlay(
        model, layer_name, view, lookup_positions
    )
    with overlay:
        result = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
            gradient_input=flat if require_gradient else None,
        )
    overlay.assert_complete()
    if _rng_identity() != before_rng:
        raise ODEBFStateError("cold target objective changed RNG")
    gradient = (
        None
        if result.input_gradient is None
        else result.input_gradient.view(target_state.shape)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if require_gradient != (gradient is not None):
        raise ODEBFContractError("cold target gradient accounting differs")
    values = tuple(
        float(value)
        for value in result.per_request_values.detach().to(device="cpu", dtype=torch.float64)
    )
    payload = {
        "schema": "ode-edit-s05-cold-target-objective/v1",
        "objective": RoutingObjective.TARGET_NEW_NLL.value,
        "value": float(result.loss.detach().to(device="cpu", dtype=torch.float64)),
        "per_request_values": list(values),
        "target_state_sha256": tensor_sha256(target_state),
        "target_span_sha256": result.target_span_sha256,
        "context_sha256": result.context_sha256,
        "lookup_plan_sha256": canonical_hash(list(lookup_positions)),
        "model_forward_count": result.model_forward_count,
        "processed_token_count": result.processed_token_count,
        "backward_count": result.backward_count,
        "target_old_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    return ColdObjectiveReceipt(
        payload["value"],
        values,
        payload["target_state_sha256"],
        result.target_span_sha256,
        result.context_sha256,
        payload["lookup_plan_sha256"],
        result.model_forward_count,
        result.processed_token_count,
        result.backward_count,
        gradient,
        canonical_hash(payload),
    )


def _cold_tensor_semantic_payload(value: torch.Tensor) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ODEBFContractError("cold semantic field tensor differs")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_sha256(value),
    }


def _cold_covariance_semantic_payload(
    receipt: Any,
) -> tuple[dict[str, Any], float]:
    payload = asdict(receipt)
    if tuple(payload) != COLD_COVARIANCE_RECEIPT_FIELDS:
        raise ODEBFContractError("cold covariance receipt schema differs")
    wall = payload["wall_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
    ):
        raise ODEBFContractError("cold covariance cost telemetry differs")
    return (
        {name: payload[name] for name in COLD_COVARIANCE_SEMANTIC_FIELDS},
        float(wall),
    )


def cold_field_semantic_receipt(
    field: P1DynamicField,
    *,
    history_solve_keys_by_layer: Mapping[int, torch.Tensor],
    history_risk_keys_by_layer: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    """Return an exact scientific field identity with cost telemetry separate.

    The exclusion schema is intentionally closed.  In particular, this does
    not recursively remove keys that happen to look like timing fields.
    """

    layers = tuple(int(item.layer) for item in field.layers)
    if (
        layers != COLD_LAYER_ORDER
        or tuple(sorted(history_solve_keys_by_layer)) != COLD_LAYER_ORDER
        or tuple(sorted(history_risk_keys_by_layer)) != COLD_LAYER_ORDER
        or field.target_state.ndim != 2
        or field.target_state.shape[1] != BATCH_SIZE
    ):
        raise ODEBFContractError("cold semantic field inventory differs")
    semantic_layers: list[dict[str, Any]] = []
    covariance_wall_seconds: list[dict[str, Any]] = []
    for item in field.layers:
        if (
            item.residual_definition != FULL_CURRENT_RESIDUAL_DEFINITION
            or item.residual_divisor != FULL_CURRENT_RESIDUAL_DIVISOR
            or item.factor.weight_name != item.weight_name
            or item.factor.layer != item.layer
        ):
            raise ODEBFContractError("cold semantic residual/factor policy differs")
        covariance, wall = _cold_covariance_semantic_payload(
            item.covariance_receipt
        )
        solve_history = history_solve_keys_by_layer[item.layer]
        risk_history = history_risk_keys_by_layer[item.layer]
        reconstructed_current_z = (
            field.target_state.detach().to(device="cpu", dtype=torch.float32)
            - item.residual.detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        factor = item.factor
        semantic_layers.append(
            {
                "layer": item.layer,
                "weight_name_sha256": hashlib.sha256(
                    item.weight_name.encode("utf-8")
                ).hexdigest(),
                "key": _cold_tensor_semantic_payload(item.key),
                "projected_key": _cold_tensor_semantic_payload(
                    item.projected_key
                ),
                "residual": _cold_tensor_semantic_payload(item.residual),
                "residual_definition": item.residual_definition,
                "residual_divisor": item.residual_divisor,
                "current_z_reconstructed_from_full_residual": (
                    _cold_tensor_semantic_payload(reconstructed_current_z)
                ),
                "q": _cold_tensor_semantic_payload(item.q),
                "factor": {
                    "identity_sha256": item.factor_identity(),
                    "layer": factor.layer,
                    "correction_cycle": factor.correction_cycle,
                    "step_in_cycle": factor.step_in_cycle,
                    "factor_ordinal": factor.factor_ordinal,
                    "theta": factor.theta,
                    "joint_batch": factor.joint_batch,
                    "left": _cold_tensor_semantic_payload(factor.left),
                    "right": _cold_tensor_semantic_payload(factor.right),
                },
                "factor_frobenius_sq": item.factor_frobenius_sq,
                "covariance_receipt": covariance,
                "covariance_action": _cold_tensor_semantic_payload(
                    item.covariance_action
                ),
                "covariance_gram": _cold_tensor_semantic_payload(
                    item.covariance_gram
                ),
                "woodbury_certificate": asdict(item.woodbury_certificate),
                "history_solve_keys": _cold_tensor_semantic_payload(
                    solve_history
                ),
                "history_risk_keys": _cold_tensor_semantic_payload(
                    risk_history
                ),
                "history_action": _cold_tensor_semantic_payload(
                    item.history_action
                ),
            }
        )
        covariance_wall_seconds.append(
            {"layer": item.layer, "wall_seconds": wall}
        )
    scientific = {
        "schema": COLD_FIELD_SEMANTIC_SCHEMA,
        "reference": P1_DYNAMIC_REFERENCE,
        "accepted_waypoint": field.accepted_waypoint,
        "request_order_sha256": field.request_order_sha256,
        "target_state": _cold_tensor_semantic_payload(field.target_state),
        "terminal_layer_current_z": _cold_tensor_semantic_payload(
            field.current_z
        ),
        "layer_order": list(layers),
        "layers": semantic_layers,
        "residual_policy": FULL_CURRENT_RESIDUAL_DEFINITION,
        "assembler": {
            "reference": W64_ASSEMBLER_REFERENCE,
            "virtual_trial": (
                "project.run_scripts.ode_bf.functional."
                "CumulativeBF16FunctionalTrial"
            ),
            "row_block": 64,
            "accumulator_dtype": str(torch.float32),
            "endpoint_dtype": str(torch.bfloat16),
        },
    }
    return {
        "schema": f"{COLD_FIELD_SEMANTIC_SCHEMA}-receipt",
        "semantic_identity_sha256": canonical_hash(scientific),
        "full_field_receipt_identity_sha256": field.identity_sha256,
        "scientific_content": scientific,
        "cost_telemetry": {
            "semantic_exclusion_allowlist": list(
                COLD_FIELD_SEMANTIC_EXCLUSION_ALLOWLIST
            ),
            "covariance_wall_seconds": covariance_wall_seconds,
            "model_forward_count": field.model_forward_count,
            "processed_token_count": field.processed_token_count,
        },
    }


@dataclass(frozen=True, slots=True)
class ColdFieldReadiness:
    reached: bool
    field_sha256: str | None
    field_semantic_sha256: str | None
    field_semantic_receipt: dict[str, Any] | None
    signed_progress: tuple[float, ...]
    maximum_feasible_progress: float
    layer_joint_ranks: tuple[int, ...]
    nonzero_layer_count: int
    reason: str
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class ColdEntryFieldContract:
    label: str
    field_semantic_receipt: dict[str, Any]
    raw_velocity: tuple[float, ...]
    pre_soft_bf_velocity: tuple[float, ...]
    requested_progress: float
    raw_velocity_sha256: str
    pre_soft_bf_velocity_sha256: str
    requested_progress_sha256: str
    signed_progress_sha256: str
    pre_treatment_identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-cold-entry-field-contract/v1",
            "label": self.label,
            "field_semantic_receipt": dict(self.field_semantic_receipt),
            "raw_velocity": list(self.raw_velocity),
            "pre_soft_bf_velocity": list(self.pre_soft_bf_velocity),
            "requested_progress": self.requested_progress,
            "raw_velocity_sha256": self.raw_velocity_sha256,
            "pre_soft_bf_velocity_sha256": self.pre_soft_bf_velocity_sha256,
            "requested_progress_sha256": self.requested_progress_sha256,
            "signed_progress_sha256": self.signed_progress_sha256,
            "pre_treatment_identity_sha256": (
                self.pre_treatment_identity_sha256
            ),
            "treatment_divergence_allowed_after": "SOFT_FIELD_TRANSFORM",
        }


def cold_entry_field_contract(
    label: str,
    active: ActiveField,
    history: P1HistoryLedger,
) -> ColdEntryFieldContract:
    if label not in COLD_PANEL_LABELS:
        raise ODEBFContractError("cold entry contract label differs")
    solve_history = _history_keys(history, COLD_LAYER_ORDER, risk=False)
    risk_history = _history_keys(history, COLD_LAYER_ORDER, risk=True)
    semantic = cold_field_semantic_receipt(
        active.field,
        history_solve_keys_by_layer=solve_history,
        history_risk_keys_by_layer=risk_history,
    )
    raw = tuple(float(value) for value in active.raw.values)
    pre_soft = tuple(float(value) for value in active.projection.values)
    requested = float(active.problem.requested_progress)
    if (
        len(raw) != len(COLD_LAYER_ORDER)
        or len(pre_soft) != len(COLD_LAYER_ORDER)
        or not all(math.isfinite(value) for value in (*raw, *pre_soft, requested))
    ):
        raise ODEBFContractError("cold entry pre-treatment values differ")
    raw_sha = canonical_hash(list(raw))
    pre_soft_sha = canonical_hash(list(pre_soft))
    requested_sha = canonical_hash({"requested_progress": requested})
    signed_sha = canonical_hash(list(active.signed_progress.signed_progress))
    common = {
        "field_semantic_sha256": semantic["semantic_identity_sha256"],
        "raw_velocity_sha256": raw_sha,
        "pre_soft_bf_velocity_sha256": pre_soft_sha,
        "requested_progress_sha256": requested_sha,
        "signed_progress_sha256": signed_sha,
    }
    return ColdEntryFieldContract(
        label,
        semantic,
        raw,
        pre_soft,
        requested,
        raw_sha,
        pre_soft_sha,
        requested_sha,
        signed_sha,
        canonical_hash(common),
    )


def validate_cold_common_entry_contracts(
    readiness: ColdFieldReadiness,
    contracts: Mapping[str, ColdEntryFieldContract],
) -> dict[str, Any]:
    if (
        not readiness.reached
        or readiness.field_semantic_sha256 is None
        or tuple(contracts) != COLD_PANEL_LABELS
    ):
        raise ODEBFContractError("cold common entry inventory differs")
    semantic = {
        str(item.field_semantic_receipt["semantic_identity_sha256"])
        for item in contracts.values()
    }
    semantic.add(readiness.field_semantic_sha256)
    pre_treatment = {
        item.pre_treatment_identity_sha256 for item in contracts.values()
    }
    if len(semantic) != 1:
        raise ODEBFContractError("cold common entry semantic field differs")
    if len(pre_treatment) != 1:
        raise ODEBFContractError("cold common entry pre-soft routing differs")
    payload = {
        "schema": "ode-edit-s05-cold-common-entry-contract/v1",
        "readiness_field_receipt_sha256": readiness.field_sha256,
        "field_semantic_sha256": readiness.field_semantic_sha256,
        "pre_treatment_identity_sha256": next(iter(pre_treatment)),
        "contracts": {
            label: contracts[label].raw_free_payload()
            for label in COLD_PANEL_LABELS
        },
        "readiness_equals_no_soft_equals_soft": True,
        "pre_soft_raw_bf_requested_progress_equal": True,
        "divergence_allowed_only_after_soft_transform_or_descendants": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def evaluate_cold_field_readiness(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    ledger: ComputeLedger,
) -> ColdFieldReadiness:
    """Check the predeclared rank/nonzero/positive/feasible bootstrap exit."""

    layers = tuple(int(layer) for layer in hparams.layers)
    history_solve = _history_keys(history, layers, risk=False)
    history_risk = _history_keys(history, layers, risk=True)
    try:
        field = build_p1_dynamic_field(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            target_state=target_state,
            accepted_waypoint=0,
            cumulative_factors_by_weight={},
            history_solve_keys_by_layer=history_solve,
            history_risk_keys_by_layer=history_risk,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            residual_tolerance=lock.residual_tolerance,
            ledger=ledger,
            residual_policy=FULL_CURRENT_RESIDUAL_DEFINITION,
        )
    except ODEBFContractError as exc:
        if "nonpositive capacity" not in str(exc):
            raise
        payload = {
            "reached": False,
            "field_sha256": None,
            "signed_progress": [],
            "maximum_feasible_progress": 0.0,
            "layer_joint_ranks": [],
            "nonzero_layer_count": 0,
            "reason": "ZERO_OR_NONPOSITIVE_FULL_RESIDUAL_FIELD",
        }
        return ColdFieldReadiness(
            reached=False,
            field_sha256=None,
            field_semantic_sha256=None,
            field_semantic_receipt=None,
            signed_progress=(),
            maximum_feasible_progress=0.0,
            layer_joint_ranks=(),
            nonzero_layer_count=0,
            reason=payload["reason"],
            identity_sha256=canonical_hash(payload),
        )
    semantic = cold_field_semantic_receipt(
        field,
        history_solve_keys_by_layer=history_solve,
        history_risk_keys_by_layer=history_risk,
    )
    ranks = tuple(
        min(
            int(torch.linalg.matrix_rank(item.residual.to(dtype=torch.float64))),
            int(torch.linalg.matrix_rank(item.q.to(dtype=torch.float64))),
        )
        for item in field.layers
    )
    nonzero = sum(
        int(
            float(torch.linalg.vector_norm(item.residual.to(dtype=torch.float64))) > 0.0
            and float(torch.linalg.vector_norm(item.q.to(dtype=torch.float64))) > 0.0
        )
        for item in field.layers
    )
    try:
        signed = signed_progress_gradient(
            model,
            tokenizer,
            requests,
            field,
            cumulative_factors_by_weight={},
            ledger=ledger,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
    except ODEBFContractError as exc:
        if "no positive signed-progress direction" not in str(exc):
            raise
        payload = {
            "reached": False,
            "field_sha256": field.identity_sha256,
            "field_semantic_sha256": semantic["semantic_identity_sha256"],
            "field_semantic_receipt": semantic,
            "signed_progress": [],
            "maximum_feasible_progress": 0.0,
            "layer_joint_ranks": list(ranks),
            "nonzero_layer_count": nonzero,
            "reason": "NO_POSITIVE_TARGET_NEW_NLL_DIRECTION",
        }
        return ColdFieldReadiness(
            reached=False,
            field_sha256=field.identity_sha256,
            field_semantic_sha256=semantic["semantic_identity_sha256"],
            field_semantic_receipt=semantic,
            signed_progress=(),
            maximum_feasible_progress=0.0,
            layer_joint_ranks=ranks,
            nonzero_layer_count=nonzero,
            reason=payload["reason"],
            identity_sha256=canonical_hash(payload),
        )
    source = build_p1_routing_problem(
        field,
        signed,
        accepted_by_layer={layer: [] for layer in layers},
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
    )
    raw = solve_matched_raw_velocity(
        source, preservation_policy=PreservationConstraintPolicy.LOCKED
    )
    barrier = build_p1_routing_problem(
        field,
        signed,
        accepted_by_layer={layer: [] for layer in layers},
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    projected = project_matched_bf_velocity(
        barrier,
        raw,
        preservation_policy=PreservationConstraintPolicy.LOCKED,
    )
    ledger.increment("qp_solve", 4)
    ledger.increment("qp_certificate", 4)
    maximum = float(projected.maximum_feasible_progress)
    reached = bool(
        len(ranks) == 5
        and all(rank > 1 for rank in ranks)
        and nonzero == 5
        and any(value > 0.0 for value in signed.signed_progress)
        and projected.status is RoutingStatus.FEASIBLE
        and projected.values is not None
        and math.isfinite(maximum)
        and maximum >= COLD_JOINT_TARGET_EPSILON
    )
    reason = "READY" if reached else "STRUCTURAL_STACK_FIELD_UNREACHED"
    payload = {
        "reached": reached,
        "field_sha256": field.identity_sha256,
        "field_semantic_sha256": semantic["semantic_identity_sha256"],
        "field_semantic_receipt": semantic,
        "signed_progress": list(signed.signed_progress),
        "maximum_feasible_progress": maximum,
        "layer_joint_ranks": list(ranks),
        "nonzero_layer_count": nonzero,
        "reason": reason,
    }
    return ColdFieldReadiness(
        reached=reached,
        field_sha256=field.identity_sha256,
        field_semantic_sha256=semantic["semantic_identity_sha256"],
        field_semantic_receipt=semantic,
        signed_progress=signed.signed_progress,
        maximum_feasible_progress=maximum,
        layer_joint_ranks=ranks,
        nonzero_layer_count=nonzero,
        reason=reason,
        identity_sha256=canonical_hash(payload),
    )


@dataclass(slots=True)
class ColdBootstrap:
    z_base: torch.Tensor
    z_boot: torch.Tensor
    metric: ColdTargetMetric
    lookup_positions: tuple[int, ...]
    accepted_steps: tuple[dict[str, Any], ...]
    rejected_trials: tuple[dict[str, Any], ...]
    readiness: ColdFieldReadiness
    n_trial: int
    k_acc: int
    entry_objective: dict[str, Any]
    terminal_objective: dict[str, Any]
    compute: dict[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-cold-target-bootstrap/v1",
            "status": "BOOTSTRAP_FIELD_READY",
            "z_base_sha256": tensor_sha256(self.z_base),
            "z_boot_sha256": tensor_sha256(self.z_boot),
            "metric": self.metric.raw_free_payload(),
            "lookup_plan_sha256": canonical_hash(list(self.lookup_positions)),
            "accepted_steps": list(self.accepted_steps),
            "rejected_trials": list(self.rejected_trials),
            "n_trial": self.n_trial,
            "k_acc": self.k_acc,
            "entry_objective": dict(self.entry_objective),
            "terminal_objective": dict(self.terminal_objective),
            "compute": dict(self.compute),
            "readiness": asdict(self.readiness),
            "target_old_access_count": 0,
            "functional_p_h_native_access_count": 0,
            "native_or_direct_z_access_count": 0,
            "weight_mutation_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def run_cold_bootstrap(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    touched: Mapping[str, torch.nn.Parameter],
    ledger: ComputeLedger,
) -> ColdBootstrap:
    """Run the no-write target-only bootstrap and return its shared snapshot."""

    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    entry_contract = _parameter_contract_sha256(touched)
    entry_rng = _rng_identity()
    z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
    metric = ColdTargetMetric.from_z_base(z_base, request_order)
    positions = cold_lookup_positions(
        tokenizer,
        requests,
        contexts,
        fact_token_strategy=hparams.fact_token,
    )
    layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    current = z_base.clone()
    current_receipt = evaluate_cold_target_objective(
        model,
        tokenizer,
        requests,
        contexts,
        layer_name=layer_name,
        lookup_positions=positions,
        target_state=current,
        require_gradient=True,
    )
    ledger.increment("backward", current_receipt.backward_count)
    ledger.increment("target_backward", current_receipt.backward_count)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    n_trial = 0
    history = P1HistoryLedger(
        layer_order=tuple(int(layer) for layer in hparams.layers),
        maximum_records=40,
    )
    initial_readiness_payload = {
        "reached": False,
        "field_sha256": None,
        "field_semantic_sha256": None,
        "field_semantic_receipt": None,
        "signed_progress": [],
        "maximum_feasible_progress": 0.0,
        "layer_joint_ranks": [],
        "nonzero_layer_count": 0,
        "reason": "AWAITING_FIRST_ACCEPTED_BOOTSTRAP_STATE",
    }
    readiness = ColdFieldReadiness(
        reached=False,
        field_sha256=None,
        field_semantic_sha256=None,
        field_semantic_receipt=None,
        signed_progress=(),
        maximum_feasible_progress=0.0,
        layer_joint_ranks=(),
        nonzero_layer_count=0,
        reason=initial_readiness_payload["reason"],
        identity_sha256=canonical_hash(initial_readiness_payload),
    )
    entry_objective = current_receipt.raw_free_payload()
    while not readiness.reached:
        if len(accepted) >= COLD_BOOT_ACCEPTED_CAP or n_trial >= COLD_BOOT_TRIAL_CAP:
            raise ODEBFContractError("COLD_BOOTSTRAP_FIELD_UNREACHED")
        if current_receipt.gradient is None:
            raise ODEBFContractError("cold bootstrap lacks target gradient")
        velocity, gradient_norms = metric.unit_descent(current_receipt.gradient)
        dt = COLD_BOOT_DT_MAX
        accepted_here = False
        for retry_index in range(COLD_BOOT_RETRY_CAP + 1):
            if n_trial >= COLD_BOOT_TRIAL_CAP:
                break
            n_trial += 1
            candidate = (current + float(dt) * velocity).contiguous()
            step_distances = metric.distance(candidate, current)
            cumulative = metric.distance(candidate, z_base)
            if any(value > float(COLD_BOOT_RADIUS) + COLD_JOINT_TARGET_EPSILON for value in cumulative):
                actual = None
                predicted = None
                rho = None
                trial_receipt = None
                passed = False
                reason = "BOOTSTRAP_RADIUS_EXCEEDED"
            else:
                trial_receipt = evaluate_cold_target_objective(
                    model,
                    tokenizer,
                    requests,
                    contexts,
                    layer_name=layer_name,
                    lookup_positions=positions,
                    target_state=candidate,
                    require_gradient=False,
                )
                predicted = -float(
                    torch.sum(
                        current_receipt.gradient.to(dtype=torch.float64)
                        * (float(dt) * velocity.to(dtype=torch.float64))
                    )
                )
                actual = current_receipt.value - trial_receipt.value
                rho = actual / predicted if predicted > 0.0 else float("nan")
                passed = bool(
                    math.isfinite(actual)
                    and math.isfinite(predicted)
                    and math.isfinite(rho)
                    and actual > COLD_JOINT_TARGET_EPSILON
                    and rho >= COLD_RHO_ACCEPT
                )
                reason = "ACCEPTED" if passed else "TARGET_PROGRESS_REJECTED"
            payload = {
                "n_trial": n_trial,
                "retry_index": retry_index,
                "delta_tau_boot": fraction_payload(dt),
                "state_before_sha256": tensor_sha256(current),
                "candidate_sha256": tensor_sha256(candidate),
                "step_g_distance": list(step_distances),
                "cumulative_g_distance": list(cumulative),
                "gradient_norms": list(gradient_norms),
                "velocity_sha256": tensor_sha256(velocity),
                "predicted_decrease": predicted,
                "actual_decrease": actual,
                "rho": rho,
                "accepted": passed,
                "reason": reason,
                "target_old_access_count": 0,
                "native_or_direct_z_access_count": 0,
            }
            if passed and trial_receipt is not None:
                current = candidate
                current_receipt = evaluate_cold_target_objective(
                    model,
                    tokenizer,
                    requests,
                    contexts,
                    layer_name=layer_name,
                    lookup_positions=positions,
                    target_state=current,
                    require_gradient=True,
                )
                ledger.increment("backward", current_receipt.backward_count)
                ledger.increment("target_backward", current_receipt.backward_count)
                readiness = evaluate_cold_field_readiness(
                    model,
                    tokenizer,
                    requests,
                    hparams,
                    projector,
                    contexts,
                    target_state=current,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    lock=lock,
                    history=history,
                    ledger=ledger,
                )
                payload["accepted_state_objective"] = (
                    current_receipt.raw_free_payload()
                )
                payload["identity_sha256"] = canonical_hash(payload)
                accepted.append(payload)
                accepted_here = True
                break
            payload["identity_sha256"] = canonical_hash(payload)
            rejected.append(payload)
            if retry_index == COLD_BOOT_RETRY_CAP or dt / 2 < COLD_BOOT_DT_MIN:
                break
            dt /= 2
        if not accepted_here:
            raise ODEBFContractError("COLD_BOOTSTRAP_FIELD_UNREACHED")
    if (
        _parameter_contract_sha256(touched) != entry_contract
        or history.version != 0
        or _rng_identity() != entry_rng
    ):
        raise ODEBFStateError("cold bootstrap mutated W/history/RNG")
    identity_payload = {
        "z_base_sha256": tensor_sha256(z_base),
        "z_boot_sha256": tensor_sha256(current),
        "metric_sha256": metric.identity_sha256,
        "lookup_plan_sha256": canonical_hash(list(positions)),
        "accepted_step_sha256": [item["identity_sha256"] for item in accepted],
        "rejected_trial_sha256": [item["identity_sha256"] for item in rejected],
        "readiness_sha256": readiness.identity_sha256,
        "entry_objective_sha256": entry_objective["identity_sha256"],
        "terminal_objective_sha256": current_receipt.identity_sha256,
        "bootstrap_compute_sha256": ledger.identity(),
    }
    return ColdBootstrap(
        z_base,
        current,
        metric,
        positions,
        tuple(accepted),
        tuple(rejected),
        readiness,
        n_trial,
        len(accepted),
        entry_objective,
        current_receipt.raw_free_payload(),
        ledger.raw_free_payload(),
        canonical_hash(identity_payload),
    )


@dataclass(frozen=True, slots=True)
class ColdTargetVelocityReceipt:
    field_sha256: str
    velocity_sha256: str
    metric_sha256: str
    gradient_norms: tuple[float, ...]
    unit_g_norms: tuple[float, ...]
    target_backward_count: int
    processed_token_count: int
    target_old_access_count: int
    native_or_direct_z_access_count: int
    layer_local_full_residual_target_overlay: bool
    target_overlay_definition: str
    target_velocity_step_semantics: str
    candidate_coupled: bool
    retry_recomputes_target_velocity: bool


class _ColdLayerLocalTargetOverlay:
    """Cold-only differentiable FR overlay at one accepted-state field."""

    DEFINITION = "R_l(z_s)+(z-z_s)"
    STEP_SEMANTICS = "STEP_INDEPENDENT_FINITE_UNIT_FIELD_LOOKAHEAD"

    def __init__(
        self,
        model: torch.nn.Module,
        field: P1DynamicField,
        coefficients: torch.Tensor,
        target_state_variable: torch.Tensor,
    ) -> None:
        self.model = model
        self.field = field
        self.coefficients = coefficients
        self.target_state_variable = target_state_variable
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _layer_left(self, index: int, device: torch.device) -> torch.Tensor:
        layer = self.field.layers[index]
        anchor = self.field.target_state.to(device=device, dtype=torch.float32)
        residual = layer.residual.to(device=device, dtype=torch.float32)
        if (
            residual.shape != anchor.shape
            or self.target_state_variable.shape != anchor.shape
        ):
            raise ODEBFContractError("cold layer-local target overlay shape differs")
        return residual + (
            self.target_state_variable.to(device=device, dtype=torch.float32)
            - anchor
        )

    def _hook(self, index: int):
        layer = self.field.layers[index]

        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise ODEBFContractError(
                    "cold layer-local target overlay Linear contract differs"
                )
            hidden = inputs[0]
            right = layer.q.to(device=hidden.device, dtype=torch.float32)
            left = self._layer_left(index, hidden.device)
            if (
                hidden.shape[-1] != right.shape[0]
                or right.shape[1] != left.shape[1]
                or output.shape[-1] != left.shape[0]
            ):
                raise ODEBFContractError(
                    "cold layer-local target overlay geometry differs"
                )
            perturbation = (hidden.float() @ right) @ left.T
            result = output.float() + self.coefficients[index] * perturbation
            return result.to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "_ColdLayerLocalTargetOverlay":
        layers = tuple(int(item.layer) for item in self.field.layers)
        weight_names = tuple(item.weight_name for item in self.field.layers)
        if (
            layers != COLD_LAYER_ORDER
            or len(set(weight_names)) != len(COLD_LAYER_ORDER)
            or self.coefficients.ndim != 1
            or self.coefficients.numel() != len(COLD_LAYER_ORDER)
            or self.target_state_variable.shape != self.field.target_state.shape
            or not self.target_state_variable.requires_grad
        ):
            raise ODEBFContractError("cold layer-local target overlay inventory differs")
        try:
            for index, layer in enumerate(self.field.layers):
                if (
                    layer.residual_definition
                    != FULL_CURRENT_RESIDUAL_DEFINITION
                    or layer.residual_divisor != FULL_CURRENT_RESIDUAL_DIVISOR
                ):
                    raise ODEBFContractError(
                        "cold layer-local target overlay residual policy differs"
                    )
                module_name = layer.weight_name[: -len(".weight")]
                module = self.model.get_submodule(module_name)
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError(
                        "cold layer-local target overlay target is not exact Linear"
                    )
                self._handles.append(
                    module.register_forward_hook(self._hook(index))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


def write_aware_cold_target_velocity(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    field: P1DynamicField,
    coefficients: Sequence[float],
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    metric: ColdTargetMetric,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, ColdTargetVelocityReceipt]:
    """Differentiate target-new NLL through the current write field only."""

    device = next(model.parameters()).device
    coefficient_tensor = torch.tensor(
        tuple(float(value) for value in coefficients),
        dtype=torch.float32,
        device=device,
    )
    flat = field.target_state.to(device=device, dtype=torch.float32).contiguous().view(-1)
    flat = flat.detach().requires_grad_(True)
    target = flat.view(field.target_state.shape)
    with _virtual_context(model, cumulative_factors_by_weight):
        with _ColdLayerLocalTargetOverlay(
            model,
            field,
            coefficient_tensor,
            target,
        ):
            result = evaluate_routing_objective(
                model,
                tokenizer,
                requests,
                objective=RoutingObjective.TARGET_NEW_NLL,
                contexts=contexts,
                gradient_input=flat,
            )
    if result.input_gradient is None or result.backward_count != BATCH_SIZE:
        raise ODEBFContractError("cold write-aware target gradient differs")
    gradient = result.input_gradient.view(field.target_state.shape).to(
        device="cpu", dtype=torch.float64
    )
    velocity, gradient_norms = metric.unit_descent(gradient)
    unit = metric.distance(velocity, torch.zeros_like(velocity))
    ledger.increment("backward", result.backward_count)
    ledger.increment("target_backward", result.backward_count)
    receipt = ColdTargetVelocityReceipt(
        field.identity_sha256,
        tensor_sha256(velocity),
        metric.identity_sha256,
        gradient_norms,
        unit,
        result.backward_count,
        result.processed_token_count,
        0,
        0,
        True,
        _ColdLayerLocalTargetOverlay.DEFINITION,
        _ColdLayerLocalTargetOverlay.STEP_SEMANTICS,
        False,
        False,
    )
    return velocity, receipt


@dataclass(slots=True)
class ColdEntryCapture:
    """Minimal entry identity consumed by the virtual adaptive machinery."""

    entry_weights: dict[str, torch.Tensor]
    entry_sha256: dict[str, str]
    direct_z_sha256: tuple[str, ...]


def cold_target_step_validator(
    metric: ColdTargetMetric,
    target_entry: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, Fraction, Fraction], Mapping[str, Any]]:
    def validate(
        current: torch.Tensor,
        candidate: torch.Tensor,
        tau_before: Fraction,
        delta_tau: Fraction,
    ) -> Mapping[str, Any]:
        step = metric.distance(candidate, current)
        cumulative = metric.distance(candidate, target_entry)
        tau_after = tau_before + delta_tau
        if any(value > float(delta_tau) + COLD_JOINT_TARGET_EPSILON for value in step):
            raise ODEBFContractError("cold target step exceeded shared delta-tau")
        if any(value > float(tau_after) + COLD_JOINT_TARGET_EPSILON for value in cumulative):
            raise ODEBFContractError("cold target path exceeded cumulative tau")
        payload = {
            "metric_sha256": metric.identity_sha256,
            "step_g_distance": list(step),
            "step_bound": float(delta_tau),
            "cumulative_g_distance": list(cumulative),
            "cumulative_bound": float(tau_after),
            "target_weight_shared_delta_tau": True,
            "clip_or_projection_count": 0,
            "native_or_direct_z_dependency_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    return validate


def _cold_capture(
    base_values: Mapping[str, torch.Tensor],
    base_receipt: ArmWeightSnapshot,
    bootstrap: ColdBootstrap,
) -> ColdEntryCapture:
    per_request = tuple(
        tensor_sha256(bootstrap.z_boot[:, index].contiguous())
        for index in range(BATCH_SIZE)
    )
    return ColdEntryCapture(
        {name: value.detach().to(device="cpu").clone() for name, value in base_values.items()},
        dict(base_receipt.parameter_sha256),
        per_request,
    )


def _cold_exact_hit_trajectory(first_hit: FirstHitTracker) -> dict[str, Any]:
    """Serialize every accepted exact-hit/loss/re-hit observation."""

    events: list[dict[str, Any]] = []
    seen_hit = False
    previous_hit = False
    hit_records = []
    for record in first_hit.records:
        exact_hit = record.success_count == BATCH_SIZE
        if exact_hit and not seen_hit:
            event = "FIRST_HIT"
            seen_hit = True
        elif exact_hit and previous_hit:
            event = "HIT_PERSISTED"
        elif exact_hit:
            event = "REHIT"
        elif previous_hit:
            event = "HIT_LOST"
        else:
            event = "NO_HIT"
        payload = {
            "accepted_index": record.accepted_index,
            "tau": fraction_payload(record.tau),
            "success_count": record.success_count,
            "exact_online_efficacy_hit": exact_hit,
            "accepted_component_feasible": record.online_component_feasible,
            "event": event,
            "snapshot_sha256": record.snapshot_sha256,
        }
        events.append(payload)
        if exact_hit:
            hit_records.append(record)
        previous_hit = exact_hit
    first_index = next(
        (
            index
            for index, item in enumerate(events)
            if item["exact_online_efficacy_hit"]
        ),
        None,
    )
    return {
        "accepted_state_events": events,
        "first_hit_tau": (
            None if not hit_records else fraction_payload(hit_records[0].tau)
        ),
        "last_hit_tau": (
            None if not hit_records else fraction_payload(hit_records[-1].tau)
        ),
        "hit_count": len(hit_records),
        "hit_loss_count": sum(item["event"] == "HIT_LOST" for item in events),
        "rehit_count": sum(item["event"] == "REHIT" for item in events),
        "persistent_from_first_through_terminal": bool(
            first_index is not None
            and all(
                bool(item["exact_online_efficacy_hit"])
                for item in events[first_index:]
            )
        ),
        "observation_only": True,
        "controller_dependency_count": 0,
    }


def _run_cold_variant(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    label: str,
    field_policy: FunctionalPFieldPolicy,
    capture: ColdEntryCapture,
    bootstrap: ColdBootstrap,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    arm_state: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: OuterEntryPretrainedCache,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    recorder: AdaptiveReceiptRecorder,
    expected_entry_contract: ColdEntryFieldContract | None = None,
) -> tuple[VariantRollout, ColdEntryFieldContract]:
    if label not in COLD_PANEL_LABELS or field_policy not in (
        FunctionalPFieldPolicy.PROBE_ONLY,
        FunctionalPFieldPolicy.SOFT_HARD,
    ):
        raise ODEBFContractError("cold panel arm policy differs")
    variant = AdaptiveVariant.FR_A8
    clock = AdaptiveTauClock(adaptive_lock(variant))
    history = arm_state.history
    ledger = arm_state.ledger
    entry_contract = _parameter_contract_sha256(touched)
    entry_history = history.snapshot().digest
    entry_sampler = schedule.state_digest
    entry_rng = _rng_identity()
    current_target = bootstrap.z_boot.clone()
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        int(layer): [] for layer in hparams.layers
    }
    entry_omega = _omega_state(accepted_by_layer)
    snapshots: list[AcceptedSnapshot] = []
    first_hit = FirstHitTracker()
    from .p1_backend import evaluate_routing_progress

    current_margin = evaluate_routing_progress(
        model,
        tokenizer,
        requests,
        cumulative_factors_by_weight=current_factors,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
    )
    entry_eval = _evaluate_rewrite(
        model,
        tokenizer,
        requests,
        alias=alias,
        factors=current_factors,
        ledger=ledger,
    )
    status = "ACTIVE"
    active: ActiveField | None = None
    replay_entry: Any | None = None
    replay_field_sha256: str | None = None
    previous_bf_velocity: tuple[float, ...] | None = None
    active_target_identity: str | None = None

    def build_active() -> ActiveField:
        nonlocal active_target_identity, replay_entry, replay_field_sha256
        if replay_entry is None:
            replay_entry = _controller_replay_entry(
                model,
                tokenizer,
                alias=alias,
                arm_state=arm_state,
                sample_waypoint=len(snapshots) + 1,
                factors=current_factors,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_entry_p_cache,
            )

        def target_builder(
            field: P1DynamicField, values: Sequence[float]
        ) -> tuple[torch.Tensor, ColdTargetVelocityReceipt]:
            return write_aware_cold_target_velocity(
                model,
                tokenizer,
                requests,
                contexts,
                field=field,
                coefficients=values,
                cumulative_factors_by_weight=current_factors,
                metric=bootstrap.metric,
                ledger=ledger,
            )

        active_target_identity = canonical_hash(
            {
                "bootstrap_sha256": bootstrap.identity_sha256,
                "metric_sha256": bootstrap.metric.identity_sha256,
                "current_target_state_sha256": tensor_sha256(current_target),
            }
        )
        built = _build_active_field(
            model,
            tokenizer,
            requests,
            variant=variant,
            accepted_index=len(snapshots),
            factors=current_factors,
            target_state=current_target,
            capture=capture,  # type: ignore[arg-type]
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            lock=lock,
            history=history,
            accepted_by_layer=accepted_by_layer,
            ledger=ledger,
            recorder=recorder,
            native_target=bootstrap.z_boot,
            target_base=bootstrap.z_base,
            routing_objective=RoutingObjective.TARGET_NEW_NLL,
            preservation_policy=PreservationConstraintPolicy.LOCKED,
            functional_p_field_policy=field_policy,
            functional_p_replay_entry=replay_entry,
            theta0_cache=theta0_cache,
            alias=alias,
            touched=touched,
            schedule=schedule,
            dynamic_entry=True,
            target_state_identity_sha256=active_target_identity,
            target_velocity_builder=target_builder,
            target_state_objective="COLD_TARGET_NEW_NLL_G_UNIT",
        )
        replay_field_sha256 = built.field.identity_sha256
        return built

    try:
        active = build_active()
    except ODEBFContractError as exc:
        if not str(exc).startswith("PROGRESS_INFEASIBLE:"):
            raise
        status = "STRUCTURAL_STACK_PROGRESS_INFEASIBLE"
    if active is None:
        raise ODEBFContractError("cold post-bootstrap entry field is unavailable")
    entry_field_contract = cold_entry_field_contract(label, active, history)
    if (
        bootstrap.readiness.field_semantic_sha256 is None
        or entry_field_contract.field_semantic_receipt[
            "semantic_identity_sha256"
        ]
        != bootstrap.readiness.field_semantic_sha256
    ):
        raise ODEBFContractError(
            "cold arm did not share bootstrap semantic entry field"
        )
    if expected_entry_contract is not None:
        validate_cold_common_entry_contracts(
            bootstrap.readiness,
            {
                COLD_PANEL_LABELS[0]: expected_entry_contract,
                COLD_PANEL_LABELS[1]: entry_field_contract,
            },
        )
    while active is not None and clock.status == "ACTIVE" and not clock.complete:
        if active_target_identity is None:
            raise ODEBFStateError("cold active target identity is absent")
        try:
            trial = clock.begin_trial()
        except ODEBFStateError:
            if clock.status == "ACTIVE":
                raise
            break
        outcome = _run_trial(
            model,
            tokenizer,
            requests,
            alias=alias,
            variant=variant,
            accepted_index=len(snapshots),
            n_trial=trial.n_trial,
            retry_index=trial.retry_index,
            tau_before=trial.tau_before,
            delta_tau=trial.delta_tau_trial,
            delta_tau_proposed=trial.delta_tau_proposed,
            remainder=trial.remainder,
            active=active,
            current_factors=current_factors,
            current_target=current_target,
            current_margin=current_margin,
            replay_entry=replay_entry,
            theta0_cache=theta0_cache,
            capture=capture,  # type: ignore[arg-type]
            lock=lock,
            ledger=ledger,
            recorder=recorder,
            touched=touched,
            history=history,
            schedule=schedule,
            previous_bf_velocity=previous_bf_velocity,
            accepted_by_layer=accepted_by_layer,
            contexts=contexts,
            routing_objective=RoutingObjective.TARGET_NEW_NLL,
            functional_p_policy=FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            preservation_policy=PreservationConstraintPolicy.LOCKED,
            target_state_identity_sha256=active_target_identity,
            target_step_validator=cold_target_step_validator(
                bootstrap.metric, bootstrap.z_boot
            ),
            target_state_objective="COLD_TARGET_NEW_NLL_G_UNIT",
        )
        if outcome.gate_accepted:
            if outcome.trust_ratio is None:
                raise ODEBFContractError("cold accepted trial lacks rho")
            transition = clock.accept(trial=trial, rho=outcome.trust_ratio)
        else:
            transition = clock.reject(trial=trial)
            ledger.increment("reject")
        transition_sha256 = recorder.transition(
            {
                "mode": "cold-adaptive-pseudo-time",
                "trial_receipt_sha256": outcome.receipt_sha256,
                "clock_trial": trial.raw_free_payload(),
                "clock_transition": transition.raw_free_payload(),
                "field_sha256": active.field.identity_sha256,
                "replay_field_sha256": replay_field_sha256,
                "same_state_retry": not outcome.gate_accepted,
                "retry_changes_only_delta_tau": True,
                "functional_p_candidate_veto_influence_count": 0,
                "first_hit_observation_only": True,
            }
        )
        if not outcome.gate_accepted:
            continue
        snapshot = _append_accepted_snapshot(
            outcome=outcome,
            active=active,
            accepted_index_before=len(snapshots),
            tau_after=clock.tau,
            delta_tau=trial.delta_tau_trial,
            transition_sha256=transition_sha256,
            accepted_by_layer=accepted_by_layer,
            ledger=ledger,
            recorder=recorder,
            first_hit=first_hit,
        )
        snapshots.append(snapshot)
        current_factors = _factor_map(outcome.candidate_factors)
        current_target = outcome.target_trial.clone()
        current_margin = outcome.margin
        previous_bf_velocity = active.bf_velocity.velocity
        replay_entry = None
        replay_field_sha256 = None
        if not clock.complete:
            try:
                active = build_active()
            except ODEBFContractError as exc:
                if not str(exc).startswith("PROGRESS_INFEASIBLE:"):
                    raise
                status = "STRUCTURAL_STACK_PROGRESS_INFEASIBLE"
                active = None
                break
    if status == "ACTIVE":
        status = clock.status
    if ledger.completed_correction_cycles == 0:
        ledger.finish_cycle(0)
    trajectory_complete = status == "TAU_COMPLETE" and clock.tau == Fraction(1, 1)
    confirmations = _terminal_confirm_snapshots(
        model,
        tokenizer,
        alias=alias,
        arm_state=arm_state,
        snapshots=snapshots,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        lock=lock,
        ledger=ledger,
        recorder=recorder,
        trajectory_complete=trajectory_complete,
        functional_p_policy=FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        preservation_policy=PreservationConstraintPolicy.LOCKED,
    )
    if (
        _parameter_contract_sha256(touched) != entry_contract
        or history.snapshot().digest != entry_history
        or schedule.state_digest != entry_sampler
        or _rng_identity() != entry_rng
        or (not snapshots and _omega_state(accepted_by_layer) != entry_omega)
    ):
        raise ODEBFStateError("cold rollout mutated model/history/sampler/RNG")
    termination = status
    rollout_payload = {
        "variant": label,
        "status": status,
        "termination_label": termination,
        "residual_policy": FULL_CURRENT_RESIDUAL_DEFINITION,
        "accepted_t": fraction_payload(clock.tau),
        "k_acc": len(snapshots),
        "n_trial": clock.n_trial,
        "n_reject": clock.n_reject,
        "field_build_count": len(recorder.field_hashes),
        "accepted_snapshot_sha256": [item.snapshot_sha256 for item in snapshots],
        "online_first_hit": None
        if first_hit.first_online is None
        else {
            **asdict(first_hit.first_online),
            "tau": fraction_payload(first_hit.first_online.tau),
        },
        "online_exact_hit_trajectory": _cold_exact_hit_trajectory(first_hit),
        "first_hit_observation_only": True,
        "full_horizon_continues_after_hit": True,
        "entry_success": entry_eval.batch_success.raw_free_payload(),
        "terminal_confirmation_count": len(confirmations),
        "receipt_links": recorder.links(),
        "compute": ledger.raw_free_payload(),
        "routing_objective": RoutingObjective.TARGET_NEW_NLL.value,
        "target_state_objective": "COLD_TARGET_NEW_NLL_G_UNIT",
        "target_old_decision_influence_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "structural_p_hard": True,
        "functional_p_decision_policy": FunctionalPDecisionPolicy.OBSERVATION_ONLY.value,
        "functional_p_candidate_veto_influence_count": 0,
        "functional_p_field_policy": field_policy.value,
        "functional_p_field_decision_influence_count": (
            len(recorder.field_hashes)
            if field_policy is FunctionalPFieldPolicy.SOFT_HARD
            else 0
        ),
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "heldout_access_count": 0,
    }
    rollout_sha = canonical_hash(rollout_payload)
    recorder.terminal(
        {
            "rollout_summary": rollout_payload,
            "rollout_sha256": rollout_sha,
            "trajectory_complete": trajectory_complete,
        }
    )
    return (
        VariantRollout(
            variant,
            status,
            termination,
            FULL_CURRENT_RESIDUAL_DEFINITION,
            clock.tau,
            len(snapshots),
            clock.n_trial,
            clock.n_reject,
            len(recorder.field_hashes),
            snapshots,
            first_hit,
            recorder,
            ledger,
            entry_eval.batch_success.raw_free_payload(),
            confirmations,
            rollout_sha,
            label,
            RoutingObjective.TARGET_NEW_NLL.value,
            FunctionalPDecisionPolicy.OBSERVATION_ONLY.value,
            PreservationConstraintPolicy.LOCKED.value,
            field_policy.value,
        ),
        entry_field_contract,
    )


def run_cold_diagnostic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
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
    theta0_cache: Theta0TeacherCache,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    artifact_guard: Any,
    artifact_receipt: Any,
    numerical_sha256: str,
    context_sha256: str,
    cuda_runtime_receipt: Mapping[str, Any],
    job_ledger: ComputeLedger,
    write_once: Any,
) -> dict[str, Any]:
    """Execute cold actions, freeze them, then open N32 and held-out surfaces."""

    from .p1_runtime import (
        ArmRuntimeState,
        _entry_parameter_snapshot_sha256,
        _evaluate_native_rewrite,
        _observed_memory,
    )

    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("cold diagnostic requires one fresh B10")
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    if request_order != stream["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("cold request/seal order differs")
    base_bytes = {name: tensor_sha256(value) for name, value in sorted(touched.items())}
    base_contract = _parameter_contract_sha256(touched)
    if tuple(sorted(base_bytes.items())) != tuple(base_receipt.parameter_sha256):
        raise ODEBFStateError("cold diagnostic did not start from W0")
    bootstrap_ledger = ComputeLedger()
    bootstrap_counter = ModelForwardCounter(model, bootstrap_ledger)
    try:
        bootstrap = run_cold_bootstrap(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            lock=controller_lock,
            touched=touched,
            ledger=bootstrap_ledger,
        )
    finally:
        bootstrap_counter.close()
    bootstrap_sha = write_once(
        raw_root / "cold" / "bootstrap.json", bootstrap.raw_free_payload()
    )
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_entry_snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    outer_counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_entry_snapshot,
        )
    finally:
        outer_counter.close()
    capture = _cold_capture(base_values, base_receipt, bootstrap)
    specs = (
        (COLD_PANEL_LABELS[0], FunctionalPFieldPolicy.PROBE_ONLY),
        (COLD_PANEL_LABELS[1], FunctionalPFieldPolicy.SOFT_HARD),
    )
    rollouts: dict[str, VariantRollout] = {}
    entry_field_contracts: dict[str, ColdEntryFieldContract] = {}
    recorders: list[AdaptiveReceiptRecorder] = []
    for label, field_policy in specs:
        if _parameter_contract_sha256(touched) != base_contract:
            raise ODEBFStateError("cold arm W0 entry differs")
        receipt = ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"label": label, "weights": base_receipt.parameter_sha256}),
        )
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(
                layer_order=tuple(int(layer) for layer in hparams.layers),
                maximum_records=40,
            ),
            ComputeLedger(),
            receipt,
            dict(base_values),
        )
        recorder = AdaptiveReceiptRecorder(
            raw_root,
            AdaptiveVariant.FR_A8,
            write_once,
            variant_label=label,
            instruction_id=COLD_INSTRUCTION_ID,
            receipt_schema=f"{COLD_SCHEMA_NAMESPACE}-receipt/v1",
        )
        recorders.append(recorder)
        counter = ModelForwardCounter(model, arm_state.ledger)
        try:
            rollout, entry_field_contract = _run_cold_variant(
                model,
                tokenizer,
                requests,
                alias=alias,
                label=label,
                field_policy=field_policy,
                capture=capture,
                bootstrap=bootstrap,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                lock=controller_lock,
                arm_state=arm_state,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_entry_p_cache,
                theta0_cache=theta0_cache,
                touched=touched,
                recorder=recorder,
                expected_entry_contract=entry_field_contracts.get(
                    COLD_PANEL_LABELS[0]
                ),
            )
        finally:
            counter.close()
        _observed_memory(arm_state.ledger)
        if arm_state.history.version != 0:
            raise ODEBFStateError("cold arm appended history")
        rollouts[label] = rollout
        entry_field_contracts[label] = entry_field_contract
        stages.record(
            f"post_{label.lower().replace('-', '_')}",
            {
                "status": rollout.status,
                "accepted_t": fraction_payload(rollout.accepted_t),
                "k_acc": rollout.k_acc,
                "n_trial": rollout.n_trial,
                "n_reject": rollout.n_reject,
                "rollout_sha256": rollout.rollout_sha256,
            },
        )
    common_entry_contract = validate_cold_common_entry_contracts(
        bootstrap.readiness, entry_field_contracts
    )
    common_entry_contract_sha = write_once(
        raw_root / "cold" / "common-entry-contract.json",
        common_entry_contract,
    )
    if {name: tensor_sha256(value) for name, value in sorted(touched.items())} != base_bytes:
        raise ODEBFStateError("cold actions did not preserve W0")
    action_freeze = {
        "schema": f"{COLD_SCHEMA_NAMESPACE}-cold-action-freeze/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "request_order_sha256": request_order,
        "bootstrap_sha256": bootstrap_sha,
        "common_entry_contract_sha256": common_entry_contract_sha,
        "common_entry_semantic_field_sha256": common_entry_contract[
            "field_semantic_sha256"
        ],
        "common_entry_pre_treatment_identity_sha256": common_entry_contract[
            "pre_treatment_identity_sha256"
        ],
        "rollout_sha256": {label: value.rollout_sha256 for label, value in rollouts.items()},
        "cold_actions_frozen_before_native": True,
        "cold_native_or_direct_z_access_count": 0,
        "heldout_controller_access_count": 0,
    }
    action_freeze_sha = write_once(raw_root / "cold" / "action-freeze.json", action_freeze)

    # Reference-only boundary: the canonical N32/direct-z primitive is opened
    # only after both cold action trajectories and their hashes are frozen.
    from .p1_backend import capture_p1_native_entry

    native_history = P1HistoryLedger(
        layer_order=tuple(int(layer) for layer in hparams.layers), maximum_records=40
    )
    native_ledger = ComputeLedger()
    native_counter = ModelForwardCounter(model, native_ledger)
    try:
        native_capture = capture_p1_native_entry(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            history_keys_by_layer=_history_keys(
                native_history,
                tuple(int(layer) for layer in hparams.layers),
                risk=False,
            ),
            mutation_lock=mutation_lock,
            ledger=native_ledger,
            residual_tolerance=controller_lock.residual_tolerance,
        )
        native_online = _evaluate_native_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            candidates=native_capture.native_candidates,
            ledger=native_ledger,
        )
    finally:
        native_counter.close()
    if native_capture.request_order_sha256 != request_order:
        raise ODEBFContractError("cold postfreeze Native order differs")
    n32_sha = write_once(
        raw_root / "cold" / "N32_NATIVE-postfreeze.json",
        {
            "schema": f"{COLD_SCHEMA_NAMESPACE}-native-reference/v1",
            "action_freeze_sha256": action_freeze_sha,
            "opened_after_cold_action_freeze": True,
            "capture": native_capture.raw_free_payload(),
            "official_success": native_online.batch_success.raw_free_payload(),
            "compute": native_ledger.raw_free_payload(),
            "controller_dependency_count": 0,
            "cold_receipt_backflow_count": 0,
        },
    )
    panel, step_receipts = _postfreeze_stepwise_panel(
        model,
        tokenizer,
        alias=alias,
        requests=requests,
        dataset_path=dataset_path,
        capture=native_capture,
        rollouts=rollouts,
        raw_root=raw_root,
        write_once=write_once,
        touched=touched,
        instruction_id=COLD_INSTRUCTION_ID,
        schema_namespace=COLD_SCHEMA_NAMESPACE,
    )
    refinement = {
        "schema": f"{COLD_SCHEMA_NAMESPACE}-contrasts/v1",
        "soft_minus_nosoft": _target_new_panel_contrast(
            native_capture,
            rollouts,
            step_receipts,
            left_label=COLD_PANEL_LABELS[0],
            right_label=COLD_PANEL_LABELS[1],
        ),
        "shared_bootstrap_sha256": bootstrap_sha,
        "cold_native_or_direct_z_access_count": 0,
        "functional_p_candidate_veto_influence_count": 0,
        "structural_p_hard": True,
        "historical_soft_active": False,
        "historical_soft_reason": "EMPTY_HISTORY",
        "scientific_promotion_authorized": False,
    }
    stepwise_sha = write_once(
        raw_root / "stepwise" / "panel.json", {**panel, "refinement": refinement}
    )
    artifact_guard.assert_unchanged()
    final_bytes = {name: tensor_sha256(value) for name, value in sorted(touched.items())}
    if final_bytes != base_bytes:
        raise ODEBFStateError("cold diagnostic final W0 restore differs")
    for recorder in recorders:
        recorder.assert_observation_complete()
    terminal = {
        "schema": f"{COLD_SCHEMA_NAMESPACE}-terminal/v1",
        "instruction_id": COLD_INSTRUCTION_ID,
        "status": "FRESH_B10_COLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION",
        "alias": alias,
        "source_head": source_head,
        "request_order_sha256": request_order,
        "stream_root_digest": stream["root_digest"],
        "bootstrap_sha256": bootstrap_sha,
        "common_entry_contract_sha256": common_entry_contract_sha,
        "common_entry_semantic_field_sha256": common_entry_contract[
            "field_semantic_sha256"
        ],
        "action_freeze_sha256": action_freeze_sha,
        "n32_postfreeze_sha256": n32_sha,
        "stepwise_panel_sha256": stepwise_sha,
        "variant_status": {label: value.status for label, value in rollouts.items()},
        "variant_rollout_sha256": {
            label: value.rollout_sha256 for label, value in rollouts.items()
        },
        "cold_native_or_direct_z_access_count": 0,
        "n32_opened_only_after_action_freeze": True,
        "structural_p_hard": True,
        "functional_p_candidate_veto_influence_count": 0,
        "functional_p_field_influence_by_label": {
            COLD_PANEL_LABELS[0]: 0,
            COLD_PANEL_LABELS[1]: rollouts[COLD_PANEL_LABELS[1]].field_build_count,
        },
        "first_hit_observation_only": True,
        "full_horizon_target_tau": 1.0,
        "no_p_trust_factorial": True,
        "numerical_lock_sha256": numerical_sha256,
        "artifact_receipt": asdict(artifact_receipt),
        "context_sha256": context_sha256,
        "cuda_preflight": dict(cuda_runtime_receipt),
        "job_compute": job_ledger.raw_free_payload(),
        "final_w0_restored": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "scientific_promotion_authorized": False,
    }
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest_sha = write_once(
        destination / "manifest.json",
        {
            "schema": f"{COLD_SCHEMA_NAMESPACE}-manifest/v1",
            "instruction_id": COLD_INSTRUCTION_ID,
            "status": terminal["status"],
            "alias": alias,
            "source_head": source_head,
            "terminal_sha256": terminal_sha,
            "bootstrap_sha256": bootstrap_sha,
            "common_entry_contract_sha256": common_entry_contract_sha,
            "action_freeze_sha256": action_freeze_sha,
            "n32_postfreeze_sha256": n32_sha,
            "stepwise_panel_sha256": stepwise_sha,
            "retry_submission_count": 0,
        },
    )
    return {
        "status": terminal["status"],
        "alias": alias,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "final_w0_restored": True,
    }
