"""Hard-B10 fixed-horizon transport and authoritative CounterFact scoring.

The action path consumes only the ten rewrite requests.  CounterFact
paraphrase/neighborhood payloads are represented only by the endpoint scorer
after the arm action has been frozen.  Exact batch success is an observation;
all locked refresh budgets continue to completion.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter

from .contracts import ControllerConfig, EventReading, MethodContractError
from .ct_k4 import (
    _build_field,
    _commit_trial,
    _measure_event,
    _transport_solution,
    event_reading_to_dict,
)
from .ct_k4_evaluation import (
    _assert_evaluation_invariants,
    _config_snapshot,
    _parameter_guards,
)
from .events import ControllerRequest
from .hard_batch_backend import JOINT_BATCH_SIZE
from .instrumentation import EditInstrumentation


COUNTERFACT_EVALUATOR_SOURCE_SHA256 = (
    "f2de63e6cc68871cb042f614193294a433c4b76622dd410de515df8cb73416b8"
)
COUNTERFACT_AGGREGATOR_SOURCE_SHA256 = (
    "9791812222b151952c16cacf28daf3e8be9645e37321e0aa84be6ab5e24a66f8"
)
ZSRE_EVALUATOR_SOURCE_SHA256 = (
    "bf5b0c79d65b54df6ea1478798dece18cbe48f3b6bb3015e3740a9ba43877e57"
)


class HardBatchArm(str, Enum):
    NATIVE_MEMIT_B10 = "native-memit-b10"
    REFRESH_K10_T1 = "refresh-k10-t1"
    REFRESH_K20_T1_RESOLUTION = "refresh-k20-t1-resolution"
    REFRESH_2XK10_T2_CORRECTION = "refresh-2xk10-t2-correction"


HARD_BATCH_ARM_ORDER = (
    HardBatchArm.NATIVE_MEMIT_B10,
    HardBatchArm.REFRESH_K10_T1,
    HardBatchArm.REFRESH_K20_T1_RESOLUTION,
    HardBatchArm.REFRESH_2XK10_T2_CORRECTION,
)


def fixed_horizon_lambdas(k_resolution: int) -> tuple[float, ...]:
    if (
        isinstance(k_resolution, bool)
        or not isinstance(k_resolution, int)
        or k_resolution <= 0
    ):
        raise MethodContractError("fixed-horizon resolution must be positive")
    return tuple(
        1.0 / (k_resolution - position)
        for position in range(k_resolution)
    )


def frozen_cumulative_fractions(k_resolution: int) -> tuple[float, ...]:
    fixed_horizon_lambdas(k_resolution)
    return tuple(
        (position + 1) / k_resolution for position in range(k_resolution)
    )


def naive_repeated_residual_fraction(k_resolution: int) -> float:
    fixed_horizon_lambdas(k_resolution)
    return (1.0 - 1.0 / k_resolution) ** k_resolution


@dataclass(frozen=True, slots=True)
class CounterFactPairSpec:
    case_id: str
    role: str
    prefix: str
    target_new: str
    target_true: str
    new_is_success: bool

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or self.role not in {"efficacy", "generalization", "locality"}
            or not self.prefix
            or not self.target_new.strip()
            or not self.target_true.strip()
            or not isinstance(self.new_is_success, bool)
        ):
            raise MethodContractError("CounterFact NLL pair specification differs")


@dataclass(frozen=True, slots=True)
class CounterFactPairReading:
    case_id: str
    role: str
    target_new_nll: float
    target_true_nll: float
    success: bool
    target_new_token_count: int
    target_true_token_count: int
    source_prefix_token_count: int
    target_new_start: int
    target_true_start: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "target_new_nll": self.target_new_nll,
            "target_true_nll": self.target_true_nll,
            "nll_slack_true_minus_new": self.target_true_nll
            - self.target_new_nll,
            "success": self.success,
            "target_new_token_count": self.target_new_token_count,
            "target_true_token_count": self.target_true_token_count,
            "source_prefix_token_count": self.source_prefix_token_count,
            "target_new_start": self.target_new_start,
            "target_true_start": self.target_true_start,
        }


@dataclass(frozen=True, slots=True)
class CounterFactJointReading:
    rows: tuple[CounterFactPairReading, ...]
    model_forward_count: int
    input_token_count: int
    evaluator_wall_seconds: float

    def __post_init__(self) -> None:
        if (
            not self.rows
            or self.model_forward_count != 1
            or self.input_token_count <= 0
            or not math.isfinite(self.evaluator_wall_seconds)
            or self.evaluator_wall_seconds < 0.0
        ):
            raise MethodContractError("CounterFact joint reading differs")

    @property
    def success_bits(self) -> tuple[bool, ...]:
        return tuple(row.success for row in self.rows)

    @property
    def success_count(self) -> int:
        return sum(self.success_bits)

    @property
    def denominator(self) -> int:
        return len(self.rows)

    @property
    def exact_hit(self) -> bool:
        return self.success_count == self.denominator

    @property
    def official_mean(self) -> float:
        return self.success_count / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": "alphaedit-counterfact-length-normalized-suffix-nll",
            "evaluator_source_sha256": COUNTERFACT_EVALUATOR_SOURCE_SHA256,
            "aggregator_source_sha256": COUNTERFACT_AGGREGATOR_SOURCE_SHA256,
            "rows": [row.to_dict() for row in self.rows],
            "success_bits": [int(value) for value in self.success_bits],
            "success_count": self.success_count,
            "denominator": self.denominator,
            "official_mean": self.official_mean,
            "exact_joint_hit": self.exact_hit,
            "model_forward_count": self.model_forward_count,
            "input_token_count": self.input_token_count,
            "evaluator_wall_seconds": self.evaluator_wall_seconds,
            "native_nll_equality_required": False,
        }


def _source_target_ids(tokenizer: Any, target: str) -> tuple[int, ...]:
    """Reproduce AlphaEdit's pinned ``tok(f" {target}")`` call exactly."""

    values = tuple(
        int(value)
        for value in tokenizer.encode(f" {target}", add_special_tokens=True)
    )
    if not values:
        raise MethodContractError("CounterFact NLL target tokenization is empty")
    return values


def _model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device(getattr(model, "device", "cpu"))


def score_counterfact_nll_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    specs: Sequence[CounterFactPairSpec],
) -> CounterFactJointReading:
    """Score all new/true suffix pairs in one authoritative model call.

    The source evaluator's mean token NLL, strict inequality, prefix length,
    target tokenization, and Llama-2-only offset branch are preserved.  This
    is deliberately source parity, not the separate Session 03 token-accuracy
    evaluator's suffix-derived policy.
    """

    locked = tuple(specs)
    if not locked:
        raise MethodContractError("CounterFact NLL panel is empty")
    original_padding = getattr(tokenizer, "padding_side", None)
    if original_padding != "right":
        raise MethodContractError("CounterFact NLL evaluator requires right padding")
    prefixes = [item.prefix for item in locked]
    prefix_payload = tokenizer(prefixes, add_special_tokens=True)
    prefix_ids = prefix_payload.get("input_ids")
    if not isinstance(prefix_ids, list) or len(prefix_ids) != len(locked):
        raise MethodContractError("CounterFact prefix tokenization differs")
    prefix_lengths = tuple(len(row) for row in prefix_ids)
    full_rows: list[tuple[int, ...]] = []
    target_rows: list[tuple[int, ...]] = []
    starts: list[int] = []
    source_name = str(
        getattr(getattr(model, "config", None), "_name_or_path", "")
    ).lower()
    llama2_offset = "llama-2" in source_name
    parameter_guards = _parameter_guards(model)
    config_snapshot = _config_snapshot(model)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    try:
        for index, item in enumerate(locked):
            for _is_new, target in (
                (True, item.target_new),
                (False, item.target_true),
            ):
                target_ids = _source_target_ids(tokenizer, target)
                full = tuple(
                    int(value)
                    for value in tokenizer.encode(
                        f"{item.prefix} {target}",
                        add_special_tokens=True,
                    )
                )
                start = prefix_lengths[index]
                if llama2_offset:
                    if len(target_ids) <= 2 or start <= 1:
                        raise MethodContractError(
                            "CounterFact Llama-2 source offset is invalid"
                        )
                    target_ids = target_ids[2:]
                    start -= 1
                if start < 1 or start + len(target_ids) - 1 > len(full):
                    raise MethodContractError("CounterFact source target span differs")
                full_rows.append(full)
                target_rows.append(target_ids)
                starts.append(start)
        pad = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise MethodContractError("CounterFact tokenizer pad id is invalid")
        width = max(len(row) for row in full_rows)
        input_ids = torch.full((len(full_rows), width), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(full_rows):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, : len(row)] = 1
        device = _model_device(model)
        wall_start = time.perf_counter()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
        if llama2_offset:
            logits = logits[:, 1:, :]
        elapsed = time.perf_counter() - wall_start
        nll_values = []
        for row_index, (target_ids, start) in enumerate(
            zip(target_rows, starts, strict=True)
        ):
            token_logits = logits[row_index, start - 1 : start - 1 + len(target_ids)]
            if token_logits.shape[0] != len(target_ids):
                raise MethodContractError("CounterFact NLL target span differs")
            accumulator = torch.tensor(0.0, dtype=torch.float32)
            for offset, token_id in enumerate(target_ids):
                value = -F.log_softmax(token_logits[offset], dim=0)[token_id]
                accumulator += float(value.detach().cpu())
            accumulator /= len(target_ids)
            observed = float(accumulator)
            if not math.isfinite(observed):
                raise MethodContractError("CounterFact NLL is non-finite")
            nll_values.append(observed)
    finally:
        tokenizer.padding_side = original_padding
        _assert_evaluation_invariants(
            model,
            parameter_guards=parameter_guards,
            config_snapshot=config_snapshot,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
        )
    readings = []
    for index, item in enumerate(locked):
        new_nll = nll_values[2 * index]
        true_nll = nll_values[2 * index + 1]
        success = new_nll < true_nll if item.new_is_success else true_nll < new_nll
        readings.append(
            CounterFactPairReading(
                case_id=item.case_id,
                role=item.role,
                target_new_nll=new_nll,
                target_true_nll=true_nll,
                success=success,
                target_new_token_count=len(target_rows[2 * index]),
                target_true_token_count=len(target_rows[2 * index + 1]),
                source_prefix_token_count=prefix_lengths[index],
                target_new_start=starts[2 * index],
                target_true_start=starts[2 * index + 1],
            )
        )
    return CounterFactJointReading(
        rows=tuple(readings),
        model_forward_count=1,
        input_token_count=int(attention_mask.sum().item()),
        evaluator_wall_seconds=elapsed,
    )


def efficacy_specs(
    requests: Sequence[ControllerRequest],
) -> tuple[CounterFactPairSpec, ...]:
    locked = tuple(requests)
    if len(locked) != JOINT_BATCH_SIZE:
        raise MethodContractError("hard-batch efficacy panel must contain ten requests")
    return tuple(
        CounterFactPairSpec(
            case_id=item.case_id,
            role="efficacy",
            prefix=item.prompt.format(item.subject),
            target_new=item.target_new,
            target_true=item.target_old,
            new_is_success=True,
        )
        for item in locked
    )


@dataclass(frozen=True, slots=True)
class CounterFactEndpointMetrics:
    efficacy: CounterFactJointReading
    generalization: CounterFactJointReading
    locality: CounterFactJointReading

    def to_dict(self) -> dict[str, Any]:
        return {
            "efficacy": self.efficacy.to_dict(),
            "generalization": self.generalization.to_dict(),
            "locality": self.locality.to_dict(),
            "controller_access": False,
        }


def endpoint_specs(
    requests: Sequence[ControllerRequest],
    private_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[
    tuple[CounterFactPairSpec, ...],
    tuple[CounterFactPairSpec, ...],
    tuple[CounterFactPairSpec, ...],
]:
    efficacy = efficacy_specs(requests)
    generalization: list[CounterFactPairSpec] = []
    locality: list[CounterFactPairSpec] = []
    for request in requests:
        try:
            payload = private_payloads[request.case_id]
            paraphrases = tuple(payload["paraphrase_prompts"])
            neighborhoods = tuple(payload["neighborhood_prompts"])
            target_true = str(payload["target_true"])
        except (KeyError, TypeError) as exc:
            raise MethodContractError("CounterFact private endpoint payload differs") from exc
        if (
            not paraphrases
            or not neighborhoods
            or target_true != request.target_old
        ):
            raise MethodContractError("CounterFact endpoint panels are empty")
        generalization.extend(
            CounterFactPairSpec(
                case_id=request.case_id,
                role="generalization",
                prefix=str(prefix),
                target_new=request.target_new,
                target_true=request.target_old,
                new_is_success=True,
            )
            for prefix in paraphrases
        )
        locality.extend(
            CounterFactPairSpec(
                case_id=request.case_id,
                role="locality",
                prefix=str(prefix),
                target_new=request.target_new,
                target_true=request.target_old,
                new_is_success=False,
            )
            for prefix in neighborhoods
        )
    return efficacy, tuple(generalization), tuple(locality)


def score_counterfact_endpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[ControllerRequest],
    private_payloads: Mapping[str, Mapping[str, Any]],
) -> CounterFactEndpointMetrics:
    """Score the three official CounterFact primary panels after action freeze."""

    efficacy, generalization, locality = endpoint_specs(
        requests, private_payloads
    )
    return CounterFactEndpointMetrics(
        efficacy=score_counterfact_nll_pairs(model, tokenizer, efficacy),
        generalization=score_counterfact_nll_pairs(
            model, tokenizer, generalization
        ),
        locality=score_counterfact_nll_pairs(model, tokenizer, locality),
    )


def native_floor_verdict(
    native: CounterFactEndpointMetrics,
    ours: CounterFactEndpointMetrics,
) -> dict[str, Any]:
    metrics = {}
    overall = True
    for name in ("efficacy", "generalization", "locality"):
        baseline = getattr(native, name)
        candidate = getattr(ours, name)
        if (
            candidate.denominator != baseline.denominator
            or tuple((row.case_id, row.role) for row in candidate.rows)
            != tuple((row.case_id, row.role) for row in baseline.rows)
        ):
            raise MethodContractError("Native-floor metric denominator differs")
        passed = candidate.success_count >= baseline.success_count
        overall = overall and passed
        losses = [
            index
            for index, (left, right) in enumerate(
                zip(baseline.success_bits, candidate.success_bits, strict=True)
            )
            if left and not right
        ]
        wins = [
            index
            for index, (left, right) in enumerate(
                zip(baseline.success_bits, candidate.success_bits, strict=True)
            )
            if not left and right
        ]
        paired = [
            {
                "position": index,
                "case_id": baseline.rows[index].case_id,
                "native_success": bool(left),
                "ours_success": bool(right),
            }
            for index, (left, right) in enumerate(
                zip(baseline.success_bits, candidate.success_bits, strict=True)
            )
        ]
        metrics[name] = {
            "native_correct_count": baseline.success_count,
            "ours_correct_count": candidate.success_count,
            "denominator": baseline.denominator,
            "delta_count": candidate.success_count - baseline.success_count,
            "sealed_noninferiority_margin_count": 0,
            "pass": passed,
            "paired_loss_indices": losses,
            "paired_win_indices": wins,
            "paired_success_vector": paired,
            "mechanistic_request_loss_review_required": bool(losses),
        }
    return {
        "pass": overall,
        "metrics": metrics,
        "secondary_diagnostics_cannot_offset_primary_loss": True,
    }


def zsre_correct_position_counts(
    predicted_token_ids: Sequence[int], expected_token_ids: Sequence[int]
) -> dict[str, Any]:
    """Future zsRE adapter's official integer numerator/denominator primitive."""

    predicted = tuple(predicted_token_ids)
    expected = tuple(expected_token_ids)
    if (
        not expected
        or len(predicted) != len(expected)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in predicted)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in expected)
    ):
        raise MethodContractError("zsRE correct-position panel differs")
    bits = tuple(left == right for left, right in zip(predicted, expected, strict=True))
    numerator = sum(bits)
    return {
        "correct_bits": [int(value) for value in bits],
        "correct_position_count": numerator,
        "denominator": len(bits),
        "official_aggregate": numerator / len(bits),
        "source_sha256": ZSRE_EVALUATOR_SOURCE_SHA256,
    }


@dataclass(frozen=True, slots=True)
class HardBatchStep:
    position: int
    correction_cycle: int
    interval_in_cycle: int
    k_resolution: int
    k_total: int
    nominal_t: float
    nominal_eta: float
    lambda_value: float
    accepted_beta: float
    t_acc: float
    source_state_id: str
    terminal_state_id: str
    direction_ids: tuple[str, ...]
    raw_coefficients: tuple[float, ...]
    corrector_coefficients: tuple[float, ...]
    applied_coefficients: tuple[float, ...]
    full_progress: float
    qp_equality_residual: float
    qp_trust_dual: float
    qp_coefficient_norm: float
    raw_radius: float
    direct_z_residual_norms: tuple[float, ...]
    direct_z_residual_fractions: tuple[float, ...]
    batch_residual_norm: float
    benchmark: CounterFactJointReading
    surrogate_event: EventReading
    request_surrogate_events: tuple[EventReading, ...]
    joint_geometry: Mapping[str, Any]
    exact_first_hit_here: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "correction_cycle": self.correction_cycle,
            "interval_in_cycle": self.interval_in_cycle,
            "K_resolution": self.k_resolution,
            "K_total": self.k_total,
            "nominal_T": self.nominal_t,
            "nominal_eta": self.nominal_eta,
            "lambda": self.lambda_value,
            "accepted_beta": self.accepted_beta,
            "T_acc": self.t_acc,
            "source_state_id": self.source_state_id,
            "terminal_state_id": self.terminal_state_id,
            "direction_ids": list(self.direction_ids),
            "raw_coefficients": list(self.raw_coefficients),
            "corrector_coefficients": list(self.corrector_coefficients),
            "applied_coefficients": list(self.applied_coefficients),
            "full_progress": self.full_progress,
            "qp_equality_residual": self.qp_equality_residual,
            "qp_trust_dual": self.qp_trust_dual,
            "qp_coefficient_norm": self.qp_coefficient_norm,
            "raw_radius": self.raw_radius,
            "direct_z_residual_norms": list(self.direct_z_residual_norms),
            "direct_z_residual_fractions": list(
                self.direct_z_residual_fractions
            ),
            "batch_residual_norm": self.batch_residual_norm,
            "benchmark": self.benchmark.to_dict(),
            "surrogate_event": event_reading_to_dict(self.surrogate_event),
            "request_surrogate_events": [
                event_reading_to_dict(value)
                for value in self.request_surrogate_events
            ],
            "joint_geometry": dict(self.joint_geometry),
            "exact_first_hit_here": self.exact_first_hit_here,
        }


@dataclass(frozen=True, slots=True)
class HardBatchArmResult:
    arm: HardBatchArm
    status: str
    entry_state_id: str
    rollout_terminal_state_id: str
    selected_terminal_state_id: str
    entry_benchmark: CounterFactJointReading
    selected_benchmark: CounterFactJointReading
    rollout_terminal_benchmark: CounterFactJointReading
    selected_request_surrogate_events: tuple[EventReading, ...]
    rollout_request_surrogate_events: tuple[EventReading, ...]
    steps: tuple[HardBatchStep, ...]
    first_exact_hit_step: int | None
    selected_earliest_exact_hit: bool
    field_build_count: int
    backward_count_expected: int
    terminal_net_energy: Mapping[int, float]
    entry_direct_z_residual_norms: tuple[float, ...]
    selected_direct_z_residual_norms: tuple[float, ...]
    rollout_direct_z_residual_norms: tuple[float, ...]
    stagnated_cycles: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "status": self.status,
            "entry_state_id": self.entry_state_id,
            "rollout_terminal_state_id": self.rollout_terminal_state_id,
            "selected_terminal_state_id": self.selected_terminal_state_id,
            "entry_benchmark": self.entry_benchmark.to_dict(),
            "selected_benchmark": self.selected_benchmark.to_dict(),
            "rollout_terminal_benchmark": self.rollout_terminal_benchmark.to_dict(),
            "selected_request_surrogate_events": [
                event_reading_to_dict(value)
                for value in self.selected_request_surrogate_events
            ],
            "rollout_request_surrogate_events": [
                event_reading_to_dict(value)
                for value in self.rollout_request_surrogate_events
            ],
            "steps": [step.to_dict() for step in self.steps],
            "first_exact_hit_step": self.first_exact_hit_step,
            "selected_earliest_exact_hit": self.selected_earliest_exact_hit,
            "field_build_count": self.field_build_count,
            "backward_count_expected": self.backward_count_expected,
            "terminal_net_energy": {
                str(layer): value for layer, value in self.terminal_net_energy.items()
            },
            "entry_direct_z_residual_norms": list(
                self.entry_direct_z_residual_norms
            ),
            "selected_direct_z_residual_norms": list(
                self.selected_direct_z_residual_norms
            ),
            "rollout_direct_z_residual_norms": list(
                self.rollout_direct_z_residual_norms
            ),
            "stagnated_cycles": list(self.stagnated_cycles),
            "online_early_stop": False,
            "surrogate_event_decision_role": "diagnostic-and-allocation-slope-only",
        }


def _arm_budget(arm: HardBatchArm) -> tuple[int, int, float]:
    if arm is HardBatchArm.REFRESH_K10_T1:
        return 10, 1, 1.0
    if arm is HardBatchArm.REFRESH_K20_T1_RESOLUTION:
        return 20, 1, 1.0
    if arm is HardBatchArm.REFRESH_2XK10_T2_CORRECTION:
        return 10, 2, 2.0
    raise MethodContractError("native arm has no refresh budget")


def _measure_benchmark(
    benchmark: Callable[[], CounterFactJointReading],
    instrumentation: EditInstrumentation,
) -> CounterFactJointReading:
    with (
        instrumentation.component("benchmark"),
        instrumentation.model_forward_scope("benchmark"),
    ):
        reading = benchmark()
    if not instrumentation.tracks_model_forwards:
        instrumentation.record_model_forward("benchmark")
    instrumentation.increment("N_benchmark_tokens", reading.input_token_count)
    return reading


def _measure_residuals(
    backend: Any,
    frozen_target: Any,
    instrumentation: EditInstrumentation,
) -> tuple[float, ...]:
    with (
        instrumentation.component("residual_diagnostic"),
        instrumentation.model_forward_scope("residual"),
    ):
        values = tuple(backend.direct_z_residual_norms(frozen_target))
    if not instrumentation.tracks_model_forwards:
        instrumentation.record_model_forward("residual")
    return values


def _fractions(
    values: Sequence[float], entry: Sequence[float]
) -> tuple[float, ...]:
    if len(values) != JOINT_BATCH_SIZE or len(entry) != JOINT_BATCH_SIZE:
        raise MethodContractError("direct-z residual panel differs")
    return tuple(
        0.0 if baseline == 0.0 and value == 0.0 else value / baseline
        for value, baseline in zip(values, entry, strict=True)
    )


def run_hard_batch_arm(
    arm: HardBatchArm,
    *,
    backend: Any,
    frozen_target: Any,
    denominators: Mapping[int, float],
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
    benchmark: Callable[[], CounterFactJointReading],
    stagnation_epsilon: float,
) -> HardBatchArmResult:
    """Run one atomic genuine-B10 arm and preserve its earliest exact hit."""

    if arm not in HARD_BATCH_ARM_ORDER:
        raise MethodContractError("hard-batch arm differs")
    if (
        not math.isfinite(stagnation_epsilon)
        or stagnation_epsilon < 0.0
        or len(backend.requests) != JOINT_BATCH_SIZE
    ):
        raise MethodContractError("hard-batch run contract differs")
    omega = {int(layer): 0.0 for layer in backend.layers}
    instrumentation.start_controller()
    entry = None
    try:
        with instrumentation.component("entry_checkpoint"):
            entry = backend.checkpoint()
        backend.prepare_event_target(frozen_target)
        before = _measure_event(
            backend, backend.request, instrumentation
        )
        entry_residuals = _measure_residuals(
            backend, frozen_target, instrumentation
        )
        entry_benchmark = _measure_benchmark(benchmark, instrumentation)
        steps: list[HardBatchStep] = []
        first_exact_step: int | None = 0 if entry_benchmark.exact_hit else None
        first_exact_checkpoint = entry if entry_benchmark.exact_hit else None
        first_exact_benchmark: CounterFactJointReading | None = (
            entry_benchmark if entry_benchmark.exact_hit else None
        )
        first_exact_residuals: tuple[float, ...] | None = (
            entry_residuals if entry_benchmark.exact_hit else None
        )
        first_exact_request_events: tuple[EventReading, ...] | None = (
            tuple(backend.last_request_event_readings)
            if entry_benchmark.exact_hit
            else None
        )
        stagnated: list[int] = []

        if arm is HardBatchArm.NATIVE_MEMIT_B10:
            with instrumentation.component("native_proposal"):
                batch = backend.build_native_terminal(frozen_target)
            instrumentation.increment("N_proposal_build")
            instrumentation.increment("N_native_sweep")
            geometry = backend.assert_genuine_joint_batch(batch)
            coefficients = tuple(1.0 for _ in batch.proposals)
            with instrumentation.component("commit_write"):
                applied = tuple(backend.commit(batch, coefficients))
            instrumentation.increment("N_write")
            instrumentation.increment("K_acc")
            after = _measure_event(backend, backend.request, instrumentation)
            residuals = _measure_residuals(
                backend, frozen_target, instrumentation
            )
            official = _measure_benchmark(benchmark, instrumentation)
            if official.exact_hit and first_exact_step is None:
                first_exact_step = 1
                first_exact_checkpoint = backend.capture_current_checkpoint()
                first_exact_benchmark = official
                first_exact_residuals = residuals
                first_exact_request_events = tuple(
                    backend.last_request_event_readings
                )
            steps.append(
                HardBatchStep(
                    position=0,
                    correction_cycle=0,
                    interval_in_cycle=0,
                    k_resolution=1,
                    k_total=1,
                    nominal_t=1.0,
                    nominal_eta=1.0,
                    lambda_value=1.0,
                    accepted_beta=1.0,
                    t_acc=1.0,
                    source_state_id=entry.state_id,
                    terminal_state_id=backend.current_state_id(),
                    direction_ids=batch.direction_ids,
                    raw_coefficients=coefficients,
                    corrector_coefficients=coefficients,
                    applied_coefficients=applied,
                    full_progress=0.0,
                    qp_equality_residual=0.0,
                    qp_trust_dual=0.0,
                    qp_coefficient_norm=math.sqrt(len(coefficients)),
                    raw_radius=math.sqrt(len(coefficients)),
                    direct_z_residual_norms=residuals,
                    direct_z_residual_fractions=_fractions(
                        residuals, entry_residuals
                    ),
                    batch_residual_norm=math.sqrt(
                        math.fsum(value * value for value in residuals)
                    ),
                    benchmark=official,
                    surrogate_event=after,
                    request_surrogate_events=tuple(
                        backend.last_request_event_readings
                    ),
                    joint_geometry=geometry,
                    exact_first_hit_here=official.exact_hit,
                )
            )
            field_builds = 0
        else:
            k_resolution, cycles, nominal_t = _arm_budget(arm)
            lambdas = fixed_horizon_lambdas(k_resolution)
            field_builds = 0
            prior_cycle_residual = math.sqrt(
                math.fsum(value * value for value in entry_residuals)
            )
            prior_cycle_success = entry_benchmark.success_count
            for cycle in range(cycles):
                for interval, lambda_value in enumerate(lambdas):
                    batch, raw = _build_field(
                        backend, frozen_target, instrumentation
                    )
                    field_builds += 1
                    geometry = backend.assert_genuine_joint_batch(batch)
                    solution = _transport_solution(
                        batch,
                        raw,
                        omega=omega,
                        denominators=denominators,
                        config=config,
                        instrumentation=instrumentation,
                    )
                    applied_coefficients = tuple(
                        lambda_value * value
                        for value in solution.qp.coefficients
                    )
                    source_state = backend.current_state_id()
                    after = _commit_trial(
                        backend,
                        backend.request,
                        batch,
                        applied_coefficients,
                        config=config,
                        instrumentation=instrumentation,
                    )
                    residuals = _measure_residuals(
                        backend, frozen_target, instrumentation
                    )
                    official = _measure_benchmark(benchmark, instrumentation)
                    position = cycle * k_resolution + interval
                    exact_here = official.exact_hit and first_exact_step is None
                    if exact_here:
                        first_exact_step = position + 1
                        first_exact_checkpoint = backend.capture_current_checkpoint()
                        first_exact_benchmark = official
                        first_exact_residuals = residuals
                        first_exact_request_events = tuple(
                            backend.last_request_event_readings
                        )
                    steps.append(
                        HardBatchStep(
                            position=position,
                            correction_cycle=cycle,
                            interval_in_cycle=interval,
                            k_resolution=k_resolution,
                            k_total=k_resolution * cycles,
                            nominal_t=nominal_t,
                            nominal_eta=1.0 / k_resolution,
                            lambda_value=lambda_value,
                            accepted_beta=lambda_value,
                            t_acc=cycle + (interval + 1) / k_resolution,
                            source_state_id=source_state,
                            terminal_state_id=backend.current_state_id(),
                            direction_ids=batch.direction_ids,
                            raw_coefficients=tuple(raw),
                            corrector_coefficients=solution.qp.coefficients,
                            applied_coefficients=applied_coefficients,
                            full_progress=solution.full_progress,
                            qp_equality_residual=solution.qp.equality_residual,
                            qp_trust_dual=solution.qp.trust_dual,
                            qp_coefficient_norm=solution.qp.coefficient_norm,
                            raw_radius=math.sqrt(
                                math.fsum(value * value for value in raw)
                            ),
                            direct_z_residual_norms=residuals,
                            direct_z_residual_fractions=_fractions(
                                residuals, entry_residuals
                            ),
                            batch_residual_norm=math.sqrt(
                                math.fsum(value * value for value in residuals)
                            ),
                            benchmark=official,
                            surrogate_event=after,
                            request_surrogate_events=tuple(
                                backend.last_request_event_readings
                            ),
                            joint_geometry=geometry,
                            exact_first_hit_here=exact_here,
                        )
                    )
                    before = after
                cycle_residual = steps[-1].batch_residual_norm
                cycle_success = steps[-1].benchmark.success_count
                if (
                    prior_cycle_residual - cycle_residual
                    <= stagnation_epsilon
                    and cycle_success <= prior_cycle_success
                ):
                    stagnated.append(cycle)
                    if cycle + 1 < cycles:
                        break
                prior_cycle_residual = cycle_residual
                prior_cycle_success = cycle_success

        expected_fields = {
            HardBatchArm.NATIVE_MEMIT_B10: 0,
            HardBatchArm.REFRESH_K10_T1: 10,
            HardBatchArm.REFRESH_K20_T1_RESOLUTION: 20,
            HardBatchArm.REFRESH_2XK10_T2_CORRECTION: 20,
        }[arm]
        if (
            not steps
            or field_builds > expected_fields
            or (
                field_builds != expected_fields
                and not (
                    arm is HardBatchArm.REFRESH_2XK10_T2_CORRECTION
                    and stagnated == [0]
                    and field_builds == 10
                )
            )
        ):
            raise MethodContractError("hard-batch fixed budget differs")
        rollout_state_id = backend.current_state_id()
        rollout_benchmark = steps[-1].benchmark
        rollout_residuals = steps[-1].direct_z_residual_norms
        rollout_request_events = steps[-1].request_surrogate_events
        if first_exact_checkpoint is not None:
            backend.restore(first_exact_checkpoint)
            backend.assert_checkpoint(first_exact_checkpoint)
            assert first_exact_benchmark is not None
            assert first_exact_residuals is not None
            assert first_exact_request_events is not None
            selected_benchmark = first_exact_benchmark
            selected_residuals = first_exact_residuals
            selected_request_events = first_exact_request_events
        else:
            selected_benchmark = rollout_benchmark
            selected_residuals = rollout_residuals
            selected_request_events = rollout_request_events
        with instrumentation.component("terminal_geometry"):
            energy = backend.terminal_net_energy(entry)
        status = (
            "EXACT_BATCH_HIT"
            if first_exact_step is not None
            else ("STAGNATED" if stagnated else "NO_EXACT_BATCH_HIT")
        )
        return HardBatchArmResult(
            arm=arm,
            status=status,
            entry_state_id=entry.state_id,
            rollout_terminal_state_id=rollout_state_id,
            selected_terminal_state_id=backend.current_state_id(),
            entry_benchmark=entry_benchmark,
            selected_benchmark=selected_benchmark,
            rollout_terminal_benchmark=rollout_benchmark,
            selected_request_surrogate_events=selected_request_events,
            rollout_request_surrogate_events=rollout_request_events,
            steps=tuple(steps),
            first_exact_hit_step=first_exact_step,
            selected_earliest_exact_hit=first_exact_step is not None,
            field_build_count=field_builds,
            backward_count_expected=field_builds,
            terminal_net_energy=dict(energy),
            entry_direct_z_residual_norms=entry_residuals,
            selected_direct_z_residual_norms=selected_residuals,
            rollout_direct_z_residual_norms=rollout_residuals,
            stagnated_cycles=tuple(stagnated),
        )
    except BaseException:
        if entry is not None:
            backend.restore(entry)
            backend.assert_checkpoint(entry)
        raise
    finally:
        instrumentation.stop_controller()


def endpoint_parameter_hashes(
    model: torch.nn.Module, weight_names: Sequence[str]
) -> Mapping[str, str]:
    from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256

    names = tuple(weight_names)
    if not names or len(names) != len(set(names)):
        raise MethodContractError("endpoint parameter hash set differs")
    return {
        name: tensor_sha256(resolve_parameter(model, name)) for name in names
    }


__all__ = [
    "COUNTERFACT_AGGREGATOR_SOURCE_SHA256",
    "COUNTERFACT_EVALUATOR_SOURCE_SHA256",
    "HARD_BATCH_ARM_ORDER",
    "ZSRE_EVALUATOR_SOURCE_SHA256",
    "CounterFactEndpointMetrics",
    "CounterFactJointReading",
    "CounterFactPairReading",
    "CounterFactPairSpec",
    "HardBatchArm",
    "HardBatchArmResult",
    "HardBatchStep",
    "efficacy_specs",
    "endpoint_parameter_hashes",
    "endpoint_specs",
    "fixed_horizon_lambdas",
    "frozen_cumulative_fractions",
    "naive_repeated_residual_fraction",
    "native_floor_verdict",
    "run_hard_batch_arm",
    "score_counterfact_endpoint",
    "score_counterfact_nll_pairs",
    "zsre_correct_position_counts",
]
