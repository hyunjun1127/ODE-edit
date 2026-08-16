"""Locked model-independent contracts for the Session 04 P1 matched pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .barriers import (
    BarrierSummary,
    HistoryItem,
    RequestKey,
    remove_obsolete_history,
    summarize_pretrained_anchors,
    summarize_small_buffer,
)
from .contracts import (
    ARM_ORDER,
    MODEL_ALIASES,
    Arm,
    ODEAllocContractError,
    SolverBudget,
    assert_common_model_policy,
    canonical_hash,
    finite,
    reject_global_strength_fields,
)


P1_INSTRUCTION_ID = "ODEEDIT-S04-ODE-ALLOC-P1-MATCHED-PAIR-V1"
SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
EXPECTED_BASE = "16d35000c49261d1fc5a5d0d2bc7ef90cd971b95"
SEAL_ROOT = "1b45cd567d5ef1a8c52bf0a5a6f8b715270f620cbbdc9949bbf7094f87ef8cab"
P1_RUN_TOKEN = "p1v1"
P1_CASE_ORDER = (6578, 5613, 13856, 16451)
P1_REQUEST_HASHES = (
    "4927599094dd1a15f0b7b4e80220a70f91ae0297e0bef9dd66dd1fa6c041fdb2",
    "ad4f5e1636203e5b0d03bf616ba2985eda55ac59a08e1fed8b5281f217517c26",
    "b5ceaf20aca52ccd709002fc834c8902caea6bf6345b15d60afcb32588ed1216",
    "efb93c7b637cce44d87987557f04c9ab058d7e2ecfae434f9bc0fd2ef4503361",
)
SEED = 41
INNER_MICROBATCH_SIZE = 2


@dataclass(frozen=True, slots=True)
class P1Policy:
    basis_energy_epsilon: float
    max_abs_centered_q: float
    rho: float
    denominator_epsilon: float
    historical_scale: float
    pretrained_scale: float
    smooth_temperature: float
    solver_budget: SolverBudget
    cbf_kappa: float
    slack_penalty_weight: float
    qp_residual_tolerance: float
    policy_id: str

    @classmethod
    def from_lock(cls, lock: Mapping[str, Any]) -> "P1Policy":
        reject_global_strength_fields(lock)
        if tuple(lock.get("model_aliases", ())) != MODEL_ALIASES:
            raise ODEAllocContractError("P1 lock model aliases differ")
        gauge = lock.get("allocation_gauge")
        rewrite = lock.get("rewrite_gate")
        barriers = lock.get("barriers")
        solver = lock.get("solver")
        transaction = lock.get("transaction")
        if not all(isinstance(value, Mapping) for value in (gauge, rewrite, barriers, solver, transaction)):
            raise ODEAllocContractError("P1 numerical lock structure differs")
        assert gauge is not None and rewrite is not None and barriers is not None
        assert solver is not None and transaction is not None
        exact = (
            gauge.get("basis_energy_epsilon") == 1.0e-24
            and gauge.get("max_abs_centered_q") == math.log(4.0)
            and gauge.get("q_chart") == "arithmetic-zero-mean"
            and gauge.get("epsilon_prior") is False
            and rewrite.get("rho") == 0.5
            and rewrite.get("denominator_epsilon") == 1.0e-8
            and barriers.get("historical_fixed_damage_scale_nats") == 1.0e-3
            and barriers.get("pretrained_fixed_damage_scale_nats") == 1.0e-3
            and barriers.get("smooth_max_temperature_nats") == 1.0e-2
            and barriers.get("aggregate") == "equal-weight-smooth-max-plus-mean"
            and barriers.get("p1_history_size_max") == 3
            and barriers.get("tail_risk_estimator") == "deferred"
            and solver.get("fixed_k") == 8
            and tuple(solver.get("backtracking_factors", ())) == (1, 0.5, 0.25)
            and solver.get("cbf_kappa") == 1
            and solver.get("cbf_slack_penalty_weight") == 100
            and tuple(solver.get("aggregated_constraint_labels", ())) == ("E", "H", "P")
            and solver.get("per_item_constraints") is False
            and solver.get("production_max_constraints") == 3
            and solver.get("active_subset_count_at_max") == 8
            and solver.get("early_stop") is False
            and transaction.get("gradient_path")
            == "fp32-factor-overlay-not-verdict-eligible"
            and transaction.get("verdict_path")
            == "one-layer-at-a-time-quantized-bf16-full-linear"
        )
        if not exact:
            raise ODEAllocContractError("P1 scientific lock differs")
        budget = SolverBudget(
            fixed_k=int(solver["fixed_k"]),
            max_trials_per_step=int(solver["max_trials_per_step"]),
            backtracking_factors=tuple(solver["backtracking_factors"]),
            context_count=6,
        )
        common = {
            "allocation_gauge": gauge,
            "rewrite_gate": rewrite,
            "barriers": barriers,
            "solver": solver,
            "transaction": transaction,
            "inner_microbatch_size": INNER_MICROBATCH_SIZE,
        }
        policy_id = assert_common_model_policy({alias: common for alias in MODEL_ALIASES})
        return cls(
            basis_energy_epsilon=float(gauge["basis_energy_epsilon"]),
            max_abs_centered_q=float(gauge["max_abs_centered_q"]),
            rho=float(rewrite["rho"]),
            denominator_epsilon=float(rewrite["denominator_epsilon"]),
            historical_scale=float(barriers["historical_fixed_damage_scale_nats"]),
            pretrained_scale=float(barriers["pretrained_fixed_damage_scale_nats"]),
            smooth_temperature=float(barriers["smooth_max_temperature_nats"]),
            solver_budget=budget,
            cbf_kappa=float(solver["cbf_kappa"]),
            slack_penalty_weight=float(solver["cbf_slack_penalty_weight"]),
            qp_residual_tolerance=float(solver["constraint_residual_tolerance"]),
            policy_id=policy_id,
        )


@dataclass(frozen=True, slots=True)
class HistoryRequest:
    case_id: int
    request_hash: str
    request_key: RequestKey
    request: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.case_id, bool)
            or not isinstance(self.case_id, int)
            or self.case_id < 0
            or len(self.request_hash) != 64
        ):
            raise ODEAllocContractError("P1 history identity differs")


def request_key(request: Mapping[str, Any]) -> RequestKey:
    return RequestKey(
        subject=str(request["subject"]),
        relation=str(request["relation_id"]),
        template=str(request["prompt"]),
    )


def remove_obsolete_requests(
    history: Sequence[HistoryRequest],
    incoming: Mapping[str, Any],
) -> tuple[tuple[HistoryRequest, ...], tuple[HistoryRequest, ...]]:
    wrapped = tuple(
        HistoryItem(item.request_key, 0.0, str(index))
        for index, item in enumerate(history)
    )
    kept_wrapped, obsolete_wrapped = remove_obsolete_history(
        wrapped, request_key(incoming)
    )
    kept_indices = {int(item.item_id) for item in kept_wrapped}
    obsolete_indices = {int(item.item_id) for item in obsolete_wrapped}
    return (
        tuple(item for index, item in enumerate(history) if index in kept_indices),
        tuple(item for index, item in enumerate(history) if index in obsolete_indices),
    )


@dataclass(frozen=True, slots=True)
class DamageAggregate:
    raw: tuple[float, ...]
    mean: float
    smooth_max: float
    aggregate: float
    fixed_scale: float
    barrier_value: float

    @property
    def feasible(self) -> bool:
        return self.barrier_value >= 0.0

    @property
    def normalized(self) -> float:
        return self.aggregate / self.fixed_scale

    def record(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "mean": self.mean,
            "max": max(self.raw) if self.raw else 0.0,
            "smooth_max": self.smooth_max,
            "aggregate": self.aggregate,
            "fixed_scale": self.fixed_scale,
            "barrier_value": self.barrier_value,
            "feasible": self.feasible,
        }


def _from_summary(summary: BarrierSummary) -> DamageAggregate:
    aggregate = 0.5 * (summary.mean + summary.smooth_max)
    return DamageAggregate(
        raw=summary.raw,
        mean=summary.mean,
        smooth_max=summary.smooth_max,
        aggregate=aggregate,
        fixed_scale=summary.fixed_scale,
        barrier_value=summary.fixed_scale - aggregate,
    )


def historical_damage_aggregate(
    values: Sequence[float], *, policy: P1Policy
) -> DamageAggregate:
    return _from_summary(
        summarize_small_buffer(
            values,
            fixed_scale=policy.historical_scale,
            smooth_temperature=policy.smooth_temperature,
        )
    )


def pretrained_damage_aggregate(
    values: Sequence[float], *, policy: P1Policy
) -> DamageAggregate:
    return _from_summary(
        summarize_pretrained_anchors(
            values,
            fixed_scale=policy.pretrained_scale,
            smooth_temperature=policy.smooth_temperature,
            expected_count=16,
        )
    )


@dataclass(frozen=True, slots=True)
class TensorDamageAggregate:
    raw: torch.Tensor
    mean: torch.Tensor
    smooth_max: torch.Tensor
    aggregate: torch.Tensor
    barrier_value: torch.Tensor


def differentiable_damage_aggregate(
    candidate: Sequence[torch.Tensor],
    teacher: Sequence[float],
    *,
    fixed_scale: float,
    smooth_temperature: float,
    zero_reference: torch.Tensor,
) -> TensorDamageAggregate:
    if len(candidate) != len(teacher):
        raise ODEAllocContractError("candidate/teacher damage panels differ")
    scale = finite("damage scale", fixed_scale)
    temperature = finite("smooth temperature", smooth_temperature)
    if scale <= 0.0 or temperature <= 0.0:
        raise ODEAllocContractError("damage scale/temperature must be positive")
    if not candidate:
        zero = zero_reference.sum() * 0.0
        raw = zero.reshape(1)[:0]
        return TensorDamageAggregate(raw, zero, zero, zero, zero + scale)
    tensors = tuple(
        torch.relu(
            torch.as_tensor(target, dtype=value.dtype, device=value.device) - value
        )
        for value, target in zip(candidate, teacher, strict=True)
    )
    raw = torch.stack(tensors)
    mean = raw.mean()
    smooth = temperature * (
        torch.logsumexp(raw / temperature, dim=0) - math.log(raw.numel())
    )
    aggregate = 0.5 * (mean + smooth)
    return TensorDamageAggregate(raw, mean, smooth, aggregate, scale - aggregate)


@dataclass(frozen=True, slots=True)
class EndpointFreezeToken:
    arm: Arm
    source_head: str
    parameter_sha256: str
    action_sha256: str
    edit_count: int
    token_id: str

    @classmethod
    def create(
        cls,
        *,
        arm: Arm,
        source_head: str,
        parameter_sha256: str,
        action_sha256: str,
        edit_count: int,
    ) -> "EndpointFreezeToken":
        if (
            arm not in ARM_ORDER
            or len(source_head) != 40
            or len(parameter_sha256) != 64
            or len(action_sha256) != 64
            or edit_count != 4
        ):
            raise ODEAllocContractError("endpoint freeze identity differs")
        payload = {
            "schema": "ode-alloc-s04-p1-endpoint-freeze/v1",
            "arm": arm.value,
            "source_head": source_head,
            "parameter_sha256": parameter_sha256,
            "action_sha256": action_sha256,
            "edit_count": edit_count,
        }
        return cls(
            arm=arm,
            source_head=source_head,
            parameter_sha256=parameter_sha256,
            action_sha256=action_sha256,
            edit_count=edit_count,
            token_id=canonical_hash(payload),
        )

    def validate(self) -> None:
        observed = EndpointFreezeToken.create(
            arm=self.arm,
            source_head=self.source_head,
            parameter_sha256=self.parameter_sha256,
            action_sha256=self.action_sha256,
            edit_count=self.edit_count,
        )
        if observed != self:
            raise ODEAllocContractError("endpoint freeze token identity differs")


def expected_p1_result_name(alias: str, numerical_sha256: str, run_token: str) -> str:
    if (
        alias not in MODEL_ALIASES
        or len(numerical_sha256) != 64
        or run_token != P1_RUN_TOKEN
    ):
        raise ODEAllocContractError("P1 result identity differs")
    return f"s04-p1-matched-{alias}-{numerical_sha256[:8]}"


def assert_p1_seal(seal: Mapping[str, Any]) -> None:
    order = tuple(item.get("case_id") for item in seal.get("p1_order", ()))
    hashes = tuple(item.get("request_hash") for item in seal.get("p1_order", ()))
    anchors = seal.get("pretrained_anchor")
    if (
        seal.get("root_digest") != SEAL_ROOT
        or order != P1_CASE_ORDER
        or hashes != P1_REQUEST_HASHES
        or not isinstance(anchors, list)
        or len(anchors) != 16
        or len({item.get("case_id") for item in anchors}) != 16
        or len({item.get("request_hash") for item in anchors}) != 16
        or set(hashes).intersection(item.get("request_hash") for item in anchors)
    ):
        raise ODEAllocContractError("approved P1 seal differs")


def arm_order_identity() -> str:
    return canonical_hash(tuple(arm.value for arm in ARM_ORDER))
