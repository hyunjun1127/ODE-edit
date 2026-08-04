"""Fail-closed contracts for Session 04 ODE-Alloc preparation.

This package contains no model loader, scheduler client, or EasyEdit writer.  It
only represents the model-independent policy approved for CPU preparation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "ode-edit-ode-alloc/v1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


class ODEAllocContractError(ValueError):
    """An input violates the common ODE-Alloc policy."""


class Arm(str, Enum):
    NATIVE = "native"
    GENERIC_ADAPTIVE = "generic-adaptive"
    ODE_ALLOC = "ode-alloc"


ARM_ORDER = (Arm.NATIVE, Arm.GENERIC_ADAPTIVE, Arm.ODE_ALLOC)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ODEAllocContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ODEAllocContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ODEAllocContractError(f"{name} must be finite")
    return result


def positive(name: str, value: Any) -> float:
    result = finite(name, value)
    if result <= 0.0:
        raise ODEAllocContractError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class SolverBudget:
    """The outcome-independent budget shared by Generic and ODE arms."""

    fixed_k: int
    max_trials_per_step: int
    backtracking_factors: tuple[float, ...]
    context_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("fixed_k", self.fixed_k),
            ("max_trials_per_step", self.max_trials_per_step),
            ("context_count", self.context_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ODEAllocContractError(f"{name} must be a positive integer")
        factors = tuple(positive("backtracking factor", item) for item in self.backtracking_factors)
        if len(factors) != self.max_trials_per_step:
            raise ODEAllocContractError("trial count and backtracking schedule differ")
        if factors[0] != 1.0 or any(value > 1.0 for value in factors):
            raise ODEAllocContractError("backtracking must start at one and never expand")
        if any(left <= right for left, right in zip(factors, factors[1:])):
            raise ODEAllocContractError("backtracking factors must be strictly decreasing")
        object.__setattr__(self, "backtracking_factors", factors)

    @property
    def euler_step(self) -> float:
        return 1.0 / self.fixed_k

    @property
    def quantized_trial_budget(self) -> int:
        return 1 + self.fixed_k * self.max_trials_per_step

    def identity(self) -> str:
        return canonical_hash(
            {
                "fixed_k": self.fixed_k,
                "max_trials_per_step": self.max_trials_per_step,
                "backtracking_factors": self.backtracking_factors,
                "context_count": self.context_count,
                "euler_grid": [index / self.fixed_k for index in range(self.fixed_k + 1)],
            }
        )


def assert_matched_budgets(
    generic: SolverBudget,
    ode: SolverBudget,
) -> None:
    if generic != ode or generic.identity() != ode.identity():
        raise ODEAllocContractError("Generic and ODE budgets are not identical")


def assert_common_model_policy(policy_by_alias: Mapping[str, Any]) -> str:
    """Require both aliases and byte-equivalent canonical policy values."""

    if tuple(sorted(policy_by_alias)) != tuple(sorted(MODEL_ALIASES)):
        raise ODEAllocContractError("both locked model aliases are required")
    identities = {canonical_hash(policy_by_alias[alias]) for alias in MODEL_ALIASES}
    if len(identities) != 1:
        raise ODEAllocContractError("model aliases received different controller policy")
    return identities.pop()


def reject_global_strength_fields(value: Mapping[str, Any]) -> None:
    forbidden = {
        "alpha",
        "global_alpha",
        "global_scale",
        "scalar_strength",
        "strength_multiplier",
    }
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            overlap = forbidden.intersection(str(key).lower() for key in item)
            if overlap:
                raise ODEAllocContractError(
                    f"global strength freedom is forbidden: {sorted(overlap)}"
                )
            stack.extend(item.values())
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            stack.extend(item)
