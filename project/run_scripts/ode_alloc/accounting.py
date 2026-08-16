"""Common compute accounting for matched-call and cost-normalized analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import ODEAllocContractError, canonical_hash, finite


@dataclass(slots=True)
class ComputeLedger:
    model_forward_calls: int = 0
    processed_tokens: int = 0
    backward_calls: int = 0
    constraint_vjp_calls: int = 0
    constraint_jvp_calls: int = 0
    quantized_trial_calls: int = 0
    rejected_trials: int = 0
    qp_calls: int = 0
    qp_cpu_seconds: float = 0.0
    controller_wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    commit_count: int = 0

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self.__dataclass_fields__:
            raise ODEAllocContractError(f"unknown accounting field: {name}")
        current = getattr(self, name)
        if isinstance(current, float):
            raise ODEAllocContractError(f"use add_seconds for {name}")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ODEAllocContractError("counter increment must be non-negative integer")
        setattr(self, name, current + amount)

    def add_seconds(self, name: str, seconds: float) -> None:
        if name not in {
            "qp_cpu_seconds",
            "controller_wall_seconds",
            "gpu_seconds",
        }:
            raise ODEAllocContractError(f"unknown timing field: {name}")
        value = finite(name, seconds)
        if value < 0.0:
            raise ODEAllocContractError("timing cannot be negative")
        setattr(self, name, getattr(self, name) + value)

    def observe_peak_memory(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ODEAllocContractError("peak memory must be non-negative integer")
        self.peak_memory_bytes = max(self.peak_memory_bytes, value)

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "ode-alloc-compute-accounting/v1"
        payload["identity"] = canonical_hash(payload)
        return payload


def assert_matched_call_identity(
    generic: ComputeLedger,
    ode: ComputeLedger,
) -> None:
    """Compare calls that are locked equal; QP cost is separately attributed."""

    fields = (
        "model_forward_calls",
        "processed_tokens",
        "backward_calls",
        "constraint_vjp_calls",
        "constraint_jvp_calls",
        "quantized_trial_calls",
        "commit_count",
    )
    differences = {
        name: (getattr(generic, name), getattr(ode, name))
        for name in fields
        if getattr(generic, name) != getattr(ode, name)
    }
    if differences:
        raise ODEAllocContractError(f"Generic/ODE matched calls differ: {differences}")
