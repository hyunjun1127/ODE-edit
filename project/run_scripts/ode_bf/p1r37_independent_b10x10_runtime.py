"""P1R37 independent B10x10 runtime with instantaneous no-carry freeze."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from .contracts import MODEL_ALIASES, ODEBFContractError
from .p1r36_independent_b10x10_runtime import (
    METHODS,
    IndependentB10x10RuntimeContract,
    run_p1r36_independent_b10x10,
)
from .p1r37_instantaneous_freeze import (
    FREEZE_POLICY,
    INSTRUCTION_ID,
    METHOD_ID,
    p1r37_target_step,
)


P1R37_RUNTIME_CONTRACT = IndependentB10x10RuntimeContract(
    instruction_id=INSTRUCTION_ID,
    method_id=METHOD_ID,
    schema_prefix="ode-edit-s05-p1r37-no-persistent-freeze-independent-b10",
    terminal_status="P1R37_NO_PERSISTENT_FREEZE_INDEPENDENT_B10X10_TERMINAL",
    freeze_policy=FREEZE_POLICY,
)


def expected_p1r37_independent_result_name(alias: str, method: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R37 independent alias differs")
    if method not in METHODS:
        raise ODEBFContractError("P1R37 independent method differs")
    token = method.lower().replace("-p1r35-", "-")
    return f"s05-p1r37-no-persistent-freeze-independent-b10x10-{alias}-{token}-v1"


def _target_step_policy_factory(
    alias: str, method: str, case_index: int
) -> Callable[..., Any]:
    return partial(
        p1r37_target_step,
        alias=alias,
        method=method,
        case_index=case_index,
    )


def run_p1r37_independent_b10x10(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs["runtime_contract"] = P1R37_RUNTIME_CONTRACT
    kwargs["target_step_policy_factory"] = _target_step_policy_factory
    return run_p1r36_independent_b10x10(*args, **kwargs)


__all__ = [
    "METHODS",
    "P1R37_RUNTIME_CONTRACT",
    "expected_p1r37_independent_result_name",
    "run_p1r37_independent_b10x10",
]
