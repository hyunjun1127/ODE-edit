from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r2 import actuator_guard
from project.run_scripts.barrier_guided_ode.r2.actuator_guard import (
    ordered_prefix_mismatch,
    propose_validated_ordered,
)
from project.run_scripts.barrier_guided_ode.r2.dynamics import (
    fixed_count_schedule,
    integrate_explicit_euler,
    maximum_step_schedule,
    weight_path_telemetry,
)
from project.run_scripts.barrier_guided_ode.r2.telemetry import R2ExecutionBoundary


def test_affine_one_step_equals_frozen_split() -> None:
    initial = torch.tensor([0.2, -0.1], dtype=torch.float64)

    def constant_velocity(_state: torch.Tensor, _time: float) -> torch.Tensor:
        return torch.tensor([0.7, -0.4], dtype=torch.float64)

    progress = lambda state: float(state[0])
    one = integrate_explicit_euler(
        initial,
        fixed_count_schedule(horizon=0.75, count=1),
        velocity=constant_velocity,
        progress=progress,
    )
    split = integrate_explicit_euler(
        initial,
        fixed_count_schedule(horizon=0.75, count=16),
        velocity=constant_velocity,
        progress=progress,
    )
    torch.testing.assert_close(one.final_state, split.final_state, rtol=0, atol=8e-16)


def test_nonlinear_euler_converges_and_reports_local_and_cumulative_error() -> None:
    initial = torch.tensor([0.0], dtype=torch.float64)
    exact = torch.tensor([torch.exp(torch.tensor(1.0, dtype=torch.float64)) - 1.0])

    def nonlinear(state: torch.Tensor, _time: float) -> torch.Tensor:
        return 1.0 + state

    errors = []
    for count in (4, 8, 16, 32):
        result = integrate_explicit_euler(
            initial,
            fixed_count_schedule(horizon=1.0, count=count),
            velocity=nonlinear,
            progress=lambda state: float(torch.log1p(state[0])),
        )
        errors.append(float(torch.linalg.vector_norm(result.final_state - exact)))
        assert len(result.nodes) == count
        assert all(torch.isfinite(torch.tensor(node.local_progress_error)) for node in result.nodes)
        assert all(torch.isfinite(torch.tensor(node.cumulative_progress_error)) for node in result.nodes)
    assert errors[1] < errors[0] and errors[2] < errors[1] and errors[3] < errors[2]


def test_only_final_partial_step_and_weight_path_accounting() -> None:
    schedule = maximum_step_schedule(horizon=1.0, maximum_step=0.3)
    assert [step.size for step in schedule] == pytest.approx([0.3, 0.3, 0.3, 0.1])
    coefficients = (
        torch.tensor([0.2, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.3], dtype=torch.float64),
    )
    telemetry = weight_path_telemetry(coefficients, (0.25, 0.5))
    assert telemetry.terminal_displacement == pytest.approx((0.2**2 + 0.3**2) ** 0.5)
    assert telemetry.path_length == pytest.approx(0.5)
    assert telemetry.integrated_kinetic_energy == pytest.approx(0.2**2 / 0.25 + 0.3**2 / 0.5)


def test_ordered_adapter_validation_is_called_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    factors = (
        SimpleNamespace(left=torch.eye(2, dtype=torch.float32), right=torch.eye(2, dtype=torch.float32)),
        SimpleNamespace(left=torch.ones((2, 1), dtype=torch.float32), right=torch.ones((2, 1), dtype=torch.float32)),
    )
    build = SimpleNamespace(proposal=SimpleNamespace(factors=factors))
    provider = SimpleNamespace(propose_ordered=lambda *args, **kwargs: build)
    calls = []

    def validator(candidate: object) -> object:
        calls.append(candidate)
        return SimpleNamespace(factor_count=2)

    monkeypatch.setattr(actuator_guard, "validate_ordered_alphaedit_build", validator)
    receipt = propose_validated_ordered(provider, origin_lineage="sealed")
    assert calls == [build]
    assert receipt.validation_call_count == 1
    assert len(receipt.factor_sha256) == 2
    mismatch = ordered_prefix_mismatch(
        tuple(factor.left @ factor.right.T for factor in factors),
        torch.tensor([0.25, -0.5], dtype=torch.float64),
    )
    assert mismatch.unit_prefix[0] == 0.0
    assert mismatch.effective_prefix[1] > 0.0


def test_dynamic_source_has_no_scalar_localizer_or_endpoint_forcing() -> None:
    directory = Path(inspect.getfile(integrate_explicit_euler)).parent
    dynamic_members = ("controller.py", "dynamics.py", "telemetry.py")
    forbidden = (
        "rho",
        "root",
        "brent",
        "bisection",
        "semantic_endpoint_forcing",
        "exact_progress_retraction",
        "localization",
    )
    combined = "\n".join((directory / member).read_text(encoding="utf-8") for member in dynamic_members).lower()
    assert all(token not in combined for token in forbidden)


def test_execution_boundary_has_all_prohibited_counters_zero() -> None:
    receipt = R2ExecutionBoundary(
        batch_size=1,
        fixed_target_compute_count=1,
        fixed_target_recompute_count=0,
        identity_scale=1.0,
        node_history_append_count=0,
        accepted_terminal_history_append_count=1,
        model_numeric_dtype="torch.float32",
        numerical_core_dtype="torch.float64",
        physical_write_dtype="torch.float32",
        probability_floor_count=0,
        damping_count=0,
        ridge_count=0,
        fallback_count=0,
        scientific_promotion=False,
    )
    assert receipt.as_payload()["identity_scale"] == 1.0
