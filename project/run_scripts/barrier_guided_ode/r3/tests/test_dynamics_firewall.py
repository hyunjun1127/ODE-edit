from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r3.dynamics import (
    dense_block_path,
    fixed_count_schedule,
    integrate_euler,
)
from project.run_scripts.barrier_guided_ode.r3.telemetry import R3ExecutionBoundary


def test_affine_one_step_equals_split_euler() -> None:
    initial = torch.tensor([0.2, -0.1], dtype=torch.float64)

    def constant(_state: torch.Tensor, _time: float) -> torch.Tensor:
        return torch.tensor([0.7, -0.4], dtype=torch.float64)

    one = integrate_euler(initial, fixed_count_schedule(horizon=0.75, count=1), velocity=constant)
    split = integrate_euler(initial, fixed_count_schedule(horizon=0.75, count=32), velocity=constant)
    torch.testing.assert_close(one.terminal, split.terminal, rtol=0, atol=2e-15)


def test_nonlinear_euler_has_first_order_convergence() -> None:
    initial = torch.tensor([1.0], dtype=torch.float64)
    exact = torch.tensor([torch.exp(torch.tensor(1.0, dtype=torch.float64))], dtype=torch.float64)
    errors = []
    for count in (4, 8, 16, 32):
        result = integrate_euler(
            initial,
            fixed_count_schedule(horizon=1.0, count=count),
            velocity=lambda state, _time: state,
        )
        errors.append(float(torch.linalg.vector_norm(result.terminal - exact)))
    assert all(right < left for left, right in zip(errors, errors[1:]))
    assert errors[0] / errors[1] > 1.6 and errors[1] / errors[2] > 1.7


def test_physical_dense_block_path_uses_actual_updates() -> None:
    nodes = (
        (torch.tensor([[0.2]], dtype=torch.float64), torch.tensor([[0.0, 0.3]], dtype=torch.float64)),
        (torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([[0.0, -0.1]], dtype=torch.float64)),
    )
    path = dense_block_path(nodes, (0.25, 0.5))
    assert path.terminal_displacement == pytest.approx((0.3**2 + 0.2**2) ** 0.5)
    assert path.path_length == pytest.approx((0.2**2 + 0.3**2) ** 0.5 + (0.1**2 + 0.1**2) ** 0.5)


def test_production_ast_has_no_forbidden_identifier_or_call_path() -> None:
    directory = Path(inspect.getfile(integrate_euler)).parent
    members = tuple(path for path in directory.glob("*.py") if path.name != "__init__.py")
    # ``root`` by itself is a filesystem-path noun in the runtime.  The R3
    # contract explicitly excludes that harmless use from the semantic
    # endpoint-root ban, so the AST gate targets controller identifiers/calls.
    forbidden = {"rho", "ridge", "damping", "floor", "early_stop", "fallback"}
    observed: set[str] = set()
    for path in members:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                observed.add(node.name.lower())
            elif isinstance(node, ast.Name):
                observed.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                observed.add(node.attr.lower())
    assert forbidden.isdisjoint(observed)
    assert not any(
        name in observed
        for name in {"endpoint_root", "root_solver", "brent", "bisection", "retraction"}
    )


def test_execution_firewall_and_history_boundary() -> None:
    receipt = R3ExecutionBoundary(
        batch_size=1,
        fixed_target_compute_count=1,
        fixed_target_recompute_count=0,
        model_numeric_dtype="torch.float32",
        event_numeric_dtype="torch.float64",
        physical_write_dtype="torch.float32",
        node_history_append_count=0,
        terminal_history_append_count=1,
        localizer_call_count=0,
        endpoint_correction_count=0,
        accepted_step_search_count=0,
        alternate_solver_count=0,
        controller_evaluator_influence_count=0,
        retained_factor_count=0,
        scientific_promotion=False,
    )
    assert receipt.payload()["terminal_history_append_count"] == 1
