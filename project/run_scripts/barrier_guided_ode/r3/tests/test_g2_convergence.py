from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r3.errors import R3ScientificBoundary
from project.run_scripts.barrier_guided_ode.r3.g2_convergence import (
    G2Endpoint,
    assess_cauchy,
    block_distance,
)


def _endpoints() -> dict[int, G2Endpoint]:
    return {
        4: G2Endpoint(4, 2.0, 1.0, 0.4),
        8: G2Endpoint(8, 2.4, 1.2, 0.5),
        16: G2Endpoint(16, 2.55, 1.28, 0.54),
        32: G2Endpoint(32, 2.60, 1.31, 0.555),
    }


def test_predeclared_cauchy_gate_accepts_first_order_sequence() -> None:
    receipt = assess_cauchy(
        _endpoints(),
        {(4, 8): 0.4, (8, 16): 0.15, (16, 32): 0.05},
    )
    assert receipt.passed
    assert receipt.first_order_trend
    assert receipt.physical_ratios[0] == pytest.approx(8.0 / 3.0)
    assert receipt.physical_ratios[1] == pytest.approx(3.0)


def test_predeclared_cauchy_gate_rejects_noncontracting_endpoint() -> None:
    receipt = assess_cauchy(
        _endpoints(),
        {(4, 8): 0.20, (8, 16): 0.25, (16, 32): 0.30},
    )
    assert not receipt.passed
    assert not receipt.physical_contraction
    assert not receipt.first_order_trend


def test_block_distance_uses_actual_dense_fp32_endpoint_bytes() -> None:
    left = (torch.tensor([1.0, 2.0], dtype=torch.float32), torch.tensor([3.0], dtype=torch.float32))
    right = (torch.tensor([1.0, 4.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32))
    assert block_distance(left, right) == pytest.approx((2.0**2 + 3.0**2) ** 0.5)
    with pytest.raises(R3ScientificBoundary):
        block_distance(left, (torch.ones(3, dtype=torch.float32),))


def test_cauchy_grid_is_exact_and_missing_cells_fail_close() -> None:
    values = _endpoints()
    values.pop(32)
    with pytest.raises(R3ScientificBoundary):
        assess_cauchy(values, {(4, 8): 0.4, (8, 16): 0.2, (16, 32): 0.1})
