from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p2r2_residual_transport_writer import ProposalQuadratics
from project.run_scripts.ode_bf.p2r5_sdrt_writer import (
    P2R5_ARMS,
    P2R5RoutingTechnicalError,
    _same_semantic_face_solve,
    build_sdrt_quadratics,
    clamp_safe_semantic_deficit,
    p2r5_forbidden_influence_receipt,
    pooled_nonnegative_realization_calibration,
    solve_sdrt_routing,
)


def _response() -> torch.Tensor:
    response = torch.zeros((10, 50), dtype=torch.float64)
    for request in range(10):
        response[request, request] = 2.0
        response[request, 10 + request] = 1.0
    return response


def _quadratics() -> SimpleNamespace:
    capacity = torch.eye(50, dtype=torch.float64)
    structural = torch.eye(50, dtype=torch.float64)
    for request in range(10):
        capacity[request, request] = 10.0
        capacity[10 + request, 10 + request] = 1.0
        structural[request, request] = 0.1
        structural[10 + request, 10 + request] = 10.0
    return SimpleNamespace(
        capacity_gram=capacity,
        capacity_cross=torch.zeros(50, dtype=torch.float64),
        prior_capacity=0.0,
        structural_p_gram=structural,
        structural_p_cross=torch.zeros(50, dtype=torch.float64),
        prior_structural_p=0.0,
    )


def test_clamp_safe_deficit_is_requestwise_positive_part() -> None:
    current = torch.arange(10, dtype=torch.float64)
    oracle = current + torch.tensor([1.0, -1.0] * 5, dtype=torch.float64)
    observed = clamp_safe_semantic_deficit(current, oracle)
    assert observed.tolist() == [0.0, 1.0] * 5


def test_pooled_nnls_has_eta_one_at_entry_and_no_floor() -> None:
    entry = pooled_nonnegative_realization_calibration([], [])
    assert entry.eta == 1.0
    assert entry.status == "ENTRY_ETA_ONE"

    predicted = torch.arange(1, 11, dtype=torch.float64)
    positive = pooled_nonnegative_realization_calibration(
        [predicted], [predicted * 0.4]
    )
    assert positive.eta == pytest.approx(0.4)

    negative = pooled_nonnegative_realization_calibration(
        [predicted], [-predicted]
    )
    assert negative.eta == 0.0
    assert negative.status == "POOLED_NNLS_CERTIFIED"

    zero = pooled_nonnegative_realization_calibration(
        [torch.zeros(10, dtype=torch.float64)],
        [torch.ones(10, dtype=torch.float64)],
    )
    assert zero.eta == 0.0
    assert zero.status == "NNLS_ZERO_DESIGN_ETA_ZERO"


def test_cap_and_structp_share_semantic_response_but_move_allocation() -> None:
    calibration = pooled_nonnegative_realization_calibration([], [])
    deficit = torch.ones(10, dtype=torch.float64)
    cap = solve_sdrt_routing(
        _response(), deficit, calibration, _quadratics(), arm="SDRT-CAP"
    )
    structp = solve_sdrt_routing(
        _response(), deficit, calibration, _quadratics(), arm="SDRT-STRUCTP"
    )
    assert torch.max(
        torch.abs(cap.calibrated_predicted_response - structp.calibrated_predicted_response)
    ) <= 2.0e-8
    assert torch.linalg.vector_norm(cap.allocation - structp.allocation) > 0.1
    for route in (cap, structp):
        assert torch.min(route.allocation) >= -1.0e-8
        mass = torch.sum(route.allocation, dim=0)
        assert torch.max(mass) <= 1.0 + 1.0e-8
        assert route.semantic_face_max_abs_residual <= 2.0e-8
        assert route.status.endswith("CERTIFIED")
        assert all(item["certificate_pass"] for item in route.solver_receipts)


def test_non_success_solver_output_requires_inherited_kkt_primal_certificate() -> None:
    start = np.zeros(50, dtype=np.float64)
    objective = np.eye(50, dtype=np.float64)
    response = np.zeros((10, 50), dtype=np.float64)
    for request in range(10):
        response[request, request] = 1.0
    def solver_result(value: np.ndarray, *, optimality: float) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.asarray(value, dtype=np.float64),
            success=False,
            status=0,
            message="maximum evaluations",
            nit=3000,
            nfev=3000,
            njev=3000,
            nhev=3000,
            optimality=optimality,
            constr_violation=0.0,
        )

    with mock.patch(
        "project.run_scripts.ode_bf.p2r5_sdrt_writer.minimize",
        side_effect=lambda function, value, **kwargs: solver_result(
            value, optimality=1.0e-10
        ),
    ):
        selected, receipt = _same_semantic_face_solve(
            start,
            objective,
            np.zeros(50, dtype=np.float64),
            response,
            np.zeros(10, dtype=np.float64),
            stage="CERTIFIED_NON_SUCCESS_FIXTURE",
        )
    assert np.array_equal(selected, start)
    assert receipt["certificate_pass"] is True
    assert receipt["non_success_certified_with_inherited_tolerance"] is True
    assert receipt["coordinate_backend"] == "SCIPY_SVD_SEMANTIC_FACE_NULLSPACE"
    assert receipt["semantic_nullity"] == 40

    with mock.patch(
        "project.run_scripts.ode_bf.p2r5_sdrt_writer.minimize",
        side_effect=lambda function, value, **kwargs: solver_result(
            value, optimality=1.0e-4
        ),
    ), pytest.raises(P2R5RoutingTechnicalError):
        _same_semantic_face_solve(
            start,
            objective,
            np.zeros(50, dtype=np.float64),
            response,
            np.zeros(10, dtype=np.float64),
            stage="UNCERTIFIED_NON_SUCCESS_FIXTURE",
        )


def test_semantic_face_nullspace_removes_equality_without_changing_solution() -> None:
    response = np.zeros((10, 50), dtype=np.float64)
    for request in range(10):
        response[request, request] = 1.0
        response[request, request + 10] = 1.0 + 1.0e-10
    start = np.zeros(50, dtype=np.float64)
    start[:10] = 0.5
    start[10:20] = 0.5 / (1.0 + 1.0e-10)
    objective = np.eye(50, dtype=np.float64)
    selected, receipt = _same_semantic_face_solve(
        start,
        objective,
        np.zeros(50, dtype=np.float64),
        response,
        response @ start,
        stage="NULLSPACE_CONDITIONING_FIXTURE",
    )
    assert receipt["coordinate_backend"] == "SCIPY_SVD_SEMANTIC_FACE_NULLSPACE"
    assert receipt["semantic_rank"] == 10
    assert receipt["semantic_nullity"] == 40
    assert receipt["semantic_basis_max_abs_residual"] <= 1.0e-8
    assert np.max(np.abs(response @ selected - response @ start)) <= 1.0e-8
    assert np.min(selected) >= -1.0e-8


def test_zero_deficit_preserves_totality_and_mass_contract() -> None:
    route = solve_sdrt_routing(
        _response(),
        torch.zeros(10, dtype=torch.float64),
        pooled_nonnegative_realization_calibration([], []),
        _quadratics(),
        arm="SDRT-CAP",
    )
    assert torch.max(torch.sum(route.allocation, dim=0)) <= 1.0 + 1.0e-8
    assert torch.max(torch.abs(route.calibrated_predicted_response)) <= 2.0e-8


def test_cumulative_capacity_matches_dense_factor_sum() -> None:
    generator = torch.Generator().manual_seed(48)
    layers = []
    prior: dict[str, tuple[WaypointFactor, ...]] = {}
    capacity = torch.zeros((50, 50), dtype=torch.float64)
    for layer in range(5):
        residual = torch.randn((3, 10), generator=generator)
        q = torch.randn((4, 10), generator=generator)
        name = f"layer{layer}.weight"
        layers.append(
            SimpleNamespace(
                weight_name=name,
                residual=residual,
                q=q,
            )
        )
        start = layer * 10
        capacity[start : start + 10, start : start + 10] = (
            residual.T.double() @ residual.double()
        ) * (q.T.double() @ q.double())
        prior[name] = (
            WaypointFactor(
                name,
                layer,
                0,
                0,
                layer,
                1.0,
                torch.randn((3, 10), generator=generator),
                torch.randn((4, 10), generator=generator),
                joint_batch=True,
                global_batch_size=10,
            ),
        )
    base = ProposalQuadratics(
        capacity,
        torch.eye(50, dtype=torch.float64),
        torch.zeros(50, dtype=torch.float64),
        0.0,
        (0.0,) * 5,
        "base",
    )
    field = SimpleNamespace(layers=layers, current_z=torch.zeros((3, 10)))
    observed = build_sdrt_quadratics(field, prior, base)
    alpha = torch.rand(50, generator=generator, dtype=torch.float64) * 0.1
    predicted = (
        observed.prior_capacity
        + 2.0 * float(observed.capacity_cross @ alpha)
        + float(alpha @ observed.capacity_gram @ alpha)
    )
    dense = 0.0
    for layer, layer_field in enumerate(layers):
        previous = prior[layer_field.weight_name][0]
        selected = alpha[layer * 10 : (layer + 1) * 10].float()
        update = (layer_field.residual * selected.unsqueeze(0)) @ layer_field.q.T
        total = previous.left @ previous.right.T + update
        dense += float(torch.sum(total.double().square()))
    assert predicted == pytest.approx(dense, rel=1.0e-6, abs=1.0e-8)


def test_forbidden_receipt_and_invalid_arm_fail_close() -> None:
    receipt = p2r5_forbidden_influence_receipt()
    assert receipt["clamp_on_decision_influence_count_per_target_microstep"] == 1
    assert receipt["clamp_off_access_count"] == 0
    assert receipt["neutral_fallback_count"] == 0
    for key in (
        "semantic_debt_input_count",
        "explicit_lag_input_count",
        "remaining_horizon_division_count",
        "hard_p_budget_influence_count",
        "functional_p_veto_count",
        "retry_count",
        "backtracking_count",
        "historical_h_influence_count",
        "sequential_state_count",
    ):
        assert receipt[key] == 0
    assert P2R5_ARMS == ("SDRT-CAP", "SDRT-STRUCTP")
    with pytest.raises(ODEBFContractError):
        solve_sdrt_routing(
            _response(),
            torch.ones(10, dtype=torch.float64),
            pooled_nonnegative_realization_calibration([], []),
            _quadratics(),
            arm="NEUTRAL",
        )
