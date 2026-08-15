from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p2r2_residual_transport_writer import (
    P2R2_ALPHA_COUNT,
    P2R2_LAYER_COUNT,
    P2R2_REQUEST_COUNT,
    ProposalQuadratics,
    _quadratic,
    _quadratic_gradient,
    build_proposal_quadratics,
    measure_request_layer_response,
    p2r2_forbidden_influence_receipt,
    p2r2_waypoint_factors,
    solve_p2r2_routing,
)


def _quadratics(*, p_costs: tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.0)):
    capacity = torch.eye(P2R2_ALPHA_COUNT, dtype=torch.float64)
    structural = torch.zeros_like(capacity)
    for layer, cost in enumerate(p_costs):
        start = layer * P2R2_REQUEST_COUNT
        structural[start : start + P2R2_REQUEST_COUNT, start : start + P2R2_REQUEST_COUNT] = (
            torch.eye(P2R2_REQUEST_COUNT, dtype=torch.float64) * cost
        )
    cross = torch.zeros(P2R2_ALPHA_COUNT, dtype=torch.float64)
    return ProposalQuadratics(
        capacity,
        structural,
        cross,
        0.0,
        (0.0,) * P2R2_LAYER_COUNT,
        canonical_hash({"fixture": "quadratics"}),
    )


def _response(*, unreachable: int | None = None) -> torch.Tensor:
    value = torch.zeros(
        (P2R2_REQUEST_COUNT, P2R2_ALPHA_COUNT), dtype=torch.float64
    )
    for request in range(P2R2_REQUEST_COUNT):
        if request == unreachable:
            continue
        for layer, progress in enumerate((1.0, 0.9, 0.8, 0.7, 0.6)):
            value[request, layer * P2R2_REQUEST_COUNT + request] = progress
    return value


class ToyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight.T


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = ToyLayer()
        self.layer1 = ToyLayer()
        self.layer2 = ToyLayer()
        self.layer3 = ToyLayer()
        self.layer4 = ToyLayer()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in (self.layer0, self.layer1, self.layer2, self.layer3, self.layer4):
            value = layer(value)
        return value


class P2R2ResidualTransportWriterTest(unittest.TestCase):
    def test_quadratic_analytic_gradient_matches_centered_difference(self) -> None:
        matrix = torch.tensor(
            [[3.0, -0.5, 0.25], [0.75, 2.0, -1.0], [0.5, 0.25, 4.0]],
            dtype=torch.float64,
        ).numpy()
        linear = torch.tensor([0.5, -0.25, 0.75], dtype=torch.float64).numpy()
        point = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64).numpy()
        objective = _quadratic(matrix, linear)
        observed = _quadratic_gradient(matrix, linear)(point)
        epsilon = 1.0e-6
        numerical = []
        for index in range(point.size):
            direction = torch.zeros(point.size, dtype=torch.float64).numpy()
            direction[index] = epsilon
            numerical.append((objective(point + direction) - objective(point - direction)) / (2.0 * epsilon))
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(observed),
                torch.tensor(numerical, dtype=torch.float64),
                atol=1.0e-8,
                rtol=1.0e-8,
            )
        )

    def test_neutral_and_soft_request_conservation_and_no_weaker(self) -> None:
        response = _response()
        neutral = solve_p2r2_routing(response, _quadratics(), arm="NEUTRAL")
        soft = solve_p2r2_routing(response, _quadratics(), arm="SOFTP")
        self.assertTrue(torch.allclose(neutral.allocation.sum(0), torch.ones(10, dtype=torch.float64), atol=1e-8))
        self.assertTrue(torch.allclose(soft.allocation.sum(0), torch.ones(10, dtype=torch.float64), atol=1e-8))
        self.assertTrue(
            torch.all(soft.predicted_progress >= neutral.predicted_progress - 1e-8)
        )
        if soft.fallback_reason is not None:
            self.assertTrue(torch.equal(soft.allocation, neutral.allocation))
            self.assertEqual(soft.status, "SOFT_NEUTRAL_FALLBACK")
        self.assertEqual(soft.raw_free_payload()["p_budget_influence_count"], 0)

    def test_unreachable_request_owns_no_allocation(self) -> None:
        response = _response(unreachable=9)
        route = solve_p2r2_routing(response, _quadratics(), arm="NEUTRAL")
        self.assertEqual(route.unreachable_requests, (9,))
        self.assertEqual(float(route.allocation[:, 9].abs().max()), 0.0)
        self.assertTrue(torch.allclose(route.allocation[:, :9].sum(0), torch.ones(9, dtype=torch.float64), atol=1e-8))

    def test_all_inactive_totalizes_to_exact_w_hold_allocation(self) -> None:
        response = torch.zeros((10, 50), dtype=torch.float64)
        for arm in ("NEUTRAL", "SOFTP"):
            route = solve_p2r2_routing(response, _quadratics(), arm=arm)
            self.assertEqual(route.unreachable_requests, tuple(range(10)))
            self.assertEqual(float(route.allocation.abs().max()), 0.0)
            self.assertEqual(float(route.predicted_progress.abs().max()), 0.0)

    def test_selected_factors_conserve_one_residual_per_request(self) -> None:
        response = _response()
        route = solve_p2r2_routing(response, _quadratics(), arm="NEUTRAL")
        layers = []
        for ordinal in range(5):
            layers.append(
                SimpleNamespace(
                    weight_name=f"layer{ordinal}.weight",
                    layer=ordinal + 4,
                    residual=torch.arange(20, dtype=torch.float32).reshape(2, 10) + 1,
                    q=torch.eye(10, dtype=torch.float32)[:2],
                )
            )
        field = SimpleNamespace(layers=tuple(layers), current_z=torch.zeros(2, 10))
        factors = p2r2_waypoint_factors(field, route, outer_step=0)
        reconstructed = sum(
            factor.left for factor in factors.values()
        )
        self.assertTrue(torch.allclose(reconstructed, layers[0].residual, atol=1e-6))
        self.assertTrue(all(factor.theta == 1.0 for factor in factors.values()))

    def test_structural_cross_and_self_match_dense_identity_covariance(self) -> None:
        residual = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        q = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        prior = WaypointFactor(
            "layer0.weight", 4, 0, 0, 0, 1.0,
            torch.tensor([[0.5], [1.5]]),
            torch.tensor([[1.0], [2.0]]),
            joint_batch=False,
        )
        layer = SimpleNamespace(
            weight_name="layer0.weight",
            residual=residual,
            q=q,
            covariance_action=q,
        )
        field = SimpleNamespace(layers=(layer,), current_z=torch.zeros(2, 2))
        prior_dense = prior.left @ prior.right.T
        prior_p = float(torch.sum(prior_dense.square()))
        observed = build_proposal_quadratics(
            field, {"layer0.weight": (prior,)}, prior_structural_p=prior_p
        )
        alpha = torch.tensor([0.25, 0.75], dtype=torch.float64)
        candidate = (residual.double() * alpha.unsqueeze(0)) @ q.double().T
        expected = float(torch.sum((prior_dense.double() + candidate).square()))
        marginal = float(
            2.0 * observed.structural_p_cross @ alpha
            + alpha @ observed.structural_p_gram @ alpha
        )
        self.assertAlmostEqual(prior_p + marginal, expected, places=10)

    def test_batched_vjp_returns_request_by_layer_request_axes(self) -> None:
        model = ToyModel()
        residual = torch.zeros(2, 10)
        residual[0] = torch.arange(1, 11)
        residual[1] = 1.0
        q = torch.zeros(2, 10)
        q[0] = 1.0
        layers = tuple(
            SimpleNamespace(
                weight_name=f"layer{ordinal}.weight",
                q=q,
                residual=residual,
            )
            for ordinal in range(5)
        )
        field = SimpleNamespace(layers=layers, current_z=torch.zeros(2, 10), identity_sha256="f" * 64)
        batch = SimpleNamespace(prepared=object())
        plan = SimpleNamespace(
            request_count=10,
            batches=(batch,),
            llama=False,
            context_sha256="c" * 64,
            processed_token_count=60,
            identity_sha256="p" * 64,
        )

        def score(observed_model, _prepared, **_kwargs):
            hidden = torch.zeros(10, 1, 2)
            hidden[:, :, 0] = torch.arange(1, 11).reshape(10, 1)
            output = observed_model(hidden)
            rows = [
                (index, output[index].sum(), 1, f"{index:064x}")
                for index in range(10)
            ]
            return rows, 60

        with mock.patch(
            "project.run_scripts.ode_bf.p2r2_residual_transport_writer._score_prepared_target_new_batch",
            side_effect=score,
        ):
            receipt = measure_request_layer_response(model, plan, field)
        self.assertEqual(tuple(receipt.response.shape), (10, 50))
        self.assertEqual(receipt.batched_vjp_count, 1)
        self.assertEqual(receipt.overlay_fire_count, (1, 1, 1, 1, 1))

    def test_forbidden_influence_is_zero(self) -> None:
        receipt = p2r2_forbidden_influence_receipt()
        for key, value in receipt.items():
            if key.endswith("_count") and key != "outer_joint_materialization_count_per_step":
                self.assertEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
