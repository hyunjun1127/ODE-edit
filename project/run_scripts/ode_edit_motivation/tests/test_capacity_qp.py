import math
import unittest

import torch

from project.run_scripts.ode_edit_motivation.capacity_qp import (
    CapacityLayerTerms,
    build_capacity_layer_terms,
    build_capacity_qp_proposal,
    capacity_state_by_layer,
    solve_capacity_qp,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import ActionDirection


def _snapshot() -> SnapshotManifest:
    return SnapshotManifest(
        model_id="toy/model@" + "a" * 40,
        context_id="b" * 64,
        request_ids=("c" * 64,),
        hparams_sha256="d" * 64,
        parameters=(
            ParameterRecord(
                name="layer1.weight",
                shape=(2, 2),
                dtype="torch.float64",
                sha256="e" * 64,
            ),
            ParameterRecord(
                name="layer2.weight",
                shape=(2, 2),
                dtype="torch.float64",
                sha256="f" * 64,
            ),
        ),
        provenance_ids=("0" * 64,),
    )


def _factor(weight_name: str, left: tuple[float, float], right: tuple[float, float]) -> LowRankFactor:
    digest = "e" * 64 if weight_name == "layer1.weight" else "f" * 64
    return LowRankFactor(
        weight_name=weight_name,
        left=torch.tensor(left, dtype=torch.float64).reshape(2, 1),
        right=torch.tensor(right, dtype=torch.float64).reshape(2, 1),
        expected_weight_sha256=digest,
    )


def _proposal(*factors: LowRankFactor, suffix: str) -> MemitFactorProposal:
    return MemitFactorProposal(
        snapshot=_snapshot(),
        factors=factors,
        semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
        solver_name=f"toy/{suffix}",
        residual_denominator=2,
    )


def _actions() -> tuple[ActionDirection, ActionDirection]:
    first = _factor("layer1.weight", (1.0, 0.0), (1.0, 0.0))
    second = _factor("layer2.weight", (0.0, 1.0), (0.0, 1.0))
    return (
        ActionDirection(
            action_id="layer_1",
            proposal=_proposal(first, suffix="unit1"),
            c_squared_norm=1.0,
        ),
        ActionDirection(
            action_id="layer_2",
            proposal=_proposal(second, suffix="unit2"),
            c_squared_norm=1.0,
        ),
    )


class CapacityQPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = _actions()
        self.covariances = {1: torch.eye(2), 2: torch.eye(2)}
        self.layer_by_weight = {"layer1.weight": 1, "layer2.weight": 2}
        self.denominators = {1: 1.0, 2: 1.0}

    def test_low_rank_capacity_polynomial_matches_dense_update(self) -> None:
        cumulative = (
            _factor("layer1.weight", (0.4, 0.0), (1.0, 0.0)),
            _factor("layer1.weight", (0.0, 0.2), (0.0, 1.0)),
        )
        terms = build_capacity_layer_terms(
            cumulative_factors=cumulative,
            unit_actions=self.actions,
            slopes={"layer_1": 1.0, "layer_2": 0.5},
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
            w0_denominators=self.denominators,
            trust_distance=0.25,
        )
        first = terms[0]
        dense_cumulative = sum(
            (factor.left @ factor.right.T for factor in cumulative),
            torch.zeros((2, 2), dtype=torch.float64),
        )
        dense_candidate = (
            self.actions[0].proposal.factors[0].left
            @ self.actions[0].proposal.factors[0].right.T
        )
        self.assertAlmostEqual(first.psi_before, float(torch.sum(dense_cumulative**2)), places=6)
        self.assertAlmostEqual(
            first.linear,
            2.0 * float(torch.sum(dense_cumulative * dense_candidate)),
            places=6,
        )
        self.assertAlmostEqual(first.quadratic, float(torch.sum(dense_candidate**2)), places=6)
        coefficient = min(0.1, first.coefficient_cap)
        dense_next = dense_cumulative + coefficient * dense_candidate
        self.assertAlmostEqual(first.next_psi(coefficient), float(torch.sum(dense_next**2)), places=6)

    def test_overloaded_aligned_layer_is_capped_and_qp_routes_elsewhere(self) -> None:
        cumulative = (
            _factor("layer1.weight", (1.0, 0.0), (1.0, 0.0)),
            _factor("layer1.weight", (1.0, 0.0), (1.0, 0.0)),
        )
        terms = build_capacity_layer_terms(
            cumulative_factors=cumulative,
            unit_actions=self.actions,
            slopes={"layer_1": 1.0, "layer_2": 1.0},
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
            w0_denominators=self.denominators,
            trust_distance=1.0,
        )
        self.assertTrue(terms[0].overloaded)
        self.assertEqual(terms[0].coefficient_cap, 0.0)
        solution = solve_capacity_qp(
            terms,
            requested_progress=0.5,
            trust_distance=1.0,
        )
        self.assertEqual(solution.coefficients[0], 0.0)
        self.assertAlmostEqual(solution.coefficients[1], 0.5, places=7)
        self.assertAlmostEqual(solution.predicted_progress, 0.5, places=7)

    def test_solver_obeys_progress_box_and_l2_trust_constraints(self) -> None:
        terms = (
            CapacityLayerTerms(
                layer=1,
                action_id="layer_1",
                weight_name="layer1.weight",
                slope=2.0,
                psi_before=0.0,
                linear=0.4,
                quadratic=1.0,
                barrier=1.0,
                coefficient_cap=0.8,
                overloaded=False,
            ),
            CapacityLayerTerms(
                layer=2,
                action_id="layer_2",
                weight_name="layer2.weight",
                slope=1.0,
                psi_before=0.0,
                linear=0.0,
                quadratic=1.0,
                barrier=1.0,
                coefficient_cap=0.8,
                overloaded=False,
            ),
        )
        solution = solve_capacity_qp(
            terms,
            requested_progress=0.9,
            trust_distance=0.6,
        )
        self.assertGreaterEqual(solution.predicted_progress + 1e-8, 0.9)
        self.assertLessEqual(solution.coefficient_norm, 0.6 + 1e-8)
        for coefficient, term in zip(solution.coefficients, terms, strict=True):
            self.assertGreaterEqual(coefficient, 0.0)
            self.assertLessEqual(coefficient, term.coefficient_cap + 1e-8)
            self.assertLessEqual(term.next_psi(coefficient), term.barrier + 1e-8)

    def test_infeasible_request_is_exposed_as_progress_slack(self) -> None:
        terms = (
            CapacityLayerTerms(
                layer=1,
                action_id="layer_1",
                weight_name="layer1.weight",
                slope=1.0,
                psi_before=0.0,
                linear=0.0,
                quadratic=1.0,
                barrier=0.04,
                coefficient_cap=0.2,
                overloaded=False,
            ),
            CapacityLayerTerms(
                layer=2,
                action_id="layer_2",
                weight_name="layer2.weight",
                slope=0.0,
                psi_before=0.0,
                linear=0.0,
                quadratic=1.0,
                barrier=1.0,
                coefficient_cap=1.0,
                overloaded=False,
            ),
        )
        solution = solve_capacity_qp(
            terms,
            requested_progress=1.0,
            trust_distance=1.0,
        )
        self.assertFalse(solution.feasible_without_slack)
        self.assertAlmostEqual(solution.predicted_progress, 0.2, places=7)
        self.assertAlmostEqual(solution.progress_slack, 0.8, places=7)

    def test_solution_builds_low_rank_proposal_and_reports_state(self) -> None:
        terms = build_capacity_layer_terms(
            cumulative_factors=(),
            unit_actions=self.actions,
            slopes={"layer_1": 2.0, "layer_2": 1.0},
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
            w0_denominators=self.denominators,
            trust_distance=0.5,
        )
        solution = solve_capacity_qp(
            terms,
            requested_progress=0.4,
            trust_distance=0.5,
        )
        synchronous = _proposal(
            self.actions[0].proposal.factors[0],
            self.actions[1].proposal.factors[0],
            suffix="sync",
        )
        proposal = build_capacity_qp_proposal(
            synchronous=synchronous,
            unit_actions=self.actions,
            terms=terms,
            solution=solution,
            solver_suffix="unit",
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(len(proposal.factors), sum(value > 0.0 for value in solution.coefficients))
        state = capacity_state_by_layer(
            cumulative_factors=proposal.factors,
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
            w0_denominators=self.denominators,
        )
        for term, coefficient in zip(terms, solution.coefficients, strict=True):
            self.assertTrue(math.isclose(state[term.layer], coefficient**2, rel_tol=1e-6, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
