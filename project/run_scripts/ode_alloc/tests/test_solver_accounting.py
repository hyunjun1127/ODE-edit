from __future__ import annotations

import inspect
import unittest

import torch

from project.run_scripts.ode_alloc.accounting import (
    ComputeLedger,
    assert_matched_call_identity,
)
from project.run_scripts.ode_alloc.contracts import (
    MODEL_ALIASES,
    ODEAllocContractError,
    SolverBudget,
    assert_matched_budgets,
)
from project.run_scripts.ode_alloc.solver import (
    CBFProjector,
    CandidateVerdict,
    ConstraintLinearization,
    EvaluationCost,
    FieldEvaluation,
    FixedGridSolver,
    GenericProjector,
    LinearizationEvaluation,
)
from project.run_scripts.ode_alloc import solver as solver_module


class SolverAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = SolverBudget(
            fixed_k=3,
            max_trials_per_step=2,
            backtracking_factors=(1.0, 0.5),
            context_count=2,
        )

    def test_soft_cbf_projection_satisfies_slack_contract(self) -> None:
        nominal = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        linearization = ConstraintLinearization(
            matrix=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
            barrier_values=torch.tensor([0.0], dtype=torch.float64),
            kappa=1.0,
            labels=("E",),
        )
        result = CBFProjector(slack_penalty_weight=100.0).project(
            nominal, linearization
        )
        residual = linearization.matrix @ result.velocity
        self.assertTrue(torch.all(residual >= -result.slack - 1.0e-12))
        self.assertAlmostEqual(float(result.velocity.mean()), 0.0, places=14)
        self.assertGreater(result.qp_cpu_seconds, 0.0)
        self.assertLess(float(torch.linalg.vector_norm(result.velocity - nominal)), 2.0)
        self.assertEqual(len(result.subset_diagnostics), 2)
        self.assertLessEqual(result.max_constraint_residual, 1.0e-10)

    def _run(self, projector):
        ledger = ComputeLedger()

        def direction(step: int, q: torch.Tensor) -> FieldEvaluation:
            del step, q
            return FieldEvaluation(
                torch.tensor([0.4, -0.4], dtype=torch.float64),
                EvaluationCost(backward_calls=1),
            )

        def linearize(step: int, q: torch.Tensor) -> LinearizationEvaluation:
            del step, q
            return LinearizationEvaluation(
                ConstraintLinearization(
                    matrix=torch.empty((0, 2), dtype=torch.float64),
                    barrier_values=torch.empty(0, dtype=torch.float64),
                    kappa=1.0,
                ),
                EvaluationCost(constraint_vjp_calls=1),
            )

        def verdict(q: torch.Tensor) -> CandidateVerdict:
            return CandidateVerdict(
                feasible=float(q[0]) >= 0.0,
                objective=-float(q[0]),
                event_id=f"event-{float(q[0]):.8f}",
                model_forward_calls=2,
                processed_tokens=12,
            )

        endpoint = FixedGridSolver(
            budget=self.budget,
            projector=projector,
        ).run(
            torch.zeros(2, dtype=torch.float64),
            direction_fn=direction,
            linearize_fn=linearize,
            verdict_fn=verdict,
            ledger=ledger,
        )
        return endpoint, ledger

    def test_generic_and_ode_exhaust_identical_call_budget(self) -> None:
        generic_endpoint, generic = self._run(GenericProjector())
        ode_endpoint, ode = self._run(
            CBFProjector(slack_penalty_weight=100.0)
        )
        self.assertEqual(generic_endpoint.completed_steps, self.budget.fixed_k)
        self.assertEqual(ode_endpoint.completed_steps, self.budget.fixed_k)
        self.assertEqual(
            generic.quantized_trial_calls, self.budget.quantized_trial_budget
        )
        self.assertEqual(ode.quantized_trial_calls, self.budget.quantized_trial_budget)
        assert_matched_call_identity(generic, ode)
        self.assertEqual(generic.qp_calls, 0)
        self.assertEqual(ode.qp_calls, self.budget.fixed_k)

    def test_no_feasible_candidate_still_finishes_and_falls_back(self) -> None:
        ledger = ComputeLedger()

        def direction(step: int, q: torch.Tensor) -> FieldEvaluation:
            del step, q
            return FieldEvaluation(
                torch.tensor([0.1, -0.1], dtype=torch.float64),
                EvaluationCost(backward_calls=1),
            )

        def linearize(step: int, q: torch.Tensor) -> LinearizationEvaluation:
            del step, q
            return LinearizationEvaluation(
                ConstraintLinearization(
                    torch.empty((0, 2), dtype=torch.float64),
                    torch.empty(0, dtype=torch.float64),
                    1.0,
                ),
                EvaluationCost(constraint_vjp_calls=1),
            )

        endpoint = FixedGridSolver(
            budget=self.budget, projector=GenericProjector()
        ).run(
            torch.zeros(2, dtype=torch.float64),
            direction_fn=direction,
            linearize_fn=linearize,
            verdict_fn=lambda q: CandidateVerdict(False, float(q.square().sum()), "no"),
            ledger=ledger,
        )
        self.assertTrue(endpoint.fallback_to_native)
        self.assertTrue(torch.equal(endpoint.q, torch.zeros_like(endpoint.q)))
        self.assertEqual(endpoint.completed_steps, self.budget.fixed_k)
        self.assertEqual(ledger.quantized_trial_calls, self.budget.quantized_trial_budget)

    def test_budget_and_source_have_no_alias_specific_controller_branch(self) -> None:
        assert_matched_budgets(self.budget, self.budget)
        with self.assertRaises(ODEAllocContractError):
            assert_matched_budgets(
                self.budget,
                SolverBudget(4, 2, (1.0, 0.5), 2),
            )
        source = inspect.getsource(solver_module).casefold()
        for alias in MODEL_ALIASES:
            self.assertNotIn(alias, source)

    def test_three_aggregated_constraints_enumerate_all_subsets_and_permute(self) -> None:
        matrix = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=torch.float64,
        )
        barrier = torch.tensor([-0.05, 0.02, -0.03], dtype=torch.float64)
        nominal = torch.tensor([-0.3, 0.1, 0.4, -0.2], dtype=torch.float64)
        projector = CBFProjector(
            slack_penalty_weight=100.0,
            residual_tolerance=1.0e-10,
        )
        baseline = projector.project(
            nominal,
            ConstraintLinearization(matrix, barrier, 1.0, ("E", "H", "P")),
        )
        self.assertEqual(len(baseline.subset_diagnostics), 8)
        for diagnostic in baseline.subset_diagnostics:
            self.assertLessEqual(diagnostic.linear_solve_residual, 1.0e-10)
            self.assertLessEqual(diagnostic.stationarity_residual, 1.0e-10)
            self.assertTrue(diagnostic.kkt_valid)
        permutation = (2, 0, 1)
        permuted = projector.project(
            nominal,
            ConstraintLinearization(
                matrix[list(permutation)],
                barrier[list(permutation)],
                1.0,
                tuple(("E", "H", "P")[index] for index in permutation),
            ),
        )
        self.assertTrue(
            torch.allclose(
                baseline.velocity,
                permuted.velocity,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                baseline.slack[list(permutation)],
                permuted.slack,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        self.assertGreater(float(baseline.slack.max()), 0.0)

    def test_per_item_or_more_than_three_constraints_are_unrepresentable(self) -> None:
        with self.assertRaises(ODEAllocContractError):
            ConstraintLinearization(
                torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
                1.0,
                ("H-item-0",),
            )
        with self.assertRaises(ODEAllocContractError):
            ConstraintLinearization(
                torch.tensor(
                    [
                        [1.0, -1.0],
                        [1.0, -1.0],
                        [1.0, -1.0],
                        [1.0, -1.0],
                    ],
                    dtype=torch.float64,
                ),
                torch.zeros(4, dtype=torch.float64),
                1.0,
                ("E", "H", "P", "E"),
            )


if __name__ == "__main__":
    unittest.main()
