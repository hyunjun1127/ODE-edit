"""Fixed-grid Generic/ODE solver skeleton with one isolated CBF-QP hook."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from .accounting import ComputeLedger
from .contracts import ODEAllocContractError, SolverBudget, finite, positive


AGGREGATED_CONSTRAINT_LABELS = frozenset({"E", "H", "P"})


@dataclass(frozen=True, slots=True)
class ConstraintLinearization:
    """Rows encode A v + kappa*h >= -slack in the q tangent space."""

    matrix: torch.Tensor
    barrier_values: torch.Tensor
    kappa: float
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.barrier_values.ndim != 1:
            raise ODEAllocContractError("constraint linearization ranks are invalid")
        if self.matrix.shape[0] != self.barrier_values.shape[0]:
            raise ODEAllocContractError("constraint rows and values differ")
        if len(self.labels) != self.matrix.shape[0]:
            raise ODEAllocContractError("aggregated constraint labels and rows differ")
        if (
            len(set(self.labels)) != len(self.labels)
            or not set(self.labels).issubset(AGGREGATED_CONSTRAINT_LABELS)
            or len(self.labels) > 3
        ):
            raise ODEAllocContractError(
                "constraints must be unique aggregated E/H/P rows"
            )
        if self.matrix.dtype != torch.float64 or self.barrier_values.dtype != torch.float64:
            raise ODEAllocContractError("constraint linearization must use float64")
        if self.matrix.device.type != "cpu" or self.barrier_values.device.type != "cpu":
            raise ODEAllocContractError("prep QP linearization must be CPU-resident")
        if not torch.isfinite(self.matrix).all() or not torch.isfinite(self.barrier_values).all():
            raise ODEAllocContractError("constraint linearization is non-finite")
        object.__setattr__(self, "kappa", positive("CBF kappa", self.kappa))
        if self.matrix.numel() and not torch.allclose(
            self.matrix.sum(dim=1),
            torch.zeros(self.matrix.shape[0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ODEAllocContractError("constraint gradients leave the zero-mean tangent")


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    velocity: torch.Tensor
    slack: torch.Tensor
    qp_cpu_seconds: float
    subset_diagnostics: tuple["ActiveSetDiagnostic", ...] = ()
    max_constraint_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class ActiveSetDiagnostic:
    active_labels: tuple[str, ...]
    linear_solve_residual: float
    stationarity_residual: float
    sign_valid: bool
    kkt_valid: bool
    objective: float


class VelocityProjector(Protocol):
    name: str

    def project(
        self,
        nominal: torch.Tensor,
        linearization: ConstraintLinearization,
    ) -> ProjectionResult: ...


class GenericProjector:
    """Matched generic optimizer: identical nominal field, no CBF projection."""

    name = "generic-no-cbf-projection"

    def project(
        self,
        nominal: torch.Tensor,
        linearization: ConstraintLinearization,
    ) -> ProjectionResult:
        slack = torch.relu(
            -(
                linearization.matrix @ nominal
                + linearization.kappa * linearization.barrier_values
            )
        )
        violation = (
            float(torch.max(torch.relu(-(
                linearization.matrix @ nominal
                + linearization.kappa * linearization.barrier_values
                + slack
            ))))
            if slack.numel()
            else 0.0
        )
        return ProjectionResult(
            nominal.clone(),
            slack,
            0.0,
            max_constraint_residual=violation,
        )


class CBFProjector:
    """Deterministic production E/H/P soft CBF-QP active-set enumerator.

    Eliminating nonnegative slack yields a convex squared-hinge problem.  The
    all at most 2^3 active sets are enumerated in float64 with explicit linear
    solve, stationarity, sign, and terminal inequality residual checks.  No
    external solver or per-item constraint is representable.
    """

    name = "soft-cbf-qp-active-set"

    def __init__(
        self,
        *,
        slack_penalty_weight: float,
        residual_tolerance: float = 1.0e-10,
    ) -> None:
        self.slack_penalty_weight = positive(
            "CBF slack penalty weight", slack_penalty_weight
        )
        self.residual_tolerance = positive(
            "CBF residual tolerance", residual_tolerance
        )

    def project(
        self,
        nominal: torch.Tensor,
        linearization: ConstraintLinearization,
    ) -> ProjectionResult:
        start = time.perf_counter()
        if nominal.ndim != 1 or nominal.dtype != torch.float64 or nominal.device.type != "cpu":
            raise ODEAllocContractError("nominal velocity must be a CPU float64 vector")
        if linearization.matrix.shape[1] != nominal.numel():
            raise ODEAllocContractError("velocity and constraint dimensions differ")
        if not torch.isfinite(nominal).all() or abs(float(nominal.sum())) > 1.0e-12:
            raise ODEAllocContractError("nominal velocity leaves the zero-mean tangent")
        count = linearization.matrix.shape[0]
        if count > 3:
            raise ODEAllocContractError("production CBF constraint cap exceeded")
        if count == 0:
            return ProjectionResult(
                nominal.clone(),
                torch.empty(0, dtype=torch.float64),
                time.perf_counter() - start,
            )
        matrix = linearization.matrix
        offset = linearization.kappa * linearization.barrier_values
        identity = torch.eye(nominal.numel(), dtype=torch.float64)
        candidates: list[
            tuple[float, tuple[float, ...], torch.Tensor, torch.Tensor]
        ] = []
        diagnostics: list[ActiveSetDiagnostic] = []
        tolerance = self.residual_tolerance
        for mask in itertools.product((False, True), repeat=count):
            active_indices = [index for index, active in enumerate(mask) if active]
            if active_indices:
                active_matrix = matrix[active_indices]
                active_offset = offset[active_indices]
                system = identity + self.slack_penalty_weight * (
                    active_matrix.transpose(0, 1) @ active_matrix
                )
                rhs = nominal - self.slack_penalty_weight * (
                    active_matrix.transpose(0, 1) @ active_offset
                )
                try:
                    velocity = torch.linalg.solve(system, rhs)
                except RuntimeError as exc:
                    raise ODEAllocContractError(
                        "CBF active-set float64 solve failed"
                    ) from exc
            else:
                system = identity
                rhs = nominal
                velocity = nominal.clone()
            residual = matrix @ velocity + offset
            sign_valid = all(
                float(residual[index]) <= tolerance if active else float(residual[index]) >= -tolerance
                for index, active in enumerate(mask)
            )
            linear_solve_residual = float(
                torch.linalg.vector_norm(system @ velocity - rhs, ord=float("inf"))
            )
            stationarity = velocity - nominal
            if active_indices:
                stationarity = stationarity + self.slack_penalty_weight * (
                    active_matrix.transpose(0, 1)
                    @ (active_matrix @ velocity + active_offset)
                )
            stationarity_residual = float(
                torch.linalg.vector_norm(stationarity, ord=float("inf"))
            )
            slack = torch.relu(-residual)
            objective = 0.5 * torch.sum((velocity - nominal).square()) + 0.5 * self.slack_penalty_weight * torch.sum(slack.square())
            objective_value = float(objective)
            kkt_valid = (
                linear_solve_residual <= tolerance
                and stationarity_residual <= tolerance
            )
            diagnostics.append(
                ActiveSetDiagnostic(
                    active_labels=tuple(
                        linearization.labels[index]
                        for index, active in enumerate(mask)
                        if active
                    ),
                    linear_solve_residual=linear_solve_residual,
                    stationarity_residual=stationarity_residual,
                    sign_valid=sign_valid,
                    kkt_valid=kkt_valid,
                    objective=objective_value,
                )
            )
            if sign_valid and kkt_valid:
                candidates.append(
                    (
                        objective_value,
                        tuple(float(value) for value in velocity),
                        velocity,
                        slack,
                    )
                )
        if len(diagnostics) != 2**count:
            raise ODEAllocContractError("CBF did not enumerate every active subset")
        if not candidates:
            raise ODEAllocContractError("CBF active-set solve found no consistent region")
        _, _, velocity, slack = min(candidates, key=lambda item: (item[0], item[1]))
        velocity = velocity - velocity.mean()
        slack = torch.relu(-(matrix @ velocity + offset))
        inequality_violation = (
            float(torch.max(torch.relu(-(matrix @ velocity + offset + slack))))
            if slack.numel()
            else 0.0
        )
        slack_identity_residual = (
            float(torch.max(torch.abs(slack - torch.relu(-(matrix @ velocity + offset)))))
            if slack.numel()
            else 0.0
        )
        max_constraint_residual = max(
            inequality_violation,
            slack_identity_residual,
            abs(float(velocity.mean())),
        )
        if max_constraint_residual > tolerance:
            raise ODEAllocContractError("CBF terminal residual exceeds R1 tolerance")
        return ProjectionResult(
            velocity=velocity,
            slack=slack,
            qp_cpu_seconds=time.perf_counter() - start,
            subset_diagnostics=tuple(diagnostics),
            max_constraint_residual=max_constraint_residual,
        )


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    feasible: bool
    objective: float
    event_id: str
    model_forward_calls: int = 0
    processed_tokens: int = 0

    def __post_init__(self) -> None:
        objective = finite("candidate objective", self.objective)
        if not self.event_id:
            raise ODEAllocContractError("candidate event identity is empty")
        for name, value in (
            ("model_forward_calls", self.model_forward_calls),
            ("processed_tokens", self.processed_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEAllocContractError(f"{name} must be non-negative integer")
        object.__setattr__(self, "objective", objective)


@dataclass(frozen=True, slots=True)
class EvaluationCost:
    """Observed call counts returned by an instrumented callback."""

    model_forward_calls: int = 0
    processed_tokens: int = 0
    backward_calls: int = 0
    constraint_vjp_calls: int = 0
    constraint_jvp_calls: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEAllocContractError(f"{name} must be non-negative integer")


@dataclass(frozen=True, slots=True)
class FieldEvaluation:
    velocity: torch.Tensor
    cost: EvaluationCost


@dataclass(frozen=True, slots=True)
class LinearizationEvaluation:
    value: ConstraintLinearization
    cost: EvaluationCost


@dataclass(frozen=True, slots=True)
class SolverEndpoint:
    q: torch.Tensor
    verdict: CandidateVerdict
    fallback_to_native: bool
    completed_steps: int
    projector_name: str


DirectionFn = Callable[[int, torch.Tensor], FieldEvaluation]
LinearizeFn = Callable[[int, torch.Tensor], LinearizationEvaluation]
VerdictFn = Callable[[torch.Tensor], CandidateVerdict]


class FixedGridSolver:
    """Run every Euler step/trial and select the best exact feasible endpoint."""

    def __init__(
        self,
        *,
        budget: SolverBudget,
        projector: VelocityProjector,
    ) -> None:
        self.budget = budget
        self.projector = projector

    @staticmethod
    def _record_verdict(ledger: ComputeLedger, verdict: CandidateVerdict) -> None:
        ledger.increment("quantized_trial_calls")
        ledger.increment("model_forward_calls", verdict.model_forward_calls)
        ledger.increment("processed_tokens", verdict.processed_tokens)
        if not verdict.feasible:
            ledger.increment("rejected_trials")

    @staticmethod
    def _record_cost(ledger: ComputeLedger, cost: EvaluationCost) -> None:
        for name in cost.__dataclass_fields__:
            ledger.increment(name, getattr(cost, name))

    def run(
        self,
        initial_q: torch.Tensor,
        *,
        direction_fn: DirectionFn,
        linearize_fn: LinearizeFn,
        verdict_fn: VerdictFn,
        ledger: ComputeLedger,
    ) -> SolverEndpoint:
        if initial_q.ndim != 1 or initial_q.dtype != torch.float64 or initial_q.device.type != "cpu":
            raise ODEAllocContractError("initial q must be a CPU float64 vector")
        if initial_q.numel() < 2 or not torch.isfinite(initial_q).all():
            raise ODEAllocContractError("initial q dimension or values are invalid")
        if abs(float(initial_q.mean())) > 1.0e-12:
            raise ODEAllocContractError("initial q must be zero-mean")
        started = time.perf_counter()
        q = initial_q.clone()
        native = verdict_fn(q.clone())
        self._record_verdict(ledger, native)
        best: tuple[torch.Tensor, CandidateVerdict] | None = (
            (q.clone(), native) if native.feasible else None
        )
        for step_index in range(self.budget.fixed_k):
            field = direction_fn(step_index, q.clone())
            if not isinstance(field, FieldEvaluation):
                raise ODEAllocContractError("direction callback omitted observed accounting")
            self._record_cost(ledger, field.cost)
            nominal = field.velocity.to(dtype=torch.float64, device="cpu")
            if nominal.shape != q.shape or not torch.isfinite(nominal).all():
                raise ODEAllocContractError("nominal allocation field is invalid")
            nominal = nominal - nominal.mean()
            observed_linearization = linearize_fn(step_index, q.clone())
            if not isinstance(observed_linearization, LinearizationEvaluation):
                raise ODEAllocContractError(
                    "constraint callback omitted observed accounting"
                )
            self._record_cost(ledger, observed_linearization.cost)
            linearization = observed_linearization.value
            if linearization.matrix.shape[1] != q.numel():
                raise ODEAllocContractError("constraint/q dimensions differ")
            projection = self.projector.project(nominal, linearization)
            if projection.velocity.shape != q.shape:
                raise ODEAllocContractError("projected allocation field has wrong shape")
            if projection.qp_cpu_seconds:
                ledger.increment("qp_calls")
                ledger.add_seconds("qp_cpu_seconds", projection.qp_cpu_seconds)
            candidates: list[tuple[torch.Tensor, CandidateVerdict]] = []
            for factor in self.budget.backtracking_factors:
                candidate_q = q + (
                    self.budget.euler_step * factor * projection.velocity
                )
                candidate_q = candidate_q - candidate_q.mean()
                verdict = verdict_fn(candidate_q.clone())
                self._record_verdict(ledger, verdict)
                candidates.append((candidate_q, verdict))
                if verdict.feasible and (
                    best is None or verdict.objective < best[1].objective
                ):
                    best = (candidate_q.clone(), verdict)
            accepted = next((item for item in candidates if item[1].feasible), None)
            if accepted is not None:
                q = accepted[0].clone()
        ledger.add_seconds("controller_wall_seconds", time.perf_counter() - started)
        if ledger.quantized_trial_calls != self.budget.quantized_trial_budget:
            raise ODEAllocContractError("solver did not exhaust the fixed trial budget")
        if best is None:
            return SolverEndpoint(
                q=initial_q.clone(),
                verdict=native,
                fallback_to_native=True,
                completed_steps=self.budget.fixed_k,
                projector_name=self.projector.name,
            )
        return SolverEndpoint(
            q=best[0],
            verdict=best[1],
            fallback_to_native=False,
            completed_steps=self.budget.fixed_k,
            projector_name=self.projector.name,
        )
