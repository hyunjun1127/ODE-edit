"""Typed, outcome-independent contracts for the HA0--HA2 implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class EngineeringBoundary(RuntimeError):
    """Source, shape, dtype, transaction, or numerical implementation fault."""


class MethodBoundary(RuntimeError):
    """A local hard-Alpha structural condition is not feasible."""


class TrajectoryBoundary(RuntimeError):
    """A finite candidate failed the fixed-z completion trajectory contract."""


class FailureLabel(str, Enum):
    ALPHA_TANGENT_RANGE_INFEASIBLE = "ALPHA_TANGENT_RANGE_INFEASIBLE"
    ALPHA_SUFFIX_RANGE_INFEASIBLE = "ALPHA_SUFFIX_RANGE_INFEASIBLE"
    ALPHA_ZERO_GUIDE_AUTHORITY = "ALPHA_ZERO_GUIDE_AUTHORITY"
    FZCB_LOCAL_CBF_INFEASIBLE = "FZCB_LOCAL_CBF_INFEASIBLE"
    NONLINEAR_CORRECTOR_FAILURE = "NONLINEAR_CORRECTOR_FAILURE"
    DISCRETE_COMPLETION_BUDGET_VIOLATION = "DISCRETE_COMPLETION_BUDGET_VIOLATION"
    BACKTRACK_EXHAUSTED = "BACKTRACK_EXHAUSTED"
    TERMINAL_FIXED_Z_FAILURE = "TERMINAL_FIXED_Z_FAILURE"
    DENSE_ALPHA_SOLVE_RESIDUAL_FAILURE = "DENSE_ALPHA_SOLVE_RESIDUAL_FAILURE"
    SENSITIVITY_NUMERICALLY_INCONCLUSIVE = "SENSITIVITY_NUMERICALLY_INCONCLUSIVE"
    PROJECTOR_ORIENTATION_FAILURE = "PROJECTOR_ORIENTATION_FAILURE"
    CACHE_NONFINITE_OR_CORRUPT = "CACHE_NONFINITE_OR_CORRUPT"
    CACHE_TRANSACTION_FAILURE = "CACHE_TRANSACTION_FAILURE"


@dataclass(frozen=True)
class NumericalLock:
    """Pre-outcome technical numerical schedule.

    The tolerances are precision-derived implementation tolerances, not
    scientific or performance thresholds.  Model-scale activation closure is
    additionally checked against the already sealed per-model FP32 endpoint
    tolerance.
    """

    schema: str = "odeedit.s06.fzcb-hard-alpha-densec.numerical-lock.v1"
    dense_dtype: str = "float32"
    reduction_dtype: str = "float64"
    dense_factorization: str = "torch.linalg.lu_factor_ex/lu_solve"
    right_solve: str = "torch.linalg.solve(S.T,Q.T).T"
    equality_solver: str = "matrix_free_lsqr"
    explicit_gram_count: int = 0
    explicit_pseudoinverse_count: int = 0
    explicit_null_basis_count: int = 0
    explicit_kronecker_count: int = 0
    dense_inverse_count: int = 0
    initial_progress_step: float = 0.25
    backtrack_factor: float = 0.5
    max_backtracks: int = 4
    corrector_max_iterations: int = 4
    corrector_line_search_factor: float = 0.5
    corrector_line_search_trials: int = 4
    cbf_kappa: float = 1.0
    lsqr_max_iterations: int = 96
    # 4096 * IEEE FP64 epsilon.
    fp64_relative_tolerance: float = 9.094947017729282e-13
    # 4096 * IEEE FP32 epsilon.  Used for backend residual/parity only.
    fp32_backend_relative_tolerance: float = 4.8828125e-4
    # 32 * sqrt(FP64 epsilon); scalar discriminant/zero authority only.
    fp64_scalar_tolerance: float = 4.76837158203125e-7
    terminal_policy: str = "EXACT_CLOSURE_ZERO_ELSE_INFINITY"
    cache_rollback: str = "IMMUTABLE_CHECKPOINT_PLUS_APPEND_ONLY_WAL_REPLAY"
    seed: int = 20260901

    def payload(self) -> dict[str, Any]:
        return asdict(self)
