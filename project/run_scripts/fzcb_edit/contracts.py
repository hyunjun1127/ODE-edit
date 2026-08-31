"""Immutable method, deployment, and numerical contracts for FzCB-Edit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FzCBEditError(RuntimeError):
    """Fail-close base error."""


class TechnicalBoundary(FzCBEditError):
    """Deployment, identity, or instrumentation invariant failed."""


class ScientificBoundary(FzCBEditError):
    """A frozen equality, range, action, or barrier contract failed."""


class SubspaceInconclusive(ScientificBoundary):
    """A reduced sensitivity sketch cannot establish a full-space verdict."""


class NumericalSensitivityFailure(ScientificBoundary):
    """Finite-difference sensitivity is not stable under the sealed sweep."""


class Arm(str, Enum):
    OFFICIAL_MEMIT = "OFFICIAL_MEMIT"
    TRUE_FROZEN_C_SPLIT = "TRUE_FROZEN_C_SPLIT"
    REFRESHED_EQUALITY_ONLY = "REFRESHED_EQUALITY_ONLY"
    FZCB = "FZCB"
    STRONG_STATIC_SAME_OBJECTIVE = "STRONG_STATIC_SAME_OBJECTIVE"
    # Source-compatibility aliases for the preserved pre-TECH-R1 runtime only.
    FROZEN_SPLIT = "TRUE_FROZEN_C_SPLIT"
    STATIC_PATH = "STRONG_STATIC_SAME_OBJECTIVE"


@dataclass(frozen=True, slots=True)
class NumericalLock:
    dtype: str = "float32"
    progress_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    macro_step: float = 0.25
    backtrack_factor: float = 0.5
    maximum_backtracks: int = 3
    corrector_iterations: int = 2
    sketch_ladder: tuple[int, ...] = (2, 8, 32, 128)
    finite_difference_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    finite_difference_repeats: int = 2
    # sqrt(eps_fp32), fixed before GPU output and used only for the explicitly
    # labeled reduced-null finite-difference correctness prototype.
    finite_difference_step: float = 0.00034526698300124393
    kappa: float = 0.0
    cg_max_iterations: int = 768
    cg_internal_refinement_count: int = 2
    relative_floor_rule: str = "64*eps_fp32*sqrt(output_dimension)"
    rank_rule: str = "max(rows,cols)*eps_fp32*largest_eigenvalue"
    metric_epsilon_rule: str = "256*eps_fp32*mean(diag(C_l^0))"
    sensitivity_method: str = "NESTED_NULL_SKETCH_WITH_DIMENSION_CORRECTION"
    direct_z_rule: str = "exactly_once_per_edit_shared_by_all_arms"
    terminal_commit_rule: str = "s_equals_1_closure_and_budget_then_atomic_commit"
    dense_inverse_count: int = 0
    explicit_kronecker_count: int = 0
    controller_output_metric_access_count: int = 0
    seed_namespace: str = "ODEEDIT-S06-FZCB-TECH-R1-JOINT-B1-CONTROLLER-VALIDITY-AUDIT-V1"

    def payload(self) -> dict[str, Any]:
        return asdict(self)


CONTRACT_PATH = Path(
    "/data/janghj/ODE-edit/local/state/fzcb-tech-r1-joint-b1-controller-validity-audit-v1/authoritative-contract.txt"
)
CONTRACT_SHA256 = "4356a2cfea04687f25a2972e3a3f4636d57d4140d76dd2db2913b5d74d0467ac"
CONTRACT_BYTES = 12248
CONTRACT_LINES = 292
PROPOSAL_PATH = Path("project/proposals/2026-08-31-fzcb-edit-method-pivot-proposal.md")
PROPOSAL_SHA256 = "47666dada9a95be2a4d4414dcb826990b94150c8dacea08a763ed063049a3853"
PROPOSAL_BYTES = 56759
PROPOSAL_LINES = 2372
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit-stock-14cea824")
EASYEDIT_HEAD = "14cea8245f06715684592ab55184939b99d70784"
EASYEDIT_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"
STREAM_ROOT = Path(
    "/data/janghj/ODE-edit/local/state/phase123-server1-sample-stream-v1/extracted-v1/"
    "phase123-server1-single-canonical-stream-v1"
)
STREAM_SHA256 = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
ORDER_SHA256 = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
SAMPLE_PAYLOAD_SHA256 = "fe7c8b0cb51abf591e0ec560c0009bdcc47f20e8c4efcf6efc3311710a373475"
EXPECTED_BASE_HEAD = "ddc178584ef14efd5d4e1271b3c324e3ebd3e443"
EXPECTED_BASE_TREE = "83889fc3d2e32119dafdffc969482de761eedfe2"
