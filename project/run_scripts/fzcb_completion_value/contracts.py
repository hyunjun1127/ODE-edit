"""Immutable execution contracts for the completion-value fast-kill screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS


class FzCBError(RuntimeError):
    """Base fail-close error."""


class TechnicalBoundary(FzCBError):
    """Deployment, identity, or instrumentation failure."""


class ScientificBoundary(FzCBError):
    """Frozen scientific equality/action/value contract failure."""


@dataclass(frozen=True, slots=True)
class NumericalLock:
    dtype: str = "float32"
    progress_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    macro_step: float = 0.25
    k_steps: int = 4
    k0_random_seed_count: int = 4
    k1_random_seed_count: int = 6
    candidate_rho: float = 0.1
    sensitivity_rho: float = 0.05
    direct_z_compute_count: int = 1
    corrector_iterations: int = 2
    cg_max_iterations: int = 24
    # Dimension-aware FP32 bounds are mechanical and locked before output.
    relative_floor_rule: str = "64*eps_fp32*sqrt(hidden_dimension)"
    absolute_floor_rule: str = "64*eps_fp32*sqrt(hidden_dimension)*(1+reference_norm)"
    rank_rule: str = "max(m,n)*eps_fp32"
    metric_epsilon_rule: str = "256*eps_fp32*mean(diag(C_l^0))"
    candidate_tie_policy: str = "finite_before_infinite_then_lexical_candidate_id"
    seed_namespace: str = "ODEEDIT-S06-FZCB-COMPLETION-VALUE-FAST-KILL-V1"
    dense_inverse_count: int = 0
    explicit_kronecker_count: int = 0
    value_gradient_count: int = 0
    cbf_count: int = 0
    full_qcqp_count: int = 0
    alphaedit_count: int = 0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


CONTRACT_PATH = Path(
    "/data/janghj/ODE-edit/local/state/fzcb-completion-value-fast-kill-v1/authoritative-contract.txt"
)
CONTRACT_SHA256 = "ac16a691280dc224c4a98640b46887ba46f9d6507c2dceda43818aae77a0b78d"
CONTRACT_BYTES = 16491
CONTRACT_LINES = 773
DATASET = Path("/data/janghj/EasyEdit/data/counterfact/counterfact.json")
DATASET_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
CASE_MANIFEST = Path("project/run_scripts/fixed_z_nonuniqueness/config/case-manifest-v1.json")
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit-stock-14cea824")
EASYEDIT_HEAD = "14cea8245f06715684592ab55184939b99d70784"
EASYEDIT_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"

# The first four sealed controller rows are K0 engineering-only.  The sealed
# screen rows are disjoint and become the exact K1 B1-8 denominator.
K0_CASE_IDS = (14148, 16872, 4164, 3002)
K1_CASE_IDS = (21100, 1477, 20838, 18707, 14288, 16426, 19041, 17609)

MODEL_ALIASES = tuple(sorted(MODEL_SPECS))
EDITABLE_LAYERS = (4, 5, 6, 7, 8)
