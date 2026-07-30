"""Reusable ODE-Edit motivation diagnostic primitives."""

from .artifacts import JsonlArtifactWriter
from .contracts import (
    ContextManifest,
    EditRequest,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    ProvenanceManifest,
    SnapshotManifest,
    preflight_pinned_files,
    sanitize_edit_requests,
)
from .direct_z import DirectZCache, FrozenDirectZ
from .diagnostic_math import (
    NativeMemitSolver,
    WoodburyMemitSolver,
    c_inner_product,
    c_norm,
    exact_top1_stop,
    finite_difference_calibration,
    joint_additivity,
    rank_turnover,
    reroutability,
    smooth_target_utility,
)
from .easyedit_bridge import (
    CovarianceCacheMissError,
    CovarianceCacheSpec,
    EasyEditBridge,
)
from .hooks import (
    ForwardCapture,
    TemporaryExactMemitApplication,
    TemporaryLowRankApplication,
)

__all__ = [
    "ContextManifest",
    "CovarianceCacheMissError",
    "CovarianceCacheSpec",
    "DirectZCache",
    "EasyEditBridge",
    "EditRequest",
    "ExpectedFileIdentity",
    "ForwardCapture",
    "FrozenDirectZ",
    "JsonlArtifactWriter",
    "LowRankFactor",
    "MemitFactorProposal",
    "NativeMemitSolver",
    "ProposalSemantics",
    "ProvenanceManifest",
    "SnapshotManifest",
    "TemporaryLowRankApplication",
    "TemporaryExactMemitApplication",
    "WoodburyMemitSolver",
    "c_inner_product",
    "c_norm",
    "exact_top1_stop",
    "finite_difference_calibration",
    "joint_additivity",
    "preflight_pinned_files",
    "rank_turnover",
    "reroutability",
    "sanitize_edit_requests",
    "smooth_target_utility",
]
