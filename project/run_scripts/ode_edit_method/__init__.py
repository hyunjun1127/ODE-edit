"""Reusable Session 02 ODE-Edit method controller.

This package lives entirely on the ODE-Edit side.  It deliberately treats
EasyEdit as a read-only actuator provider and keeps evaluation-only fields out
of controller-facing contracts.
"""

from .contracts import (
    ARM_ORDER,
    Arm,
    ArmRunResult,
    ControllerConfig,
    EventReading,
    LayerProposal,
    ProposalBatch,
    ProposalSemantics,
)
from .controller import OmegaLedger, ScalarFirstHitSearch, solve_progress_qp
from .derivatives import (
    ActuatorDirectionalHook,
    DirectionalField,
    all_layer_directional_derivatives,
    directional_gradient_scope,
)
from .functional_trial import LowRankFunctionalTrial
from .instrumentation import EditInstrumentation, InstrumentationSnapshot
from .retry import RejectedRetryCache
from .cached_graph import (
    MEMIT_PROPOSAL_CAPABILITIES,
    CachedAcceptedTrialSequence,
    assess_cached_graph_feasibility,
)

__all__ = [
    "ARM_ORDER",
    "Arm",
    "ArmRunResult",
    "ControllerConfig",
    "ActuatorDirectionalHook",
    "CachedAcceptedTrialSequence",
    "DirectionalField",
    "EditInstrumentation",
    "EventReading",
    "InstrumentationSnapshot",
    "LayerProposal",
    "LowRankFunctionalTrial",
    "MEMIT_PROPOSAL_CAPABILITIES",
    "OmegaLedger",
    "ProposalBatch",
    "ProposalSemantics",
    "ScalarFirstHitSearch",
    "RejectedRetryCache",
    "all_layer_directional_derivatives",
    "assess_cached_graph_feasibility",
    "directional_gradient_scope",
    "solve_progress_qp",
]
