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
    ScalarGateDirectionalReference,
    all_layer_directional_derivatives,
    assert_scalar_gate_matches_hook,
    directional_gradient_scope,
)
from .functional_trial import LowRankFunctionalTrial
from .hooks import apply_accepted_factors
from .instrumentation import EditInstrumentation, InstrumentationSnapshot
from .lock import controller_config, dry_plan, load_lock
from .retry import RejectedRetryCache
from .runtime import FiveArmRunner, MethodBackend
from .cached_graph import (
    MEMIT_PROPOSAL_CAPABILITIES,
    CachedAcceptedTrialSequence,
    assess_cached_graph_feasibility,
)
from .easyedit_backend import EasyEditMemitBackend
from .preflight import PreparedConcreteEnvironment

__all__ = [
    "ARM_ORDER",
    "Arm",
    "ArmRunResult",
    "ControllerConfig",
    "ActuatorDirectionalHook",
    "CachedAcceptedTrialSequence",
    "DirectionalField",
    "EditInstrumentation",
    "EasyEditMemitBackend",
    "EventReading",
    "InstrumentationSnapshot",
    "LayerProposal",
    "LowRankFunctionalTrial",
    "FiveArmRunner",
    "MethodBackend",
    "MEMIT_PROPOSAL_CAPABILITIES",
    "OmegaLedger",
    "ProposalBatch",
    "ProposalSemantics",
    "PreparedConcreteEnvironment",
    "ScalarFirstHitSearch",
    "ScalarGateDirectionalReference",
    "RejectedRetryCache",
    "all_layer_directional_derivatives",
    "assert_scalar_gate_matches_hook",
    "apply_accepted_factors",
    "assess_cached_graph_feasibility",
    "directional_gradient_scope",
    "controller_config",
    "dry_plan",
    "load_lock",
    "solve_progress_qp",
]
