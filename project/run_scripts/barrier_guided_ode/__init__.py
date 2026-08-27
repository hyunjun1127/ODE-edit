"""CPU scientific core for BGODE-R1.

BGODE-R1 is deliberately a one-request method.  This package contains only
the event-space and equality-Rayleighian mathematics needed before a model
runtime is authorized.  It does not import EasyEdit or mutate model weights.
"""

from .errors import (
    BGODEScientificBoundary,
    NumericalRankBoundary,
    UnsupportedBatchBoundary,
)
from .event_moments import EventScoreMoments, aggregate_event_score_moments
from .event_trie import (
    EventDistribution,
    EventKind,
    JointFirstDepartureTrie,
    TerminationConvention,
)
from .policies import NativeBypassReceipt, explicit_native_bypass
from .rayleighian_controller import RayleighianSolution, solve_equality_rayleighian
from .reference_path import SingleCoordinateReference, single_coordinate_reference

__all__ = [
    "BGODEScientificBoundary",
    "EventDistribution",
    "EventKind",
    "EventScoreMoments",
    "JointFirstDepartureTrie",
    "NativeBypassReceipt",
    "NumericalRankBoundary",
    "RayleighianSolution",
    "SingleCoordinateReference",
    "TerminationConvention",
    "UnsupportedBatchBoundary",
    "aggregate_event_score_moments",
    "explicit_native_bypass",
    "single_coordinate_reference",
    "solve_equality_rayleighian",
]
