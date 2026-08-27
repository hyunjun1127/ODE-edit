"""BGODE-R2 termination-free event geometry and FP64 controller core."""

from .controller import (
    PlainMinimumNormSolution,
    R2NumericalReceipt,
    R2RayleighianSolution,
    solve_plain_minimum_norm,
    solve_two_equality_rayleighian,
)
from .events import PrefixEventEvaluation, PrefixEventLayout, evaluate_prefix_events
from .moments import R2EventMoments, aggregate_r2_event_moments, barrier_decomposition

__all__ = [
    "PlainMinimumNormSolution",
    "PrefixEventEvaluation",
    "PrefixEventLayout",
    "R2EventMoments",
    "R2NumericalReceipt",
    "R2RayleighianSolution",
    "aggregate_r2_event_moments",
    "barrier_decomposition",
    "evaluate_prefix_events",
    "solve_plain_minimum_norm",
    "solve_two_equality_rayleighian",
]
