"""Functional H/P verifier receipts; structural proxies never authorize a trial."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import ODEBFContractError, canonical_hash, finite, nonnegative, positive


@dataclass(frozen=True, slots=True)
class ReplayRiskReceipt:
    barrier: str
    item_count: int
    mean_positive_damage: float
    smooth_max_positive_damage: float
    raw_max_positive_damage: float
    signed_mean_damage: float
    budget: float
    passed: bool
    cvar_diagnostic: float | None
    cvar_decision_enabled: bool
    sample_order_sha256: str

    def __post_init__(self) -> None:
        if self.barrier not in ("historical-current-teacher", "pretrained-theta0-teacher"):
            raise ODEBFContractError("functional replay barrier identity differs")
        if self.item_count < 0:
            raise ODEBFContractError("functional replay item count is invalid")
        for name in (
            "mean_positive_damage",
            "smooth_max_positive_damage",
            "raw_max_positive_damage",
            "budget",
        ):
            nonnegative(name, getattr(self, name))
        finite("signed_mean_damage", self.signed_mean_damage)
        if self.cvar_decision_enabled:
            raise ODEBFContractError("CVaR is not a V1/P1 decision gate")
        if self.cvar_diagnostic is not None:
            nonnegative("cvar_diagnostic", self.cvar_diagnostic)
        if len(self.sample_order_sha256) != 64:
            raise ODEBFContractError("functional replay order identity is invalid")


def _smooth_max(values: np.ndarray, temperature: float) -> float:
    tau = positive("smooth-max temperature", temperature)
    if values.size == 0:
        return 0.0
    maximum = float(values.max())
    return maximum + tau * math.log(float(np.exp((values - maximum) / tau).mean()))


def functional_replay_risk(
    *,
    barrier: str,
    entry_values: Sequence[float],
    trial_values: Sequence[float],
    sample_sha256: Sequence[str],
    budget: float,
    smooth_max_temperature: float = 1.0e-2,
) -> ReplayRiskReceipt:
    if len(entry_values) != len(trial_values) or len(entry_values) != len(sample_sha256):
        raise ODEBFContractError("functional replay arrays/orders differ")
    if len(set(sample_sha256)) != len(sample_sha256) or any(
        len(identity) != 64 for identity in sample_sha256
    ):
        raise ODEBFContractError("functional replay sample identities differ")
    entry = np.asarray(entry_values, dtype=np.float64)
    trial = np.asarray(trial_values, dtype=np.float64)
    if entry.ndim != 1 or trial.ndim != 1 or not np.isfinite(entry).all() or not np.isfinite(trial).all():
        raise ODEBFContractError("functional replay values must be finite vectors")
    signed = trial - entry
    positive_part = np.maximum(signed, 0.0)
    mean_positive = float(positive_part.mean()) if positive_part.size else 0.0
    smooth = _smooth_max(positive_part, smooth_max_temperature)
    raw_max = float(positive_part.max()) if positive_part.size else 0.0
    locked_budget = nonnegative("functional replay budget", budget)

    # CVaR is diagnostic only, and even that requires at least eight history
    # items and at least two members in the deterministic upper tail.
    tail_count = max(1, math.ceil(0.2 * positive_part.size)) if positive_part.size else 0
    cvar = None
    if positive_part.size >= 8 and tail_count >= 2:
        cvar = float(np.sort(positive_part)[-tail_count:].mean())
    passed = (
        mean_positive <= locked_budget
        and smooth <= locked_budget
        and raw_max <= locked_budget
    )
    return ReplayRiskReceipt(
        barrier,
        int(positive_part.size),
        mean_positive,
        smooth,
        raw_max,
        float(signed.mean()) if signed.size else 0.0,
        locked_budget,
        passed,
        cvar,
        False,
        canonical_hash(list(sample_sha256)),
    )


@dataclass(frozen=True, slots=True)
class FunctionalHPVerdict:
    historical: ReplayRiskReceipt
    pretrained: ReplayRiskReceipt
    structural_h_pass: bool
    structural_p_pass: bool
    trust_pass: bool

    def __post_init__(self) -> None:
        if self.historical.barrier != "historical-current-teacher":
            raise ODEBFContractError("H verifier uses the wrong teacher")
        if self.pretrained.barrier != "pretrained-theta0-teacher":
            raise ODEBFContractError("P verifier uses the wrong teacher")

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.structural_h_pass,
                self.structural_p_pass,
                self.trust_pass,
                self.historical.passed,
                self.pretrained.passed,
            )
        )

    def identity(self) -> str:
        return canonical_hash(
            {
                "historical": self.historical.sample_order_sha256,
                "pretrained": self.pretrained.sample_order_sha256,
                "historical_pass": self.historical.passed,
                "pretrained_pass": self.pretrained.passed,
                "structural_h_pass": self.structural_h_pass,
                "structural_p_pass": self.structural_p_pass,
                "trust_pass": self.trust_pass,
                "accepted": self.accepted,
            }
        )
