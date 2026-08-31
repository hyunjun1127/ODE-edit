"""Unit-separated controller tolerances sealed before joint B1 execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .contracts import ScientificBoundary, TechnicalBoundary
from .hashing import canonical_hash, file_sha256


LOCK_PATH = Path(__file__).with_name("config") / "controller-validity-tolerances-tech-r1-v1.json"


@dataclass(frozen=True, slots=True)
class UnitTolerancePolicy:
    tau_range: float
    tau_null: float
    tau_z: float
    tau_cbf_over_a0: float
    tau_budget_over_a0: float
    tau_grad_over_sqrt_a0: float
    fd_base_epsilon: float
    fd_multipliers: tuple[float, ...]
    fd_repeats: int
    calibration_rule: str

    @classmethod
    def load(cls, path: Path = LOCK_PATH) -> "UnitTolerancePolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = cls(
            tau_range=float(payload["tau_range"]),
            tau_null=float(payload["tau_null"]),
            tau_z=float(payload["tau_z"]),
            tau_cbf_over_a0=float(payload["tau_cbf_over_a0"]),
            tau_budget_over_a0=float(payload["tau_budget_over_a0"]),
            tau_grad_over_sqrt_a0=float(payload["tau_grad_over_sqrt_a0"]),
            fd_base_epsilon=float(payload["fd_base_epsilon"]),
            fd_multipliers=tuple(float(value) for value in payload["fd_multipliers"]),
            fd_repeats=int(payload["fd_repeats"]),
            calibration_rule=str(payload["calibration_rule"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        values = (
            self.tau_range, self.tau_null, self.tau_z, self.tau_cbf_over_a0,
            self.tau_budget_over_a0, self.tau_grad_over_sqrt_a0, self.fd_base_epsilon,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise TechnicalBoundary("unit tolerance lock contains non-positive/nonfinite values")
        if self.fd_multipliers != (0.25, 0.5, 1.0, 2.0) or self.fd_repeats != 2:
            raise TechnicalBoundary("FD sweep/repeat lock differs from TECH-R1 contract")

    def resolve(self, initial_budget: float) -> "ResolvedTolerances":
        if not math.isfinite(initial_budget) or initial_budget <= 0.0:
            raise ScientificBoundary("initial action budget must be finite positive")
        return ResolvedTolerances(
            tau_range=self.tau_range,
            tau_null=self.tau_null,
            tau_z=self.tau_z,
            tau_cbf=self.tau_cbf_over_a0 * initial_budget,
            tau_budget=self.tau_budget_over_a0 * initial_budget,
            tau_grad=self.tau_grad_over_sqrt_a0 * math.sqrt(initial_budget),
            initial_budget=initial_budget,
            policy_identity=canonical_hash(asdict(self)),
        )

    def receipt(self, path: Path = LOCK_PATH) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "policy": asdict(self),
            "policy_identity": canonical_hash(asdict(self)),
            "sealed_before_b1": True,
            "outcome_dependent_adjustment_count": 0,
        }


@dataclass(frozen=True, slots=True)
class ResolvedTolerances:
    tau_range: float
    tau_null: float
    tau_z: float
    tau_cbf: float
    tau_budget: float
    tau_grad: float
    initial_budget: float
    policy_identity: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def synthetic_calibration_receipt(policy: UnitTolerancePolicy) -> dict[str, Any]:
    """A deterministic no-op/repeat/precision calibration with no model outcome."""

    eps = float(torch.finfo(torch.float32).eps)
    matrix = torch.tensor([[1.0, 0.25], [0.25, 1.5]], dtype=torch.float32)
    rhs = torch.tensor([0.5, -0.25], dtype=torch.float32)
    first = torch.linalg.solve(matrix, rhs)
    second = torch.linalg.solve(matrix, rhs)
    repeat_noise = float(torch.linalg.vector_norm(first - second).item())
    no_op_noise = float(torch.linalg.vector_norm((matrix @ torch.zeros_like(rhs))).item())
    return {
        "schema": "odeedit.s06.fzcb-tech-r1.tolerance-calibration.v1",
        "policy": policy.receipt(),
        "fp32_epsilon": eps,
        "synthetic_no_op_noise": no_op_noise,
        "synthetic_repeat_solve_noise": repeat_noise,
        "precision_multipliers": {
            "range_and_null": policy.tau_range / eps,
            "activation": policy.tau_z / eps,
            "action_rate_relative": policy.tau_cbf_over_a0 / eps,
            "budget_relative": policy.tau_budget_over_a0 / eps,
            "gradient_relative": policy.tau_grad_over_sqrt_a0 / eps,
        },
        "unit_separation": {
            "dimensionless": ["tau_range", "tau_null"],
            "activation": ["tau_z"],
            "action_rate": ["tau_cbf"],
            "accumulated_action": ["tau_budget"],
            "projected_sensitivity": ["tau_grad"],
        },
        "model_result_access_count": 0,
    }

