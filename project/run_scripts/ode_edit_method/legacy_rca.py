"""Read-only R2 trajectory pins for the V2 P1 legacy RCA replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.ode_edit_motivation.contracts import ExpectedFileIdentity

from .continuation import (
    CANONICAL_CASE_ORDER,
    DEFAULT_R2_SOURCE_SPECS,
    R2_PROPOSAL_ID,
    ContinuationSelection,
    ValidatedR2Source,
    _strict_jsonl,
    validate_r2_sources,
)
from .contracts import Arm, MethodContractError


LEGACY_RCA_ARMS = (Arm.NATIVE_MEMIT, Arm.FULL_ODE_EDIT)


@dataclass(frozen=True, slots=True)
class LegacyR2Trajectory:
    model_alias: str
    arm: Arm
    case_id: str
    order_position: int
    controller_record: Mapping[str, Any]
    compute_record: Mapping[str, Any]
    direct_z_path: Path
    direct_z_identity: ExpectedFileIdentity
    direct_z_tensor_sha256: str
    direct_z_source_state_id: str

    def raw_free_identity(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "arm": self.arm.value,
            "case_id": self.case_id,
            "order_position": self.order_position,
            "direct_z_relative_path": str(
                self.direct_z_path.relative_to(self.direct_z_path.parents[1])
            ),
            "direct_z_sha256": self.direct_z_identity.sha256,
            "direct_z_size": self.direct_z_identity.size,
            "direct_z_tensor_sha256": self.direct_z_tensor_sha256,
            "direct_z_source_state_id": self.direct_z_source_state_id,
        }


def _source_records(
    source: ValidatedR2Source,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    controller = _strict_jsonl(source.spec.root / "controller_steps.jsonl")
    compute = _strict_jsonl(source.spec.root / "compute.jsonl")
    controller_by_key = {
        (str(row.get("arm")), str(row.get("case_id"))): row
        for row in controller
    }
    compute_by_key = {
        (str(row.get("arm")), str(row.get("case_id"))): row for row in compute
    }
    if len(controller_by_key) != 16 or len(compute_by_key) != 16:
        raise MethodContractError("R2 legacy RCA record matrix differs")
    return controller_by_key, compute_by_key


def legacy_r2_trajectories(
    repo: str | Path,
    model_alias: str,
    *,
    selection: ContinuationSelection | None = None,
) -> tuple[LegacyR2Trajectory, ...]:
    root = Path(repo).resolve(strict=True)
    locked = selection or validate_r2_sources(root, DEFAULT_R2_SOURCE_SPECS)
    source = locked.source_for_alias(model_alias)
    controller, compute = _source_records(source)
    rows: list[LegacyR2Trajectory] = []
    for arm in LEGACY_RCA_ARMS:
        for order_position, case_id in enumerate(CANONICAL_CASE_ORDER):
            key = (arm.value, case_id)
            record = controller.get(key)
            compute_record = compute.get(key)
            if not isinstance(record, Mapping) or not isinstance(
                compute_record, Mapping
            ):
                raise MethodContractError("R2 legacy RCA row is absent")
            if (
                record.get("order_position") != order_position
                or compute_record.get("order_position") != order_position
                or record.get("hashes", {}).get("proposal_id") != R2_PROPOSAL_ID
                or record.get("status") != record.get("result", {}).get("status")
                or record.get("pre_edit_state_id")
                != record.get("hashes", {}).get("direct_z", {}).get(
                    "source_state_id"
                )
                or compute_record.get("counters", {}).get("N_eval") != 0
            ):
                raise MethodContractError("R2 legacy RCA row provenance differs")
            relative = (
                f"direct_z/{model_alias}-{arm.value}-position-{order_position}-"
                f"case-{case_id}.pt"
            )
            identity = source.identity_for(relative)
            direct = record["hashes"]["direct_z"]
            if (
                direct.get("artifact_sha256") != identity.sha256
                or direct.get("artifact_size") != identity.size
            ):
                raise MethodContractError("R2 legacy direct-z identity differs")
            rows.append(
                LegacyR2Trajectory(
                    model_alias=model_alias,
                    arm=arm,
                    case_id=case_id,
                    order_position=order_position,
                    controller_record=record,
                    compute_record=compute_record,
                    direct_z_path=source.spec.root / relative,
                    direct_z_identity=identity,
                    direct_z_tensor_sha256=str(direct["tensor_sha256"]),
                    direct_z_source_state_id=str(direct["source_state_id"]),
                )
            )
    return tuple(rows)


__all__ = [
    "LEGACY_RCA_ARMS",
    "LegacyR2Trajectory",
    "legacy_r2_trajectories",
]
