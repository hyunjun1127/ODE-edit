"""Fail-closed lineage for reusing one W0 direct-z target at descendants.

``FrozenDirectZ`` deliberately binds its tensor artifact to the model snapshot
where EasyEdit computed it.  MV-2 must keep that semantic target fixed while
relinearizing factor directions at W1.  The quarter-step Motivation diagnostic
extends the same invariant to four verified descendants.  This module records
that distinction explicitly: the target remains a W0 artifact, while an
immutable ancestry receipt authorizes using the same bytes at an exact
descendant.

The lineage contains hashes and scalar metadata only.  It never serializes
direct-z values, target tokens, prompts, or model weights.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import (
    ContractError,
    MemitFactorProposal,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)
from .direct_z import FrozenDirectZ
from .hooks import TemporaryLowRankApplication, tensor_sha256


FROZEN_TARGET_LINEAGE_SCHEMA = "ode-edit-frozen-target-lineage/v1"
QUARTER_STEP_LINEAGE_SCHEMA = "ode-edit-quarter-step-target-lineage/v1"
ADAPTIVE_STEP_LINEAGE_SCHEMA = "ode-edit-adaptive-step-target-lineage/v1"
_ALLOWED_HOPS = {
    ("h0_sham", 0.0),
    ("partial_joint", 0.5),
}
_QUARTER_STEP_LABELS = frozenset(
    {
        "common_step_1",
        *(f"refreshed_step_{index}" for index in range(2, 5)),
        *(f"coefficient_step_{index}" for index in range(2, 5)),
        *(f"fixed_step_{index}" for index in range(2, 5)),
        *(f"native_step_{index}" for index in range(1, 5)),
    }
)
_ADAPTIVE_STEP_LABELS = tuple(f"capacity_round_{index}" for index in range(1, 5))
_ADAPTIVE_STEP_SCALE_TOLERANCE = 1e-5


def _is_adaptive_step(label: str, scale: float) -> bool:
    return (
        label in _ADAPTIVE_STEP_LABELS
        and 0.0 < scale <= 0.25 + _ADAPTIVE_STEP_SCALE_TOLERANCE
    )


def _full_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a full SHA-256 digest")
    normalized = value.lower()
    if (
        len(normalized) != 64
        or any(character not in string.hexdigits for character in normalized)
    ):
        raise ContractError(f"{name} must be a full SHA-256 digest")
    return normalized


def _finite_scale(value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError("lineage step scale must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("lineage step scale must be finite") from exc
    if not math.isfinite(normalized):
        raise ContractError("lineage step scale must be finite")
    return normalized


def _token_identity(target_token_ids: torch.Tensor) -> tuple[str, tuple[int, ...], str]:
    if (
        not isinstance(target_token_ids, torch.Tensor)
        or target_token_ids.ndim != 1
        or target_token_ids.numel() <= 0
        or target_token_ids.dtype
        not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    ):
        raise ContractError("target token IDs must be one non-empty integer vector")
    frozen = target_token_ids.detach().cpu().contiguous()
    return tensor_sha256(frozen), tuple(frozen.shape), str(frozen.dtype)


def _parameter_hashes(snapshot: SnapshotManifest) -> dict[str, str]:
    return {record.name: record.sha256 for record in snapshot.parameters}


def proposal_direction_payload(proposal: MemitFactorProposal) -> dict[str, Any]:
    """Return the exact tensor-byte identity of one proposal direction."""

    if not isinstance(proposal, MemitFactorProposal):
        raise ContractError("proposal direction identity requires MemitFactorProposal")
    return {
        "semantics": proposal.semantics.value,
        "residual_denominator": proposal.residual_denominator,
        "factors": [
            {
                "weight_name": factor.weight_name,
                "left_sha256": tensor_sha256(factor.left),
                "right_sha256": tensor_sha256(factor.right),
                "left_shape": list(factor.left.shape),
                "right_shape": list(factor.right.shape),
                "left_dtype": str(factor.left.dtype),
                "right_dtype": str(factor.right.dtype),
                "native_update_transposed": factor.native_update_transposed,
            }
            for factor in proposal.factors
        ],
    }


def proposal_direction_hash(proposal: MemitFactorProposal) -> str:
    return sha256_bytes(
        canonical_json(proposal_direction_payload(proposal)).encode("utf-8")
    )


@dataclass(frozen=True, slots=True)
class TargetLineageHop:
    """One fully observed model-state transition under the frozen target."""

    label: str
    step_scale: float
    parent_snapshot_id: str
    parent_state_id: str
    child_snapshot_id: str
    child_state_id: str
    action_hash: str
    child_parameter_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        scale = _finite_scale(self.step_scale)
        if (
            (self.label, scale) not in _ALLOWED_HOPS
            and not (self.label in _QUARTER_STEP_LABELS and scale == 0.25)
            and not _is_adaptive_step(self.label, scale)
        ):
            raise ContractError("target-lineage hop is outside the fixed MV-2 envelope")
        for field_name in (
            "parent_snapshot_id",
            "parent_state_id",
            "child_snapshot_id",
            "child_state_id",
            "action_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _full_sha256(field_name, getattr(self, field_name)),
            )
        hashes = tuple(self.child_parameter_hashes)
        names = [name for name, _digest in hashes]
        if (
            not hashes
            or len(names) != len(set(names))
            or names != sorted(names)
        ):
            raise ContractError(
                "lineage child parameter hashes must be non-empty, unique, and sorted"
            )
        normalized = tuple(
            (name, _full_sha256(f"child parameter {name}", digest))
            for name, digest in hashes
        )
        if scale == 0.0 and self.child_state_id != self.parent_state_id:
            raise ContractError("h=0 lineage must preserve exact state identity")
        if scale > 0.0 and self.child_state_id == self.parent_state_id:
            raise ContractError("positive-step lineage must reach a distinct descendant state")
        object.__setattr__(self, "step_scale", scale)
        object.__setattr__(self, "child_parameter_hashes", normalized)

    @property
    def hop_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "step_scale": self.step_scale,
            "parent_snapshot_id": self.parent_snapshot_id,
            "parent_state_id": self.parent_state_id,
            "child_snapshot_id": self.child_snapshot_id,
            "child_state_id": self.child_state_id,
            "action_hash": self.action_hash,
            "child_parameter_hashes": dict(self.child_parameter_hashes),
        }


@dataclass(frozen=True, slots=True)
class FrozenTargetLineage:
    """Identity of a W0 target plus a verified bounded ancestry.

    The original MV-2 envelope remains exactly zero or one ``h=1/2`` hop.  The
    independent quarter-step envelope permits exactly chained ``h=1/4`` hops,
    up to four.  The capacity controller has a separate envelope of up to
    three accepted positive hops, each no larger than ``h=1/4``.  Mixing
    envelopes is rejected.
    """

    model_id: str
    context_id: str
    request_ids: tuple[str, ...]
    hparams_sha256: str
    origin_snapshot_id: str
    origin_state_id: str
    direct_z_tensor_sha256: str
    direct_z_artifact_sha256: str
    direct_z_artifact_size: int
    target_token_sha256: str
    target_token_shape: tuple[int, ...]
    target_token_dtype: str
    origin_parameter_hashes: tuple[tuple[str, str], ...]
    hops: tuple[TargetLineageHop, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id
            or not isinstance(self.context_id, str)
            or not self.context_id
            or not self.request_ids
            or len(set(self.request_ids)) != len(self.request_ids)
        ):
            raise ContractError("frozen-target identity is incomplete")
        for field_name in (
            "hparams_sha256",
            "origin_snapshot_id",
            "origin_state_id",
            "direct_z_tensor_sha256",
            "direct_z_artifact_sha256",
            "target_token_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _full_sha256(field_name, getattr(self, field_name)),
            )
        if (
            isinstance(self.direct_z_artifact_size, bool)
            or not isinstance(self.direct_z_artifact_size, int)
            or self.direct_z_artifact_size <= 0
        ):
            raise ContractError("direct-z artifact size must be a positive integer")
        if (
            len(tuple(self.target_token_shape)) != 1
            or tuple(self.target_token_shape)[0] <= 0
            or tuple(self.target_token_shape)
            != (tuple(self.target_token_shape)[0],)
            or not isinstance(self.target_token_dtype, str)
            or not self.target_token_dtype
        ):
            raise ContractError("target-token metadata is invalid")
        origin_hashes = tuple(self.origin_parameter_hashes)
        names = [name for name, _digest in origin_hashes]
        if (
            not origin_hashes
            or names != sorted(names)
            or len(names) != len(set(names))
        ):
            raise ContractError("origin parameter hashes are not canonical")
        normalized_origin = tuple(
            (name, _full_sha256(f"origin parameter {name}", digest))
            for name, digest in origin_hashes
        )
        hops = tuple(self.hops)
        legacy_envelope = bool(
            len(hops) <= 1
            and all((hop.label, hop.step_scale) in _ALLOWED_HOPS for hop in hops)
        )
        quarter_envelope = bool(
            1 <= len(hops) <= 4
            and all(
                hop.label in _QUARTER_STEP_LABELS and hop.step_scale == 0.25
                for hop in hops
            )
        )
        adaptive_envelope = bool(
            1 <= len(hops) <= len(_ADAPTIVE_STEP_LABELS)
            and tuple(hop.label for hop in hops)
            == _ADAPTIVE_STEP_LABELS[: len(hops)]
            and all(_is_adaptive_step(hop.label, hop.step_scale) for hop in hops)
        )
        if hops and not (legacy_envelope or quarter_envelope or adaptive_envelope):
            raise ContractError("frozen-target lineage mixes or exceeds its fixed envelope")
        if hops:
            hop = hops[0]
            if (
                hop.parent_snapshot_id != self.origin_snapshot_id
                or hop.parent_state_id != self.origin_state_id
                or tuple(name for name, _digest in hop.child_parameter_hashes)
                != tuple(name for name, _digest in normalized_origin)
            ):
                raise ContractError("lineage hop does not descend from the target origin")
            for previous, current in zip(hops, hops[1:]):
                if (
                    current.parent_snapshot_id != previous.child_snapshot_id
                    or current.parent_state_id != previous.child_state_id
                    or tuple(name for name, _digest in current.child_parameter_hashes)
                    != tuple(name for name, _digest in normalized_origin)
                ):
                    raise ContractError("quarter-step lineage is not an exact chain")
        object.__setattr__(self, "origin_parameter_hashes", normalized_origin)
        object.__setattr__(self, "hops", hops)

    @classmethod
    def start(
        cls,
        *,
        direct_z: FrozenDirectZ,
        origin_snapshot: SnapshotManifest,
        target_token_ids: torch.Tensor,
    ) -> "FrozenTargetLineage":
        """Bind target bytes/tokens to the exact W0 source snapshot."""

        if not isinstance(direct_z, FrozenDirectZ):
            raise ContractError("frozen target requires a canonical FrozenDirectZ")
        token_hash, token_shape, token_dtype = _token_identity(target_token_ids)
        if (
            direct_z.source_snapshot_id != origin_snapshot.snapshot_id
            or direct_z.source_state_id != origin_snapshot.state_id
            or direct_z.model_id != origin_snapshot.model_id
            or direct_z.context_id != origin_snapshot.context_id
            or direct_z.request_ids != origin_snapshot.request_ids
        ):
            raise ContractError("direct-z target and W0 source snapshot differ")
        if tensor_sha256(direct_z.values) != direct_z.tensor_sha256:
            raise ContractError("direct-z values changed after target construction")
        return cls(
            model_id=origin_snapshot.model_id,
            context_id=origin_snapshot.context_id,
            request_ids=origin_snapshot.request_ids,
            hparams_sha256=origin_snapshot.hparams_sha256,
            origin_snapshot_id=origin_snapshot.snapshot_id,
            origin_state_id=origin_snapshot.state_id,
            direct_z_tensor_sha256=direct_z.tensor_sha256,
            direct_z_artifact_sha256=direct_z.artifact.sha256,
            direct_z_artifact_size=direct_z.artifact.size,
            target_token_sha256=token_hash,
            target_token_shape=token_shape,
            target_token_dtype=token_dtype,
            origin_parameter_hashes=tuple(
                sorted(_parameter_hashes(origin_snapshot).items())
            ),
        )

    @property
    def terminal_snapshot_id(self) -> str:
        return self.hops[-1].child_snapshot_id if self.hops else self.origin_snapshot_id

    @property
    def terminal_state_id(self) -> str:
        return self.hops[-1].child_state_id if self.hops else self.origin_state_id

    @property
    def lineage_id(self) -> str:
        return sha256_bytes(
            canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        )

    def _derive_verified(
        self,
        *,
        parent_snapshot: SnapshotManifest,
        child_snapshot: SnapshotManifest,
        action_hash: str,
        step_scale: float,
        label: str,
        observed_child_parameter_hashes: Mapping[str, str],
    ) -> "FrozenTargetLineage":
        """Construct one already-bound hop after checking its observed state."""

        legacy_step = (label, step_scale) in _ALLOWED_HOPS
        quarter_step = label in _QUARTER_STEP_LABELS and step_scale == 0.25
        adaptive_step = _is_adaptive_step(label, step_scale)
        if legacy_step and self.hops:
            raise ContractError("MV-2 lineage cannot be extended beyond one hop")
        if quarter_step and (
            len(self.hops) >= 4
            or any(
                hop.label not in _QUARTER_STEP_LABELS or hop.step_scale != 0.25
                for hop in self.hops
            )
        ):
            raise ContractError("quarter-step lineage cannot exceed or mix four hops")
        if adaptive_step and (
            len(self.hops) >= len(_ADAPTIVE_STEP_LABELS)
            or label != _ADAPTIVE_STEP_LABELS[len(self.hops)]
            or any(
                not _is_adaptive_step(hop.label, hop.step_scale)
                for hop in self.hops
            )
        ):
            raise ContractError("adaptive lineage cannot exceed, reorder, or mix four hops")
        if not (legacy_step or quarter_step or adaptive_step):
            raise ContractError("lineage step is outside its fixed envelope")
        if (
            parent_snapshot.snapshot_id != self.terminal_snapshot_id
            or parent_snapshot.state_id != self.terminal_state_id
        ):
            raise ContractError("lineage parent is not the current verified terminal")
        for field_name in ("model_id", "context_id", "request_ids", "hparams_sha256"):
            if getattr(parent_snapshot, field_name) != getattr(child_snapshot, field_name):
                raise ContractError(f"lineage changed immutable {field_name}")
        parent_records = {
            record.name: (record.shape, record.dtype)
            for record in parent_snapshot.parameters
        }
        child_records = {
            record.name: (record.shape, record.dtype)
            for record in child_snapshot.parameters
        }
        if parent_records != child_records:
            raise ContractError("lineage changed parameter names, shapes, or dtypes")
        observed = dict(observed_child_parameter_hashes)
        expected = _parameter_hashes(child_snapshot)
        parent_hashes = _parameter_hashes(parent_snapshot)
        if not observed or not set(observed).issubset(expected):
            raise ContractError("observed descendant hash set is empty or out of scope")
        if any(
            _full_sha256(f"observed parameter {name}", observed[name])
            != expected[name]
            for name in observed
        ):
            raise ContractError("observed descendant hashes differ from the child snapshot")
        if any(
            expected[name] != parent_hashes[name]
            for name in set(expected) - set(observed)
        ):
            raise ContractError(
                "unobserved parameter changed during the descendant transition"
            )
        hop = TargetLineageHop(
            label=label,
            step_scale=step_scale,
            parent_snapshot_id=parent_snapshot.snapshot_id,
            parent_state_id=parent_snapshot.state_id,
            child_snapshot_id=child_snapshot.snapshot_id,
            child_state_id=child_snapshot.state_id,
            action_hash=action_hash,
            child_parameter_hashes=tuple(sorted(expected.items())),
        )
        return FrozenTargetLineage(
            model_id=self.model_id,
            context_id=self.context_id,
            request_ids=self.request_ids,
            hparams_sha256=self.hparams_sha256,
            origin_snapshot_id=self.origin_snapshot_id,
            origin_state_id=self.origin_state_id,
            direct_z_tensor_sha256=self.direct_z_tensor_sha256,
            direct_z_artifact_sha256=self.direct_z_artifact_sha256,
            direct_z_artifact_size=self.direct_z_artifact_size,
            target_token_sha256=self.target_token_sha256,
            target_token_shape=self.target_token_shape,
            target_token_dtype=self.target_token_dtype,
            origin_parameter_hashes=self.origin_parameter_hashes,
            hops=(*self.hops, hop),
        )

    def derive_h0(self, *, snapshot: SnapshotManifest) -> "FrozenTargetLineage":
        """Run the zero-hop sham through the same lineage state machinery."""

        if self.hops:
            raise ContractError("MV-2 lineage cannot be extended beyond one hop")
        if (
            snapshot.snapshot_id != self.origin_snapshot_id
            or snapshot.state_id != self.origin_state_id
            or _parameter_hashes(snapshot) != dict(self.origin_parameter_hashes)
        ):
            raise ContractError("h=0 sham must use the exact target-origin snapshot")
        action_hash = sha256_bytes(
            canonical_json(
                {
                    "label": "h0_sham",
                    "step_scale": 0.0,
                    "origin_state_id": self.origin_state_id,
                }
            ).encode("utf-8")
        )
        return self._derive_verified(
            parent_snapshot=snapshot,
            child_snapshot=snapshot,
            action_hash=action_hash,
            step_scale=0.0,
            label="h0_sham",
            observed_child_parameter_hashes=_parameter_hashes(snapshot),
        )

    def derive_partial(
        self,
        *,
        parent_snapshot: SnapshotManifest,
        child_snapshot: SnapshotManifest,
        application: TemporaryLowRankApplication,
        full_step_proposal: MemitFactorProposal,
    ) -> "FrozenTargetLineage":
        """Bind W1 to the exact pre-scaled proposal currently being applied.

        Callers cannot supply an action hash or an observed hash mapping.  Both
        are taken from the active application receipt, and the application
        scale must be one so the proposal tensor bytes are the complete action.
        """

        if type(application) is not TemporaryLowRankApplication:
            raise ContractError("partial lineage requires the canonical application")
        if application.scale != 1.0:
            raise ContractError(
                "partial lineage requires a pre-scaled proposal applied at scale one"
            )
        proposal = application.proposal
        if (
            not isinstance(full_step_proposal, MemitFactorProposal)
            or not full_step_proposal.is_synchronous
        ):
            raise ContractError("partial lineage requires a synchronous full-step action")
        expected_partial = MemitFactorProposal(
            snapshot=full_step_proposal.snapshot,
            factors=tuple(
                factor.scaled(0.5) for factor in full_step_proposal.factors
            ),
            semantics=full_step_proposal.semantics,
            solver_name="frozen-target-lineage/canonical-h-1-2",
            residual_denominator=full_step_proposal.residual_denominator,
        )
        if (
            not isinstance(proposal, MemitFactorProposal)
            or not proposal.is_synchronous
            or proposal_direction_hash(proposal)
            != proposal_direction_hash(expected_partial)
        ):
            raise ContractError(
                "active proposal is not the canonical h=1/2 full-step action"
            )
        if (
            full_step_proposal.snapshot.state_id != parent_snapshot.state_id
            or _parameter_hashes(full_step_proposal.snapshot)
            != _parameter_hashes(parent_snapshot)
        ):
            raise ContractError("full-step action does not start at the lineage parent")
        if (
            proposal.snapshot.state_id != parent_snapshot.state_id
            or proposal.snapshot.model_id != parent_snapshot.model_id
            or proposal.snapshot.context_id != parent_snapshot.context_id
            or proposal.snapshot.request_ids != parent_snapshot.request_ids
            or proposal.snapshot.hparams_sha256 != parent_snapshot.hparams_sha256
            or _parameter_hashes(proposal.snapshot)
            != _parameter_hashes(parent_snapshot)
        ):
            raise ContractError("applied proposal does not start at the lineage parent")
        factor_names = tuple(factor.weight_name for factor in proposal.factors)
        parent_hashes = _parameter_hashes(parent_snapshot)
        if (
            not factor_names
            or len(set(factor_names)) != len(factor_names)
            or not set(factor_names).issubset(parent_hashes)
            or any(
                factor.expected_weight_sha256 != parent_hashes[factor.weight_name]
                for factor in proposal.factors
            )
        ):
            raise ContractError("applied proposal factors are outside the lineage parent")
        observed = application.applied_hashes
        if set(observed) != set(factor_names):
            raise ContractError(
                "partial lineage requires the active application's complete hash receipt"
            )
        return self._derive_verified(
            parent_snapshot=parent_snapshot,
            child_snapshot=child_snapshot,
            action_hash=proposal_direction_hash(proposal),
            step_scale=0.5,
            label="partial_joint",
            observed_child_parameter_hashes=observed,
        )

    def derive_quarter_step(
        self,
        *,
        parent_snapshot: SnapshotManifest,
        child_snapshot: SnapshotManifest,
        application: TemporaryLowRankApplication,
        label: str,
    ) -> "FrozenTargetLineage":
        """Bind one exact ``h=1/4`` transition for the K=4 diagnostic.

        C-distance matching is checked by the runner against the immutable
        covariance.  This method owns only ancestry, action-byte, and observed
        child-state identity.
        """

        if label not in _QUARTER_STEP_LABELS:
            raise ContractError("quarter-step lineage label is outside the lock")
        if type(application) is not TemporaryLowRankApplication:
            raise ContractError("quarter-step lineage requires canonical application")
        if application.scale != 1.0:
            raise ContractError("quarter-step proposal must be pre-scaled and applied at one")
        proposal = application.proposal
        if not isinstance(proposal, MemitFactorProposal):
            raise ContractError("quarter-step lineage requires MemitFactorProposal")
        if (
            proposal.snapshot.state_id != parent_snapshot.state_id
            or proposal.snapshot.model_id != parent_snapshot.model_id
            or proposal.snapshot.context_id != parent_snapshot.context_id
            or proposal.snapshot.request_ids != parent_snapshot.request_ids
            or proposal.snapshot.hparams_sha256 != parent_snapshot.hparams_sha256
            or _parameter_hashes(proposal.snapshot) != _parameter_hashes(parent_snapshot)
        ):
            raise ContractError("quarter-step proposal does not start at its lineage parent")
        factor_names = tuple(factor.weight_name for factor in proposal.factors)
        parent_hashes = _parameter_hashes(parent_snapshot)
        if (
            not factor_names
            or len(set(factor_names)) != len(factor_names)
            or not set(factor_names).issubset(parent_hashes)
            or any(
                factor.expected_weight_sha256 != parent_hashes[factor.weight_name]
                for factor in proposal.factors
            )
        ):
            raise ContractError("quarter-step factors are outside the lineage parent")
        observed = application.applied_hashes
        if set(observed) != set(factor_names):
            raise ContractError("quarter-step application receipt is incomplete")
        return self._derive_verified(
            parent_snapshot=parent_snapshot,
            child_snapshot=child_snapshot,
            action_hash=proposal_direction_hash(proposal),
            step_scale=0.25,
            label=label,
            observed_child_parameter_hashes=observed,
        )

    def derive_adaptive_step(
        self,
        *,
        parent_snapshot: SnapshotManifest,
        child_snapshot: SnapshotManifest,
        application: TemporaryLowRankApplication,
        step_scale: float,
        label: str,
    ) -> "FrozenTargetLineage":
        """Bind one accepted capacity-controller transition.

        ``step_scale`` is the observed C-distance divided by the edit's native
        C-distance.  The runner owns that metric check; this lineage binds the
        exact proposal bytes and descendant hashes without pretending every
        accepted trust-region step consumed the full ``D/4`` envelope.  A
        ``1e-5`` scale tolerance covers the QP solver's already-bounded numeric
        trust-region residual without changing or rounding the applied action.
        """

        scale = _finite_scale(step_scale)
        if not _is_adaptive_step(label, scale):
            raise ContractError("adaptive-step lineage label/scale is outside the lock")
        if not isinstance(application, TemporaryLowRankApplication):
            raise ContractError("adaptive-step lineage requires canonical application")
        if application.scale != 1.0:
            raise ContractError("adaptive proposal must be pre-scaled and applied at one")
        proposal = application.proposal
        if not isinstance(proposal, MemitFactorProposal):
            raise ContractError("adaptive-step lineage requires MemitFactorProposal")
        if (
            proposal.snapshot.state_id != parent_snapshot.state_id
            or proposal.snapshot.model_id != parent_snapshot.model_id
            or proposal.snapshot.context_id != parent_snapshot.context_id
            or proposal.snapshot.request_ids != parent_snapshot.request_ids
            or proposal.snapshot.hparams_sha256 != parent_snapshot.hparams_sha256
            or _parameter_hashes(proposal.snapshot) != _parameter_hashes(parent_snapshot)
        ):
            raise ContractError("adaptive proposal does not start at its lineage parent")
        factor_names = tuple(factor.weight_name for factor in proposal.factors)
        parent_hashes = _parameter_hashes(parent_snapshot)
        if (
            not factor_names
            or len(set(factor_names)) != len(factor_names)
            or not set(factor_names).issubset(parent_hashes)
            or any(
                factor.expected_weight_sha256 != parent_hashes[factor.weight_name]
                for factor in proposal.factors
            )
        ):
            raise ContractError("adaptive factors are outside the lineage parent")
        observed = application.applied_hashes
        if set(observed) != set(factor_names):
            raise ContractError("adaptive application receipt is incomplete")
        return self._derive_verified(
            parent_snapshot=parent_snapshot,
            child_snapshot=child_snapshot,
            action_hash=proposal_direction_hash(proposal),
            step_scale=scale,
            label=label,
            observed_child_parameter_hashes=observed,
        )

    def assert_target_tokens(self, target_token_ids: torch.Tensor) -> None:
        digest, shape, dtype = _token_identity(target_token_ids)
        if (
            digest != self.target_token_sha256
            or shape != self.target_token_shape
            or dtype != self.target_token_dtype
        ):
            raise ContractError("target-token identity changed along the trajectory")

    def assert_terminal_state(self, snapshot: SnapshotManifest) -> None:
        """Authorize the terminal parameter state independent of provenance IDs."""

        if (
            snapshot.model_id != self.model_id
            or snapshot.context_id != self.context_id
            or snapshot.request_ids != self.request_ids
            or snapshot.hparams_sha256 != self.hparams_sha256
            or snapshot.state_id != self.terminal_state_id
            or _parameter_hashes(snapshot)
            != dict(
                self.hops[-1].child_parameter_hashes
                if self.hops
                else self.origin_parameter_hashes
            )
        ):
            raise ContractError("snapshot state is outside the frozen-target lineage")

    def assert_authorizes(
        self,
        *,
        direct_z: FrozenDirectZ,
        snapshot: SnapshotManifest,
        target_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Validate target bytes plus the exact terminal model-state identity."""

        if not isinstance(direct_z, FrozenDirectZ):
            raise ContractError("lineage authorization requires FrozenDirectZ")
        if (
            direct_z.model_id != self.model_id
            or direct_z.context_id != self.context_id
            or direct_z.request_ids != self.request_ids
            or direct_z.source_snapshot_id != self.origin_snapshot_id
            or direct_z.source_state_id != self.origin_state_id
            or direct_z.tensor_sha256 != self.direct_z_tensor_sha256
            or direct_z.artifact.sha256 != self.direct_z_artifact_sha256
            or direct_z.artifact.size != self.direct_z_artifact_size
            or tensor_sha256(direct_z.values) != self.direct_z_tensor_sha256
        ):
            raise ContractError("frozen direct-z identity changed along the trajectory")
        self.assert_terminal_state(snapshot)
        if snapshot.snapshot_id != self.terminal_snapshot_id:
            raise ContractError("requested proposal state is outside target lineage")
        if target_token_ids is not None:
            self.assert_target_tokens(target_token_ids)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        quarter_envelope = bool(
            self.hops
            and all(
                hop.label in _QUARTER_STEP_LABELS and hop.step_scale == 0.25
                for hop in self.hops
            )
        )
        adaptive_envelope = bool(
            self.hops
            and tuple(hop.label for hop in self.hops)
            == _ADAPTIVE_STEP_LABELS[: len(self.hops)]
            and all(_is_adaptive_step(hop.label, hop.step_scale) for hop in self.hops)
        )
        payload: dict[str, Any] = {
            "schema_version": (
                ADAPTIVE_STEP_LINEAGE_SCHEMA
                if adaptive_envelope
                else QUARTER_STEP_LINEAGE_SCHEMA
                if quarter_envelope
                else FROZEN_TARGET_LINEAGE_SCHEMA
            ),
            "model_id": self.model_id,
            "context_id": self.context_id,
            "request_ids": list(self.request_ids),
            "hparams_sha256": self.hparams_sha256,
            "origin_snapshot_id": self.origin_snapshot_id,
            "origin_state_id": self.origin_state_id,
            "direct_z_tensor_sha256": self.direct_z_tensor_sha256,
            "direct_z_artifact": {
                "sha256": self.direct_z_artifact_sha256,
                "size": self.direct_z_artifact_size,
            },
            "target_token": {
                "sha256": self.target_token_sha256,
                "shape": list(self.target_token_shape),
                "dtype": self.target_token_dtype,
            },
            "origin_parameter_hashes": dict(self.origin_parameter_hashes),
            "hops": [
                {**hop.to_dict(), "hop_id": hop.hop_id}
                for hop in self.hops
            ],
        }
        if include_id:
            payload["lineage_id"] = self.lineage_id
        return payload
