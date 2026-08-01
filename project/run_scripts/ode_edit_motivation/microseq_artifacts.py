"""Local-only tensor artifacts for micro-sequential action replay.

Only tensor dictionaries are stored with ``torch.save`` and they are loaded
with ``weights_only=True``.  A strict JSON sidecar binds every tensor byte,
factor field, entry state, and proposal direction hash before replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import (
    ContractError,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    SnapshotManifest,
    canonical_json,
)
from .frozen_target_lineage import proposal_direction_hash
from .hooks import tensor_sha256


ARTIFACT_SCHEMA = "ode-edit-microseq-proposal-artifact/v1"
_ACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX = frozenset("0123456789abcdef")


class MicroseqArtifactError(ContractError):
    """A local replay artifact differs from its strict sidecar contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MicroseqArtifactError(f"{path}: lowercase SHA-256 required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise MicroseqArtifactError(f"{path}: keys differ from artifact contract")


def _safe_action_id(value: Any) -> str:
    if not isinstance(value, str) or _ACTION_ID.fullmatch(value) is None:
        raise MicroseqArtifactError("action_id is outside the safe local filename contract")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = canonical_json(payload) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def write_proposal_artifact(
    directory: str | Path,
    *,
    action_id: str,
    proposal: MemitFactorProposal,
) -> Path:
    """Exclusively persist one proposal under an ignored local directory."""

    safe_id = _safe_action_id(action_id)
    if not isinstance(proposal, MemitFactorProposal):
        raise MicroseqArtifactError("proposal artifact requires MemitFactorProposal")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir() or root.is_symlink():
        raise MicroseqArtifactError("proposal artifact root must be a real directory")
    tensor_path = root / f"{safe_id}.pt"
    manifest_path = root / f"{safe_id}.json"
    if tensor_path.exists() or tensor_path.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise MicroseqArtifactError("proposal artifact destination already exists")

    tensors: dict[str, torch.Tensor] = {}
    factors: list[dict[str, Any]] = []
    for index, factor in enumerate(proposal.factors):
        left_key = f"factor_{index}_left"
        right_key = f"factor_{index}_right"
        left = factor.left.detach().cpu().contiguous()
        right = factor.right.detach().cpu().contiguous()
        tensors[left_key] = left
        tensors[right_key] = right
        factors.append(
            {
                "weight_name": factor.weight_name,
                "left_key": left_key,
                "right_key": right_key,
                "left_sha256": tensor_sha256(left),
                "right_sha256": tensor_sha256(right),
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
                "left_dtype": str(left.dtype),
                "right_dtype": str(right.dtype),
                "expected_weight_sha256": factor.expected_weight_sha256,
                "native_update_transposed": factor.native_update_transposed,
            }
        )
    try:
        with tensor_path.open("xb") as handle:
            torch.save(tensors, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tensor_path, 0o600)
        payload = {
            "schema_version": ARTIFACT_SCHEMA,
            "action_id": safe_id,
            "entry_state_id": proposal.entry_snapshot_id,
            "semantics": proposal.semantics.value,
            "residual_denominator": proposal.residual_denominator,
            "proposal_direction_sha256": proposal_direction_hash(proposal),
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": _sha256_file(tensor_path),
            "tensor_file_size": tensor_path.stat().st_size,
            "factors": factors,
            "raw_prompt_persisted": False,
            "raw_logits_persisted": False,
        }
        _write_json_exclusive(manifest_path, payload)
        os.chmod(manifest_path, 0o600)
    except BaseException:
        if manifest_path.exists() and not manifest_path.is_symlink():
            manifest_path.unlink()
        if tensor_path.exists() and not tensor_path.is_symlink():
            tensor_path.unlink()
        raise
    return manifest_path


def load_proposal_artifact(
    manifest: str | Path,
    *,
    snapshot: SnapshotManifest,
) -> MemitFactorProposal:
    """Verify and rebuild one proposal for an exact replay entry state."""

    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise MicroseqArtifactError("proposal manifest must be a real file")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MicroseqArtifactError("proposal manifest is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise MicroseqArtifactError("proposal manifest must be a mapping")
    _exact_keys(
        payload,
        {
            "schema_version",
            "action_id",
            "entry_state_id",
            "semantics",
            "residual_denominator",
            "proposal_direction_sha256",
            "tensor_file",
            "tensor_file_sha256",
            "tensor_file_size",
            "factors",
            "raw_prompt_persisted",
            "raw_logits_persisted",
        },
        "manifest",
    )
    _safe_action_id(payload["action_id"])
    if (
        payload["schema_version"] != ARTIFACT_SCHEMA
        or payload["entry_state_id"] != snapshot.state_id
        or payload["raw_prompt_persisted"] is not False
        or payload["raw_logits_persisted"] is not False
    ):
        raise MicroseqArtifactError("proposal manifest identity/state differs")
    expected_direction = _sha256(
        payload["proposal_direction_sha256"], "proposal_direction_sha256"
    )
    tensor_name = payload["tensor_file"]
    if (
        not isinstance(tensor_name, str)
        or Path(tensor_name).name != tensor_name
        or tensor_name != f"{payload['action_id']}.pt"
    ):
        raise MicroseqArtifactError("tensor filename differs from action identity")
    tensor_path = manifest_path.parent / tensor_name
    if not tensor_path.is_file() or tensor_path.is_symlink():
        raise MicroseqArtifactError("tensor artifact must be a real file")
    if (
        tensor_path.stat().st_size != payload["tensor_file_size"]
        or _sha256_file(tensor_path)
        != _sha256(payload["tensor_file_sha256"], "tensor_file_sha256")
    ):
        raise MicroseqArtifactError("tensor artifact byte identity differs")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensors, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in tensors.items()
    ):
        raise MicroseqArtifactError("tensor artifact must contain only named tensors")

    factors_raw = payload["factors"]
    if not isinstance(factors_raw, list) or not factors_raw:
        raise MicroseqArtifactError("factor manifest must be a non-empty list")
    factors: list[LowRankFactor] = []
    expected_tensor_keys: set[str] = set()
    for index, raw in enumerate(factors_raw):
        if not isinstance(raw, Mapping):
            raise MicroseqArtifactError(f"factors[{index}]: mapping required")
        _exact_keys(
            raw,
            {
                "weight_name",
                "left_key",
                "right_key",
                "left_sha256",
                "right_sha256",
                "left_shape",
                "right_shape",
                "left_dtype",
                "right_dtype",
                "expected_weight_sha256",
                "native_update_transposed",
            },
            f"factors[{index}]",
        )
        left_key = raw["left_key"]
        right_key = raw["right_key"]
        if left_key != f"factor_{index}_left" or right_key != f"factor_{index}_right":
            raise MicroseqArtifactError("factor tensor keys differ from canonical order")
        expected_tensor_keys.update((left_key, right_key))
        left = tensors.get(left_key)
        right = tensors.get(right_key)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise MicroseqArtifactError("factor tensor is missing")
        if (
            list(left.shape) != raw["left_shape"]
            or list(right.shape) != raw["right_shape"]
            or str(left.dtype) != raw["left_dtype"]
            or str(right.dtype) != raw["right_dtype"]
            or tensor_sha256(left) != _sha256(raw["left_sha256"], "left_sha256")
            or tensor_sha256(right) != _sha256(raw["right_sha256"], "right_sha256")
            or raw["expected_weight_sha256"]
            != snapshot.parameter(raw["weight_name"]).sha256
            or type(raw["native_update_transposed"]) is not bool
        ):
            raise MicroseqArtifactError("factor metadata/tensor identity differs")
        factors.append(
            LowRankFactor(
                weight_name=raw["weight_name"],
                left=left,
                right=right,
                expected_weight_sha256=raw["expected_weight_sha256"],
                native_update_transposed=raw["native_update_transposed"],
            )
        )
    if set(tensors) != expected_tensor_keys:
        raise MicroseqArtifactError("tensor artifact contains missing or extra tensors")
    try:
        semantics = ProposalSemantics(payload["semantics"])
    except ValueError as exc:
        raise MicroseqArtifactError("proposal semantics is invalid") from exc
    proposal = MemitFactorProposal(
        snapshot=snapshot,
        factors=tuple(factors),
        semantics=semantics,
        solver_name=f"microseq-replay/{payload['action_id']}",
        residual_denominator=payload["residual_denominator"],
    )
    if proposal_direction_hash(proposal) != expected_direction:
        raise MicroseqArtifactError("replayed proposal direction hash differs")
    return proposal
