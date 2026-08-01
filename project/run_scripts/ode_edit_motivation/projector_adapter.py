"""Read-only AlphaEdit projector adapter for reusable ODE-Edit proposals.

The adapter never imports or calls AlphaEdit's mutation entrypoint.  It consumes
the exact projector bytes already pinned by :mod:`manifests`, memory-maps the
tensor read-only, and transforms an existing low-rank proposal ``B`` into
``B @ P`` without materializing ``B``.  EasyEdit, its projector, and its
Wikipedia statistics are never written or recomputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .contracts import (
    ContractError,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProvenanceManifest,
    SnapshotManifest,
    preflight_pinned_files,
)
from .manifests import FIXED_FILE_IDENTITIES, FixedModelSpec


@dataclass(frozen=True, slots=True)
class ProjectedProposal:
    """One projected low-rank proposal plus outcome-free norm diagnostics."""

    proposal: MemitFactorProposal
    right_norm_retention: Mapping[str, float]
    projector_manifest_id: str


class AlphaEditProjectorBank:
    """Pinned, mmap-backed bank of per-layer AlphaEdit projectors."""

    def __init__(
        self,
        *,
        path: Path,
        identity: ExpectedFileIdentity,
        layers: tuple[int, ...],
        manifest: ProvenanceManifest,
        tensor: torch.Tensor,
    ) -> None:
        self.path = path
        self.identity = identity
        self.layers = layers
        self.manifest = manifest
        self._tensor = tensor
        stat = path.stat()
        self._stat_token = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    @classmethod
    def open(
        cls,
        easyedit_root: str | Path,
        spec: FixedModelSpec,
    ) -> "AlphaEditProjectorBank":
        root = Path(easyedit_root).expanduser().resolve(strict=True)
        relative = spec.projector_path
        identity = FIXED_FILE_IDENTITIES.get(relative)
        if identity is None:
            raise ContractError("fixed model has no pinned AlphaEdit projector")
        manifest = preflight_pinned_files(
            {relative: identity},
            label=f"AlphaEdit projector for {spec.alias}",
            base_dir=root,
        )
        path = (root / relative).resolve(strict=True)
        # mmap avoids a 4--7 GiB eager host copy.  weights_only prevents pickle
        # object construction; the pinned file must contain one plain tensor.
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.ndim != 3
            or value.shape[0] != len(spec.layers)
            or value.shape[1] != value.shape[2]
            or value.requires_grad
        ):
            raise ContractError("pinned AlphaEdit projector tensor shape/type is invalid")
        return cls(
            path=path,
            identity=identity,
            layers=tuple(spec.layers),
            manifest=manifest,
            tensor=value,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._tensor.shape)

    def metadata(self) -> dict[str, object]:
        return {
            "relative_name": self.path.name,
            "sha256": self.identity.sha256,
            "size": self.identity.size,
            "shape": list(self.shape),
            "dtype": str(self._tensor.dtype),
            "layers": list(self.layers),
            "manifest_id": self.manifest.manifest_id,
            "load_mode": "torch.load(weights_only=True,mmap=True,map_location=cpu)",
            "write_or_recompute": False,
        }

    def assert_stat_current(self) -> None:
        stat = self.path.stat()
        observed = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        if observed != self._stat_token:
            raise ContractError("AlphaEdit projector changed during the run")

    def assert_hash_current(self) -> None:
        """Expensive end-of-run byte verification; call once, not per event."""

        self.manifest.assert_current()
        self.assert_stat_current()

    def project_proposal(
        self,
        proposal: MemitFactorProposal,
        *,
        layer_by_weight: Mapping[str, int],
        solver_suffix: str,
    ) -> ProjectedProposal:
        """Return ``proposal @ P`` through the proposal's right factors.

        If ``B = L R.T``, then ``B P = L (P.T R).T``.  The transpose is kept
        explicitly even though the pinned AlphaEdit matrices are intended to
        be symmetric projectors.
        """

        if not isinstance(proposal, MemitFactorProposal):
            raise ContractError("AlphaEdit projection requires a factor proposal")
        if not isinstance(solver_suffix, str) or not solver_suffix.strip():
            raise ContractError("AlphaEdit projection solver suffix is empty")
        factor_names = tuple(factor.weight_name for factor in proposal.factors)
        if len(set(factor_names)) != len(factor_names):
            raise ContractError("AlphaEdit proposal contains duplicate factors")
        if not set(factor_names).issubset(layer_by_weight):
            raise ContractError("AlphaEdit proposal contains an unmapped weight")
        self.assert_stat_current()

        projected: list[LowRankFactor] = []
        retention: dict[str, float] = {}
        with torch.no_grad():
            for factor in proposal.factors:
                layer = int(layer_by_weight[factor.weight_name])
                try:
                    index = self.layers.index(layer)
                except ValueError as exc:
                    raise ContractError(
                        f"AlphaEdit projector does not cover layer {layer}"
                    ) from exc
                if factor.right.shape[0] != self._tensor.shape[1]:
                    raise ContractError(
                        "AlphaEdit projector dimension differs from factor input"
                    )
                matrix = self._tensor[index].to(
                    device=factor.right.device,
                    dtype=factor.right.dtype,
                    non_blocking=False,
                )
                right = matrix.transpose(0, 1) @ factor.right
                del matrix
                if not bool(torch.isfinite(right).all()):
                    raise ContractError("AlphaEdit projection produced non-finite values")
                source_norm = float(torch.linalg.vector_norm(factor.right).item())
                projected_norm = float(torch.linalg.vector_norm(right).item())
                if not (
                    math.isfinite(source_norm)
                    and math.isfinite(projected_norm)
                    and source_norm > 0.0
                    and projected_norm > 0.0
                ):
                    raise ContractError("AlphaEdit projection produced a zero/invalid direction")
                retention[factor.weight_name] = projected_norm / source_norm
                projected.append(
                    LowRankFactor(
                        weight_name=factor.weight_name,
                        left=factor.left,
                        right=right,
                        expected_weight_sha256=factor.expected_weight_sha256,
                        native_update_transposed=factor.native_update_transposed,
                    )
                )

        # The model snapshot itself is unchanged.  Projector provenance is
        # committed separately in the run manifest and solver name so direct-z
        # lineage identity remains exact.
        transformed = MemitFactorProposal(
            snapshot=SnapshotManifest(
                model_id=proposal.snapshot.model_id,
                context_id=proposal.snapshot.context_id,
                request_ids=proposal.snapshot.request_ids,
                hparams_sha256=proposal.snapshot.hparams_sha256,
                parameters=proposal.snapshot.parameters,
                provenance_ids=proposal.snapshot.provenance_ids,
            ),
            factors=tuple(projected),
            semantics=proposal.semantics,
            solver_name=(
                f"{proposal.solver_name}/alphaedit-projector/"
                f"{self.manifest.manifest_id[:16]}/{solver_suffix}"
            ),
            residual_denominator=proposal.residual_denominator,
        )
        self.assert_stat_current()
        return ProjectedProposal(
            proposal=transformed,
            right_norm_retention=retention,
            projector_manifest_id=self.manifest.manifest_id,
        )
