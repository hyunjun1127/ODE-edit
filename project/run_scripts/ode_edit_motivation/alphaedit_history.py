"""Low-rank, branch-local history state matching AlphaEdit ``cache_c``.

Upstream AlphaEdit stores a dense per-layer matrix and appends post-edit keys as
``cache_c[l] += H_l @ H_l.T``.  ODE-Edit keeps the equivalent key columns
``H_l`` instead, so the solver can use Woodbury algebra without materializing or
mutating the upstream global cache.  History is immutable within one atomic edit
and is appended exactly once after its accepted endpoint has been committed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .contracts import ContractError, canonical_json, sha256_bytes


ALPHAEDIT_HISTORY_SCHEMA = "ode-edit-alphaedit-history/v1"


class AlphaEditHistoryError(ContractError):
    """The low-rank historical cache violates canonical append semantics."""


def _tensor_sha256(value: torch.Tensor) -> str:
    normalized = value.detach().cpu().float().contiguous()
    return hashlib.sha256(normalized.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class AlphaEditHistoryAppend:
    edit_id: str
    rank_by_layer: Mapping[int, int]
    tensor_sha256_by_layer: Mapping[int, str]

    def __post_init__(self) -> None:
        if not isinstance(self.edit_id, str) or not self.edit_id.strip():
            raise AlphaEditHistoryError("history edit_id must not be empty")
        if set(self.rank_by_layer) != set(self.tensor_sha256_by_layer):
            raise AlphaEditHistoryError("history append layer sets differ")
        for layer, rank in self.rank_by_layer.items():
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
            ):
                raise AlphaEditHistoryError("history append rank is invalid")
            digest = self.tensor_sha256_by_layer[layer]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise AlphaEditHistoryError("history append digest is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "edit_id": self.edit_id,
            "rank_by_layer": {
                str(layer): self.rank_by_layer[layer]
                for layer in sorted(self.rank_by_layer)
            },
            "tensor_sha256_by_layer": {
                str(layer): self.tensor_sha256_by_layer[layer]
                for layer in sorted(self.tensor_sha256_by_layer)
            },
        }


class AlphaEditHistoryBank:
    """Branch-local low-rank representation of canonical AlphaEdit history."""

    def __init__(self, layers: Sequence[int]) -> None:
        normalized = tuple(int(layer) for layer in layers)
        if not normalized or tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
            raise AlphaEditHistoryError("history layers must be unique and ascending")
        self._layers = normalized
        self._chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in normalized}
        self._widths: dict[int, int] = {}
        self._appends: list[AlphaEditHistoryAppend] = []
        self._edit_ids: set[str] = set()

    @property
    def layers(self) -> tuple[int, ...]:
        return self._layers

    @property
    def edit_count(self) -> int:
        return len(self._appends)

    @property
    def appends(self) -> tuple[AlphaEditHistoryAppend, ...]:
        return tuple(self._appends)

    def rank(self, layer: int) -> int:
        self._require_layer(layer)
        return sum(chunk.shape[1] for chunk in self._chunks[int(layer)])

    def _require_layer(self, layer: int) -> int:
        normalized = int(layer)
        if normalized not in self._chunks:
            raise AlphaEditHistoryError("history layer is outside the bank")
        return normalized

    def matrix(
        self,
        layer: int,
        *,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``H_l`` without exposing mutable bank storage."""

        normalized = self._require_layer(layer)
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise AlphaEditHistoryError("history width must be positive")
        known = self._widths.get(normalized)
        if known is not None and known != width:
            raise AlphaEditHistoryError("history width differs from current key width")
        chunks = self._chunks[normalized]
        if not chunks:
            return torch.empty((width, 0), device=device, dtype=dtype)
        return torch.cat(chunks, dim=1).to(device=device, dtype=dtype).clone()

    def append_post_edit_keys(
        self,
        *,
        edit_id: str,
        keys_by_layer: Mapping[int, torch.Tensor],
    ) -> AlphaEditHistoryAppend:
        """Atomically append one accepted edit's post-edit keys for every layer."""

        if not isinstance(edit_id, str) or not edit_id.strip() or edit_id in self._edit_ids:
            raise AlphaEditHistoryError("history edit_id is empty or already appended")
        if set(keys_by_layer) != set(self._layers):
            raise AlphaEditHistoryError("history append must cover every configured layer")
        normalized: dict[int, torch.Tensor] = {}
        for layer in self._layers:
            value = keys_by_layer[layer]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] <= 0
                or value.shape[1] <= 0
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise AlphaEditHistoryError("post-edit history keys are invalid")
            chunk = value.detach().cpu().float().contiguous().clone()
            known = self._widths.get(layer)
            if known is not None and known != chunk.shape[0]:
                raise AlphaEditHistoryError("post-edit key width changed")
            normalized[layer] = chunk

        # Validate all layers before mutating any branch-local state.
        record = AlphaEditHistoryAppend(
            edit_id=edit_id.strip(),
            rank_by_layer={layer: int(normalized[layer].shape[1]) for layer in self._layers},
            tensor_sha256_by_layer={
                layer: _tensor_sha256(normalized[layer]) for layer in self._layers
            },
        )
        for layer in self._layers:
            self._widths.setdefault(layer, int(normalized[layer].shape[0]))
            self._chunks[layer].append(normalized[layer])
        self._edit_ids.add(record.edit_id)
        self._appends.append(record)
        return record

    @property
    def history_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ALPHAEDIT_HISTORY_SCHEMA,
            "layers": list(self._layers),
            "edit_count": self.edit_count,
            "rank_by_layer": {str(layer): self.rank(layer) for layer in self._layers},
            "width_by_layer": {
                str(layer): self._widths.get(layer) for layer in self._layers
            },
            "appends": [record.to_dict() for record in self._appends],
            "storage": "cpu-float32-key-columns-no-dense-cache_c",
            "append_policy": "post-accepted-edit-once-history-fixed-within-edit",
        }
        if include_id:
            payload["history_id"] = self.history_id
        return payload


__all__ = [
    "ALPHAEDIT_HISTORY_SCHEMA",
    "AlphaEditHistoryAppend",
    "AlphaEditHistoryBank",
    "AlphaEditHistoryError",
]
