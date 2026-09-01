"""Arm-local dense-cache append-once transaction and WAL replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import torch

from .contracts import EngineeringBoundary, FailureLabel


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheAppend:
    version_before: int
    version_after: int
    request_order: tuple[int, ...]
    post_keys: dict[int, torch.Tensor]
    post_key_digests: dict[int, str]


class DenseCacheTransaction:
    """One arm's immutable checkpoint plus append-only post-key WAL."""

    def __init__(self, initial: Mapping[int, torch.Tensor]):
        if not initial:
            raise EngineeringBoundary("dense cache checkpoint is empty")
        self._checkpoint = {int(layer): value.detach().to("cpu", torch.float32).clone() for layer, value in initial.items()}
        if any(value.ndim != 2 or value.shape[0] != value.shape[1] or not torch.isfinite(value).all() for value in self._checkpoint.values()):
            raise EngineeringBoundary(FailureLabel.CACHE_NONFINITE_OR_CORRUPT.value)
        self._wal: list[CacheAppend] = []
        self._active = False
        self._trial_mutation_count = 0

    @property
    def version(self) -> int:
        return len(self._wal)

    @property
    def wal(self) -> tuple[CacheAppend, ...]:
        return tuple(self._wal)

    @property
    def trial_mutation_count(self) -> int:
        return self._trial_mutation_count

    def materialize(self) -> dict[int, torch.Tensor]:
        state = {layer: value.clone() for layer, value in self._checkpoint.items()}
        for append in self._wal:
            for layer, key in append.post_keys.items():
                state[layer] = torch.add(state[layer], key @ key.T, alpha=1.0).to(torch.float32)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise EngineeringBoundary(FailureLabel.CACHE_NONFINITE_OR_CORRUPT.value)
        return state

    def begin_batch(self) -> tuple[int, dict[int, str]]:
        if self._active:
            raise EngineeringBoundary(FailureLabel.CACHE_TRANSACTION_FAILURE.value)
        self._active = True
        state = self.materialize()
        return self.version, {layer: tensor_digest(value) for layer, value in state.items()}

    def observe_trial(self) -> int:
        if not self._active:
            raise EngineeringBoundary(FailureLabel.CACHE_TRANSACTION_FAILURE.value)
        # Deliberately no cache object is returned to a candidate mutation path.
        return self.version

    def reject_batch(self) -> None:
        if not self._active:
            raise EngineeringBoundary(FailureLabel.CACHE_TRANSACTION_FAILURE.value)
        self._active = False

    def commit_batch(self, post_keys: Mapping[int, torch.Tensor], request_order: tuple[int, ...]) -> CacheAppend:
        if not self._active or set(post_keys) != set(self._checkpoint):
            raise EngineeringBoundary(FailureLabel.CACHE_TRANSACTION_FAILURE.value)
        prepared: dict[int, torch.Tensor] = {}
        digests: dict[int, str] = {}
        for layer, raw in post_keys.items():
            key = raw.detach().to("cpu", torch.float32).contiguous().clone()
            expected = self._checkpoint[int(layer)].shape[0]
            if key.ndim != 2 or key.shape[0] != expected or not torch.isfinite(key).all():
                raise EngineeringBoundary(FailureLabel.CACHE_NONFINITE_OR_CORRUPT.value)
            prepared[int(layer)] = key
            digests[int(layer)] = tensor_digest(key)
        before = self.version
        append = CacheAppend(before, before + 1, tuple(int(value) for value in request_order), prepared, digests)
        self._wal.append(append)
        self._active = False
        return append

    def rolling_digest(self) -> str:
        digest = sha256()
        digest.update(b"fzcb-dense-cache-wal.v1\0")
        for append in self._wal:
            digest.update(str(append.version_before).encode())
            digest.update(str(append.version_after).encode())
            digest.update(str(append.request_order).encode())
            for layer in sorted(append.post_key_digests):
                digest.update(str(layer).encode())
                digest.update(append.post_key_digests[layer].encode())
        return digest.hexdigest()

    def replay_from_checkpoint(self) -> dict[int, torch.Tensor]:
        return self.materialize()
