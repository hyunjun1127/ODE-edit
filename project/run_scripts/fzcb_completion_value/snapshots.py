"""Exact-copy virtual candidate and rollback snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import ScientificBoundary
from .hashing import canonical_hash, tensor_sha256


@dataclass(slots=True)
class WeightSnapshot:
    names: tuple[str, ...]
    tensors: dict[str, torch.Tensor]
    pointers: dict[str, int]
    versions: dict[str, int]
    root: str

    @classmethod
    def capture(cls, model: Any, names: tuple[str, ...]) -> "WeightSnapshot":
        params = dict(model.named_parameters())
        tensors = {name: params[name].detach().clone() for name in names}
        metadata = [
            {"name": name, "sha256": tensor_sha256(tensors[name]), "shape": list(tensors[name].shape)}
            for name in names
        ]
        return cls(
            names, tensors, {name: params[name].data_ptr() for name in names},
            {name: params[name]._version for name in names}, canonical_hash(metadata),
        )

    def restore(self, model: Any) -> dict[str, Any]:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in self.names:
                params[name].copy_(self.tensors[name])
        exact = all(torch.equal(params[name], self.tensors[name]) for name in self.names)
        pointers = all(params[name].data_ptr() == self.pointers[name] for name in self.names)
        if not exact or not pointers:
            raise ScientificBoundary("exact candidate rollback failed")
        return {"bytes_exact": exact, "pointer_exact": pointers, "entry_root": self.root}

    def apply_tangents(self, model: Any, tangents: tuple[torch.Tensor, ...], scale: float) -> None:
        if len(tangents) != len(self.names):
            raise ScientificBoundary("weight tangent cardinality mismatch")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, tangent in zip(self.names, tangents, strict=True):
                params[name].copy_(self.tensors[name] + float(scale) * tangent.to(self.tensors[name]))

    def current_root(self, model: Any) -> str:
        params = dict(model.named_parameters())
        return canonical_hash([
            {"name": name, "sha256": tensor_sha256(params[name]), "shape": list(params[name].shape)}
            for name in self.names
        ])
