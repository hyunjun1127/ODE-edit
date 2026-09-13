"""Exact-copy virtual state and terminal-only atomic commit primitives."""

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
    root: str

    @classmethod
    def capture(cls, model: Any, names: tuple[str, ...]) -> "WeightSnapshot":
        parameters = dict(model.named_parameters())
        tensors = {name: parameters[name].detach().clone() for name in names}
        rows = [
            {"name": name, "shape": list(tensors[name].shape), "sha256": tensor_sha256(tensors[name])}
            for name in names
        ]
        return cls(names, tensors, {name: parameters[name].data_ptr() for name in names}, canonical_hash(rows))

    def restore(self, model: Any) -> dict[str, Any]:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name in self.names:
                parameters[name].copy_(self.tensors[name])
        exact = all(torch.equal(parameters[name], self.tensors[name]) for name in self.names)
        pointer_exact = all(parameters[name].data_ptr() == self.pointers[name] for name in self.names)
        if not exact or not pointer_exact:
            raise ScientificBoundary("exact-copy W rollback failed")
        return {"bytes_exact": exact, "pointer_exact": pointer_exact, "root": self.root}

    def set_displacement(self, model: Any, tangents: tuple[torch.Tensor, ...], scale: float) -> None:
        if len(tangents) != len(self.names):
            raise ScientificBoundary("tangent/weight cardinality mismatch")
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, tangent in zip(self.names, tangents, strict=True):
                parameters[name].copy_(self.tensors[name] + float(scale) * tangent.to(self.tensors[name]))

    def current_root(self, model: Any) -> str:
        parameters = dict(model.named_parameters())
        return canonical_hash([
            {"name": name, "shape": list(parameters[name].shape), "sha256": tensor_sha256(parameters[name])}
            for name in self.names
        ])


@dataclass(slots=True)
class AtomicEditTransaction:
    model: Any
    entry: WeightSnapshot
    committed: bool = False

    def rollback(self) -> dict[str, Any]:
        self.committed = False
        return self.entry.restore(self.model)

    def commit_terminal(self, terminal: WeightSnapshot) -> dict[str, Any]:
        terminal.restore(self.model)
        observed = terminal.current_root(self.model)
        if observed != terminal.root:
            raise ScientificBoundary("terminal atomic commit identity mismatch")
        self.committed = True
        return {"commit_count": 1, "terminal_root": observed, "intermediate_commit_count": 0}

