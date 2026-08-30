"""Exact endpoint copy/apply/restore with bitwise receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import ScientificBoundary
from .hashing import canonical_hash, tensor_sha256


@dataclass
class EndpointSnapshot:
    tensors: dict[str, torch.Tensor]
    member_hashes: dict[str, str]
    root: str

    @classmethod
    def capture(cls, model: Any, names: list[str]) -> "EndpointSnapshot":
        params = dict(model.named_parameters())
        tensors = {name: params[name].detach().to("cpu").clone() for name in names}
        hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
        return cls(tensors=tensors, member_hashes=hashes, root=canonical_hash(hashes))

    def copy_to_model(self, model: Any) -> dict[str, object]:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in self.tensors.items():
                params[name].copy_(value.to(params[name].device, params[name].dtype))
        observed = {name: tensor_sha256(params[name]) for name in self.tensors}
        exact = observed == self.member_hashes
        if not exact:
            raise ScientificBoundary("exact endpoint snapshot copy/hash failed")
        return {"expected_root": self.root, "observed_root": canonical_hash(observed), "exact": True}

    def apply_correction(self, model: Any, name: str, correction: torch.Tensor) -> str:
        self.copy_to_model(model)
        parameter = dict(model.named_parameters())[name]
        with torch.no_grad():
            parameter.add_(correction.to(parameter.device, parameter.dtype))
        return tensor_sha256(parameter)

    def restore_after_candidate(self, model: Any) -> dict[str, object]:
        return self.copy_to_model(model)
