"""Validated all-layer staging and one logical commit with exact rollback."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping

import torch

from .accounting import ComputeLedger
from .contracts import ODEAllocContractError


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    touched_weights: tuple[str, ...]
    commit_count: int
    rolled_back: bool


class AtomicLayerTransaction:
    """Logical atomicity under a caller-shared exclusive model mutation lock."""

    def __init__(
        self,
        parameters: Mapping[str, torch.nn.Parameter],
        *,
        mutation_lock: threading.RLock,
    ) -> None:
        if not parameters:
            raise ODEAllocContractError("transaction target set is empty")
        self.parameters = dict(parameters)
        self.mutation_lock = mutation_lock
        self._staged: dict[str, torch.Tensor] = {}

    def stage(self, name: str, candidate_bf16: torch.Tensor) -> None:
        if name not in self.parameters or name in self._staged:
            raise ODEAllocContractError("transaction stage target is absent or repeated")
        parameter = self.parameters[name]
        if (
            parameter.dtype is not torch.bfloat16
            or candidate_bf16.dtype is not torch.bfloat16
            or parameter.shape != candidate_bf16.shape
            or parameter.grad is not None
        ):
            raise ODEAllocContractError("transaction staged tensor contract differs")
        if not torch.isfinite(candidate_bf16).all():
            raise ODEAllocContractError("transaction staged tensor is non-finite")
        # All-layer atomicity needs all candidates staged, but the model-side
        # contract permits only one effective BF16 target weight live at once.
        # Store the validated candidates on CPU and stream one layer per write.
        self._staged[name] = candidate_bf16.detach().to(device="cpu").clone()

    def validate_complete(self) -> None:
        if set(self._staged) != set(self.parameters):
            raise ODEAllocContractError("all touched layers must be staged before commit")

    def commit(
        self,
        *,
        ledger: ComputeLedger | None = None,
        fault_after_writes: int | None = None,
    ) -> CommitReceipt:
        self.validate_complete()
        if fault_after_writes is not None and (
            isinstance(fault_after_writes, bool)
            or not isinstance(fault_after_writes, int)
            or fault_after_writes < 0
        ):
            raise ODEAllocContractError("fault injection index is invalid")
        names = tuple(sorted(self.parameters))
        with self.mutation_lock:
            pointers = {name: self.parameters[name].data_ptr() for name in names}
            snapshots = {
                name: self.parameters[name].detach().to(device="cpu").clone()
                for name in names
            }
            writes = 0
            try:
                with torch.no_grad():
                    for name in names:
                        if fault_after_writes is not None and writes == fault_after_writes:
                            raise RuntimeError("injected all-layer commit fault")
                        self.parameters[name].copy_(self._staged[name])
                        writes += 1
                if any(self.parameters[name].data_ptr() != pointers[name] for name in names):
                    raise ODEAllocContractError("commit changed parameter storage pointer")
            except BaseException:
                with torch.no_grad():
                    for name in names:
                        self.parameters[name].copy_(snapshots[name])
                if any(
                    self.parameters[name].data_ptr() != pointers[name]
                    or not torch.equal(
                        self.parameters[name].detach().to(device="cpu"), snapshots[name]
                    )
                    for name in names
                ):
                    raise ODEAllocContractError("transaction rollback was not byte exact")
                self._staged.clear()
                raise
            finally:
                snapshots.clear()
        self._staged.clear()
        if ledger is not None:
            ledger.increment("commit_count")
        return CommitReceipt(names, 1, False)
