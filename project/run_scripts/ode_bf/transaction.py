"""One logical all-layer batch commit with complete rollback."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError
from .functional import tensor_sha256


@dataclass(frozen=True, slots=True)
class BatchCommitReceipt:
    transaction_id: str
    touched_weights: tuple[str, ...]
    commit_count: int
    rollback_count: int
    post_commit_verified: bool
    parameter_sha256: tuple[tuple[str, str], ...]


class AtomicBatchTransaction:
    def __init__(
        self,
        parameters: Mapping[str, torch.nn.Parameter],
        *,
        transaction_id: str,
        mutation_lock: threading.RLock,
    ) -> None:
        if not parameters:
            raise ODEBFContractError("batch transaction target set is empty")
        if not transaction_id:
            raise ODEBFContractError("batch transaction identity is empty")
        self.parameters = dict(parameters)
        self.transaction_id = transaction_id
        self.mutation_lock = mutation_lock
        self._staged: dict[str, torch.Tensor] = {}
        self._consumed = False

    def stage(self, name: str, candidate_bf16: torch.Tensor) -> None:
        if self._consumed:
            raise ODEBFStateError("batch transaction is already consumed")
        if name not in self.parameters or name in self._staged:
            raise ODEBFContractError("batch transaction stage target is absent or repeated")
        parameter = self.parameters[name]
        if (
            parameter.dtype is not torch.bfloat16
            or candidate_bf16.dtype is not torch.bfloat16
            or parameter.shape != candidate_bf16.shape
            or parameter.grad is not None
            or not torch.isfinite(candidate_bf16).all()
        ):
            raise ODEBFContractError("batch transaction staged tensor contract differs")
        self._staged[name] = candidate_bf16.detach().to(device="cpu").clone()

    def validate_complete(self) -> None:
        if set(self._staged) != set(self.parameters):
            raise ODEBFContractError("all touched layers must be staged before batch commit")

    def commit(
        self,
        *,
        post_commit_verify: Callable[[], bool],
        fault_after_writes: int | None = None,
    ) -> BatchCommitReceipt:
        if self._consumed:
            raise ODEBFStateError("batch transaction is already consumed")
        self.validate_complete()
        if fault_after_writes is not None and (
            isinstance(fault_after_writes, bool)
            or not isinstance(fault_after_writes, int)
            or fault_after_writes < 0
        ):
            raise ODEBFContractError("batch transaction fault index is invalid")
        names = tuple(sorted(self.parameters))
        rollback_count = 0
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
                            raise RuntimeError("injected joint batch commit fault")
                        candidate = self._staged[name].to(device=self.parameters[name].device)
                        self.parameters[name].copy_(candidate)
                        del candidate
                        writes += 1
                if any(self.parameters[name].data_ptr() != pointers[name] for name in names):
                    raise ODEBFContractError("batch commit changed a parameter storage pointer")
                if not bool(post_commit_verify()):
                    raise ODEBFStateError("post-commit joint verification failed")
            except BaseException:
                with torch.no_grad():
                    for name in names:
                        snapshot = snapshots[name].to(device=self.parameters[name].device)
                        self.parameters[name].copy_(snapshot)
                        del snapshot
                rollback_count = 1
                if any(
                    self.parameters[name].data_ptr() != pointers[name]
                    or not torch.equal(
                        self.parameters[name].detach().to(device="cpu"), snapshots[name]
                    )
                    for name in names
                ):
                    raise ODEBFStateError("joint batch rollback was not byte exact")
                self._staged.clear()
                self._consumed = True
                raise
            finally:
                snapshots.clear()
        self._staged.clear()
        self._consumed = True
        return BatchCommitReceipt(
            self.transaction_id,
            names,
            1,
            rollback_count,
            True,
            tuple((name, tensor_sha256(self.parameters[name])) for name in names),
        )
