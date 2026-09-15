"""CPU FP32, in-memory commit reference; no disk/crash-recovery guarantee.

One object snapshots a logical batch's entry W/M. Share its receipt dictionary
across batch objects to detect batch-ID conflicts during this process lifetime.
Production code still needs a durable atomic W/M/checkpoint/commit-ID protocol.
"""
from dataclasses import dataclass, replace
import hashlib
import math

import torch


class CommitError(RuntimeError):
    pass


class CommitParityError(CommitError):
    pass


class CommitConflict(CommitError):
    pass


class CommitBudgetError(CommitError):
    pass


def _digest(tensor):
    # Own exactly this tensor's storage, including when the input is a slice.
    # This CPU reference avoids an additional NumPy dependency.
    value = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
    text = str((tuple(value.shape), str(value.dtype))).encode()
    return hashlib.sha256(text + bytes(value.untyped_storage())).hexdigest()


def _fp32_cpu(value, name):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(name + " must be CPU FP32 in this reference")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def materialized_cost(delta, history, lambda_w=1.0, request_count=1):
    """Measure actual rounded delta in FP64 against the entry history geometry."""
    if not math.isfinite(lambda_w) or lambda_w < 0 or not isinstance(request_count, int) or request_count <= 0:
        raise ValueError("invalid cost geometry")
    if delta.ndim != 2 or history.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError("delta/history dimensions differ")
    d, m = delta.double(), history.double()
    return float((((d @ m) * d).sum() + lambda_w * d.square().sum()) / (2 * request_count))


@dataclass(frozen=True)
class CommitReceipt:
    batch_id: str
    payload_sha256: str
    entry_weight_sha256: str
    entry_history_sha256: str
    weight_sha256: str
    history_sha256: str
    actual_cost: float
    history_append: int = 1
    replayed: bool = False


class InMemoryBatchTransaction:
    """Single-threaded exactly-once commit with exclusive ownership of W/M.

    parity_checker(candidate_weight, actual_delta)->bool sees detached clones
    before any W/M write. Candidate W is FP32; actual_delta is the FP64
    subtraction of the two FP32 endpoints, without extra subtraction rounding.
    A checker exception or mutation restores owned W/M. Other external state,
    suffix parameters and RNG are outside this reference's rollback boundary.
    Candidate/rejected flow states should never call commit. Identical replay
    returns the existing receipt without checking or changing current W/M;
    a different payload under the same batch ID raises CommitConflict.
    """
    def __init__(self, batch_id, weight, history, receipts=None):
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("batch_id must be a nonempty string")
        _fp32_cpu(weight, "weight")
        _fp32_cpu(history, "history")
        if weight.ndim != 2 or history.shape != (weight.shape[1], weight.shape[1]):
            raise ValueError("history must have shape [d_in,d_in]")
        self.batch_id = batch_id
        self._weight, self._history = weight, history
        self._entry_weight = weight.detach().clone()
        self._entry_history = history.detach().clone()
        self._receipts = {} if receipts is None else receipts
        self._entry_weight_hash = _digest(self._entry_weight)
        self._entry_history_hash = _digest(self._entry_history)

    def commit(self, x, writer, keys, parity_checker, *, cost_budget=None,
               lambda_w=1.0, request_count=None, cost_tolerance=0.0):
        for value, name in ((x, "X"), (writer, "writer"), (keys, "keys")):
            _fp32_cpu(value, name)
        if (x.ndim != 2 or writer.ndim != 2 or keys.ndim != 2
                or x.shape[0] != self._weight.shape[0] or x.shape[1] != writer.shape[0]
                or writer.shape[1] != self._weight.shape[1] or keys.shape[0] != writer.shape[1]):
            raise ValueError("inconsistent X/writer/key dimensions")
        if not callable(parity_checker):
            raise ValueError("a parity checker is required")
        if request_count is None:
            request_count = x.shape[1]
        if (cost_budget is not None and (not math.isfinite(cost_budget) or cost_budget < 0)
                or not math.isfinite(cost_tolerance) or cost_tolerance < 0):
            raise ValueError("budget and tolerance must be finite and nonnegative")
        payload = "|".join((self._entry_weight_hash, self._entry_history_hash,
                            _digest(x), _digest(writer), _digest(keys),
                            repr((cost_budget, lambda_w, request_count, cost_tolerance))))
        payload_hash = hashlib.sha256(payload.encode()).hexdigest()
        previous = self._receipts.get(self.batch_id)
        if previous is not None:
            if previous.payload_sha256 != payload_hash:
                raise CommitConflict("batch ID already has a different committed payload")
            return replace(previous, replayed=True)
        if not torch.equal(self._weight, self._entry_weight) or not torch.equal(self._history, self._entry_history):
            raise CommitConflict("live weight/history differ from the batch entry snapshot")
        with torch.no_grad():
            candidate = self._entry_weight + x @ writer
            actual_delta = candidate.double() - self._entry_weight.double()
            # Native history order: CPU FP32 K@K.T, then entry M + Gram.
            candidate_history = self._entry_history + keys @ keys.T
        if not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(candidate_history).all()):
            raise CommitError("nonfinite materialized weight/history")
        cost = materialized_cost(actual_delta, self._entry_history, lambda_w, request_count)
        if not math.isfinite(cost):
            raise CommitBudgetError("nonfinite materialized cost")
        if cost_budget is not None and cost > cost_budget + cost_tolerance:
            raise CommitBudgetError("actual materialized delta exceeds the cost budget")
        try:
            parity = parity_checker(candidate.detach().clone(), actual_delta.detach().clone())
        except BaseException as error:
            with torch.no_grad():
                self._weight.copy_(self._entry_weight)
                self._history.copy_(self._entry_history)
            if isinstance(error, Exception):
                raise CommitParityError("external parity checker raised an exception") from error
            raise
        if not torch.equal(self._weight, self._entry_weight) or not torch.equal(self._history, self._entry_history):
            with torch.no_grad():
                self._weight.copy_(self._entry_weight)
                self._history.copy_(self._entry_history)
            raise CommitConflict("parity checker changed live weight/history")
        if parity is not True:
            raise CommitParityError("external full-write/suffix parity check failed")
        receipt = CommitReceipt(self.batch_id, payload_hash, self._entry_weight_hash,
                                self._entry_history_hash, _digest(candidate),
                                _digest(candidate_history), cost)
        # Best-effort Python exception rollback. Process death remains outside
        # this in-memory reference and needs the durable transaction adapter.
        try:
            with torch.no_grad():
                self._weight.copy_(candidate)
                self._history.copy_(candidate_history)
            self._receipts[self.batch_id] = receipt
        except BaseException:
            with torch.no_grad():
                self._weight.copy_(self._entry_weight)
                self._history.copy_(self._entry_history)
            self._receipts.pop(self.batch_id, None)
            raise
        return receipt
