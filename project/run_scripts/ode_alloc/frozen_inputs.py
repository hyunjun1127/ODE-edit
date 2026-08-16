"""Per-edit one-shot capture for direct-z, keys, covariance, and Native factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import ODEAllocContractError, canonical_hash


REQUIRED_INPUTS = ("direct_z", "key", "covariance", "native_factors")
Builder = Callable[[], Any]
IdentityFn = Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class FrozenInputReceipt:
    edit_id: str
    identities: tuple[tuple[str, str], ...]
    capture_counts: tuple[tuple[str, int], ...]
    receipt_id: str


@dataclass(slots=True)
class _Captured:
    payload: Any
    identity: str
    identity_fn: IdentityFn


class OneShotEditInputs:
    """Build each expensive Native input once, then reuse the same object."""

    def __init__(self, edit_id: str) -> None:
        if not isinstance(edit_id, str) or not edit_id:
            raise ODEAllocContractError("frozen input edit identity is empty")
        self.edit_id = edit_id
        self._captured: dict[str, _Captured] = {}
        self._sealed: FrozenInputReceipt | None = None

    def capture_from(
        self,
        name: str,
        builder: Builder,
        identity_fn: IdentityFn,
    ) -> Any:
        if name not in REQUIRED_INPUTS:
            raise ODEAllocContractError("unknown frozen edit input")
        if self._sealed is not None or name in self._captured:
            raise ODEAllocContractError(f"frozen edit input {name} was already captured")
        payload = builder()
        identity = identity_fn(payload)
        if not isinstance(identity, str) or len(identity) != 64:
            raise ODEAllocContractError("frozen edit input identity must be SHA-256")
        self._captured[name] = _Captured(payload, identity, identity_fn)
        return payload

    def read(self, name: str) -> Any:
        try:
            captured = self._captured[name]
        except KeyError as exc:
            raise ODEAllocContractError("frozen edit input was not captured") from exc
        self._assert_one_current(name, captured)
        return captured.payload

    @staticmethod
    def _assert_one_current(name: str, captured: _Captured) -> None:
        observed = captured.identity_fn(captured.payload)
        if observed != captured.identity:
            raise ODEAllocContractError(f"frozen edit input {name} changed after capture")

    def assert_current(self) -> None:
        for name, captured in self._captured.items():
            self._assert_one_current(name, captured)

    def seal(self) -> FrozenInputReceipt:
        if self._sealed is not None:
            self.assert_current()
            return self._sealed
        if set(self._captured) != set(REQUIRED_INPUTS):
            missing = sorted(set(REQUIRED_INPUTS) - set(self._captured))
            raise ODEAllocContractError(f"frozen edit inputs are incomplete: {missing}")
        self.assert_current()
        identities = tuple(
            (name, self._captured[name].identity) for name in REQUIRED_INPUTS
        )
        counts = tuple((name, 1) for name in REQUIRED_INPUTS)
        receipt_id = canonical_hash(
            {
                "schema": "ode-alloc-frozen-edit-inputs/v1",
                "edit_id": self.edit_id,
                "identities": identities,
                "capture_counts": counts,
            }
        )
        self._sealed = FrozenInputReceipt(
            edit_id=self.edit_id,
            identities=identities,
            capture_counts=counts,
            receipt_id=receipt_id,
        )
        return self._sealed
