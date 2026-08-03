"""Outcome-independent compute accounting for one outer edit.

The counters in this module describe work performed by the method runtime;
they do not contain scientific outcomes.  GPU timing is opt-in so CPU unit
tests never initialize CUDA.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .contracts import MethodContractError, canonical_hash


COUNTER_NAMES = (
    "N_z",
    "N_state_fwd",
    "N_field",
    "N_bw",
    "K_acc",
    "N_trial",
    "N_reject",
    "N_eval",
    "N_write",
)

COMPONENT_NAMES = (
    "direct_z",
    "field",
    "qp",
    "trial",
    "event",
    "commit_write",
    "evaluation",
)

# A first hit freezes controller work.  Outcome evaluation is deliberately
# separate and may run only after the information firewall opens.
_FORBIDDEN_AFTER_HIT = frozenset(
    {
        "N_z",
        "N_state_fwd",
        "N_field",
        "N_bw",
        "K_acc",
        "N_trial",
        "N_reject",
        "N_write",
    }
)
_CONTROLLER_COMPONENTS = frozenset(
    {"direct_z", "field", "qp", "trial", "commit_write"}
)


@dataclass(frozen=True, slots=True)
class InstrumentationSnapshot:
    edit_id: str
    counters: tuple[tuple[str, int], ...]
    cpu_seconds: tuple[tuple[str, float], ...]
    gpu_seconds: tuple[tuple[str, float], ...]
    gpu_seconds_per_edit: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    first_hit: bool
    snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-compute-accounting/v1",
            "edit_id": self.edit_id,
            "counters": dict(self.counters),
            "component_cpu_seconds": dict(self.cpu_seconds),
            "component_gpu_seconds": dict(self.gpu_seconds),
            "gpu_seconds_per_edit": self.gpu_seconds_per_edit,
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": self.peak_memory_reserved_bytes,
            "first_hit": self.first_hit,
            "snapshot_id": self.snapshot_id,
        }


class EditInstrumentation:
    """Mutable edit-local recorder that freezes into a hashable snapshot."""

    def __init__(
        self,
        edit_id: str,
        *,
        gpu_timing: bool = False,
        gpu_device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(edit_id, str) or not edit_id:
            raise MethodContractError("instrumentation edit identity is empty")
        self.edit_id = edit_id
        self._counters = {name: 0 for name in COUNTER_NAMES}
        self._cpu_seconds = {name: 0.0 for name in COMPONENT_NAMES}
        self._gpu_seconds = {name: 0.0 for name in COMPONENT_NAMES}
        self._gpu_event_pairs: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._first_hit = False
        self._finalized = False
        self._active_components: set[str] = set()
        self._gpu_device: torch.device | None = None
        if gpu_timing:
            if not torch.cuda.is_available():
                raise MethodContractError("GPU timing requested without available CUDA")
            resolved = torch.device(gpu_device or "cuda")
            if resolved.type != "cuda":
                raise MethodContractError("GPU timing device is not CUDA")
            self._gpu_device = resolved
            torch.cuda.reset_peak_memory_stats(resolved)

    @property
    def first_hit(self) -> bool:
        return self._first_hit

    def increment(self, name: str, amount: int = 1) -> None:
        if self._finalized:
            raise MethodContractError("instrumentation is already finalized")
        if name not in self._counters:
            raise MethodContractError(f"unknown compute counter: {name}")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise MethodContractError("compute counter increment must be positive integer")
        if self._first_hit and name in _FORBIDDEN_AFTER_HIT:
            raise MethodContractError(f"controller work {name} attempted after first hit")
        self._counters[name] += amount

    @contextmanager
    def component(self, name: str) -> Iterator[None]:
        if self._finalized:
            raise MethodContractError("instrumentation is already finalized")
        if name not in self._cpu_seconds:
            raise MethodContractError(f"unknown timed component: {name}")
        if name in self._active_components:
            raise MethodContractError(f"timed component is recursively active: {name}")
        if self._first_hit and name in _CONTROLLER_COMPONENTS:
            raise MethodContractError(f"controller component {name} started after first hit")
        self._active_components.add(name)
        cpu_start = time.perf_counter()
        gpu_start: torch.cuda.Event | None = None
        gpu_end: torch.cuda.Event | None = None
        if self._gpu_device is not None:
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self._gpu_device):
                gpu_start.record()
        try:
            yield
        finally:
            if self._gpu_device is not None:
                assert gpu_start is not None and gpu_end is not None
                with torch.cuda.device(self._gpu_device):
                    gpu_end.record()
                self._gpu_event_pairs.append((name, gpu_start, gpu_end))
            self._cpu_seconds[name] += time.perf_counter() - cpu_start
            self._active_components.remove(name)

    def mark_first_hit(self) -> None:
        if self._finalized:
            raise MethodContractError("instrumentation is already finalized")
        if self._first_hit:
            raise MethodContractError("first hit may be marked only once")
        if self._active_components:
            raise MethodContractError("first hit cannot freeze active components")
        self._first_hit = True

    @contextmanager
    def state_forward(self, component: str = "event") -> Iterator[None]:
        """Count and time one full-model state forward."""

        self.increment("N_state_fwd")
        with self.component(component):
            yield

    def finalize(self) -> InstrumentationSnapshot:
        if self._finalized:
            raise MethodContractError("instrumentation may finalize only once")
        if self._active_components:
            raise MethodContractError("cannot finalize active component timers")
        peak_allocated = 0
        peak_reserved = 0
        if self._gpu_device is not None:
            torch.cuda.synchronize(self._gpu_device)
            for name, start, end in self._gpu_event_pairs:
                self._gpu_seconds[name] += float(start.elapsed_time(end)) / 1000.0
            peak_allocated = int(torch.cuda.max_memory_allocated(self._gpu_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self._gpu_device))
        self._finalized = True
        payload = {
            "schema_version": "ode-edit-compute-accounting/v1",
            "edit_id": self.edit_id,
            "counters": self._counters,
            "component_cpu_seconds": self._cpu_seconds,
            "component_gpu_seconds": self._gpu_seconds,
            "gpu_seconds_per_edit": sum(self._gpu_seconds.values()),
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "first_hit": self._first_hit,
        }
        return InstrumentationSnapshot(
            edit_id=self.edit_id,
            counters=tuple((name, self._counters[name]) for name in COUNTER_NAMES),
            cpu_seconds=tuple(
                (name, self._cpu_seconds[name]) for name in COMPONENT_NAMES
            ),
            gpu_seconds=tuple(
                (name, self._gpu_seconds[name]) for name in COMPONENT_NAMES
            ),
            gpu_seconds_per_edit=sum(self._gpu_seconds.values()),
            peak_memory_allocated_bytes=peak_allocated,
            peak_memory_reserved_bytes=peak_reserved,
            first_hit=self._first_hit,
            snapshot_id=canonical_hash(payload),
        )
