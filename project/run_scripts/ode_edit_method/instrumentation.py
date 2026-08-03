"""Outcome-independent, non-overlapping compute accounting for one outer edit.

``N_trial`` counts logical low-rank trials and is intentionally independent of
full-model forwards.  ``N_model_fwd`` is the actual number of top-level model
calls; the event/field counters classify a subset of those calls.  Attaching
the recorder to a model is therefore the production path.  CPU fixtures may
record explicit calls through :meth:`record_model_forward`.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .contracts import MethodContractError, canonical_hash


COUNTER_NAMES = (
    "N_model_fwd",
    "N_event_fwd",
    "N_field_state_fwd",
    "N_field",
    "N_proposal_build",
    "N_native_sweep",
    "N_bw",
    "N_reference_gate_fwd",
    "N_reference_gate_bw",
    "N_trial",
    "N_reject",
    "N_eval",
    "N_write",
    "K_acc",
    "N_z",
)

COMPONENT_NAMES = (
    "context_setup",
    "entry_checkpoint",
    "restore",
    "terminal_geometry",
    "direct_z",
    "native_proposal",
    "proposal",
    "trust_scale",
    "event",
    "field",
    "backward_hook",
    "reference_gate",
    "qp",
    "trial",
    "commit_write",
    "evaluation",
)

_FORBIDDEN_AFTER_HIT = frozenset(
    {
        "N_model_fwd",
        "N_event_fwd",
        "N_field_state_fwd",
        "N_field",
        "N_proposal_build",
        "N_native_sweep",
        "N_bw",
        "N_reference_gate_fwd",
        "N_reference_gate_bw",
        "N_trial",
        "N_reject",
        "N_write",
        "K_acc",
        "N_z",
    }
)
_FORBIDDEN_COMPONENTS_AFTER_HIT = frozenset(
    {
        "direct_z",
        "native_proposal",
        "proposal",
        "trust_scale",
        "field",
        "backward_hook",
        "reference_gate",
        "qp",
        "trial",
        "commit_write",
    }
)
_FORWARD_CATEGORIES = {
    "event": "N_event_fwd",
    "field": "N_field_state_fwd",
    "reference_gate": "N_reference_gate_fwd",
}


@dataclass(frozen=True, slots=True)
class InstrumentationSnapshot:
    edit_id: str
    counters: tuple[tuple[str, int], ...]
    cpu_seconds: tuple[tuple[str, float], ...]
    gpu_seconds: tuple[tuple[str, float], ...]
    controller_wall_seconds: float
    setup_wall_seconds: float
    editor_only_seconds_per_edit: float
    setup_amortized_seconds_per_edit: float
    controller_gpu_seconds: float
    evaluation_gpu_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    first_hit: bool
    snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-compute-accounting/v2",
            "edit_id": self.edit_id,
            "counters": dict(self.counters),
            "component_wall_seconds": dict(self.cpu_seconds),
            "component_gpu_seconds": dict(self.gpu_seconds),
            "controller_wall_seconds": self.controller_wall_seconds,
            "setup_wall_seconds": self.setup_wall_seconds,
            "editor_only_seconds_per_edit": self.editor_only_seconds_per_edit,
            "setup_amortized_seconds_per_edit": self.setup_amortized_seconds_per_edit,
            "controller_gpu_seconds": self.controller_gpu_seconds,
            "evaluation_gpu_seconds": self.evaluation_gpu_seconds,
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
        self._gpu_event_pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._first_hit = False
        self._finalized = False
        self._active_components: set[str] = set()
        self._forward_categories: list[str | None] = []
        self._model_hook: torch.utils.hooks.RemovableHandle | None = None
        self._controller_started: float | None = None
        self._controller_wall_seconds = 0.0
        self._controller_gpu_start: torch.cuda.Event | None = None
        self._controller_gpu_end: torch.cuda.Event | None = None
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

    @property
    def tracks_model_forwards(self) -> bool:
        return self._model_hook is not None

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

    def start_controller(self) -> None:
        if self._controller_started is not None:
            raise MethodContractError("controller wall timer is already active")
        self._controller_started = time.perf_counter()
        if self._gpu_device is not None:
            self._controller_gpu_start = torch.cuda.Event(enable_timing=True)
            self._controller_gpu_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self._gpu_device):
                self._controller_gpu_start.record()

    def stop_controller(self) -> None:
        if self._controller_started is None:
            raise MethodContractError("controller wall timer is not active")
        self._controller_wall_seconds += time.perf_counter() - self._controller_started
        if self._gpu_device is not None:
            assert self._controller_gpu_end is not None
            with torch.cuda.device(self._gpu_device):
                self._controller_gpu_end.record()
        self._controller_started = None

    def add_wall_seconds(self, component: str, seconds: float) -> None:
        """Attach a pre-measured run-level component such as setup amortization."""

        value = float(seconds)
        if component not in self._cpu_seconds or not math.isfinite(value) or value < 0.0:
            raise MethodContractError("external wall component is invalid")
        if self._finalized or component in self._active_components:
            raise MethodContractError("cannot attach wall time to active/final recorder")
        self._cpu_seconds[component] += value

    @contextmanager
    def component(self, name: str) -> Iterator[None]:
        if self._finalized:
            raise MethodContractError("instrumentation is already finalized")
        if name not in self._cpu_seconds:
            raise MethodContractError(f"unknown timed component: {name}")
        if name in self._active_components:
            raise MethodContractError(f"timed component is recursively active: {name}")
        if self._first_hit and name in _FORBIDDEN_COMPONENTS_AFTER_HIT:
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

    @contextmanager
    def model_forward_scope(self, category: str | None) -> Iterator[None]:
        if category is not None and category not in _FORWARD_CATEGORIES:
            raise MethodContractError(f"unknown model-forward category: {category}")
        self._forward_categories.append(category)
        try:
            yield
        finally:
            self._forward_categories.pop()

    def record_model_forward(self, category: str | None = None) -> None:
        selected = category if category is not None else (
            self._forward_categories[-1] if self._forward_categories else None
        )
        self.increment("N_model_fwd")
        if selected is not None:
            self.increment(_FORWARD_CATEGORIES[selected])

    @contextmanager
    def state_forward(self, category: str = "field") -> Iterator[None]:
        """CPU-fixture helper for one explicit full-model call.

        Production code attaches the top-level model hook instead.  This
        helper does not time the surrounding operation; callers use a
        component timer separately when required.
        """

        self.record_model_forward(category)
        yield

    def attach_model(self, model: torch.nn.Module) -> None:
        """Count actual top-level model calls until :meth:`detach_model`."""

        if self._model_hook is not None:
            raise MethodContractError("full-model forward hook is already attached")

        def count(_module: torch.nn.Module, _inputs: Any) -> None:
            self.record_model_forward()

        self._model_hook = model.register_forward_pre_hook(count)

    def detach_model(self) -> None:
        if self._model_hook is None:
            raise MethodContractError("full-model forward hook is not attached")
        self._model_hook.remove()
        self._model_hook = None

    def mark_first_hit(self) -> None:
        if self._finalized:
            raise MethodContractError("instrumentation is already finalized")
        if self._first_hit:
            raise MethodContractError("first hit may be marked only once")
        if self._active_components:
            raise MethodContractError("first hit cannot freeze active components")
        self._first_hit = True

    def finalize(self) -> InstrumentationSnapshot:
        if self._finalized:
            raise MethodContractError("instrumentation may finalize only once")
        if self._active_components or self._controller_started is not None:
            raise MethodContractError("cannot finalize active timers")
        if self._model_hook is not None:
            raise MethodContractError("detach full-model counter before finalize")
        categorized = sum(
            self._counters[counter] for counter in _FORWARD_CATEGORIES.values()
        )
        if categorized > self._counters["N_model_fwd"]:
            raise MethodContractError("categorized forwards exceed actual model forwards")
        peak_allocated = 0
        peak_reserved = 0
        if self._gpu_device is not None:
            torch.cuda.synchronize(self._gpu_device)
            for name, start, end in self._gpu_event_pairs:
                self._gpu_seconds[name] += float(start.elapsed_time(end)) / 1000.0
            peak_allocated = int(torch.cuda.max_memory_allocated(self._gpu_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self._gpu_device))
        controller_gpu_seconds = 0.0
        if self._gpu_device is not None:
            assert self._controller_gpu_start is not None
            assert self._controller_gpu_end is not None
            controller_gpu_seconds = float(
                self._controller_gpu_start.elapsed_time(self._controller_gpu_end)
            ) / 1000.0
        evaluation_gpu_seconds = self._gpu_seconds["evaluation"]
        setup_wall_seconds = self._cpu_seconds["context_setup"]
        self._finalized = True
        payload = {
            "schema_version": "ode-edit-compute-accounting/v2",
            "edit_id": self.edit_id,
            "counters": self._counters,
            "component_wall_seconds": self._cpu_seconds,
            "component_gpu_seconds": self._gpu_seconds,
            "controller_wall_seconds": self._controller_wall_seconds,
            "setup_wall_seconds": setup_wall_seconds,
            "editor_only_seconds_per_edit": self._controller_wall_seconds,
            "setup_amortized_seconds_per_edit": self._controller_wall_seconds
            + setup_wall_seconds,
            "controller_gpu_seconds": controller_gpu_seconds,
            "evaluation_gpu_seconds": evaluation_gpu_seconds,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "first_hit": self._first_hit,
        }
        return InstrumentationSnapshot(
            edit_id=self.edit_id,
            counters=tuple((name, self._counters[name]) for name in COUNTER_NAMES),
            cpu_seconds=tuple((name, self._cpu_seconds[name]) for name in COMPONENT_NAMES),
            gpu_seconds=tuple((name, self._gpu_seconds[name]) for name in COMPONENT_NAMES),
            controller_wall_seconds=self._controller_wall_seconds,
            setup_wall_seconds=setup_wall_seconds,
            editor_only_seconds_per_edit=self._controller_wall_seconds,
            setup_amortized_seconds_per_edit=self._controller_wall_seconds
            + setup_wall_seconds,
            controller_gpu_seconds=controller_gpu_seconds,
            evaluation_gpu_seconds=evaluation_gpu_seconds,
            peak_memory_allocated_bytes=peak_allocated,
            peak_memory_reserved_bytes=peak_reserved,
            first_hit=self._first_hit,
            snapshot_id=canonical_hash(payload),
        )
