"""Call-scoped stock EasyEdit observer and independent call counters."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_bf.functional import tensor_sha256

from .contracts import LAYERS, Method, ObservationBoundary, ObservationLock
from .metrics import residual_debt_metrics, weight_action_energy


def _output_component(method: Method, value: Any) -> torch.Tensor:
    if method is Method.MEMIT:
        if not isinstance(value, torch.Tensor):
            raise ObservationBoundary("MEMIT activation return type differs")
        return value
    if not isinstance(value, (tuple, list)) or len(value) < 2 or not isinstance(value[1], torch.Tensor):
        raise ObservationBoundary("AlphaEdit activation return type differs")
    return value[1]


class OfficialLayerObserver:
    """Clone z and existing layer-loop outputs without changing return values."""

    def __init__(
        self,
        *,
        method: Method,
        request_sha256: Sequence[str],
        capture_layers: bool,
        replay_z: Sequence[torch.Tensor] | None = None,
    ) -> None:
        self.method = method
        self.request_sha256 = tuple(str(value) for value in request_sha256)
        self.capture_layers = bool(capture_layers)
        self._replay_z = (
            None
            if replay_z is None
            else tuple(
                value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                for value in replay_z
            )
        )
        self._z: list[torch.Tensor] = []
        self._z_hashes: list[str] = []
        self._layer_rows: list[torch.Tensor] = []
        self._terminal: torch.Tensor | None = None
        self._last_activation_args: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self._original_compute_z: Any = None
        self._original_activation: Any = None
        self._compute_wrapper: Any = None
        self._activation_wrapper: Any = None
        self._entered = False
        self._restored = False
        self._pass_through_layer_calls = 0
        self._captured_layer_calls = 0
        self._terminal_forward_calls = 0
        self._weight_action: dict[str, Any] | None = None
        self._z_copy_wall_seconds = 0.0
        self._layer_copy_wall_seconds = 0.0
        self._terminal_forward_wall_seconds = 0.0
        self._weight_action_wall_seconds = 0.0
        self._optimizer_compute_count = 0
        self._replay_count = 0

    @contextlib.contextmanager
    def observe(self, module: Any):
        if self._entered:
            raise ObservationBoundary("observer cannot be re-entered")
        self._entered = True
        original_compute_z = module.compute_z
        original_activation = module.get_module_input_output_at_words
        self._original_compute_z = original_compute_z
        self._original_activation = original_activation

        def compute_z_observer(*args: Any, **kwargs: Any) -> Any:
            if self._replay_z is None:
                value = original_compute_z(*args, **kwargs)
                self._optimizer_compute_count += 1
            else:
                index = self._replay_count
                if index >= len(self._replay_z):
                    raise ObservationBoundary("fixed-z replay call count exceeds seal")
                if not args or not hasattr(args[0], "parameters"):
                    raise ObservationBoundary("fixed-z replay model binding differs")
                device = next(args[0].parameters()).device
                value = self._replay_z[index].to(device=device, dtype=torch.float32)
                self._replay_count += 1
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ObservationBoundary("Official compute_z output differs/nonfinite")
            started = time.perf_counter()
            captured = value.detach().to("cpu", torch.float32).contiguous().clone()
            self._z.append(captured)
            self._z_hashes.append(tensor_sha256(captured))
            self._z_copy_wall_seconds += time.perf_counter() - started
            return value

        def activation_observer(*args: Any, **kwargs: Any) -> Any:
            value = original_activation(*args, **kwargs)
            self._pass_through_layer_calls += 1
            if self.capture_layers:
                output = _output_component(self.method, value)
                if not torch.isfinite(output).all():
                    raise ObservationBoundary("layer-loop activation is nonfinite")
                started = time.perf_counter()
                self._layer_rows.append(
                    output.detach().to("cpu", torch.float32).contiguous().clone()
                )
                self._layer_copy_wall_seconds += time.perf_counter() - started
                self._captured_layer_calls += 1
                self._last_activation_args = (args, dict(kwargs))
            return value

        self._compute_wrapper = compute_z_observer
        self._activation_wrapper = activation_observer
        module.compute_z = compute_z_observer
        module.get_module_input_output_at_words = activation_observer
        try:
            yield self
        finally:
            module.compute_z = original_compute_z
            module.get_module_input_output_at_words = original_activation
            self._restored = (
                module.compute_z is original_compute_z
                and module.get_module_input_output_at_words is original_activation
            )
            if not self._restored:
                raise ObservationBoundary("observer module-global identity restore failed")

    def capture_terminal(self, model: Any, tokenizer: Any) -> None:
        if not self.capture_layers:
            return
        if self._last_activation_args is None or self._original_activation is None:
            raise ObservationBoundary("layer-loop activation arguments are absent")
        args, kwargs = self._last_activation_args
        replaced = list(args)
        if len(replaced) >= 2:
            replaced[0], replaced[1] = model, tokenizer
        else:
            kwargs = {**kwargs, "model": model, "tok": tokenizer}
        started = time.perf_counter()
        value = self._original_activation(*tuple(replaced), **kwargs)
        output = _output_component(self.method, value)
        if not torch.isfinite(output).all():
            raise ObservationBoundary("terminal activation is nonfinite")
        self._terminal = output.detach().to("cpu", torch.float32).contiguous().clone()
        self._terminal_forward_calls += 1
        self._terminal_forward_wall_seconds += time.perf_counter() - started

    def capture_weight_action(
        self,
        touched: Mapping[str, torch.nn.Parameter],
        originals: Mapping[str, torch.Tensor],
    ) -> None:
        if self.capture_layers:
            started = time.perf_counter()
            self._weight_action = weight_action_energy(touched, originals)
            self._weight_action_wall_seconds += time.perf_counter() - started

    def payload(self, module: Any) -> dict[str, Any]:
        lock = ObservationLock()
        if not self._restored or module.compute_z is not self._original_compute_z or module.get_module_input_output_at_words is not self._original_activation:
            raise ObservationBoundary("observer restore receipt differs")
        if len(self._z) != len(self.request_sha256):
            raise ObservationBoundary("compute_z request count differs")
        common = {
            "schema": "odeedit.s06.official-layer-realization-debt.observer.v1",
            "method": self.method.value,
            "layer_capture_enabled": self.capture_layers,
            "request_count": len(self.request_sha256),
            "direct_z_compute_count": len(self._z),
            "direct_z_optimizer_compute_count": self._optimizer_compute_count,
            "direct_z_shared_replay_count": self._replay_count,
            "direct_z_recompute_count": 0,
            "z_sha256": list(self._z_hashes),
            "layer_loop_pass_through_call_count": self._pass_through_layer_calls,
            "layer_loop_observation_copy_count": self._captured_layer_calls,
            "terminal_post_L8_forward_count": self._terminal_forward_calls,
            "decision_or_update_tensor_mutation_count": 0,
            "raw_prompt_logit_publish_count": 0,
            "module_global_identity_restored": True,
            "overhead_wall_seconds": {
                "direct_z_hash_copy": self._z_copy_wall_seconds,
                "layer_activation_copy": self._layer_copy_wall_seconds,
                "terminal_post_L8_forward": self._terminal_forward_wall_seconds,
                "weight_action_reduction": self._weight_action_wall_seconds,
                "additional_z_optimization": 0,
                "additional_key_compute": 0,
                "additional_closed_form_solve": 0,
            },
        }
        if not self.capture_layers:
            if self._captured_layer_calls or self._terminal_forward_calls:
                raise ObservationBoundary("parity control unexpectedly captured activations")
            return {**common, "status": "PARITY_CONTROL_NO_LAYER_COPY"}
        if (
            self._captured_layer_calls != lock.expected_layer_loop_calls
            or self._pass_through_layer_calls != lock.expected_layer_loop_calls
            or self._terminal_forward_calls != lock.expected_terminal_forward_calls
            or self._terminal is None
            or self._weight_action is None
        ):
            raise ObservationBoundary("observer layer/terminal/action count differs")
        metrics = residual_debt_metrics(
            z_rows=self._z,
            pre_layer_rows=self._layer_rows,
            terminal_rows=self._terminal,
            request_sha256=self.request_sha256,
        )
        return {
            **common,
            "status": "TERMINAL_OBSERVATION_VALID",
            "activation_sha256": [tensor_sha256(value) for value in self._layer_rows],
            "terminal_activation_sha256": tensor_sha256(self._terminal),
            "residual_debt": metrics,
            "weight_action": self._weight_action,
        }

    def replay_tensors(self) -> tuple[torch.Tensor, ...]:
        if len(self._z) != len(self.request_sha256):
            raise ObservationBoundary("fixed-z replay tensor inventory differs")
        return tuple(value.detach().clone() for value in self._z)

    def raw_capture(self) -> dict[str, Any]:
        if (
            not self.capture_layers
            or len(self._layer_rows) != len(LAYERS)
            or self._terminal is None
        ):
            raise ObservationBoundary("raw observer capture is incomplete")
        return {
            "z": self.replay_tensors(),
            "pre_layer": tuple(value.detach().clone() for value in self._layer_rows),
            "terminal": self._terminal.detach().clone(),
        }


class OfficialCallAudit:
    """Count key and solve calls separately from the layer observer."""

    def __init__(self) -> None:
        self.compute_ks_calls = 0
        self.solve_calls = 0
        self._original_compute_ks: Any = None
        self._original_solve: Any = None
        self._restored = False

    @contextlib.contextmanager
    def observe(self, module: Any):
        original_compute_ks = module.compute_ks
        original_solve = torch.linalg.solve
        self._original_compute_ks = original_compute_ks
        self._original_solve = original_solve

        def counted_compute_ks(*args: Any, **kwargs: Any) -> Any:
            self.compute_ks_calls += 1
            return original_compute_ks(*args, **kwargs)

        def counted_solve(*args: Any, **kwargs: Any) -> Any:
            self.solve_calls += 1
            return original_solve(*args, **kwargs)

        module.compute_ks = counted_compute_ks
        torch.linalg.solve = counted_solve
        try:
            yield self
        finally:
            module.compute_ks = original_compute_ks
            torch.linalg.solve = original_solve
            self._restored = module.compute_ks is original_compute_ks and torch.linalg.solve is original_solve
            if not self._restored:
                raise ObservationBoundary("call-audit identity restore failed")

    def payload(self, module: Any) -> dict[str, Any]:
        if not self._restored or module.compute_ks is not self._original_compute_ks or torch.linalg.solve is not self._original_solve:
            raise ObservationBoundary("call-audit restore receipt differs")
        return {
            "compute_ks_call_count": self.compute_ks_calls,
            "torch_linalg_solve_call_count": self.solve_calls,
            "module_compute_ks_identity_restored": True,
            "torch_linalg_solve_identity_restored": True,
            "output_or_decision_mutation_count": 0,
        }
