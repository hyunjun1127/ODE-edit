"""Current-state terminal-response JVP for one official-form layer factor.

The production operation is forward-mode AD through a temporary low-rank
module-output hook.  It never materializes or mutates a parameter.  Central
finite differences are exposed only for the preregistered runtime preamble.
"""

from __future__ import annotations

import math
import hashlib
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from project.run_scripts.ode_edit_motivation.hooks import resolve_module

from .adapters import LayerBuild
from .contracts import NumericalMethodBoundary, StaleStateBoundary, TechnicalBoundary
from .fp32_overlay import GroupedFP32Overlay


@contextmanager
def forward_ad_attention(model: torch.nn.Module) -> Iterator[bool]:
    """Assert the production-wide eager backend without per-observation switching."""

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation_internal"):
        # Compact synthetic fixtures need not implement the Transformers
        # attention contract; production models are sealed by
        # :func:`seal_eager_attention` immediately after load.
        yield False
        return
    if config._attn_implementation != "eager":
        raise TechnicalBoundary("production attention backend drifted from eager")
    yield False
    if config._attn_implementation != "eager":
        raise TechnicalBoundary("production attention backend changed during observation")


def seal_eager_attention(model: torch.nn.Module) -> dict[str, Any]:
    """Seal the backend selected by ``from_pretrained(attn_implementation='eager')``."""

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation_internal"):
        raise TechnicalBoundary("production model exposes no attention implementation seal")
    observed = getattr(config, "_attn_implementation", None)
    internal = getattr(config, "_attn_implementation_internal", None)
    if observed != "eager" or internal != "eager":
        raise TechnicalBoundary(
            f"model was not constructed with eager attention: observed={observed}, internal={internal}"
        )
    source_file = inspect.getsourcefile(type(model))
    source_sha256 = None
    if source_file is not None:
        try:
            with open(source_file, "rb") as handle:
                source_sha256 = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            source_file = None
    if source_file is None or source_sha256 is None:
        raise TechnicalBoundary("model class source identity is unavailable")
    config_payload = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "observed_backend": observed,
        "internal_backend": internal,
    }
    config_identity = hashlib.sha256(
        (json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "schema": "orbode.production-attention-backend.v1",
        "configured_at_model_load": True,
        "requested_backend": "eager",
        "observed_backend": observed,
        "internal_backend": internal,
        **config_payload,
        "config_identity_sha256": config_identity,
        "model_class_source_path": source_file,
        "model_class_source_sha256": source_sha256,
        "per_observation_backend_switch_count": 0,
        "terminal_jvp_evaluator_shared_backend": True,
    }


@dataclass(frozen=True, slots=True)
class TerminalJVPResult:
    terminal: torch.Tensor
    response: torch.Tensor
    built_state_version: int
    attention_backend_switched: bool
    wall_seconds: float

    def __post_init__(self) -> None:
        if (
            self.terminal.shape != self.response.shape
            or self.terminal.ndim != 2
            or self.terminal.dtype is not torch.float32
            or self.response.dtype is not torch.float32
            or self.terminal.requires_grad
            or self.response.requires_grad
            or not bool(torch.isfinite(self.terminal).all())
            or not bool(torch.isfinite(self.response).all())
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0.0
        ):
            raise NumericalMethodBoundary("terminal JVP output differs from FP32 contract")


@dataclass(frozen=True, slots=True)
class FiniteDifferenceReceipt:
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float
    maximum_absolute_error: float
    maximum_relative_error: float
    relative_l2_error: float
    cosine_similarity: float
    sign_agreement_fraction: float
    reference_l2_norm: float
    finite_difference_l2_norm: float
    rounding_l2_envelope: float
    allclose: bool


@dataclass(slots=True)
class TerminalJVPLedger:
    jvp_call_count: int = 0
    model_forward_invocation_count: int = 0
    finite_difference_forward_count: int = 0
    temporary_direction_hook_count: int = 0
    attention_backend_switch_count: int = 0
    physical_write_count: int = 0
    wall_seconds: float = 0.0


class TerminalResponseObserver:
    """Measure ``D H(W)[B]`` against the exact active grouped overlay."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        overlay: GroupedFP32Overlay,
        capture_terminal_graph: Callable[[], torch.Tensor],
    ) -> None:
        self.model = model
        self.overlay = overlay
        self.capture_terminal_graph = capture_terminal_graph
        self.ledger = TerminalJVPLedger()

    def _function(self, build: LayerBuild) -> Callable[[torch.Tensor], torch.Tensor]:
        module = resolve_module(self.model, build.weight_name.removesuffix(".weight"))
        device = next(self.model.parameters()).device
        left = build.left.to(device=device, dtype=torch.float32)
        right = build.right.to(device=device, dtype=torch.float32)

        def function(coefficient: torch.Tensor) -> torch.Tensor:
            if coefficient.shape or coefficient.dtype is not torch.float32 or coefficient.device != device:
                raise TechnicalBoundary("JVP coefficient must be a scalar FP32 on model device")

            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
                if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                    raise TechnicalBoundary("response hook saw unexpected module IO")
                hidden = inputs[0]
                if hidden.dtype is not torch.float32 or output.dtype is not torch.float32:
                    raise TechnicalBoundary("response hook left FULL_FP32")
                return output + coefficient * ((hidden @ right) @ left.transpose(0, 1))

            handle = module.register_forward_hook(hook)
            self.ledger.temporary_direction_hook_count += 1
            try:
                value = self.capture_terminal_graph()
                self.ledger.model_forward_invocation_count += 1
            finally:
                handle.remove()
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype is not torch.float32:
                raise TechnicalBoundary("terminal graph callback must return FP32 [D,B]")
            return value

        return function

    def observe(self, build: LayerBuild, *, expected_state_version: int) -> TerminalJVPResult:
        if build.built_state_version != expected_state_version or self.overlay.state_version != expected_state_version:
            raise StaleStateBoundary("JVP factor/current overlay state differs")
        self.overlay.assert_w0_unchanged(full_bytes=False)
        started = time.perf_counter()
        function = self._function(build)
        device = next(self.model.parameters()).device
        zero = torch.zeros((), dtype=torch.float32, device=device)
        one = torch.ones_like(zero)
        with forward_ad_attention(self.model) as switched:
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(zero, one)
                output = function(dual)
                primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
        if tangent is None:
            raise TechnicalBoundary("terminal forward JVP returned no tangent")
        if next(self.model.parameters()).device.type == "cuda":
            torch.cuda.synchronize(next(self.model.parameters()).device)
        elapsed = time.perf_counter() - started
        self.ledger.jvp_call_count += 1
        self.ledger.attention_backend_switch_count += int(switched)
        self.ledger.wall_seconds += elapsed
        self.overlay.assert_w0_unchanged(full_bytes=False)
        return TerminalJVPResult(
            terminal=primal.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            response=tangent.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            built_state_version=build.built_state_version,
            attention_backend_switched=bool(switched),
            wall_seconds=elapsed,
        )

    def audit_central_difference(
        self,
        build: LayerBuild,
        *,
        expected_state_version: int,
        epsilon: float,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> FiniteDifferenceReceipt:
        """Preamble-only identity check; never called by the main trajectory."""

        if build.built_state_version != expected_state_version or self.overlay.state_version != expected_state_version:
            raise StaleStateBoundary("FD factor/current overlay state differs")
        if not all(math.isfinite(value) and value > 0.0 for value in (epsilon, absolute_tolerance, relative_tolerance)):
            raise TechnicalBoundary("FD numerical lock is invalid")
        reference = self.observe(build, expected_state_version=expected_state_version)
        function = self._function(build)
        device = next(self.model.parameters()).device
        plus = function(torch.tensor(epsilon, dtype=torch.float32, device=device)).detach().cpu()
        minus = function(torch.tensor(-epsilon, dtype=torch.float32, device=device)).detach().cpu()
        self.ledger.finite_difference_forward_count += 2
        finite = (plus - minus) / (2.0 * epsilon)
        delta = (reference.response - finite).abs()
        denominator = torch.maximum(reference.response.abs(), finite.abs()).clamp_min(torch.finfo(torch.float32).tiny)
        maximum_absolute = float(delta.max().item()) if delta.numel() else 0.0
        maximum_relative = float((delta / denominator).max().item()) if delta.numel() else 0.0
        reference64 = reference.response.double().flatten()
        finite64 = finite.double().flatten()
        reference_norm = float(torch.linalg.vector_norm(reference64).item())
        finite_norm = float(torch.linalg.vector_norm(finite64).item())
        rounding_envelope = float(absolute_tolerance * math.sqrt(max(1, reference64.numel())))
        if reference_norm <= rounding_envelope:
            raise NumericalMethodBoundary(
                "ZERO_RESPONSE_ROUNDING_ENVELOPE: "
                f"norm={reference_norm}, envelope={rounding_envelope}"
            )
        relative_l2 = float(
            torch.linalg.vector_norm(reference64 - finite64).item()
            / max(reference_norm, finite_norm, torch.finfo(torch.float64).tiny)
        )
        cosine = float(
            torch.dot(reference64, finite64).item()
            / max(reference_norm * finite_norm, torch.finfo(torch.float64).tiny)
        )
        sign_active = torch.maximum(reference64.abs(), finite64.abs()) > absolute_tolerance
        sign_agreement = float(
            ((torch.sign(reference64[sign_active]) == torch.sign(finite64[sign_active])).double().mean()).item()
        ) if bool(sign_active.any()) else 1.0
        allclose = bool(torch.allclose(reference.response, finite, atol=absolute_tolerance, rtol=relative_tolerance))
        receipt = FiniteDifferenceReceipt(
            epsilon=epsilon,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_absolute_error=maximum_absolute,
            maximum_relative_error=maximum_relative,
            relative_l2_error=relative_l2,
            cosine_similarity=cosine,
            sign_agreement_fraction=sign_agreement,
            reference_l2_norm=reference_norm,
            finite_difference_l2_norm=finite_norm,
            rounding_l2_envelope=rounding_envelope,
            allclose=allclose,
        )
        if (
            not allclose
            or not math.isfinite(cosine)
            or cosine <= 0.0
            or sign_agreement != 1.0
        ):
            raise TechnicalBoundary(f"terminal JVP/FD preamble failed: {receipt}")
        return receipt


__all__ = [
    "FiniteDifferenceReceipt",
    "TerminalJVPResult",
    "TerminalJVPLedger",
    "TerminalResponseObserver",
    "forward_ad_attention",
    "seal_eager_attention",
]
