"""Functional full-model observation of W(theta) with serial forward JVPs."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch

from project.run_scripts.barrier_guided_ode.r3.events import FineEventLayout, PrefixDirectionalObservation
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import _forward_ad_attention
from project.run_scripts.ode_edit_motivation.contracts import MemitFactorProposal
from project.run_scripts.ode_edit_motivation.hooks import resolve_module


class FunctionalObservationBoundary(RuntimeError):
    """The fixed-basis functional observation failed."""


@dataclass(slots=True)
class ObservationLedger:
    model_forward_invocations: int = 0
    jvp_call_count: int = 0
    primal_prefix_count: int = 0
    temporary_materialization_count: int = 0
    physical_write_count: int = 0
    wall_seconds: float = 0.0


class FixedBasisObserver:
    """Measure logits/JVP at a combined five-layer affine state without mutating W."""

    def __init__(self, model: torch.nn.Module, tokenization: Any, proposal: MemitFactorProposal) -> None:
        self.model = model
        self.tokenization = tokenization
        self.proposal = proposal
        self.device = next(model.parameters()).device
        self.ledger = ObservationLedger()
        self._factors = tuple(
            (
                factor.weight_name,
                factor.left.to(device=self.device, dtype=torch.float32),
                factor.right.to(device=self.device, dtype=torch.float32),
            )
            for factor in proposal.factors
        )

    def _function(self, prefix: tuple[int, ...]):
        ids = torch.tensor([self.tokenization.prompt_token_ids + prefix], dtype=torch.long, device=self.device)

        def function(theta: torch.Tensor) -> torch.Tensor:
            if theta.dtype != torch.float32 or theta.shape != (len(self._factors),) or theta.device != self.device:
                raise FunctionalObservationBoundary("functional theta has wrong dtype/shape/device")
            with ExitStack() as stack:
                for index, (weight_name, left, right) in enumerate(self._factors):
                    module = resolve_module(self.model, weight_name.removesuffix(".weight"))

                    def hook(_module: torch.nn.Module, args: tuple[Any, ...], output: Any, *, i=index, l=left, r=right):
                        if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
                            raise FunctionalObservationBoundary("fixed actuator hook saw unexpected module IO")
                        return output + theta[i] * ((args[0] @ r) @ l.transpose(0, 1))

                    stack.callback(module.register_forward_hook(hook).remove)
                self.ledger.model_forward_invocations += 1
                return self.model(input_ids=ids, use_cache=False).logits[0, -1].float()

        return function

    def observe(self, layout: FineEventLayout, theta: torch.Tensor, *, with_jvp: bool) -> dict[tuple[int, ...], PrefixDirectionalObservation]:
        if theta.dtype != torch.float64 or theta.shape != (len(self._factors),) or not bool(torch.isfinite(theta).all()):
            raise FunctionalObservationBoundary("observation theta must be detached finite FP64")
        started = time.perf_counter()
        theta32 = theta.to(device=self.device, dtype=torch.float32)
        result: dict[tuple[int, ...], PrefixDirectionalObservation] = {}
        with _forward_ad_attention(self.model):
            for prefix in layout.internal_prefixes:
                function = self._function(prefix)
                if not with_jvp:
                    with torch.inference_mode():
                        primal = function(theta32).detach()
                    tangent_matrix = None
                else:
                    primals: list[torch.Tensor] = []
                    tangents: list[torch.Tensor] = []
                    for index in range(theta.numel()):
                        direction = torch.zeros_like(theta32)
                        direction[index] = 1.0
                        with torch.autograd.forward_ad.dual_level():
                            dual = torch.autograd.forward_ad.make_dual(theta32, direction)
                            output = function(dual)
                            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
                        if tangent is None:
                            raise FunctionalObservationBoundary("forward JVP returned no tangent")
                        self.ledger.jvp_call_count += 1
                        primals.append(primal.detach())
                        tangents.append(tangent.detach())
                    primal = primals[0]
                    if any(not torch.equal(primal, other) for other in primals[1:]):
                        raise FunctionalObservationBoundary("JVP primal bytes differ by direction")
                    tangent_matrix = torch.stack(tangents, dim=1).to(dtype=torch.float32)
                self.ledger.primal_prefix_count += 1
                result[prefix] = PrefixDirectionalObservation(
                    logits=primal.cpu().contiguous(),
                    tangent_logits=None if tangent_matrix is None else tangent_matrix.cpu().contiguous(),
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.ledger.wall_seconds += time.perf_counter() - started
        return result
