"""Adapter-neutral cached suffix oracle; this is not a Llama runtime adapter.

The callback consumes hidden states [sequence, token, hidden] and returns full
logits [sequence, token, vocabulary]. All labels refer to prediction positions
in those logits: the adapter, not this module, owns tokenization and label shift.
Caches and teachers are detached, privately cloned at construction. The caller
must keep the suffix and all forward-affecting state fixed during a trajectory.
"""
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MicrobatchCache:
    h_entry: torch.Tensor                 # [batch, tokens, d_out]
    a: torch.Tensor                       # [batch, tokens, requests]
    edit_positions: torch.Tensor          # [edit_predictions, 2]: batch, token
    edit_labels: torch.Tensor             # [edit_predictions]
    edit_weights: torch.Tensor            # global request/context/token weights
    kl_positions: Optional[torch.Tensor] = None
    teacher_log_probs: Optional[torch.Tensor] = None
    kl_weights: Optional[torch.Tensor] = None


def make_affine_cache(h_entry, all_token_keys, writer, **labels):
    """Form A=BK for every token; keys are [batch,tokens,d_in], B is [m,d_in].

    This helper does not run a model prefix. Canonical native writer keys and
    the all-token training keys remain distinct inputs to the overall method.
    """
    if all_token_keys.ndim != 3 or writer.ndim != 2:
        raise ValueError("keys must be rank 3 and writer rank 2")
    if all_token_keys.shape[:2] != h_entry.shape[:2]:
        raise ValueError("hidden states and keys must cover the same tokens")
    if all_token_keys.shape[-1] != writer.shape[-1]:
        raise ValueError("writer/key input dimensions differ")
    with torch.no_grad():
        a = all_token_keys @ writer.T
    return MicrobatchCache(h_entry=h_entry, a=a, **labels)


def _clone_cache(cache):
    values = {}
    for name in MicrobatchCache.__dataclass_fields__:
        value = getattr(cache, name)
        values[name] = None if value is None else value.detach().clone()
    return MicrobatchCache(**values)


def _positions(positions, h, name):
    if positions.dtype != torch.long or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(name + " must be int64 [count,2]")
    if positions.device != h.device:
        raise ValueError(name + " and hidden states must share a device")
    if positions.numel() and (bool((positions < 0).any()) or
            bool((positions[:, 0] >= h.shape[0]).any()) or
            bool((positions[:, 1] >= h.shape[1]).any())):
        raise ValueError(name + " contains an invalid prediction position")


def _weights(weights, count, h, name):
    if weights.shape != (count,) or weights.device != h.device or not weights.is_floating_point():
        raise ValueError(name + " must be a floating vector on the cache device")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError(name + " must be finite and nonnegative")


class CachedSuffixOracle:
    """Return one complete logical (loss, X-gradient) without writing parameters.

    Edit weights must sum to one over ALL microbatches. For a request with c
    contexts and T target tokens the usual token weight is 1/(m*c*T).
    Optional KL weights independently sum to one. KL is the native reverse
    direction KL(p_current || p_entry), with a fixed entry teacher and beta=.0625.
    The quadratic preservation cost belongs to the integrator, not this oracle.
    """
    def __init__(self, suffix: Callable, microbatches: Iterable[MicrobatchCache], beta=0.0625):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self._suffix = suffix
        self._microbatches = tuple(_clone_cache(c) for c in microbatches)
        self.beta = float(beta)
        if not self._microbatches:
            raise ValueError("at least one microbatch is required")
        first = self._microbatches[0]
        if first.h_entry.ndim != 3 or first.a.ndim != 3:
            raise ValueError("h_entry and a must be rank 3")
        self.shape = (first.h_entry.shape[-1], first.a.shape[-1])
        self.device, self.dtype = first.h_entry.device, first.h_entry.dtype
        if self.dtype not in (torch.float32, torch.float64) or min(self.shape) <= 0:
            raise ValueError("reference caches require FP32/FP64 and nonempty dimensions")
        edit_total = kl_total = 0.0
        has_kl = False
        for c in self._microbatches:
            h = c.h_entry
            if (h.ndim != 3 or c.a.ndim != 3 or h.shape[:2] != c.a.shape[:2]
                    or (h.shape[-1], c.a.shape[-1]) != self.shape
                    or h.device != self.device or c.a.device != self.device
                    or h.dtype != self.dtype or c.a.dtype != self.dtype):
                raise ValueError("inconsistent affine cache shape/device/dtype")
            if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(c.a).all()):
                raise ValueError("nonfinite affine cache")
            _positions(c.edit_positions, h, "edit_positions")
            n = len(c.edit_positions)
            if c.edit_labels.shape != (n,) or c.edit_labels.dtype != torch.long or c.edit_labels.device != h.device:
                raise ValueError("edit_labels must be an int64 vector on the cache device")
            if bool((c.edit_labels < 0).any()):
                raise ValueError("edit labels cannot be negative")
            _weights(c.edit_weights, n, h, "edit_weights")
            edit_total += float(c.edit_weights.double().sum())
            optional = (c.kl_positions, c.teacher_log_probs, c.kl_weights)
            if any(v is not None for v in optional):
                if any(v is None for v in optional):
                    raise ValueError("KL positions, fixed teacher and weights are required together")
                has_kl = True
                _positions(c.kl_positions, h, "kl_positions")
                q = len(c.kl_positions)
                teacher = c.teacher_log_probs
                if (teacher.ndim != 2 or teacher.shape[0] != q or teacher.shape[1] == 0
                        or teacher.device != h.device or not teacher.is_floating_point()
                        or not bool(torch.isfinite(teacher).all())):
                    raise ValueError("invalid fixed teacher log probabilities")
                if not torch.allclose(teacher.logsumexp(-1), torch.zeros(q, device=h.device, dtype=teacher.dtype), atol=1e-5, rtol=0):
                    raise ValueError("teacher log probabilities must be normalized")
                _weights(c.kl_weights, q, h, "kl_weights")
                kl_total += float(c.kl_weights.double().sum())
        if not math.isclose(edit_total, 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("edit weights must sum to one over the logical batch")
        if has_kl and not math.isclose(kl_total, 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("KL weights must sum to one over the logical batch")

    def __call__(self, x):
        if x.shape != self.shape or x.dtype != self.dtype or x.device != self.device:
            raise ValueError("X shape/device/dtype does not match the cache")
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("nonfinite X")
        # Do not alter x.requires_grad, x.grad, caller caches, or parameter .grad.
        coordinate = x.detach().clone().requires_grad_(True)
        gradient = torch.zeros_like(coordinate)
        total = 0.0
        with torch.enable_grad():
            for c in self._microbatches:
                hidden = c.h_entry + c.a @ coordinate.T
                logits = self._suffix(hidden)
                if logits.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
                    raise ValueError("suffix must return full logits [batch,tokens,vocab]")
                pos = c.edit_positions
                selected = logits[pos[:, 0], pos[:, 1]]
                loss = (F.cross_entropy(selected, c.edit_labels, reduction="none") * c.edit_weights).sum()
                if c.kl_positions is not None and self.beta:
                    pos = c.kl_positions
                    student = logits[pos[:, 0], pos[:, 1]].log_softmax(-1)
                    if student.shape != c.teacher_log_probs.shape:
                        raise ValueError("teacher and current vocabulary dimensions differ")
                    reverse_kl = F.kl_div(c.teacher_log_probs, student,
                                        log_target=True, reduction="none").sum(-1)
                    loss = loss + self.beta * (reverse_kl * c.kl_weights).sum()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite suffix loss")
                current_gradient, = torch.autograd.grad(loss, coordinate)
                gradient.add_(current_gradient.detach())
                total += float(loss.detach())
                del logits, hidden, loss, current_gradient
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite X-gradient")
        return total, gradient
