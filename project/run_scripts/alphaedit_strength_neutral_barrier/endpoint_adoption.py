"""Exact endpoint adoption and FP32 replay diagnostics.

The guided writer temporarily constructs the authoritative endpoint and then
restores its internal W0 transaction.  This module deliberately stores the
endpoint itself, rather than a numerically lossy ``endpoint - W0`` delta.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Dict, Iterable

import torch

from easyeditor.util.device import copy_to_param


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def selected_sha(weights: Dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(weights):
        value = weights[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_exact_endpoint(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Capture one same-sized CPU snapshot of the temporary endpoint."""

    return {name: value.detach().cpu().clone() for name, value in weights.items()}


def adopt_exact_endpoint(
    weights: Dict[str, torch.Tensor], endpoint: Dict[str, torch.Tensor]
) -> Dict[str, object]:
    """Copy exact endpoint bytes into authoritative parameters, fail closed."""

    if set(weights) != set(endpoint):
        raise RuntimeError("endpoint/weight member mismatch")
    with torch.no_grad():
        for name, weight in weights.items():
            source = endpoint[name]
            if tuple(source.shape) != tuple(weight.shape) or source.dtype != weight.dtype:
                raise RuntimeError(f"endpoint identity mismatch: {name}")
            copy_to_param(weight, source)
    component_equal = {
        name: bool(torch.equal(weights[name].detach().cpu(), endpoint[name]))
        for name in weights
    }
    temporary_sha = selected_sha(endpoint)
    authoritative_sha = selected_sha(weights)
    passed = bool(all(component_equal.values()) and temporary_sha == authoritative_sha)
    if not passed:
        raise RuntimeError("exact temporary/authoritative endpoint adoption mismatch")
    return {
        "temporary_endpoint_selected_sha256": temporary_sha,
        "authoritative_endpoint_selected_sha256": authoritative_sha,
        "component_equal": component_equal,
        "pass": passed,
    }


def _ordered_ints(value: torch.Tensor) -> torch.Tensor:
    bits = value.detach().cpu().contiguous().view(torch.int32).flatten().to(torch.int64)
    return torch.where(bits < 0, -bits - 1, bits + 2**31)


def fp32_comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    ulp_thresholds: Iterable[int] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
) -> Dict[str, object]:
    """Return bounded exact/max-absolute/ULP diagnostics for two FP32 tensors."""

    if left.shape != right.shape or left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("FP32 comparison requires shape-matched FP32 tensors")
    left_cpu = left.detach().cpu().contiguous()
    right_cpu = right.detach().cpu().contiguous()
    left_flat = left_cpu.flatten()
    right_flat = right_cpu.flatten()
    left_ordered = _ordered_ints(left_cpu)
    right_ordered = _ordered_ints(right_cpu)
    ulp = (left_ordered - right_ordered).abs()
    absolute = (left_flat - right_flat).abs()
    max_ulp, ulp_index = torch.max(ulp, dim=0)
    max_abs, abs_index = torch.max(absolute, dim=0)
    ulp_flat_index = int(ulp_index.item())
    abs_flat_index = int(abs_index.item())
    return {
        "left_sha256": tensor_sha(left_cpu),
        "right_sha256": tensor_sha(right_cpu),
        "bitwise_equal": bool(torch.equal(left_cpu, right_cpu)),
        "max_abs": float(max_abs.item()),
        "max_abs_flat_index": abs_flat_index,
        "max_abs_left": float(left_flat[abs_flat_index].item()),
        "max_abs_right": float(right_flat[abs_flat_index].item()),
        "max_ulp": int(max_ulp.item()),
        "max_ulp_flat_index": ulp_flat_index,
        "max_ulp_left": float(left_flat[ulp_flat_index].item()),
        "max_ulp_right": float(right_flat[ulp_flat_index].item()),
        "ulp_threshold_exceed_count": {
            str(int(threshold)): int((ulp > int(threshold)).sum().item())
            for threshold in ulp_thresholds
        },
        "numel": int(left_flat.numel()),
    }
