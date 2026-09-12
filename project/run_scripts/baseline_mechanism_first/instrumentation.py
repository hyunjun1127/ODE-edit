"""Compact E0/E1 observation primitives; baseline source remains unmodified.

Capture semantics derive from C3 ForwardCapture but deliberately expose no
edit_output callback. C4 restore is owned by fixtures, not these observers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Iterator

import torch


class ReadOnlyCapture:
    """Clone every call's input/output, returning None to the native forward."""
    def __init__(self, module: torch.nn.Module, *, destination: str | torch.device = "cpu"):
        self.module, self.destination = module, destination
        self.records: list[dict[str, torch.Tensor]] = []
        self.handle = None

    def _hook(self, module, args, kwargs, output):
        value = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(value, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError("native capture requires tensor input/output")
        self.records.append({"input": value.detach().to(self.destination, copy=True),
                             "output": output.detach().to(self.destination, copy=True)})
        return None

    def __enter__(self):
        if self.handle is not None:
            raise ValueError("capture already active")
        self.handle = self.module.register_forward_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


@dataclass(frozen=True)
class WriterObservation:
    """Detached native tensors; no inferred request/context-column mapping."""
    K: torch.Tensor
    R: torch.Tensor
    actual_delta: torch.Tensor
    column_metadata: tuple[dict[str, Any], ...]
    endpoint_delta: torch.Tensor | None = None
    endpoint_delta_fp64: torch.Tensor | None = None

    @classmethod
    def from_native(cls, K: torch.Tensor, R: torch.Tensor, delta: torch.Tensor, *,
                    column_metadata: list[dict[str, Any]], weight_before: torch.Tensor | None = None,
                    weight_after: torch.Tensor | None = None) -> "WriterObservation":
        if K.ndim != 2 or R.ndim != 2 or delta.shape != (R.shape[0], K.shape[0]):
            raise ValueError("writer observation requires explicit [output,input] direction")
        if R.shape[1] != K.shape[1] or len(column_metadata) != K.shape[1]:
            raise ValueError("every actual K/R column requires metadata")
        if (weight_before is None) != (weight_after is None):
            raise ValueError("endpoint subtraction requires both weights")
        endpoint = endpoint_fp64 = None
        if weight_before is not None:
            if weight_before.shape != delta.shape or weight_after.shape != delta.shape:
                raise ValueError("endpoint weight orientation mismatch")
            endpoint = (weight_after.detach() - weight_before.detach()).clone()
            # Stored endpoint operands converted BEFORE subtraction: distinct
            # from promoting an already rounded FP32 difference to FP64.
            endpoint_fp64 = weight_after.detach().double() - weight_before.detach().double()
        return cls(K.detach().clone(), R.detach().clone(), delta.detach().clone(),
                   tuple(dict(m) for m in column_metadata), endpoint, endpoint_fp64)

    def materialization_receipt(self) -> dict[str, Any]:
        if self.endpoint_delta is None:
            return {"endpoint_delta_status": "NOT_OBSERVED"}
        difference = self.endpoint_delta - self.actual_delta
        return {"endpoint_delta_status": "OBSERVED", "endpoint_subtraction_includes_addition_rounding": True,
                "endpoint_minus_materialized_norm": float(difference.norm()),
                "endpoint_minus_materialized_max_abs": float(difference.abs().max()),
                "endpoint_subtraction_dtype": str(self.endpoint_delta.dtype),
                "endpoint_fp64_subtraction_minus_materialized_norm": float((self.endpoint_delta_fp64 - self.actual_delta.double()).norm()),
                "endpoint_equals_materialized_bitwise": torch.equal(self.endpoint_delta, self.actual_delta)}


class NativeTargetObserver:
    """Receive source-observed scalars only; absent optimizer facts stay absent.

    Source wrappers may pass terminal locals/events without changing compute-z.
    A cached z supplies no evidence about iteration/NLL/clamp and uses defaults.
    """
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def record(self, *, case_id: Any, source: str, training_nll: float | None = None,
               iterations: int | None = None, stop_reason: str | None = None,
               clamp_reached: bool | None = None) -> dict[str, Any]:
        if training_nll is not None and not math.isfinite(training_nll):
            raise ValueError("NONFINITE_TARGET_NLL")
        if iterations is not None and (isinstance(iterations, bool) or iterations < 0):
            raise ValueError("invalid observed iteration count")
        row = {"case_id": case_id, "source": source}
        for name, value in (("training_nll", training_nll), ("iterations", iterations),
                            ("stop_reason", stop_reason), ("clamp_reached", clamp_reached)):
            row[name] = value
            row[name + "_status"] = "NOT_OBSERVED" if value is None else "OBSERVED"
        self.rows.append(row)
        return dict(row)


@torch.no_grad()
def exposure_diagnostics(K: torch.Tensor, F: torch.Tensor, actual_delta: torch.Tensor,
                         queries: torch.Tensor, *, position_mask: torch.Tensor,
                         position_tags: list[str] | None = None) -> dict[str, Any]:
    """q rows include every true/new teacher-forced position, including padding.

    Caller binds token/position/mask/sequence identity. Raw, included and
    excluded summaries are all retained; diagnostic masks never affect write.
    """
    if queries.ndim != 2 or K.ndim != 2 or F.shape != K.shape or queries.shape[1] != K.shape[0]:
        raise ValueError("exposure geometry mismatch")
    if actual_delta.ndim != 2 or actual_delta.shape[1] != K.shape[0]:
        raise ValueError("dense direction orientation mismatch")
    if position_mask.shape != (queries.shape[0],) or position_mask.dtype != torch.bool:
        raise ValueError("all-position inclusion mask required")
    if position_tags is not None and len(position_tags) != queries.shape[0]:
        raise ValueError("position tags must cover all rows")
    raw_overlap = queries @ K
    weighted = queries @ F
    response = queries @ actual_delta.T
    if not all(bool(torch.isfinite(v).all()) for v in (raw_overlap, weighted, response)):
        raise ValueError("NONFINITE_EXPOSURE")
    def summary(mask):
        return {"position_count": int(mask.sum()),
                "raw_Ktq_squared_sum": float(raw_overlap[mask].square().sum()),
                "writer_Ftq_squared_sum": float(weighted[mask].square().sum()),
                "actual_delta_q_squared_sum": float(response[mask].square().sum())}
    result = {"raw": summary(torch.ones_like(position_mask)), "included": summary(position_mask),
              "excluded": summary(~position_mask), "writer_replaced": False,
              "formula": "raw=K.T@q; writer=F.T@q; actual=actual_dense_delta@q"}
    if position_tags is not None:
        result["position_tags_raw"] = {tag: summary(torch.tensor([v == tag for v in position_tags],
                                                                 device=queries.device))
                                       for tag in sorted(set(position_tags))}
    return result


@contextmanager
def measured_component(rows: list[dict[str, Any]], component: str, *,
                       parent_component: str | None = None) -> Iterator[None]:
    """Append host wall cost even on technical exceptions; no nested double sum."""
    start, status = time.perf_counter(), "COMPLETED"
    try:
        yield
    except BaseException:
        status = "TECHNICAL_EXCEPTION"
        raise
    finally:
        rows.append({"component": component, "parent_component": parent_component,
                     "wall_seconds": time.perf_counter() - start, "status": status,
                     "timing_kind": "HOST_WALL_NO_IMPLICIT_CUDA_SYNC",
                     "sum_as_top_level": parent_component is None})
