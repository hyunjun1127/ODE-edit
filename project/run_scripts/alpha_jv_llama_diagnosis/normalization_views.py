"""Frozen source N0 and explicitly named D-only NRMS objective views.

Neither view changes the native RHS/request inventory. Inputs preserve source
FP32 subtraction/norm rounding; only weighting and the NRMS RMS use FP64.
"""
from dataclasses import dataclass
import hashlib
import json

import torch

from project.run_scripts.native_response_ode_v31.algebra import (
    FrozenNormalization, NativeMetricBoundary,
)


class NormalizationBoundary(NativeMetricBoundary):
    pass


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + value.numpy().tobytes()).hexdigest()


def source_residual(target, terminal):
    """Do not replace this operation by double(target)-double(terminal)."""
    if (not isinstance(target, torch.Tensor) or not isinstance(terminal, torch.Tensor)
            or target.dtype != torch.float32 or terminal.dtype != torch.float32
            or target.ndim != 2 or target.shape != terminal.shape):
        raise NormalizationBoundary("SOURCE_FP32_TARGET_TERMINAL_SHAPE")
    result = target - terminal
    if not torch.isfinite(result).all():
        raise NormalizationBoundary("NONFINITE_SOURCE_RESIDUAL")
    return result


@dataclass(frozen=True)
class NormalizationView:
    """Clone/seal source scales; delegated N0 bytes are unchanged."""
    _source: FrozenNormalization
    normalization_id: str
    _source_scales_sha256: str
    _active_mask_sha256: str

    @classmethod
    def from_source(cls, source, normalization_id="N0_SOURCE"):
        if normalization_id not in ("N0_SOURCE", "NRMS_ENTRY"):
            raise NormalizationBoundary("UNAUTHORIZED_NORMALIZATION_ID")
        if (not isinstance(source, FrozenNormalization)
                or source.scales.dtype != torch.float32
                or source.active.dtype != torch.bool
                or source.scales.ndim != 1 or source.active.shape != source.scales.shape
                or not torch.isfinite(source.scales).all()
                or bool((source.scales < 0).any())
                or not torch.equal(source.active, source.scales > 0)):
            raise NormalizationBoundary("SOURCE_NORMALIZATION_INVENTORY")
        copied = FrozenNormalization(source.scales.detach().cpu().clone(),
                                     source.active.detach().cpu().clone(), source.entry_sha)
        return cls(copied, normalization_id, tensor_identity(copied.scales),
                   tensor_identity(copied.active))

    def assert_frozen(self):
        if (tensor_identity(self._source.scales) != self._source_scales_sha256
                or tensor_identity(self._source.active) != self._active_mask_sha256):
            raise NormalizationBoundary("FROZEN_NORMALIZATION_MUTATED")

    @property
    def scales(self):
        self.assert_frozen()
        return self._source.scales.clone()

    @property
    def active(self):
        self.assert_frozen()
        return self._source.active.clone()

    @property
    def entry_sha(self):
        return self._source.entry_sha

    @property
    def denominators(self):
        """Full B inventory; inactive rows are explicitly NA, never divided."""
        self.assert_frozen()
        active = self._source.active
        count = int(active.sum())
        result = torch.full(self._source.scales.shape, float("nan"), dtype=torch.float64)
        if count:
            scales64 = self._source.scales[active].double()
            if self.normalization_id == "N0_SOURCE":
                result[active] = scales64 * count ** 0.5
            else:
                rms = scales64.square().mean().sqrt()
                result[active] = rms * count ** 0.5
            if not torch.isfinite(result[active]).all() or bool((result[active] <= 0).any()):
                raise NormalizationBoundary("NONFINITE_POSITIVE_DENOMINATOR")
        return result

    def weight(self, tensor):
        self.assert_frozen()
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.ndim != 2 or tensor.shape[1] != self._source.scales.numel()):
            raise NormalizationBoundary("SOURCE_FP32_WEIGHT_INPUT_SHAPE")
        if not torch.isfinite(tensor).all():
            raise NormalizationBoundary("NONFINITE_WEIGHT_INPUT")
        if self.normalization_id == "N0_SOURCE":
            return self._source.weight(tensor)
        active = self._source.active
        return (tensor.detach().cpu().double()[:, active]
                / self.denominators[active]).reshape(-1)

    def receipt(self):
        denominators = self.denominators
        active = self.active
        return dict(normalization_id=self.normalization_id,
                    entry_sha256=self.entry_sha,
                    source_scales_sha256=self._source_scales_sha256,
                    active_mask_sha256=self._active_mask_sha256,
                    applied_denominator_sha256=tensor_identity(denominators),
                    applied_denominators=[float(denominators[i]) if active[i] else None
                                          for i in range(len(active))],
                    request_count=len(active), active_request_count=int(active.sum()),
                    flatten_order="D_BY_B_ACTIVE_ROW_MAJOR",
                    source_norm_rounding="FP32", weighting_dtype="torch.float64",
                    native_rhs_change_count=0, request_removal_count=0,
                    normalization_reinitialization_count=0)
