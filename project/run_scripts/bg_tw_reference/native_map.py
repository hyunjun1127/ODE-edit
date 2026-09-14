"""Frozen BLUE singleton native residual operator; no model or vendor mutation.

Native shapes are K[d_in, B*q], R[d_out, B], P/M[d_in, d_in].
The native RHS path is solve(G, (P K) repeat(R).T).T.  The separately
constructed differentiable map is R A, A=sum_q(solve(G, P K).T).  These
are algebraically equivalent, NOT promised bitwise identical in FP32.
The actual native candidate must be retained by the caller for RAW1.
"""
from dataclasses import dataclass
import math

import torch


class NativeMapError(RuntimeError):
    pass


def _fp32_finite(name, tensor, ndim=2):
    if tensor.ndim != ndim or tensor.dtype != torch.float32:
        raise NativeMapError(f"{name}_FP32_SHAPE")
    if not bool(torch.isfinite(tensor).all()):
        raise NativeMapError(f"{name}_NONFINITE")


@dataclass(frozen=True)
class FrozenNativeMap:
    """Own detached copies of the linear system and immutable A.

    Only Llama's out-by-in weight layout is accepted.  No assumption that G
    is symmetric/SPD, no inverse/Cholesky, and no autograd through the solve.
    Native compute_ks averages context types before returning one column per
    request in the pinned BLUE source (q=1); an explicit request-major repeat
    identity is mandatory if a different capture has q>1.
    """
    system: torch.Tensor
    projected_keys: torch.Tensor
    A: torch.Tensor
    request_count: int
    repeat_factor: int
    weight_shape: tuple
    source_identity: str
    _versions: tuple

    @classmethod
    def from_native_system(cls, system, projected_keys, *, request_count,
                           weight_shape, source_identity,
                           request_major_repeat_verified=False):
        _fp32_finite("SYSTEM", system)
        _fp32_finite("PROJECTED_KEYS", projected_keys)
        if not source_identity:
            raise NativeMapError("SOURCE_IDENTITY_REQUIRED")
        if type(request_count) is not int or request_count <= 0:
            raise NativeMapError("REQUEST_COUNT")
        if len(weight_shape) != 2 or weight_shape[0] == weight_shape[1]:
            # The pinned vendor's match-shape branch is ambiguous for squares.
            raise NativeMapError("LLAMA_NON_SQUARE_OUT_IN_LAYOUT_REQUIRED")
        d_in = weight_shape[1]
        if (system.shape != (d_in, d_in) or
                projected_keys.shape[0] != d_in or
                projected_keys.shape[1] % request_count or
                projected_keys.device != system.device):
            raise NativeMapError("NATIVE_SYSTEM_SHAPE_OR_DEVICE")
        q = projected_keys.shape[1] // request_count
        if q < 1 or (q != 1 and not request_major_repeat_verified):
            raise NativeMapError("REPEAT_IDENTITY_NOT_VERIFIED")
        with torch.no_grad():
            g = system.detach().clone()
            pk = projected_keys.detach().clone()
            solved = torch.linalg.solve(g, pk)
            a = solved.T.contiguous()
            if q != 1:
                a = a.reshape(request_count, q, d_in).sum(dim=1)
            _fp32_finite("FIXED_A", a)
        return cls(g, pk, a, request_count, q, tuple(weight_shape),
                   source_identity, (g._version, pk._version, a._version))

    @classmethod
    def from_frozen_kpm(cls, keys, projector, history, *, request_count,
                        weight_shape, source_identity, l2=1.,
                        request_major_repeat_verified=False):
        for name, value in (("KEYS", keys), ("P", projector), ("M", history)):
            _fp32_finite(name, value)
        d_in = keys.shape[0]
        if (projector.shape != (d_in, d_in) or history.shape != projector.shape
                or any(x.device != keys.device for x in (projector, history))):
            raise NativeMapError("KPM_SHAPE_OR_DEVICE")
        if l2 != 1.:
            raise NativeMapError("BG1_NATIVE_L2_MUST_EQUAL_ONE")
        # Exact native multiplication/addition association; no symmetry repair.
        with torch.no_grad():
            system = projector @ (keys @ keys.T + history) + l2 * torch.eye(
                d_in, dtype=torch.float32, device=keys.device)
            pk = projector @ keys
        return cls.from_native_system(
            system, pk, request_count=request_count, weight_shape=weight_shape,
            source_identity=source_identity,
            request_major_repeat_verified=request_major_repeat_verified)

    def assert_frozen(self):
        if (self.system._version, self.projected_keys._version,
                self.A._version) != self._versions:
            raise NativeMapError("FROZEN_NATIVE_OPERATOR_MUTATED")

    def _check_residual(self, residual):
        self.assert_frozen()
        _fp32_finite("RESIDUAL", residual)
        if (tuple(residual.shape) != (self.weight_shape[0], self.request_count)
                or residual.device != self.A.device):
            raise NativeMapError("RESIDUAL_SHAPE_OR_DEVICE")

    def __call__(self, residual):
        """Differentiable R A; caller must apply it to the full down-projection."""
        self._check_residual(residual)
        action = residual @ self.A
        _fp32_finite("MAP_ACTION", action)
        return action

    def vjp(self, weight_cotangent):
        self.assert_frozen()
        _fp32_finite("WEIGHT_COTANGENT", weight_cotangent)
        if (tuple(weight_cotangent.shape) != self.weight_shape or
                weight_cotangent.device != self.A.device):
            raise NativeMapError("WEIGHT_COTANGENT_SHAPE")
        result = weight_cotangent @ self.A.T
        _fp32_finite("RESIDUAL_VJP", result)
        return result

    def direct_native_update(self, residual):
        """Reference RHS solve, not an A-based replacement of RAW1."""
        self._check_residual(residual)
        with torch.no_grad():
            repeated = residual.detach().repeat_interleave(self.repeat_factor, dim=1)
            update = torch.linalg.solve(
                self.system, self.projected_keys @ repeated.T).T.contiguous()
            _fp32_finite("DIRECT_NATIVE_UPDATE", update)
        return update

    def comparison(self, residual, parent, *, captured_native_endpoint=None):
        """Report arithmetic differences without asserting model-level parity."""
        self._check_residual(residual)
        _fp32_finite("PARENT", parent)
        if tuple(parent.shape) != self.weight_shape or parent.device != self.A.device:
            raise NativeMapError("PARENT_SHAPE_OR_DEVICE")
        with torch.no_grad():
            rhs = self.direct_native_update(residual)
            mapped = self(residual)
            native_endpoint = parent + rhs
            mapped_endpoint = parent + mapped
            result = {
                "request_count": self.request_count,
                "repeat_factor": self.repeat_factor,
                "weight_layout": "out_in",
                "system_is_assumed_spd": False,
                "map_solve_rhs": "P K",
                "native_solve_rhs": "(P K) repeat(R).T",
                "map_native_update_max_abs": float((mapped-rhs).abs().max()),
                "map_native_update_l2": float((mapped-rhs).double().norm()),
                "map_native_endpoint_max_abs": float((mapped_endpoint-native_endpoint).abs().max()),
                "map_native_endpoint_exact": bool(torch.equal(mapped_endpoint, native_endpoint)),
                "source_identity": self.source_identity,
                "model_forward_parity": "NOT_TESTED",
            }
            if captured_native_endpoint is not None:
                _fp32_finite("CAPTURED_NATIVE_ENDPOINT", captured_native_endpoint)
                if captured_native_endpoint.shape != parent.shape:
                    raise NativeMapError("CAPTURED_NATIVE_ENDPOINT_SHAPE")
                captured = captured_native_endpoint.to(parent.device)
                result.update(
                    rhs_captured_endpoint_exact=bool(torch.equal(native_endpoint, captured)),
                    rhs_captured_endpoint_max_abs=float((native_endpoint-captured).abs().max()))
        if not all(math.isfinite(v) for v in result.values() if type(v) is float):
            raise NativeMapError("NONFINITE_NATIVE_MAP_COMPARISON")
        return result
