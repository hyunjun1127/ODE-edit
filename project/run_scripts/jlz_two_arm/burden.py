"""FP64 quadratic write burden, with no eigenvalue clipping or model mutation.

R convention is [output_dim, requests]. CPU/GPU tensors are both supported;
torch is imported lazily to permit metadata-only tooling without model imports.
Final actual-energy evaluation is row streamed and never stores a dense dW.
"""
from __future__ import annotations

import math


def _torch():
    import torch
    return torch


def _finite(value, label):
    if not bool(_torch().isfinite(value).all()):
        raise FloatingPointError(f"NONFINITE_{label}")


def _fp64(value, label):
    if value.dtype != _torch().float64 or value.ndim != 2:
        raise ValueError(f"{label}_MUST_BE_FP64_MATRIX")
    _finite(value, label)


def build_burden_matrix(A, K, P, *, verify_direct=True, rtol=1e-8, atol=1e-12,
                        previously_verified_route=None):
    """Return (M, diagnostic) for M=P.T A P via S-S² when verified stable.

    The fixed defaults classify numerical diagnostics, never science validity.
    A finite mismatch falls back to the symmetric direct quadratic form and
    retains FAIL_DIRECT_FALLBACK. No negative eigenvalue is silently removed.
    ``A=None`` is permitted only for a previously verified small-route binding;
    callers must bind the verification to dtype/source/geometry implementation.
    """
    torch = _torch()
    _fp64(K, "K")
    _fp64(P, "P")
    if K.shape != P.shape or K.device != P.device:
        raise ValueError("BURDEN_K_P_IDENTITY")
    if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("BURDEN_DIAGNOSTIC_TOLERANCE")
    if not verify_direct and not previously_verified_route:
        raise ValueError("BURDEN_SMALL_ROUTE_NOT_VERIFIED")
    Sraw = K.T @ P
    S = (Sraw + Sraw.T) * 0.5
    small = S - S @ S
    _finite(small, "BURDEN_SMALL_M")
    small = (small + small.T) * 0.5
    evidence = {"dtype": "float64", "shape": list(small.shape),
                "rtol": rtol, "atol": atol, "eigenvalue_clipping": False,
                "s_asymmetry_max": float((Sraw - Sraw.T).abs().max()),
                "route": "S-S2", "direct_verified": False,
                "prior_route_binding": previously_verified_route,
                "numerical_certification": "LIMITED_TO_RECORDED_COMPARISON"}
    if verify_direct:
        if A is None:
            raise ValueError("BURDEN_DIRECT_A_REQUIRED")
        _fp64(A, "A")
        if A.shape != (K.shape[0], K.shape[0]) or A.device != K.device:
            raise ValueError("BURDEN_A_IDENTITY")
        direct_raw = P.T @ (A @ P)
        _finite(direct_raw, "BURDEN_DIRECT_M")
        direct = (direct_raw + direct_raw.T) * 0.5
        difference = small - direct
        maximum = float(difference.abs().max())
        relative = float(difference.norm() / direct.norm().clamp_min(1e-300))
        passed = bool(torch.allclose(small, direct, rtol=rtol, atol=atol))
        evidence.update(direct_verified=True, maximum_abs_error=maximum,
                        relative_frobenius_error=relative,
                        direct_asymmetry_max=float((direct_raw - direct_raw.T).abs().max()),
                        original_verdict="PASS" if passed else "FAIL_DIRECT_FALLBACK",
                        action="USE_SMALL" if passed else "RECORD_WARNING_USE_DIRECT")
        if not passed:
            evidence["route"] = "P.T@A@P"
            small = direct
    else:
        evidence.update(original_verdict="NOT_RECHECKED_PRIOR_VERIFIED_ROUTE",
                        action="USE_SMALL_PRIOR_VERIFIED")
    return small.detach(), evidence


def anchor_scale_squared(anchors):
    """Mean_r ||h_r||² at the current arm's batch entry, not W0/global scale."""
    if anchors.ndim != 2 or anchors.shape[1] < 1:
        raise ValueError("BURDEN_ANCHOR_SHAPE")
    _finite(anchors, "BURDEN_ANCHORS")
    scale = float(anchors.double().square().sum(0).mean())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("BURDEN_ANCHOR_SCALE")
    return scale


def omega_and_grad(R, M, s_squared, eta=1.0):
    """Return (raw Ω tensor in FP64, eta-scaled gradient in R dtype).

    The caller adds ``eta * omega`` to the smooth objective. A still computes
    raw Ω for diagnostic accounting, but eta=0 gives exact zero extra gradient.
    Symmetric M is required: otherwise R M is not the quadratic derivative.
    """
    torch = _torch()
    _fp64(M, "M")
    if R.ndim != 2 or M.shape != (R.shape[1], R.shape[1]) or R.device != M.device:
        raise ValueError("BURDEN_R_M_SHAPE_DEVICE")
    _finite(R, "BURDEN_R")
    if eta not in (0, 1) or not math.isfinite(float(s_squared)) or float(s_squared) <= 0:
        raise ValueError("BURDEN_ETA_OR_SCALE")
    if not torch.equal(M, M.T):
        raise ValueError("BURDEN_M_NOT_SYMMETRIC")
    RM = R.double() @ M
    omega = 0.5 * (RM * R.double()).sum() / float(s_squared)
    gradient = (RM / float(s_squared)).to(dtype=R.dtype) if eta == 1 else torch.zeros_like(R)
    _finite(omega, "BURDEN_OMEGA")
    _finite(gradient, "BURDEN_GRADIENT")
    return omega, gradient


def actual_energy_chunked(committed, entry, A, scale_squared, *, row_chunk=32):
    """Final-only tr(dW A dW.T), using exact committed/entry FP32 bytes.

    Subtraction is in FP64, so it measures the difference of the two actual
    FP32 endpoints rather than a separately rounded FP32 subtraction. A can be
    CPU resident; only each bounded dW row block is transferred to A's device.
    No extra model forward is performed. Output is a scalar-only receipt.
    """
    torch = _torch()
    _fp64(A, "A")
    if committed.dtype != torch.float32 or entry.dtype != torch.float32:
        raise ValueError("ACTUAL_ENDPOINT_NOT_FP32")
    if committed.ndim != 2 or committed.shape != entry.shape:
        raise ValueError("ACTUAL_ENDPOINT_SHAPE")
    if A.shape != (committed.shape[1], committed.shape[1]):
        raise ValueError("ACTUAL_ENERGY_A_SHAPE")
    if not isinstance(row_chunk, int) or row_chunk < 1:
        raise ValueError("ACTUAL_ENERGY_CHUNK")
    if not math.isfinite(float(scale_squared)) or float(scale_squared) <= 0:
        raise ValueError("ACTUAL_ENERGY_SCALE")
    energy = torch.zeros((), dtype=torch.float64, device=A.device)
    norm2 = torch.zeros_like(energy)
    chunks = 0
    with torch.no_grad():
        for start in range(0, committed.shape[0], row_chunk):
            stop = min(start + row_chunk, committed.shape[0])
            current = committed[start:stop].to(device=A.device, dtype=torch.float64)
            original = entry[start:stop].to(device=A.device, dtype=torch.float64)
            delta = current - original
            _finite(delta, "ACTUAL_DELTA")
            action = delta @ A
            _finite(action, "ACTUAL_ACTION")
            energy.add_((action * delta).sum())
            norm2.add_(delta.square().sum())
            chunks += 1
        _finite(energy, "ACTUAL_ENERGY")
    scalar = float(energy)
    return {"actual_raw_energy": scalar,
            "actual_omega": 0.5 * scalar / float(scale_squared),
            "actual_delta_frobenius": math.sqrt(float(norm2)),
            "row_chunk": row_chunk, "chunks": chunks,
            "shape": list(committed.shape), "compute_dtype": "float64",
            "endpoint_dtype": "float32", "final_only": True,
            "checkpoint_saved": False}
