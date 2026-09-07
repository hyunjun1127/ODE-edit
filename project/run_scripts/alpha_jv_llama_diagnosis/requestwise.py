"""Observer-only raw response attribution; no model/overlay/write dependency.

Per-request activation blocks use the production [D,B_active] flatten order.
All-row influence zero-weights an objective block without changing denominators,
the native B100 RHS, targets, dictionary, qref, or metric. It is not an editor.
"""
from dataclasses import dataclass
import math

import torch

from project.run_scripts.native_response_ode_v31.algebra import (
    NativeMetricBoundary, nnls_response, turning,
)


class RequestwiseBoundary(NativeMetricBoundary):
    pass


def _max_abs(value):
    return float(value.abs().max()) if value.numel() else 0.


def _reconstruction(actual, expected, label):
    error = _max_abs(actual - expected)
    scale = max(1., _max_abs(actual), _max_abs(expected))
    if error > 1e-10 * scale:
        raise RequestwiseBoundary(f"REQUESTWISE_{label}_RECONSTRUCTION")
    return dict(max_absolute_residual=error, scaled_residual=error / scale,
                tolerance=1e-10, status="PASS")


@dataclass(frozen=True)
class ResponseContributions:
    e: torch.Tensor
    psi: torch.Tensor
    e_by_request: torch.Tensor
    psi_by_request: torch.Tensor
    g_by_request: torch.Tensor
    H_by_request: torch.Tensor
    active: torch.Tensor
    layer_ids: tuple
    q_layers: torch.Tensor
    raw_response_norms: torch.Tensor
    residual_norms: torch.Tensor
    normalization_receipt: dict
    reconstruction: dict

    @property
    def g(self):
        return self.psi.T @ self.e

    @property
    def H(self):
        return self.psi.T @ self.psi


def contributions(residual32, raw_responses32, q_layers, normalization, layer_ids):
    """[m,D,B] raw FP32 JVP -> frozen weighting -> FP64 whitening."""
    if (not isinstance(residual32, torch.Tensor) or residual32.dtype != torch.float32
            or residual32.ndim != 2 or not isinstance(raw_responses32, torch.Tensor)
            or raw_responses32.dtype != torch.float32 or raw_responses32.ndim != 3
            or raw_responses32.shape[1:] != residual32.shape):
        raise RequestwiseBoundary("RAW_RESPONSE_FP32_M_D_B_SHAPE")
    if not torch.isfinite(residual32).all() or not torch.isfinite(raw_responses32).all():
        raise RequestwiseBoundary("NONFINITE_RAW_RESPONSE")
    m, width, batch = raw_responses32.shape
    q = torch.as_tensor(q_layers, dtype=torch.float64, device="cpu").detach().clone()
    layers = tuple(layer_ids)
    if (q.shape != (m,) or len(layers) != m or len(set(layers)) != m
            or any(layer not in (4, 5, 6, 7, 8) for layer in layers)
            or not torch.isfinite(q).all() or bool((q <= 0).any())):
        raise RequestwiseBoundary("ACTIVE_LAYER_Q_INVENTORY")
    active = normalization.active
    if active.shape != (batch,):
        raise RequestwiseBoundary("REQUEST_INVENTORY_DRIFT")
    count = int(active.sum())
    e = normalization.weight(residual32)
    psi = (torch.stack([normalization.weight(raw_responses32[j]) / q[j].sqrt()
                        for j in range(m)], dim=1) if m
           else torch.empty((e.numel(), 0), dtype=torch.float64))
    er = torch.zeros((batch, width), dtype=torch.float64)
    pr = torch.zeros((batch, width, m), dtype=torch.float64)
    er[active] = e.reshape(width, count).T
    pr[active] = psi.reshape(width, count, m).permute(1, 0, 2)
    g_rows = torch.einsum("bdm,bd->bm", pr, er)
    h_rows = torch.einsum("bdm,bdn->bmn", pr, pr)
    reconstruction = dict(g=_reconstruction(g_rows.sum(0), psi.T @ e, "G"),
                          H=_reconstruction(h_rows.sum(0), psi.T @ psi, "H"))
    return ResponseContributions(e, psi, er, pr, g_rows, h_rows, active, layers, q,
        raw_responses32.detach().cpu().double().square().sum(dim=1).sqrt().T,
        residual32.detach().cpu().double().square().sum(dim=0).sqrt(),
        normalization.receipt(), reconstruction)


def publication_rows(bundle, request_ids, coefficients):
    """Zero-pad only publication layer columns, never the solve coordinates."""
    ids = tuple(str(value) for value in request_ids)
    if len(ids) != len(bundle.active) or len(set(ids)) != len(ids):
        raise RequestwiseBoundary("REQUEST_ID_INVENTORY")
    c = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    if c.shape != bundle.q_layers.shape or not torch.isfinite(c).all():
        raise RequestwiseBoundary("COEFFICIENT_INVENTORY")
    traces = bundle.H_by_request.diagonal(dim1=1, dim2=2).sum(dim=1)
    total = float(traces.sum())
    shares = traces / total if total else None
    signed = bundle.g_by_request @ c
    positive, negative = signed.clamp(min=0).sum(), (-signed.clamp(max=0)).sum()
    rows = []
    for i, case in enumerate(ids):
        padded_g = torch.zeros(5, dtype=torch.float64)
        padded_H = torch.zeros((5, 5), dtype=torch.float64)
        raw_norm, weighted_norm = [0.] * 5, [0.] * 5
        for j, layer in enumerate(bundle.layer_ids):
            padded_g[layer-4] = bundle.g_by_request[i, j]
            raw_norm[layer-4] = float(bundle.raw_response_norms[i, j])
            weighted_norm[layer-4] = float(bundle.psi_by_request[i, :, j].norm())
            for k, other in enumerate(bundle.layer_ids):
                padded_H[layer-4, other-4] = bundle.H_by_request[i, j, k]
        rows.append(dict(request_index=i, case_id=case, active=bool(bundle.active[i]),
            applied_denominator=bundle.normalization_receipt["applied_denominators"][i],
            residual_norm=float(bundle.residual_norms[i]), g_i=padded_g.tolist(),
            H_i=padded_H.tolist(), raw_response_norm_by_layer=raw_norm,
            weighted_whitened_response_norm_by_layer=weighted_norm,
            signed_gain=float(signed[i]), trace_share=float(shares[i]) if shares is not None else None,
            global_positive_gain=float(positive), global_negative_gain=float(negative),
            global_gain_cancellation_fraction=float(negative/(positive+negative))
                if positive+negative > 0 else None,
            scope="SAME_STATE_RESPONSE_OBSERVATION_ONLY", controller_influence_count=0))
    return rows


def global_summary(bundle, coefficients):
    c = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    traces = bundle.H_by_request.diagonal(dim1=1, dim2=2).sum(dim=1)
    total = float(traces.sum())
    ordered = traces.sort(descending=True).values
    spectrum = torch.linalg.eigvalsh(bundle.H) if bundle.H.numel() else torch.empty(0)
    condition = (float(spectrum[-1]/spectrum[0]) if spectrum.numel() and spectrum[0] > 0 else None)
    # This SVD describes the raw response span, not an alternate controller.
    # Its rank never changes the native NNLS coordinates or retained requests.
    if bundle.psi.numel():
        left, singular, _ = torch.linalg.svd(bundle.psi, full_matrices=False)
        threshold = max(bundle.psi.shape)*torch.finfo(torch.float64).eps*float(singular[0])
        retained = singular > threshold
        projected = left[:, retained] @ (left[:, retained].T @ bundle.e)
        residual = bundle.e-projected
        span = dict(singular_values=singular.tolist(), diagnostic_rank=int(retained.sum()),
            threshold=threshold, threshold_rule="max(rows,columns)*eps64*sigma_max",
            error_energy=float(bundle.e.square().sum()), unexplained_error_energy=float(residual.square().sum()),
            explained_error_energy=float(projected.square().sum()), controller_influence_count=0)
    else:
        span = dict(singular_values=[], diagnostic_rank=0, threshold=0.,
            threshold_rule="EMPTY_RESPONSE_SPACE", error_energy=float(bundle.e.square().sum()),
            unexplained_error_energy=float(bundle.e.square().sum()), explained_error_energy=0.,
            controller_influence_count=0)
    return dict(trace_H=total, top1_trace_share=float(ordered[:1].sum())/total if total else None,
        top5_trace_share=float(ordered[:5].sum())/total if total else None,
        effective_request_count=total**2/float(traces.square().sum()) if total else None,
        H_eigenvalues=spectrum.tolist(), H_condition=condition,
        error_span_observation=span,
        raw_physical_coefficients=(c/bundle.q_layers.sqrt()).tolist(),
        active_layer_ids=list(bundle.layer_ids), reconstruction=bundle.reconstruction,
        normalization=bundle.normalization_receipt, controller_influence_count=0)


def all_row_influence(bundle, metric, lambda_response, *, frobenius_metric=None):
    """Finite CPU shadow records only; native solver is reused unchanged.

    No callable writer/model/overlay is accepted, returned, or imported. Returned
    coefficient lists are labelled observations and never used by this module.
    """
    if not math.isfinite(lambda_response) or lambda_response <= 0:
        raise RequestwiseBoundary("LAMBDA_RESPONSE_BOUNDARY")
    metric = torch.as_tensor(metric, dtype=torch.float64, device="cpu")
    baseline = nnls_response(bundle.e, bundle.psi, metric, lambda_response)
    width = bundle.e_by_request.shape[1]
    count = int(bundle.active.sum())
    active_positions = {int(index): pos for pos, index in enumerate(bundle.active.nonzero().flatten())}
    result = []
    for i in range(len(bundle.active)):
        e = bundle.e.clone().reshape(width, count)
        psi = bundle.psi.clone().reshape(width, count, len(bundle.layer_ids))
        if i in active_positions:
            e[:, active_positions[i]] = 0
            psi[:, active_positions[i], :] = 0
        flat_e, flat_psi = e.reshape(-1), psi.reshape(-1, len(bundle.layer_ids)) if bundle.layer_ids else torch.empty((e.numel(), 0), dtype=torch.float64)
        # Re-summing retained request blocks avoids large-term subtraction as a
        # numerical authority. The subtraction residual is observation only.
        keep = torch.arange(len(bundle.active)) != i
        expected_g = bundle.g_by_request[keep].sum(0)
        expected_H = bundle.H_by_request[keep].sum(0)
        reconstruction = dict(g=_reconstruction(flat_psi.T@flat_e, expected_g, "INFLUENCE_G"),
                              H=_reconstruction(flat_psi.T@flat_psi, expected_H, "INFLUENCE_H"))
        solution = nnls_response(flat_e, flat_psi, metric, lambda_response)
        row = dict(observer="ALL_ROW_OBJECTIVE_INFLUENCE_OBSERVER", request_index=i,
            active=bool(bundle.active[i]), lambda_response=lambda_response,
            original_active_request_count=count, denominator_reinitialization_count=0,
            request_removal_count=0, native_rhs_change_count=0, controller_influence_count=0,
            coefficients_observed=solution.coefficients.tolist(),
            raw_coefficients_observed=(solution.coefficients/bundle.q_layers.sqrt()).tolist(),
            coefficient_delta_observed=(solution.coefficients-baseline.coefficients).tolist(),
            native_angle=turning(baseline.coefficients, solution.coefficients, metric),
            KKT_stationarity=solution.stationarity, KKT_complementarity=solution.complementarity,
            reconstruction=reconstruction,
            subtraction_H_max_abs_residual=_max_abs(bundle.H-bundle.H_by_request[i]-expected_H),
            subtraction_g_max_abs_residual=_max_abs(bundle.g-bundle.g_by_request[i]-expected_g))
        if frobenius_metric is not None:
            row["frobenius_angle"] = turning(baseline.coefficients, solution.coefficients, frobenius_metric)
        result.append(row)
    return result
