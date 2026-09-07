"""D's bounded inventory and CPU-only same-state attribution orchestration.

No model loading, historical replay, endpoint evaluation, or write authority.
Missing historical bytes do not block S. NRMS is an explicitly separate D view.
"""
from pathlib import Path
import hashlib
import stat

import torch

from project.run_scripts.native_response_ode_v31.algebra import (
    FrozenNormalization, nnls_response, identities, turning,
)
from .normalization_views import NormalizationView, source_residual, tensor_identity
from .requestwise import (RequestwiseBoundary, contributions, publication_rows,
                          global_summary, all_row_influence)


def capture_repeatability(target32, captures32, *, semantic_identities):
    """Three observations; the first alone remains the initialization source."""
    if len(captures32) != 3:
        raise RequestwiseBoundary("ENTRY_REPEAT_CAPTURE_COUNT")
    if len(semantic_identities) != 3:
        raise RequestwiseBoundary("CAPTURE_SEMANTIC_IDENTITY_COUNT")
    residuals = [source_residual(target32, value) for value in captures32]
    reference = captures32[0]
    rows = []
    for i, value in enumerate(captures32):
        arithmetic64 = target32.detach().cpu().double() - value.detach().cpu().double()
        primary = residuals[i].detach().cpu().double()
        difference = value.detach().cpu().double()-reference.detach().cpu().double()
        rounding = primary-arithmetic64
        rows.append(dict(capture=i, terminal_sha256=tensor_identity(value),
            primary_residual_sha256=tensor_identity(residuals[i]),
            max_abs_repeat_deviation=float(difference.abs().max()) if difference.numel() else 0.,
            repeat_l2_deviation=float(difference.norm()),
            fp32_subtract_vs_fp64_subtract_max_abs=float(rounding.abs().max()) if rounding.numel() else 0.,
            fp32_subtract_vs_fp64_subtract_l2=float(rounding.norm()),
            semantic_identity=semantic_identities[i]))
    return dict(status="OBSERVATION_ONLY", captures=rows, initialization_capture_index=0,
        normalization_initialization_count=1, repeated_capture_mean_used=False,
        deterministic_repeat_does_not_exclude_capture_bias=True,
        native_compute_z_capture_comparison="REQUIRES_SEPARATELY_RECORDED_NATIVE_CAPTURE",
        model_forward_invocation_by_this_function=0, controller_influence_count=0)


def same_state_attribution(*, source_normalization, residual32, raw_responses32,
                           q_layers, layer_ids, metric, lambda_response,
                           request_ids, frobenius_metric=None, include_all_row=False):
    """Compare N0/NRMS on exactly one frozen raw bundle; never a trajectory."""
    views, records, solutions = {}, {}, {}
    for normalization_id in ("N0_SOURCE", "NRMS_ENTRY"):
        view = NormalizationView.from_source(source_normalization, normalization_id)
        bundle = contributions(residual32, raw_responses32, q_layers, view, layer_ids)
        solution = nnls_response(bundle.e, bundle.psi, metric, lambda_response)
        views[normalization_id], solutions[normalization_id] = bundle, solution
        records[normalization_id] = dict(
            label="SAME_STATE_OBJECTIVE_SHADOW_NOT_ACTUAL_HPARAM_ENDPOINT",
            lambda_response=lambda_response,
            normalization=view.receipt(), summary=global_summary(bundle, solution.coefficients),
            request_rows=publication_rows(bundle, request_ids, solution.coefficients),
            coefficients_observed=solution.coefficients.tolist(),
            KKT_stationarity=solution.stationarity, KKT_complementarity=solution.complementarity,
            **identities(bundle.e, bundle.psi, metric, solution.coefficients, lambda_response))
        if include_all_row:
            records[normalization_id]["all_row_influence"] = all_row_influence(
                bundle, metric, lambda_response, frobenius_metric=frobenius_metric)
    n0, nrms = solutions["N0_SOURCE"].coefficients, solutions["NRMS_ENTRY"].coefficients
    result = dict(views=records, native_angle=turning(n0, nrms, metric),
        fixed_raw_residual_sha256=tensor_identity(residual32),
        fixed_raw_response_sha256=tensor_identity(raw_responses32),
        fixed_q_layers_sha256=tensor_identity(torch.as_tensor(q_layers, dtype=torch.float64)),
        raw_bundle_recomputation_count=0, actual_write_count=0,
        native_rhs_change_count=0, request_removal_count=0, controller_influence_count=0)
    if frobenius_metric is not None:
        result["frobenius_angle"] = turning(n0, nrms, frobenius_metric)
    return result


def cross_objective_scores(target32, endpoint_terminal32, source_normalization):
    """Call only with a verified actual endpoint capture; this does no capture."""
    residual = source_residual(target32, endpoint_terminal32)
    result = {}
    for normalization_id in ("N0_SOURCE", "NRMS_ENTRY"):
        view = NormalizationView.from_source(source_normalization, normalization_id)
        result[normalization_id] = dict(V=float(view.weight(residual).square().sum()/2),
                                       normalization=view.receipt())
    return dict(objectives=result, endpoint_terminal_sha256=tensor_identity(endpoint_terminal32),
        raw_residual_norm_by_request=residual.detach().cpu().double().norm(dim=0).tolist(),
        model_forward_count=0, history_append_count=0, writer_recompute_count=0,
        controller_influence_count=0)


def inspect_sealed_local_member(path, expected_sha256, expected_bytes):
    """One bounded read-only exact-path check; never scans or contacts a host."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("ABSOLUTE_SEALED_PATH_REQUIRED")
    row = dict(path=str(path), expected_sha256=expected_sha256,
               expected_bytes=int(expected_bytes), host_scope="SERVER1_LOCAL_ONLY")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current = current/component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                return dict(row, status="SYMLINK_BOUNDARY", boundary_path=str(current))
    except FileNotFoundError:
        return dict(row, status="ABSENT_ON_SERVER1", missing_component=str(current))
    if not stat.S_ISREG(info.st_mode):
        return dict(row, status="NOT_REGULAR_FILE")
    if info.st_size != int(expected_bytes):
        return dict(row, status="SIZE_MISMATCH", bytes=info.st_size)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    return dict(row, sha256=observed, bytes=info.st_size, mode=f"{stat.S_IMODE(info.st_mode):04o}",
                status="SEALED_BYTES_REHASH_PASS" if observed == expected_sha256 else "SHA_MISMATCH")


def historical_availability(*, checkpoint_rows, context_rows, model="llama3-8b-inst",
                            arm="JV_NATIVE", chain="chain-2"):
    """Publication is an inventory, not proof that local W9/z exists."""
    selected = [row for row in checkpoint_rows if row["alias"] == model and row["arm"] == arm]
    contexts = [row for row in context_rows if row["chain"] == chain]
    checks = []
    for row in selected:
        if int(row["batch"]) == 5:
            checks.append(dict(kind="W5_M5", **inspect_sealed_local_member(
                row["path"], row["sha256"], row["bytes"])))
    for row in contexts:
        checks.append(dict(kind="EXACT_NATIVE_CONTEXT", **inspect_sealed_local_member(
            row["raw_local_path"], row["file_sha256"], row["bytes"])))
    w5 = any(row["kind"] == "W5_M5" and row["status"] == "SEALED_BYTES_REHASH_PASS" for row in checks)
    context = any(row["kind"] == "EXACT_NATIVE_CONTEXT" and row["status"] == "SEALED_BYTES_REHASH_PASS" for row in checks)
    return dict(status=("W5_CONTEXT_BYTES_AVAILABLE_REPLAY_NOT_VALIDATED" if w5 and context
                        else "HISTORICAL_EXACT_STATE_UNAVAILABLE_ON_SERVER1"),
        published_checkpoint_batches=sorted(int(row["batch"]) for row in selected),
        W9_in_selected_publication=any(int(row["batch"]) == 9 for row in selected),
        fixed_z_tensor_availability="NOT_RECORDED_IN_CHECKPOINT_SCHEMA",
        journal_replay_status=sorted(set(row["journal_replay_parity"] for row in selected)),
        local_checks=checks, exact_state_replay_pass=False, replay_action_count=0,
        remote_access_count=0, repeated_poll_count=0, model_action_count=0,
        S_blocked_by_D=False, W10_as_W9_allowed=False, W0_to_W9_rescue_allowed=False)
