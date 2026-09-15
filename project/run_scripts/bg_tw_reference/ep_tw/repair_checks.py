"""Saved-episode EP derivative diagnostics, separately versioned from old FD.

No native target/solve, policy change, scheduler, or scientific execution lives
here. GPU observations occur only when an authorized caller supplies its loaded
adapter. Pure reducers and deterministic CPU fixtures are usable independently.
All probe payloads (including private prompt rows) must stay local-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model_adapter import anchored_weight, tensor_sha


class RepairCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairNumerics:
    """Pre-GPU bounded diagnostic contract, not a positive quality allowance."""
    schema_version: str = "EP_SAVED_EPISODE_FD_REPAIR_V1"
    factors: tuple = tuple(2. ** -k for k in range(10))
    relative_weight_step: float = 1e-3
    derivative_relative_tolerance: float = .15
    derivative_absolute_tolerance: float = 1e-7
    convergence_relative_tolerance: float = .15
    minimum_adjacent_resolved_scales: int = 2
    first_window_k: int = 1
    baseline_observations_per_objective: int = 3
    extra_rechecks: int = 0
    signed_grid_observations_max: int = 80
    fp32_signal_multiple: float = 8.
    observed_zero_span_multiple: float = 4.
    loss_scale_floor: float = 1e-6
    analytic_near_zero: float = 1e-7
    action_near_zero: float = 1e-12
    direct_norm_relative_tolerance: float = 1e-3
    direct_max_absolute_floor: float = 1e-7
    bilinear_relative_tolerance: float = 1e-3
    bilinear_absolute_tolerance: float = 1e-7
    independent_seed_E: int = 2026091501
    independent_seed_D: int = 2026091502
    diagnostic_seed: int = 2026091503

    def __post_init__(self):
        fixed = {
            "factors": tuple(2. ** -k for k in range(10)),
            "relative_weight_step": 1e-3,
            "derivative_relative_tolerance": .15,
            "derivative_absolute_tolerance": 1e-7,
            "convergence_relative_tolerance": .15,
            "minimum_adjacent_resolved_scales": 2, "first_window_k": 1,
            "baseline_observations_per_objective": 3, "extra_rechecks": 0,
            "signed_grid_observations_max": 80, "fp32_signal_multiple": 8.,
            "observed_zero_span_multiple": 4., "analytic_near_zero": 1e-7,
            "action_near_zero": 1e-12, "direct_norm_relative_tolerance": 1e-3,
            "direct_max_absolute_floor": 1e-7, "bilinear_relative_tolerance": 1e-3,
            "bilinear_absolute_tolerance": 1e-7, "independent_seed_E": 2026091501,
            "independent_seed_D": 2026091502, "diagnostic_seed": 2026091503,
        }
        for key, expected in fixed.items():
            if getattr(self, key) != expected:
                raise RepairCheckError("UNLOCKED_REPAIR_NUMERICS:" + key)
        if self.loss_scale_floor != 1e-6:
            raise RepairCheckError("OLD_LOSS_SCALE_FLOOR_CHANGED")

    def to_dict(self):
        return dict(asdict(self),
            window_selection="coarsest qualifying adjacent [k,k+1], k>=1; retain all10",
            derivative_allowance="max(1e-7,.15*abs(AD))",
            convergence_denominator="max(abs(AD),1e-7)",
            resolution="signal>max(8*eps32*loss_scale,4*observed_zero_span) AND actual-distinct-action",
            fp32_signal_threshold_is_model_error_upper_bound=False,
            independent_rule="CPU torch.Generator, fixed objective seed, Rademacher, Frobenius normalize",
            baseline_rule="existing gradient-sweep value then exactly2 no-grad C0 repeats; total3",
            extrapolation_or_isolated_point_for_PASS=False,
            E_both_directions_PASS_before_D_FD=True,
            same_neural_backward_not_independently_proven_by_direct_route=True,
            old_coarse_FAIL_preserved=True, current_quality_allowance=0.)


DEFAULT_REPAIR_NUMERICS = RepairNumerics()


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    return value


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(_safe(value), f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=_sha_file(path))


def _save_tensors(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(payload, f)
        f.flush(); os.fsync(f.fileno())
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=_sha_file(path))


def _norm(value):
    return float(value.detach().double().norm())


def _dot(first, second):
    return float((first.detach().double() * second.detach().double()).sum())


def _matrix(name, value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim != 2:
        raise RepairCheckError(name + "_FP32_MATRIX")
    if not bool(torch.isfinite(value).all()):
        raise RepairCheckError(name + "_NONFINITE")


def make_directions(gradient, *, objective, config=DEFAULT_REPAIR_NUMERICS):
    """The independent direction uses no gradient values or objective scores."""
    if objective not in ('E', 'D'):
        raise RepairCheckError('OBJECTIVE_NOT_E_D')
    _matrix('GRADIENT', gradient)
    norm = _norm(gradient)
    seed = config.independent_seed_E if objective == 'E' else config.independent_seed_D
    gen = torch.Generator(device='cpu').manual_seed(seed)
    independent = torch.randint(0, 2, tuple(gradient.shape), generator=gen,
                                dtype=torch.int64, device='cpu').float().mul_(2).sub_(1)
    independent /= _norm(independent)
    independent = independent.to(gradient.device)
    own = gradient.detach() / norm if norm > 0 else None
    return {
        'self_gradient': dict(direction=own, seed=None,
            definition='recomputed g_objective / FP64 Frobenius norm; FP32 direction',
            sha256=None if own is None else tensor_sha(own),
            status='DIRECTION_READY' if own is not None else 'UNRESOLVED_ZERO_GRADIENT'),
        'independent': dict(direction=independent, seed=seed,
            definition='CPU seeded Rademacher, normalized Frobenius; independent of gradient values',
            sha256=tensor_sha(independent), status='DIRECTION_READY'),
    }


def check_direct_route(gC, gW, A, directions, *, config=DEFAULT_REPAIR_NUMERICS):
    """Compare independent direct-weight leaf gradient to custom residual VJP.

    This is NOT independent validation of all neural backward kernels. Caller
    must source gW from the direct leaf path, without the custom affine node.
    """
    for name, value in [('GC', gC), ('GW', gW), ('A', A)]:
        _matrix(name, value)
    if (gC.shape[0] != gW.shape[0] or A.shape != (gC.shape[1], gW.shape[1])
            or gC.device != gW.device or gC.device != A.device):
        raise RepairCheckError('DIRECT_GRADIENT_ORIENTATION_DEVICE')
    with torch.no_grad():
        mapped = gW @ A.T
        difference = gC - mapped
        norm_c, norm_m, error = _norm(gC), _norm(mapped), _norm(difference)
        relative = error / max(norm_c, norm_m, 1e-30)
        maximum = float(difference.abs().max())
        scale = max(float(gC.abs().max()), float(mapped.abs().max()))
        abs_allowance = config.direct_norm_relative_tolerance * scale + config.direct_max_absolute_floor
        matrix_pass = relative <= config.direct_norm_relative_tolerance and maximum <= abs_allowance
        rows = []
        for role, spec in directions.items():
            direction = spec['direction'] if isinstance(spec, dict) else spec
            if direction is None:
                rows.append(dict(direction=role, status='UNRESOLVED_ZERO_DIRECTION'))
                continue
            _matrix('BILINEAR_DIRECTION', direction)
            if direction.shape != gC.shape or direction.device != gC.device:
                raise RepairCheckError('BILINEAR_DIRECTION_SCHEMA')
            lhs, rhs = _dot(gC, direction), _dot(gW, direction @ A)
            allowance = max(config.bilinear_absolute_tolerance,
                            config.bilinear_relative_tolerance * max(abs(lhs), abs(rhs)))
            status = 'PASS' if abs(lhs-rhs) <= allowance else 'FAIL_DIRECT_BILINEAR'
            if min(abs(lhs), abs(rhs)) <= config.analytic_near_zero:
                status = 'UNRESOLVED_NEAR_ZERO_BILINEAR'
            rows.append(dict(direction=role, lhs_gC_v=lhs, rhs_gW_vA=rhs,
                absolute_error=abs(lhs-rhs), allowance=allowance, status=status,
                direction_sha256=tensor_sha(direction)))
    status = 'PASS' if matrix_pass and rows and all(x['status'] == 'PASS' for x in rows) else 'FAIL_OR_UNRESOLVED'
    return dict(status=status, matrix_pass=matrix_pass, gradient_norm=norm_c,
        direct_mapped_norm=norm_m, difference_norm=error, difference_relative=relative,
        max_abs_difference=maximum, max_abs_allowance=abs_allowance, bilinear=rows,
        gC_sha256=tensor_sha(gC), gW_sha256=tensor_sha(gW), A_sha256=tensor_sha(A),
        direct_mapped_sha256=tensor_sha(mapped), numerical=config.to_dict(),
        tensor_schema={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                       for name,value in [('gC',gC),('gW',gW),('A',A)]},
        tensor_sha_convention='raw_contiguous_tensor_bytes; dtype/shape recorded separately',
        full_neural_backward_independently_proven=False)


def reduce_fd_grid(rows, *, analytic, baseline_values, config=DEFAULT_REPAIR_NUMERICS):
    """Pure full-grid reduction; never hides coarse failures or uses FD scale.

    Each input row contains k,h,plus,minus, plus/minus actual weight SHA/norm,
    and directional dot products with the common unsigned mapped direction.
    Missing action evidence cannot become a resolved point.
    """
    if len(baseline_values) != config.baseline_observations_per_objective:
        raise RepairCheckError('EXACT_THREE_BASELINE_OBSERVATIONS_REQUIRED')
    if not all(math.isfinite(float(x)) for x in baseline_values) or not math.isfinite(analytic):
        raise RepairCheckError('BASELINE_OR_AD_NONFINITE')
    if len(rows) != 10 or [r['k'] for r in rows] != list(range(10)):
        raise RepairCheckError('FULL_TEN_SCALE_GRID_REQUIRED')
    baseline = float(baseline_values[0])
    span = max(baseline_values) - min(baseline_values)
    ad_scale = max(abs(analytic), config.derivative_absolute_tolerance)
    ad_near_zero = abs(analytic) <= config.analytic_near_zero
    result, seen_hash_pairs = [], set()
    h0 = float(rows[0]['h'])
    for source in rows:
        row = dict(source)
        k, h, fp, fm = row['k'], float(row['h']), float(row['plus']), float(row['minus'])
        if not math.isfinite(h) or h <= 0 or h != h0 * config.factors[k]:
            raise RepairCheckError('UNLOCKED_FD_INTERVAL')
        finite = math.isfinite(fp) and math.isfinite(fm)
        slope = (fp-fm)/(2*h) if finite else math.nan
        signal = abs(fp-fm) if finite else math.nan
        loss_scale = max(abs(fp), abs(fm), abs(baseline), config.loss_scale_floor) if finite else math.inf
        nominal_threshold = config.fp32_signal_multiple * torch.finfo(torch.float32).eps * loss_scale
        jitter_threshold = config.observed_zero_span_multiple * span
        threshold = max(nominal_threshold, jitter_threshold)
        pair = (row.get('plus_weight_sha256'), row.get('minus_weight_sha256'))
        hashes_present = all(isinstance(x, str) and x for x in pair) and isinstance(row.get('raw_sha256'), str) and bool(row['raw_sha256'])
        distinct = hashes_present and pair[0] != pair[1] and row.get('raw_sha256') not in pair
        amplitude_new = distinct and pair not in seen_hash_pairs
        if hashes_present:
            seen_hash_pairs.add(pair)
        actions = (float(row.get('plus_actual_norm', 0.)), float(row.get('minus_actual_norm', 0.)))
        nonzero = all(math.isfinite(x) and x > config.action_near_zero for x in actions)
        sign_distinct = row.get('plus_actual_direction_dot', 0.) > 0 and row.get('minus_actual_direction_dot', 0.) < 0
        resolved = (finite and signal > threshold and distinct and amplitude_new and nonzero and sign_distinct)
        error = abs(slope-analytic) if finite else math.inf
        allowance = max(config.derivative_absolute_tolerance, config.derivative_relative_tolerance * abs(analytic))
        row.update(analytic=analytic, central_difference=slope, absolute_error=error,
            relative_error=error/ad_scale, derivative_allowance=allowance,
            derivative_pass=finite and error <= allowance and not ad_near_zero,
            signal=signal, signal_threshold=threshold, fp32_auxiliary_threshold=nominal_threshold,
            observed_jitter_threshold=jitter_threshold, baseline_span=span,
            actual_nonzero=nonzero, actual_signs_distinct=sign_distinct,
            actual_weight_bytes_distinct=distinct, actual_amplitude_not_duplicate=amplitude_new,
            signal_resolved=resolved, analytic_near_zero=ad_near_zero,
            taylor_plus=fp-baseline-h*analytic if finite else math.nan,
            taylor_minus=fm-baseline+h*analytic if finite else math.nan,
            plus_one_sided=(fp-baseline)/h if finite else math.nan,
            minus_one_sided=(baseline-fm)/h if finite else math.nan)
        result.append(row)
    windows = []
    for k in range(config.first_window_k, 9):
        first, second = result[k:k+2]
        convergence = abs(first['central_difference']-second['central_difference']) / ad_scale
        okay = (first['signal_resolved'] and second['signal_resolved']
                and first['derivative_pass'] and second['derivative_pass']
                and convergence <= config.convergence_relative_tolerance)
        windows.append(dict(k_start=k, k_stop=k+1, convergence_relative=convergence,
                            convergence_denominator=ad_scale, valid=okay))
    valid_windows = [x for x in windows if x['valid']]
    selected = valid_windows[0] if valid_windows else None
    reason = ('AD_CONSISTENT_ADJACENT_RESOLVED_WINDOW' if selected else
              'NEAR_ZERO_AD_UNRESOLVED' if ad_near_zero else 'NO_AD_CONSISTENT_RESOLVED_WINDOW')
    return _safe(dict(status='PASS' if selected else 'UNRESOLVED', reason=reason,
        selected_window=selected, rows=result, windows=windows, baseline_values=baseline_values,
        baseline_anchor_index=0, baseline_span=span, analytic=analytic,
        signed_grid_observations=20, all_coarse_failures_preserved=True,
        numerical=config.to_dict()))


def _capture_rng(device):
    ns = np.random.get_state()
    return dict(torch_cpu=torch.random.get_rng_state().clone(),
                torch_device=torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None,
                python=random.getstate(), numpy=(ns[0], torch.from_numpy(ns[1].copy()), *ns[2:]))


def _restore_rng(state, device):
    torch.random.set_rng_state(state['torch_cpu'])
    if state['torch_device'] is not None:
        torch.cuda.set_rng_state(state['torch_device'], device)
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], ns[1].numpy().copy(), *ns[2:]))


def _rng_digest(state):
    summary = dict(cpu=tensor_sha(state['torch_cpu']),
        device=None if state['torch_device'] is None else tensor_sha(state['torch_device']),
        python=state['python'], numpy=[state['numpy'][0], tensor_sha(state['numpy'][1]), *state['numpy'][2:]])
    return hashlib.sha256(json.dumps(summary, sort_keys=True, allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def _probe_materialization(raw, A, direction, correction, *, sign, h, mapped_direction):
    """One selected-layer action only; no full-weight-per-probe persistence."""
    weight = anchored_weight(correction, raw, A)
    actual = weight - raw
    nominal = mapped_direction * (sign * h)
    error = actual - nominal
    # Deterministic bounded coordinates, not data-selected largest deviations.
    flat_raw, flat_weight, flat_delta = raw.flatten(), weight.flatten(), actual.flatten()
    indices = torch.linspace(0, raw.numel()-1, min(raw.numel(), 64), device=raw.device).long()
    sample_raw, sample_delta = flat_raw[indices], flat_delta[indices]
    ulp = (torch.nextafter(sample_raw, torch.full_like(sample_raw, math.inf)) - sample_raw).abs()
    samples = dict(indices=indices.cpu().tolist(), raw=sample_raw.cpu().tolist(),
        weight=flat_weight[indices].cpu().tolist(), actual_delta=sample_delta.cpu().tolist(),
        next_positive_ulp=ulp.cpu().tolist())
    evidence = dict(weight_sha256=tensor_sha(weight), actual_delta_sha256=tensor_sha(actual),
        correction_sha256=tensor_sha(correction), actual_norm=_norm(actual),
        nominal_norm=_norm(nominal), nominal_actual_difference_norm=_norm(error),
        nominal_actual_relative_error=_norm(error)/max(_norm(nominal), 1e-30),
        nominal_actual_max_abs=float(error.abs().max()), actual_nonzero_elements=int(torch.count_nonzero(actual)),
        actual_direction_dot=_dot(actual, mapped_direction), actual_numel=actual.numel(),
        executable_C_map_norm=_norm(correction @ A), bounded_fixed_coordinate_ulp=samples,
        weight_reconstructible_as='anchored_weight(stored_C,exact_saved_Vp,exact_saved_A)',
        full_weight_per_probe_saved=False)
    del weight, actual, nominal, error
    return evidence


def _row_deltas(observation, baseline, *, objective):
    """Pair exact row identities; no per-request AD/Jacobian is invented."""
    def identity(row):
        if objective == 'E':
            return (row['case_id'], row.get('kind'), row.get('prompt_index'),
                    row.get('prompt'), row.get('target'), tuple(row.get('target_token_ids', [])))
        return row['index'], row['role'], row['source_row_id']
    first = {identity(row): row for row in baseline['rows']}
    if (len(first) != len(baseline['rows']) or len(observation['rows']) != len(first)
            or len({identity(row) for row in observation['rows']}) != len(first)):
        raise RepairCheckError('PROBE_ROW_DENOMINATOR_OR_DUPLICATE')
    field = 'nll' if objective == 'E' else 'kl'
    values = []
    for row in observation['rows']:
        key = identity(row)
        if key not in first:
            raise RepairCheckError('PROBE_ROW_IDENTITY')
        values.append(dict(row=row, baseline_value=first[key][field],
                           loss_delta=row[field]-first[key][field],
                           per_row_AD_or_Taylor='NOT_RECORDED_NO_PER_REQUEST_JACOBIAN'))
    return values


def run_directional_grid(adapter, records, gradient, *, objective, direction_id,
                         direction, baseline_observations, output,
                         config=DEFAULT_REPAIR_NUMERICS):
    """Save direction, every signed C/action/observation, then reduce all10.

    Each probe restores a newly sealed diagnostic RNG state, independent of the
    missing original post-fit RNG. Every completed observation is saved before
    any reduction/raising. Unexpected exceptions also get a create-once receipt.
    Caller, not this function, owns E-both-PASS -> D execution ordering.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    start = time.monotonic()
    completed = 0
    rng = None
    try:
        if objective not in ('E', 'D') or direction_id not in ('self_gradient', 'independent'):
            raise RepairCheckError('UNAPPROVED_OBJECTIVE_DIRECTION')
        if len(baseline_observations) != 3:
            raise RepairCheckError('BASELINE_REPEAT_COUNT')
        adapter._assert_episode()
        _matrix('GRADIENT', gradient)
        baseline_values = [float(x[objective]) for x in baseline_observations]
        if not all(math.isfinite(x) for x in baseline_values):
            raise RepairCheckError('BASELINE_OBJECTIVE_NONFINITE')
        baseline = baseline_observations[0]
        expected = 100 if objective == 'E' else 64
        if any(x['denominator'] != expected for x in baseline_observations):
            raise RepairCheckError('BASELINE_OBJECTIVE_MASS')
        _save_json(root/'numerical.json', config.to_dict())
        _save_json(root/'baseline-observations.json', baseline_observations)
        if direction is None:
            receipt = dict(status='UNRESOLVED', reason='ZERO_GRADIENT_SELF_DIRECTION',
                objective=objective, direction=direction_id, signed_grid_observations=0)
            _save_json(root/'receipt.json', receipt)
            return receipt
        _matrix('DIRECTION', direction)
        if direction.shape != gradient.shape or direction.device != gradient.device:
            raise RepairCheckError('DIRECTION_GRADIENT_SCHEMA')
        action = direction @ adapter.fixed_a
        action_norm, weight_norm = _norm(action), _norm(adapter.raw)
        analytic = _dot(gradient, direction)
        raw_sha = tensor_sha(adapter.raw)
        raw_version, map_version = adapter.raw._version, adapter.fixed_a._version
        rng = _capture_rng(adapter.device)
        rng_identity = _rng_digest(rng)
        _save_tensors(root/'direction-rng.pt', dict(direction=direction.detach().cpu(), rng=rng))
        metadata = dict(objective=objective, direction=direction_id, direction_sha256=tensor_sha(direction),
            raw_sha256=raw_sha, A_sha256=tensor_sha(adapter.fixed_a), gradient_sha256=tensor_sha(gradient),
            gradient_norm=_norm(gradient), analytic=analytic, direction_action_norm=action_norm,
            raw_weight_norm=weight_norm, rng_sha256=rng_identity,
            tensor_schema={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                for name,value in [('raw',adapter.raw),('A',adapter.fixed_a),('direction',direction),('gradient',gradient)]},
            tensor_sha_convention='raw_contiguous_tensor_bytes; dtype/shape recorded separately',
            original_failed_gradient_or_postfit_rng_byte_identity_claim=False,
            diagnostic_native_target_calls=0, diagnostic_native_solves=0)
        _save_json(root/'direction.json', metadata)
        if not math.isfinite(action_norm) or action_norm <= config.action_near_zero:
            receipt = dict(metadata, status='UNRESOLVED', reason='NEAR_ZERO_OR_NONFINITE_EXECUTABLE_ACTION',
                           signed_grid_observations=0)
            _save_json(root/'receipt.json', receipt)
            return receipt
        h0 = config.relative_weight_step * max(weight_norm, 1.) / action_norm
        pairs, forwards, tokens = [], 0, 0
        for k, factor in enumerate(config.factors):
            h, signed = h0 * factor, {}
            for sign, name in ((1., 'plus'), (-1., 'minus')):
                probe = root/f'k{k:02d}'/name
                probe.mkdir(parents=True, exist_ok=False, mode=0o700)
                _restore_rng(rng, adapter.device)
                if (adapter.raw._version, adapter.fixed_a._version) != (raw_version, map_version):
                    raise RepairCheckError('SAVED_EPISODE_MUTATED')
                probe_started = time.monotonic()
                correction = direction * (sign * h)
                materialization = _probe_materialization(adapter.raw, adapter.fixed_a, direction, correction,
                    sign=sign, h=h, mapped_direction=action)
                materialization_seconds = time.monotonic()-probe_started
                write_begin = time.monotonic()
                c_member = _save_tensors(probe/'correction.pt', {'C': correction.detach().cpu()})
                _save_json(probe/'input.json', dict(metadata, k=k, h=h, factor=factor, sign=sign,
                    materialization=materialization, correction_member=c_member))
                input_save_seconds = time.monotonic()-write_begin
                forward_begin = time.monotonic()
                observation = (adapter.current(records, correction) if objective == 'E'
                               else adapter.generic('S64', correction))
                observe_seconds = time.monotonic()-forward_begin
                output_save_begin = time.monotonic()
                # Preserve the actual result BEFORE checks/reducers can fail.
                _save_json(probe/'observation.json', observation)
                completed += 1
                value = float(observation[objective])
                _save_json(probe/'scalar-diagnostics.json', dict(objective=objective, value=value,
                    baseline_anchor=baseline_values[0], analytic=analytic, signed_step=sign*h,
                    taylor_remainder=value-baseline_values[0]-sign*h*analytic,
                    observed_zero_span=max(baseline_values)-min(baseline_values),
                    per_request_AD='NOT_RECORDED_NO_PER_REQUEST_JACOBIAN'))
                _save_json(probe/'paired-rows.json', _row_deltas(observation, baseline, objective=objective))
                after_rng = _capture_rng(adapter.device)
                guard = dict(rng_before=rng_identity, rng_after=_rng_digest(after_rng),
                    rng_exact=_rng_digest(after_rng) == rng_identity,
                    raw_map_versions_exact=(adapter.raw._version, adapter.fixed_a._version) == (raw_version, map_version),
                    adapter_parameter_hook_rng_nonmutation=observation.get('parameter_hook_rng_nonmutation'))
                _save_json(probe/'state-guard.json', guard)
                _save_json(probe/'probe-cost.json', dict(materialization_hash_diagnostics_seconds=materialization_seconds,
                    input_tensor_metadata_save_seconds=input_save_seconds,
                    objective_observation_wall_seconds=observe_seconds,
                    output_paired_guard_save_seconds=time.monotonic()-output_save_begin,
                    observation_internal_seconds_are_nested=observation.get('seconds'),
                    native_target_calls=0, native_solves=0, backward_calls=0,
                    diagnostic_C_map_matmuls=2, shared_direction_map_matmul_once_per_direction=True,
                    fullweight_per_probe_saved=False))
                if not guard['rng_exact'] or not guard['raw_map_versions_exact'] or guard['adapter_parameter_hook_rng_nonmutation'] is not True:
                    raise RepairCheckError('PROBE_STATE_OR_RNG_MUTATION')
                if observation['denominator'] != expected:
                    raise RepairCheckError('PROBE_OBJECTIVE_MASS')
                signed[name] = (observation, materialization)
                forwards += observation['counts']['forwards']
                tokens += observation['counts'].get('scored_tokens', 0)
                del correction
            row = dict(k=k, factor=factor, h=h, raw_sha256=raw_sha,
                       plus=float(signed['plus'][0][objective]), minus=float(signed['minus'][0][objective]))
            for name in ('plus', 'minus'):
                for key in ('weight_sha256', 'actual_norm', 'actual_direction_dot'):
                    row[name+'_'+key] = signed[name][1][key]
            _save_json(root/f'k{k:02d}'/'pair.json', row)
            pairs.append(row)
        receipt = reduce_fd_grid(pairs, analytic=analytic, baseline_values=baseline_values, config=config)
        receipt.update(metadata, status=receipt['status'], forward_groups_or_documents=forwards,
            scored_tokens=tokens, backward_calls=0, seconds=time.monotonic()-start,
            actual_signed_observations=completed, repeat_observations_owned_by_parent=3,
            raw_gradient_recomputed_not_old_tensor_replay=True)
        _save_json(root/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        _save_json(root/'failure.json', dict(status='TECHNICAL_EXCEPTION',
            exception_type=type(error).__name__, error=str(error), objective=objective,
            direction=direction_id, completed_signed_observations=completed,
            seconds=time.monotonic()-start, completed_probe_outputs_preserved=True))
        raise
    finally:
        if rng is not None:
            _restore_rng(rng, adapter.device)


def run_objective_fd(adapter, records, gradient, *, objective, baseline_observations,
                     output, config=DEFAULT_REPAIR_NUMERICS, E_receipt=None):
    """Both preregistered directions; no score-dependent direction replacement."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    if objective == 'D' and not (isinstance(E_receipt, dict) and E_receipt.get('objective') == 'E'
            and E_receipt.get('status') == 'PASS'
            and set(E_receipt.get('directions', {})) == {'self_gradient', 'independent'}
            and all(x.get('status') == 'PASS' for x in E_receipt['directions'].values())):
        _save_json(root/'failure.json', dict(status='NOT_RUN_BLOCKED',
            reason='E_BOTH_DIRECTIONS_PASS_REQUIRED_BEFORE_D_FD', signed_grid_observations=0))
        raise RepairCheckError('E_BOTH_DIRECTIONS_PASS_REQUIRED_BEFORE_D_FD')
    try:
        directions = make_directions(gradient, objective=objective, config=config)
    except BaseException as error:
        _save_json(root/'failure.json', dict(status='TECHNICAL_EXCEPTION', error=str(error),
            objective=objective, stage='direction-construction', signed_grid_observations=0))
        raise
    _save_json(root/'directions.json', {key: {k:v for k,v in spec.items() if k != 'direction'}
                                       for key,spec in directions.items()})
    receipts = {}
    for role in ('self_gradient', 'independent'):
        receipts[role] = run_directional_grid(adapter, records, gradient, objective=objective,
            direction_id=role, direction=directions[role]['direction'],
            baseline_observations=baseline_observations, output=root/role, config=config)
    receipt = dict(status='PASS' if all(x['status'] == 'PASS' for x in receipts.values()) else 'UNRESOLVED',
        objective=objective, directions=receipts,
        signed_grid_observations=sum(x.get('actual_signed_observations', 0) for x in receipts.values()),
        next_objective_may_start=objective == 'E' and all(x['status'] == 'PASS' for x in receipts.values()),
        backward_calls=0, native_target_calls=0, native_solves=0)
    _save_json(root/'receipt.json', receipt)
    return receipt
