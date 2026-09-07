"""One entry/changed-state technical gate using the pinned v3.1 primitives.

The family already owns fixed stock z and the dictionary already owns entry
qref. This module never computes z, captures qref, runs Official, appends history,
or creates a scientific endpoint. FD is validation-only, never a field backend.
"""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import time

import torch

from project.run_scripts.native_response_ode_v31.fidelity import full_logits, _parity
from project.run_scripts.native_response_ode_v31.provenance import save
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import (
    GroupedFP32Overlay, tensor_set_sha256, tensor_sha256,
)
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .contracts import TrajectoryConfig, LAYERS, BindingBoundary
from .fixture import RNGSnapshot
from .trajectory import solve_current

FD_EPSILONS = (2**-7, 2**-8, 2**-9)


class FidelityBoundary(BindingBoundary):
    def __init__(self, status, receipt):
        super().__init__(status)
        self.status = status
        self.receipt = deepcopy(receipt)


def _sync(family):
    # Unlike legacy runtime._sync this preserves a genuine CPU fixture boundary.
    if family.device.type == 'cuda':
        torch.cuda.synchronize(family.device)


def _finite_fp32(value, label):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or not torch.isfinite(value).all()):
        raise FidelityBoundary('NONFINITE_OR_DTYPE_' + label, {})


def _fd_layer(family, overlay, observer, build, terminal, state_label, output):
    """Preserve the original v3.1 three-epsilon numerical policy exactly."""
    version = overlay.state_version
    row = dict(state_label=state_label, state_version=version, layer=build.layer,
               build_identity=build.build_identity, built_state_version=build.built_state_version,
               solve_backward_error=build.solve_backward_error, eps=[])
    started = time.perf_counter()
    try:
        result = observer.observe(build, expected_state_version=version)
        row['primal_parity'] = _parity(terminal, result.terminal)
        if not row['primal_parity']['pass_check']:
            raise FidelityBoundary('JVP_CURRENT_PRIMAL_BOUNDARY', row)
        function = observer._function(build)  # Existing source's serial raw-direction hook.
        for epsilon in FD_EPSILONS:
            row['current_epsilon'] = epsilon
            plus = function(torch.tensor(epsilon, device=family.device, dtype=torch.float32)).detach().float().cpu()
            observer.ledger.finite_difference_forward_count += 1
            minus = function(torch.tensor(-epsilon, device=family.device, dtype=torch.float32)).detach().float().cpu()
            observer.ledger.finite_difference_forward_count += 1
            _finite_fp32(plus, 'FD_PLUS'); _finite_fp32(minus, 'FD_MINUS')
            finite = (plus.double()-minus.double())/(2*epsilon)
            reference = result.response.double()
            signal = float(torch.linalg.vector_norm(plus.double()-minus.double()))
            noise = 64*torch.finfo(torch.float32).eps*max(
                float(torch.linalg.vector_norm(plus.double())), float(torch.linalg.vector_norm(minus.double())))
            nr, nf = float(reference.norm()), float(finite.norm())
            cosine = float((finite*reference).sum())/(nr*nf) if nr > 0 and nf > 0 else None
            relative = float((finite-reference).norm())/nr if nr > 0 else None
            row['eps'].append(dict(epsilon=epsilon, signal=signal, rounding_envelope=noise,
                sufficient_signal=signal > noise, cosine=cosine, relative_L2=relative,
                passed=bool(signal > noise and cosine is not None and cosine >= .99 and relative <= .05)))
        row['adjacent_pass'] = any(row['eps'][i]['passed'] and row['eps'][i+1]['passed'] for i in range(2))
        row['adjacent_sufficient_signal'] = any(row['eps'][i]['sufficient_signal'] and
            row['eps'][i+1]['sufficient_signal'] for i in range(2))
        row['raw_response_sha256'] = tensor_sha256(result.response)
        _sync(family)
        row['wall_seconds'] = time.perf_counter()-started
        row['ledger_at_layer_boundary'] = asdict(observer.ledger)
        # Save BEFORE a failing layer stops the gate, independent of terminal publication.
        save(output/f'fd-{state_label}-layer-{build.layer}.json', row)
        if not row['adjacent_pass']:
            status = ('FD_RESPONSE_IDENTITY_BOUNDARY' if row['adjacent_sufficient_signal']
                      else 'JVP_NUMERICALLY_UNRESOLVED')
            raise FidelityBoundary(status, row)
        return result.response, row
    except BaseException as exc:
        save(output/f'fd-{state_label}-layer-{build.layer}-boundary.json', dict(row,
            original_exception=dict(type=type(exc).__name__, message=str(exc)),
            ledger_at_failure=asdict(observer.ledger), state_version_at_failure=overlay.state_version))
        raise


def _observe_parity(family, overlay, logits_observer, state_label, output, counters):
    """Temporary copy/observe/copy-back only; no terminal/history finalization."""
    virtual = family.terminal()
    virtual_logits = logits_observer(family)
    _finite_fp32(virtual, 'VIRTUAL_ACTIVATION'); _finite_fp32(virtual_logits, 'VIRTUAL_LOGITS')
    shadow = overlay.materialize_shadow(device='cpu')
    endpoint_sha = tensor_set_sha256(shadow)
    changed = sum(int((shadow[name] != family.w0[name].cpu()).sum()) for name in shadow)
    counters['temporary_materialization_attempt_count'] += 1
    try:
        with overlay.suspend(authoritative=False):
            try:
                family._apply_shadow(shadow)
                if tensor_set_sha256(family.parameters) != endpoint_sha:
                    raise FidelityBoundary('SHADOW_PHYSICAL_BYTES_BOUNDARY', {})
                physical = family.terminal()
                physical_logits = logits_observer(family)
                _finite_fp32(physical, 'PHYSICAL_ACTIVATION'); _finite_fp32(physical_logits, 'PHYSICAL_LOGITS')
            finally:
                old._restore_selected(family.parameters, family.w0)
        counters['temporary_materialization_completed_count'] += 1
    except BaseException:
        raise
    row = dict(state_label=state_label, state_version=overlay.state_version,
        activation_parity=_parity(virtual, physical), logit_parity=_parity(virtual_logits, physical_logits),
        shadow_endpoint_sha256=endpoint_sha, materialized_nonzero_elements=changed,
        restored_W0_sha256=tensor_set_sha256(family.parameters), authoritative_write_count=0,
        history_append_count=0, temporary_materialization_count=1)
    save(output/f'parity-{state_label}.json', row)
    if not row['activation_parity']['pass_check'] or not row['logit_parity']['pass_check']:
        raise FidelityBoundary('OVERLAY_MATERIALIZED_PARITY_BOUNDARY', row)
    return row


def run_gpu_fidelity(family, dictionary, normalization, output, *, logits_observer=full_logits,
                     logits_observer_identity='SOURCE_FULL_CANONICAL_REWRITE_LOGITS'):
    """Injected family only; source N0/.1/T2/N4 gives one actual h=.5 step.

    Call once per model/fixture, not once per candidate. The caller may inject a
    sealed first-canonical-request full-vocab logits observer to bound memory;
    all raw JVP/FD activation rows still cover the complete bound cohort.
    """
    output = Path(output)
    receipt = dict(schema='alpha-jv-ds.gpu-fidelity.v1', status='RUNNING', states=[],
        execution_device_type=family.device.type,
        receipt_scope='GPU_RUNTIME_FIDELITY' if family.device.type == 'cuda' else 'CPU_SYNTHETIC_FIXTURE',
        fd_epsilons=list(FD_EPSILONS), fd_adjacent_required=True, fd_cosine_min=.99,
        fd_relative_L2_max=.05, logits_observer_identity=logits_observer_identity,
        normalization_id=normalization.normalization_id, fixed_z_recompute_count=0,
        qref_capture_count=0, official_run_count=0, authoritative_write_count=0,
        history_append_count=0, science_change_count=0, tolerance_change_count=0,
        retry_count=0, counters=dict(joint_overlay_step_attempt_count=0,
            joint_overlay_step_completed_count=0, temporary_materialization_attempt_count=0,
            temporary_materialization_completed_count=0))
    stage = 'BINDING'; overlay = observer = rng = None
    started = time.perf_counter(); original = None
    pointers = {name: value.data_ptr() for name, value in family.parameters.items()}
    build_before, solve_before = dictionary.build_count, dictionary.solve_count
    try:
        if (family.fixed_z is None or dictionary.family is not family
                or dictionary.qref is None or dictionary.qfref is None
                or normalization.entry_sha != family.w0_sha256
                or normalization.normalization_id != 'N0_SOURCE'
                or family.tokenizer.padding_side != 'right'):
            raise FidelityBoundary('FIXTURE_FIXED_Z_QREF_N0_BINDING', receipt)
        family.reset_entry()
        reference = (dictionary.qref, dictionary.qfref)
        fixed_z_sha = tensor_sha256(family.fixed_z.values)
        entry_M = family.method_state_identity()
        cache_flag = bool(family.module.cache_c_new)
        dtype_counts = {}
        for parameter in family.model.parameters():
            dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0)+1
        if any(dtype != 'torch.float32' for dtype in dtype_counts) or torch.is_autocast_enabled():
            raise FidelityBoundary('FULL_FP32_PARAMETER_AUTOCAST_BOUNDARY', dtype_counts)
        receipt.update(entry_W_sha256=family.w0_sha256, entry_M_identity=entry_M,
            fixed_z_sha256=fixed_z_sha, qN_ref=dictionary.qref, qF_ref=dictionary.qfref,
            parameter_dtype_counts=dtype_counts, context_identity=family.fixed_z.target_context_identity_sha256,
            request_order_sha256=family.fixed_z.request_order_sha256, normalization=normalization.receipt())
        rng = RNGSnapshot()
        names = {layer:f'{family.hparams.rewrite_module_tmp.format(layer)}.weight' for layer in LAYERS}
        overlay = GroupedFP32Overlay(family.model, names)
        observer = TerminalResponseObserver(model=family.model, overlay=overlay,
            capture_terminal_graph=family.terminal_graph)
        config = TrajectoryConfig(.1, 2., 4)
        with overlay, torch.no_grad():
            for state_label in ('entry', 'joint-step-1'):
                stage = state_label
                terminal = family.terminal(); version = overlay.state_version
                builds = dictionary.build(terminal, version)
                active, q, _ = dictionary.whiten(builds)
                if (dictionary.qref, dictionary.qfref) != reference:
                    raise FidelityBoundary('FROZEN_ENTRY_QREF_CHANGED', {})
                response_values, layer_rows = [], []
                for build in active:
                    stage = f'{state_label}/layer-{build.layer}'
                    response, row = _fd_layer(family, overlay, observer, build, terminal, state_label, output)
                    response_values.append(response); layer_rows.append(row)
                stage = f'{state_label}/parity'
                parity = _observe_parity(family, overlay, logits_observer, state_label, output, receipt['counters'])
                state = dict(state_label=state_label, state_version=version,
                    active_layers=[build.layer for build in active], q_layers=q.tolist(),
                    build_ids=[build.build_identity for build in builds], layers=layer_rows, parity=parity)
                receipt['states'].append(state)
                if family.method_state_identity() != entry_M or bool(family.module.cache_c_new) != cache_flag:
                    raise FidelityBoundary('FIDELITY_HISTORY_MUTATION', {})
                if state_label == 'entry':
                    stage = 'joint-step-1/apply'
                    _, _, _, solution, fact = solve_current(family.fixed_z.values, terminal,
                        response_values, q, normalization, config)
                    if not bool(solution.coefficients.any()):
                        raise FidelityBoundary('FINITE_NO_ACTION_NONZERO_GATE_NOT_OBSERVED',
                                               dict(fact, coefficients=solution.coefficients.tolist()))
                    receipt['counters']['joint_overlay_step_attempt_count'] += 1
                    token = overlay.seal_sweep_entry(0)
                    for i, build in enumerate(active):
                        overlay.append(build.overlay_delta(float(config.h*solution.coefficients[i]/q[i].sqrt())),
                                       sweep_token=token)
                    overlay.close_sweep(token)
                    receipt['counters']['joint_overlay_step_completed_count'] += 1
                    state['one_joint_step'] = dict(config=config.receipt(), coefficients=solution.coefficients.tolist(),
                        step_coefficients=(config.h*solution.coefficients/q.sqrt()).tolist(),
                        exit_state_version=overlay.state_version)
                elif parity['materialized_nonzero_elements'] == 0:
                    raise FidelityBoundary('FINITE_NO_ACTION_MATERIALIZED_GATE_NOT_OBSERVED', parity)
            overlay.assert_w0_unchanged(full_bytes=True)
            if tensor_sha256(family.fixed_z.values) != fixed_z_sha or not rng.matches():
                raise FidelityBoundary('FIDELITY_FIXED_Z_OR_RNG_MUTATION', {})
        receipt['status'] = 'PASS'
    except BaseException as exc:
        original = exc
        receipt['status'] = exc.status if isinstance(exc, FidelityBoundary) else 'TECHNICAL_FIDELITY_EXCEPTION'
        receipt['original_exception'] = dict(type=type(exc).__name__, message=str(exc),
            receipt=getattr(exc, 'receipt', {}))
    finally:
        # Boundary evidence is published AFTER exact W/M restoration, regardless
        # of how early FD, parity, shape, source or injected callbacks failed.
        restore_error = None
        try:
            family.reset_entry()
        except BaseException as exc:
            restore_error = dict(type=type(exc).__name__, message=str(exc))
        rng_matched = rng.matches() if rng is not None else None
        if rng is not None:
            rng.restore()
        restored_W = tensor_set_sha256(family.parameters) == family.w0_sha256
        restored_pointers = pointers == {name:value.data_ptr() for name, value in family.parameters.items()}
        try:
            restored_M = family.method_state_identity() == family._prepared_method_state_identity
        except BaseException:
            restored_M = False
        receipt.update(stage=stage, W0_bytes_restore=restored_W, W0_pointer_restore=restored_pointers,
            M_restore=restored_M, restore_error=restore_error, RNG_unchanged=rng_matched,
            dictionary_build_count=dictionary.build_count-build_before,
            solve_count=dictionary.solve_count-solve_before,
            jvp_ledger=asdict(observer.ledger) if observer is not None else {},
            overlay=overlay.receipt() if overlay is not None else {},
            wall_seconds=time.perf_counter()-started)
        if not restored_W or not restored_pointers or not restored_M or restore_error:
            receipt['status'] = 'FIDELITY_RESTORE_BOUNDARY'
        if receipt['status'] == 'PASS':
            save(output/'gpu_fidelity_checks.json', receipt)
        else:
            save(output/'gpu-fidelity-boundary-after-restore.json', receipt)
    if receipt['status'] != 'PASS':
        raise FidelityBoundary(receipt['status'], receipt) from original
    return receipt
