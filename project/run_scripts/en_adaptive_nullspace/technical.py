"""Fresh bounded T0 evidence; no historical waiver and no tensor persistence.

Exactly four identity-ordered current requests and four R512 documents. Numeric
residuals are observations, not an invented admission threshold. Finite/identity
failures raise after the scalar failure receipt and exact RAM restoration.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import types

import numpy as np
import torch

from .current import capture_current, digest
from .geometry import build_geometry
from .native import NATIVE_SHA256, requests_from_records
from .selector import select_arms
from .controller import _endpoint
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter
from project.run_scripts.single_layer_mechanism_first.z_hook_parity import compare_native_z_paths


def fixed_indices(values, identity, count=4):
    """SHA256 canonical identity ascending, independent of every model outcome."""
    if len(values) < count:
        raise ValueError('T0_INSUFFICIENT_PANEL')
    ids = [digest(identity(value)) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError('T0_DUPLICATE_PANEL_IDENTITY')
    return sorted(range(len(values)), key=lambda i: ids[i])[:count]


def residual(reference, candidate):
    a = torch.as_tensor(reference).detach().double().cpu()
    b = torch.as_tensor(candidate).detach().double().cpu()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('T0_NONFINITE_OR_SHAPE_RESIDUAL')
    error = float((b-a).norm()); norm = float(a.norm())
    return dict(max_abs=float((b-a).abs().max()), l2=error, reference_l2=norm,
                relative_l2=error/norm if norm else None, reference_zero=norm == 0,
                bitwise_equal=torch.equal(torch.as_tensor(reference), torch.as_tensor(candidate)))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return dict(value=None, status='UNDEFINED_NONFINITE_DIAGNOSTIC')
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        raise TypeError('T0_TENSOR_DISK_PERSISTENCE_FORBIDDEN')
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False)+'\n')


class _TargetsOnlyFitter(NativeSingletonFitter):
    """Same native solve with already observed T0 z, no additional optimization."""
    def __init__(self, *args, targets, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = targets
        self.position = 0

    def _functions(self, counts, capture):
        fit, finalize = super()._functions(counts, capture)
        namespace = dict(fit.__globals__)
        def observed_z(model, *args, **kwargs):
            if self.position >= self.targets.shape[1]:
                raise ValueError('T0_TARGET_QUEUE_OVERCONSUMED')
            target = self.targets[:,self.position].to(next(model.parameters()).device)
            self.position += 1
            counts['reused_T0_z'] = counts.get('reused_T0_z',0)+1
            counts['compute_z'] = counts.get('compute_z',0)+1
            return target
        namespace['compute_z'] = observed_z
        return types.FunctionType(fit.__code__, namespace, fit.__name__, fit.__defaults__, fit.__closure__), finalize


def _z_trace_scalars(vectors):
    output = {}
    entries = [('native',vectors['native_rows']),
               ('cache_batch1',[dict(losses=r['losses'][0],gradients=r['gradients'][0]) for r in vectors['cache_batch1_receipts']]),
               ('cache_batched',[dict(losses=vectors['cache_batched_receipt']['losses'][i],
                                      gradients=vectors['cache_batched_receipt']['gradients'][i]) for i in range(4)])]
    for name, rows in entries:
        output[name] = [dict(losses=r['losses'], gradient_l2=[float(g.double().norm()) for g in r['gradients']]) for r in rows]
    return output


def _cut_parity(oracle, index, center, delta):
    """Actual native H0 + delta K versus independently captured physical cut."""
    cache = oracle.caches[index]
    base = center.double()-delta.double()
    keys = cache.keys.to(oracle.device)
    with torch.no_grad():
        h0 = cache.residual.to(oracle.device) + torch.nn.functional.linear(keys,base.to(oracle.device,dtype=torch.float32))
        predicted = h0.double() + torch.nn.functional.linear(keys.double(),delta.to(oracle.device,dtype=torch.float64))
        captured = []
        with oracle.physical_weight(center.to(oracle.device)):
            handle = oracle.decoder.layers[4].register_forward_hook(lambda module,args,result: captured.append(result[0].detach().clone()))
            try:
                oracle._physical_hidden(index)
            finally:
                handle.remove()
        if len(captured) != 1:
            raise RuntimeError('T0_PHYSICAL_CUT_CAPTURE_CARDINALITY')
        absolute = cache.residual.to(oracle.device) + torch.nn.functional.linear(keys,center.to(oracle.device))
        return dict(H0_plus_actual_DK_vs_physical=residual(captured[0],predicted),
                    absolute_FP32_linear_vs_physical=residual(captured[0],absolute),
                    physical_layout='one exact packed input; all its input positions')


def run_t0(rt, objective, V, output):
    """Return scalar evidence and restore entry W/RNG on success or exception.

    Runtime supplies model/tok/etok/module/hp/context/W/W0/records/install and
    native_runner. The temporary four-request native solve consumes already
    measured batched z. It neither appends M nor increments science counters.
    """
    output = Path(output)
    started = time.monotonic()
    saved = rt.W.detach().cpu().clone()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_rng, np_rng = random.getstate(), np.random.get_state()
    report = dict(schema='EN_ADAPTIVE_FRESH_T0_V1', precision_status='NOT_ESTABLISHED',
                  exploratory=True, historical_waiver_inherited=False, numerical_policy_changed=False,
                  tensor_files_written=0, history_appends=0, science_native_batches=0,
                  finite_identity_status='RUNNING')
    current = None
    try:
        if not torch.equal(saved, rt.W0.detach().cpu()):
            raise ValueError('T0_REQUIRES_COLD_W0')
        records = list(rt.records[:300])
        selected = fixed_indices(records,lambda r: dict(case_id=r['case_id'], requested_rewrite=r['requested_rewrite']))
        panel_records = [records[i] for i in selected]
        requests = requests_from_records(panel_records)
        ref_candidates = list(objective.store.indices('R512'))
        order = fixed_indices(ref_candidates,lambda i: objective.ref._capsule_shas[i])
        ref_indices = [ref_candidates[i] for i in order]
        report['panel'] = dict(current_prefix_indices=selected, case_ids=[r['case_id'] for r in panel_records],
            current_identity_sha256=[digest(dict(case_id=r['case_id'], requested_rewrite=r['requested_rewrite'])) for r in panel_records],
            reference_indices=ref_indices, reference_identities=[objective.ref._capsule_shas[i] for i in ref_indices],
            selection='ascending SHA256(canonical full request identity / reference capsule SHA), first four; no model outcome')
        _write(output/'panel.json', report['panel'])
        z_receipt, vectors = compare_native_z_paths(rt.model,rt.tok,rt.module,rt.hp,rt.context,requests)
        report['z_source_comparison'] = z_receipt
        report['z_source_comparison']['inherited_gate_interpretation'] = 'historical diagnostics only; no blanket new PASS, no prior waiver'
        _write(output/'z-source-comparison.json', dict(receipt=z_receipt,traces=_z_trace_scalars(vectors)))
        if not torch.equal(rt.W.detach().cpu(),saved):
            raise RuntimeError('T0_Z_MODEL_MUTATION')
        fitter = _TargetsOnlyFitter(rt.module,expected_source_sha256=NATIVE_SHA256,contexts=rt.context,
                                    targets=vectors['cache_batched_targets'])
        history = torch.zeros_like(rt.native_runner.projector,device='cpu')
        fit = fitter.fit(rt.model,rt.tok,rt.hp,history,rt.native_runner.projector,requests,layer=4,capture=False)
        if fitter.position != 4 or bool(history.count_nonzero()):
            raise RuntimeError('T0_TARGET_QUEUE_OR_HISTORY_MISMATCH')
        center = fit['weight'].detach().cpu().clone()
        delta = center.double()-saved.double()
        report['temporary_native_solve'] = dict(receipt=fit['receipt'], target_calls=0, reused_T0_z=4,
            actual_delta_norm=float(delta.norm()), persisted=False, restored_before_science=True)
        del history, fit, fitter, vectors
        objective.rebind(center)
        current = capture_current(rt.model,rt.tok,rt.etok,requests,rt.context)
        manifest = current['manifest']
        report['current_weighting'] = {k:manifest[k] for k in ('case_ids','request_weights','K_shape','weight_sum','actual_byte_columns','logical_prefix_groups','original_cache_positions','original_occurrence_positions','byte_variant_columns','identity_sha256')}
        oracle = current['oracle']
        report['current_parity'] = []
        for case in report['panel']['case_ids']:
            matches = [r for r in current['rows'] if r['case_id']==case and r['kind']=='canonical' and r['branch']=='new']
            if len(matches) != 1:
                raise ValueError('T0_CANONICAL_CURRENT_SELECTION')
            index = matches[0]['cache']
            cut = _cut_parity(oracle,index,center,delta)
            logits = oracle.compare_logits(index,center,center)
            stationarity = oracle.key_stationarity(index,center)
            report['current_parity'].append(dict(case_id=case,input_sha256=matches[0]['full_input_sha256'],cut=cut,logits=logits,stationarity=stationarity))
        objective.rebind(center)
        report['reference_parity'] = []
        for index in ref_indices:
            cut = _cut_parity(objective.ref,index,center,delta)
            logits = objective.ref.compare_logits(index,center,center)
            stationarity = objective.ref.key_stationarity(index,center)
            report['reference_parity'].append(dict(index=index,cut=cut,logits=logits,stationarity=stationarity))
        j, gradient, cached = objective.evaluate(center,indices=ref_indices,gradient=True,kind='technical',route='cached')
        jp, physical_g, physical = objective.evaluate(center,indices=ref_indices,gradient=True,kind='technical',route='physical')
        report['AD'] = dict(cached_objective=cached,physical_objective=physical,objective_abs=abs(j-jp),
                            gradient=residual(physical_g,gradient),stationary_entry='temporary native four-request endpoint')
        rho = float(delta.norm())
        if not math.isfinite(rho) or rho == 0:
            report['directional_derivative'] = dict(status='NOT_ESTABLISHED_ZERO_NATIVE_DELTA')
        else:
            direction = delta / rho
            # One prospectively fixed central step, no finite-difference sweep.
            step = .01*rho
            plus = _endpoint(center,step*direction.numpy())
            minus = _endpoint(center,-step*direction.numpy())
            jplus,_,rp = objective.evaluate(plus,indices=ref_indices,gradient=False,kind='technical')
            jminus,_,rm = objective.evaluate(minus,indices=ref_indices,gradient=False,kind='technical')
            actual_span = plus.double()-minus.double()
            ad_difference = float((gradient*actual_span).sum())
            observed_difference = jplus-jminus
            report['directional_derivative'] = dict(status='OBSERVED_NO_NEW_THRESHOLD',fixed_step_fraction_native_norm=.01,
                step=step,nominal_AD=float((gradient*direction).sum()),actual_materialized_AD_difference=ad_difference,
                central_objective_difference=observed_difference,absolute_residual=abs(observed_difference-ad_difference),
                relative_residual=abs(observed_difference-ad_difference)/abs(ad_difference) if ad_difference else None,
                plus_objective=rp,minus_objective=rm,additional_step_search=0)
            del plus,minus,actual_span,direction
        geo = build_geometry(current['K'],current['weights'],current['representative_indices'],V)
        spectrum = geo.gradient_spectrum(gradient,delta,j)
        selection = select_arms(spectrum) if j >= 0 else dict(selected={},status='NOT_ESTABLISHED_NEGATIVE_SIGNED_KL')
        report['geometry'] = dict(diagnostic=geo.diagnostic,spectrum=spectrum,negative_signed_KL_observed=j<0,
            signed_KL_clamping=False,selector=selection,projection_rows=[])
        for name, row in selection['selected'].items():
            h = geo.direction(gradient,row)
            projected_twice = geo.space(row['released_modes']).project(h)
            energy = float(np.sum(h*h)); inner = float(np.sum(gradient.numpy()*h))
            ideal = -row['eta']*h
            actual = _endpoint(center,ideal).double().numpy()-center.double().numpy()
            report['geometry']['projection_rows'].append(dict(arm=name,released_modes=row['released_modes'],
                energy_scalar=row['gradient_energy'],energy_direct=energy,energy_abs=abs(energy-row['gradient_energy']),
                self_adjoint_witness_inner=inner,self_adjoint_witness_abs=abs(inner-energy),
                idempotence=residual(h,projected_twice),direct_unit_action=geo.action_norm(h),
                scalar_unit_action=math.sqrt(row['unit_coefficient_action_squared']),
                materialization=geo.diagnostics(ideal,actual)))
            del h,projected_twice,ideal,actual
        report['finite_identity_status'] = 'PASS'
        report['precision_status'] = 'NOT_ESTABLISHED'
        report['precision_reason'] = ('Fresh residuals and historical z diagnostics are reported; design supplies no new global numerical acceptance threshold. '
                                      'No old waiver or automatic PASS is assigned; scientific continuation is exploratory under the explicit design provision.')
        report['additional_science_denominator_cost'] = 0
        report['reference_technical_sweeps'] = 4 if rho else 2
        _write(output/'t0-observations.json',report)
    except BaseException as exc:
        report.update(finite_identity_status='FAILED',exception_type=type(exc).__name__,exception=str(exc))
        _write(output/'t0-failure.json',report)
        raise
    finally:
        rt.install(saved)
        objective.rebind(saved)
        if not torch.equal(rt.W.detach().cpu(),saved):
            raise RuntimeError('T0_RESTORE_NOT_EXACT')
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng);np.random.set_state(np_rng)
        report['entry_weight_exact_restored'] = True
        report['rng_restored'] = True
        report['seconds'] = time.monotonic()-started
        _write(output/'t0-final-state.json',dict(entry_weight_exact_restored=True,rng_restored=True,
            tensor_files_written=0,history_appends=0,seconds=report['seconds'],status=report['finite_identity_status']))
    _write(output/'t0-observations.json',report)
    return report
