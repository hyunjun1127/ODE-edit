"""Versioned physical L4..L8 adapter around the unchanged native BLUE body.

The original helper's AST split, fit equations and history loop are reused.
Only its explicit physical-layer allowlist is generalized here. Trace-based
measurements read native locals and never alter an optimizer/target operation.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import sys
import textwrap
import time

from project.run_scripts.low_cost_write_donor_pilot.fitting import (
    FitBoundary, NativeSingletonFitter, select_projector, tensor_sha,
)


class NativeTrace:
    """Observe exact loss/clamp lines in an unmodified compute_z code object."""
    def __init__(self, compute_z):
        self.fn = compute_z
        lines, start = inspect.getsourcelines(compute_z)
        source = textwrap.dedent(''.join(lines))
        tree = ast.parse(source)
        loss_lines = [start+n.lineno-1 for n in ast.walk(tree)
                      if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                      and isinstance(n.test.left, ast.Name) and n.test.left.id == 'loss']
        clamp_lines = [start+n.lineno-1 for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and any(
                           isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == 'delta' for t in n.targets)]
        if len(loss_lines) != 1 or len(clamp_lines) != 1:
            raise FitBoundary('NATIVE_TRACE_LAYOUT')
        self.loss_line, self.clamp_line = loss_lines[0], clamp_lines[0]
        self.rows = []; self.active = {}; self.failed = []
        self.evidence = dict(method='READ_ONLY_COMPUTE_Z_LINE_AND_RETURN_TRACE',
            compute_z_sha256=hashlib.sha256(source.encode()).hexdigest(),
            loss_line=self.loss_line, clamp_line=self.clamp_line,
            numeric_source_modified=False, extra_model_forwards=0, extra_backwards=0)

    @staticmethod
    def _step(locals_):
        step = locals_['opt'].state.get(locals_['delta'],{}).get('step',0)
        return int(step.item()) if hasattr(step,'item') else int(step)

    def _trace(self, frame, event, arg):
        if frame.f_code is not self.fn.__code__:
            return None
        key = id(frame); d = frame.f_locals
        if event == 'call':
            self.active[key] = dict(losses=[], clamp_events=[])
        if event == 'line' and frame.f_lineno == self.loss_line:
            row = dict(iteration=int(d['it']),actual_adam_step=self._step(d),
                total_loss=float(d['loss'].detach()),nll=float(d['nll_loss'].detach()),
                kl=float(d['kl_loss'].detach()),decay=float(d['weight_decay'].detach()),
                context_nll=d['nll_loss_each'].detach().cpu().tolist(),
                delta_norm=float(d['delta'].detach().norm()),
                anchor_norm=float(d['target_init'].detach().norm()))
            self.active[key]['losses'].append(row)
        elif event == 'line' and frame.f_lineno == self.clamp_line:
            self.active[key]['clamp_events'].append(dict(iteration=int(d['it']),
                actual_adam_step=self._step(d),pre_clamp_norm=float(d['delta'].detach().norm()),
                radius=float(d['max_norm'].detach())))
        elif event == 'return':
            state = self.active.pop(key,dict(losses=[],clamp_events=[]))
            # Exceptions return None. They propagate, never become successful rows.
            if arg is None or not hasattr(arg,'detach'):
                self.failed.append(dict(case_id=d.get('request',{}).get('case_id'),
                    layer=d.get('layer'),iteration=d.get('it'),**state))
                return self._trace
            hp = d['hparams']; step = self._step(d)
            loss = float(d['loss'].detach())
            row = dict(case_id=int(d['request']['case_id']),layer=int(d['layer']),
                adam_updates=step,loss_evaluations=int(d['it'])+1,
                stop_reason='LOSS_LT_0.05' if loss < .05 else 'LOSS_BUDGET_25',
                final_loss=loss,final_nll=float(d['nll_loss'].detach()),
                final_kl=float(d['kl_loss'].detach()),final_decay=float(d['weight_decay'].detach()),
                final_context_nll=d['nll_loss_each'].detach().cpu().tolist(),
                anchor=d['target_init'].detach().cpu().clone(),
                delta=d['delta'].detach().cpu().clone(),target=arg.detach().cpu().clone(),
                radius=float((hp.clamp_norm_factor*d['target_init'].norm()).detach()),
                lookup_indices=list(d['lookup_idxs']),teacher_sha256=tensor_sha(d['kl_distr_init']),
                clamp_hits=len(state['clamp_events']),**state)
            self.rows.append(row)
        return self._trace

    def __enter__(self):
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise FitBoundary('EXISTING_NATIVE_TRACE_OR_PROFILE_OWNER')
        sys.settrace(self._trace)
        return self

    def __exit__(self, *unused):
        sys.settrace(None)


class GeneralizedNativeFitter(NativeSingletonFitter):
    """Unchanged numerical fit/finalizer, physical singleton layers 4..8."""
    def _check(self, model, tok, hp, history, projector, layer):
        t = self.torch
        if hp.layers != [layer] or not hp.blue or hp.L2 != 1 or layer not in (4,5,6,7,8):
            raise FitBoundary('V2_SINGLETON_HPARAMS')
        if (hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay,hp.clamp_norm_factor,hp.kl_factor) != (25,.1,.5,.75,.0625):
            raise FitBoundary('V2_NATIVE_OPTIMIZER_CONTRACT')
        if tok.padding_side != 'right' or self.module.CONTEXT_TEMPLATES_CACHE != self.contexts:
            raise FitBoundary('V2_NATIVE_CONTEXT_OR_PADDING')
        name = hp.rewrite_module_tmp.format(layer)+'.weight'
        weights = dict(model.named_parameters()); w = weights[name]
        if (any(x.dtype != t.float32 for x in weights.values()) or
                history.dtype != t.float32 or projector.dtype != t.float32 or
                history.shape != (1,w.shape[1],w.shape[1]) or projector.shape != history.shape):
            raise FitBoundary('V2_FP32_SINGLETON_SCHEMA')
        if any(not t.isfinite(x).all().item() for x in (w,history,projector)):
            raise FitBoundary('V2_NONFINITE_ENTRY')
        return name,w

    def fit_capture(self, model, tok, hp, history, projector, requests, *, layer, instrument=True):
        if not instrument:
            result = self.fit(model,tok,hp,history,projector,requests,layer=layer,capture=True)
            result['receipt']['instrumentation'] = 'DISABLED_TECHNICAL_PARITY_REFERENCE'
            return result
        trace = NativeTrace(self.module.compute_z)
        try:
            with trace:
                result = self.fit(model,tok,hp,history,projector,requests,layer=layer,capture=True)
        except BaseException as exc:
            # Caller persists this local-only record before rethrow. No target,
            # gradient, optimizer or numerical operation is altered by tracing.
            exc.native_partial = dict(completed_requests=trace.rows,
                failed_requests=trace.failed,active_requests=list(trace.active.values()),
                instrumentation=trace.evidence,layer=layer)
            raise
        rows = trace.rows
        def failure(label):
            exc=FitBoundary(label)
            exc.native_partial=dict(completed_requests=trace.rows,failed_requests=trace.failed,
                active_requests=list(trace.active.values()),instrumentation=trace.evidence,
                layer=layer,native_fit_receipt=result.get('receipt'))
            return exc
        if [r['case_id'] for r in rows] != [int(r['case_id']) for r in requests]:
            raise failure('V2_NATIVE_REQUEST_TRACE_ORDER')
        if any(r['adam_updates'] > 24 or r['loss_evaluations'] > 25 or
               len(r['losses']) != r['loss_evaluations'] for r in rows):
            raise failure('V2_NATIVE_QUOTA_OR_TRACE_COVERAGE')
        z = self.torch.stack(result['captures']['compute_z'],dim=1)
        observed = self.torch.stack([r['target'] for r in rows],dim=1)
        if not self.torch.equal(z,observed):
            raise failure('V2_NATIVE_RETURN_CAPTURE_BRIDGE')
        result.update(target=z,anchors=self.torch.stack([r['anchor'] for r in rows],dim=1),
            radii=self.torch.tensor([r['radius'] for r in rows],dtype=self.torch.float32),
            target_observations=rows)
        result['receipt'].update(adam_updates=sum(r['adam_updates'] for r in rows),
            loss_evaluations=sum(r['loss_evaluations'] for r in rows),clamp_hits=sum(r['clamp_hits'] for r in rows),
            instrumentation=trace.evidence,extra_target_calls=0)
        return result

    def multi_layer_reference(self, model, tok, hp, history, projector, requests, *, include_history=False, capture=False):
        """Actual native BLUE ascending loop, no singleton reimplementation.

        History is excluded for endpoint parity by the already verified AST
        split. Optional final history invokes the exact extracted native loop.
        Caller owns full entry/return state restoration and technical cost.
        """
        layers = list(hp.layers)
        if not layers or layers != sorted(set(layers)) or any(l not in range(4,9) for l in layers):
            raise FitBoundary('V2_REFERENCE_LAYERS')
        if history.shape[0] != len(layers) or projector.shape != history.shape:
            raise FitBoundary('V2_REFERENCE_MP_STACK')
        params = dict(model.named_parameters())
        names = [hp.rewrite_module_tmp.format(l)+'.weight' for l in layers]
        for i,l in enumerate(layers):
            singleton = deepcopy(hp); singleton.layers=[l]
            self._check(model,tok,singleton,history[i:i+1],projector[i:i+1],l)
        weights_before={n:tensor_sha(params[n]) for n in names}
        h_before,p_before=tensor_sha(history),tensor_sha(projector)
        counts={}; captures={} if capture else None
        fit,finalizer=self._functions(counts,captures)
        start=time.monotonic()
        returned,cache=fit(model,tok,requests,hp,cache_template=None,cache_c=history,P=projector)
        if returned is not model or cache is not history or tensor_sha(history)!=h_before or tensor_sha(projector)!=p_before:
            raise FitBoundary('V2_MULTILAYER_REFERENCE_MUTATION')
        if counts.get('compute_z')!=len(layers)*len(requests) or counts.get('solve')!=len(layers):
            raise FitBoundary('V2_MULTILAYER_REFERENCE_COUNTS')
        if include_history:
            finalizer(model,tok,requests,hp,cache_template=None,cache_c=history,P=projector)
        for name in names:
            if not self.torch.isfinite(params[name]).all():raise FitBoundary('V2_REFERENCE_NONFINITE')
        return dict(weights={l:params[n].detach().cpu().clone() for l,n in zip(layers,names)},captures=captures,
            receipt=dict(**counts,layers=layers,entry_weights=weights_before,
                endpoint_weights={n:tensor_sha(params[n]) for n in names},
                history_before=h_before,history_after=tensor_sha(history),projector_sha256=p_before,
                history_appends=len(layers) if include_history else 0,seconds=time.monotonic()-start,
                source=self.source_evidence,route='EXACT_NATIVE_BLUE_LOOP_AST_HISTORY_SPLIT'))
