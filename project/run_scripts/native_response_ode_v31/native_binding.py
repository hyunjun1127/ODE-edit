"""Thin native residual/metric binding. Historical ORBFH remains unmodified."""
import time
import torch
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.adapters import LayerBuild
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256
from .algebra import NativeMetricBoundary, factor_inner


class NativeDictionary:
    def __init__(self, family):
        self.family = family
        self.qref = None
        self.qfref = None
        self.build_count = 0
        self.solve_count = 0
        self.wall = 0.
        self.projector_receipts = []
        self._operators = {}

    def build(self, terminal, version, *, audit_residual_divisor=1):
        f = self.family
        if f.tokenizer.padding_side != 'right':
            raise RuntimeError('TOKENIZER_RIGHT_PADDING_BOUNDARY')
        result = []
        started = time.perf_counter()
        with torch.no_grad(), old._model_name(f.model, str(f.hparams.model_name)):
            for layer in (4,5,6,7,8):
                if f.family == 'AlphaEdit':
                    # Exact accepted P-inside native-form primitive, divisor=1.
                    build = f.build_layer(layer=layer,current_terminal=terminal,fixed_z=f.fixed_z,
                                          residual_denominator=audit_residual_divisor,state_version=version)
                else:
                    # Preserve stock MEMIT's ephemeral FP64 solve. The model,
                    # detached overlay factors, primal and tangent remain FP32.
                    keys = f.module.compute_ks(f.model,f.tokenizer,list(f.requests),f.hparams,
                                               layer,[list(g) for g in f.contexts]).T.detach()
                    residual = (f.fixed_z.values-terminal).to(f.device)
                    if keys.shape[1] % residual.shape[1]:
                        raise RuntimeError('KEY_MULTIPLICITY_BOUNDARY')
                    residual = residual.repeat_interleave(keys.shape[1]//residual.shape[1],dim=1)
                    residual = residual / float(audit_residual_divisor)
                    cov = f.module.get_cov(f.model,f.tokenizer,f.hparams.rewrite_module_tmp.format(layer),
                        f.hparams.mom2_dataset,f.hparams.mom2_n_samples,f.hparams.mom2_dtype,
                        force_recompute=False,hparams=f.hparams)
                    kd = keys.double()
                    matrix = float(f.hparams.mom2_update_weight)*cov.double() + kd@kd.T
                    solved = torch.linalg.solve(matrix,kd)
                    err = float(torch.linalg.norm(matrix@solved-kd)/(torch.linalg.norm(matrix)*torch.linalg.norm(solved)+torch.linalg.norm(kd)))
                    name = f'{f.hparams.rewrite_module_tmp.format(layer)}.weight'
                    if (residual.shape[0],solved.shape[0]) != tuple(f.parameters[name].shape):
                        raise RuntimeError('NATIVE_ORIENTATION_BOUNDARY')
                    build = LayerBuild(layer,name,residual.float().cpu(),solved.float().cpu(),version,audit_residual_divisor,
                        tensor_sha256(residual.float().cpu()),tensor_sha256(keys.float().cpu()),
                        canonical_hash(dict(family='MEMIT',solve='stock_FP64',divisor=1,layer=layer)),err)
                    del matrix,solved,cov,kd,keys
                result.append(build)
                self.build_count += 1
                self.solve_count += 1
        old._sync()
        self.wall += time.perf_counter()-started
        return result

    def operator(self, layer):
        if layer in self._operators:
            return self._operators[layer]
        f = self.family
        position = layer-4
        if f.family == 'AlphaEdit':
            matrix = f._alpha_cache_entry[position]
            lam = float(f.hparams.L2)
            coefficient = 1.
        else:
            with old._model_name(f.model,str(f.hparams.model_name)):
                matrix = f.module.get_cov(f.model,f.tokenizer,f.hparams.rewrite_module_tmp.format(layer),
                    f.hparams.mom2_dataset,f.hparams.mom2_n_samples,f.hparams.mom2_dtype,
                    force_recompute=False,hparams=f.hparams).detach().cpu()
            lam = 0.
            coefficient = float(f.hparams.mom2_update_weight)
        def apply(value):
            value = value.double().cpu()
            output = torch.empty_like(value)
            # Avoid a persistent FP64 copy of the dense native metric.
            for start in range(0,matrix.shape[0],512):
                stop = min(start+512,matrix.shape[0])
                output[start:stop] = coefficient*(matrix[start:stop].double()@value)+lam*value[start:stop]
            return output
        self._operators[layer] = apply
        return apply

    def raw(self, a, b=None, *, frobenius=False):
        if b is None:
            b = a
        if a.layer != b.layer:
            return 0.
        op = (lambda v:v) if frobenius else self.operator(a.layer)
        return factor_inner(a.left,a.right,b.left,b.right,op)

    def capture_reference(self, builds):
        if self.qref is not None:
            raise RuntimeError('QREF_RECAPTURE_BOUNDARY')
        raw = [self.raw(b) for b in builds]
        fraw = [self.raw(b,frobenius=True) for b in builds]
        for native,frob in zip(raw,fraw,strict=True):
            if frob != 0 and (native <= 0 or not torch.isfinite(torch.tensor(native,dtype=torch.float64))):
                raise NativeMetricBoundary('NATIVE_METRIC_UNRESOLVED')
        self.qref, self.qfref = sum(raw),sum(fraw)
        return dict(qN_ref=self.qref,qF_ref=self.qfref,capture_count=1,raw_layer_actions=raw,
                    metric='lambda_mom2_C' if self.family.family=='MEMIT' else 'entry_M_plus_L2_I')

    def whiten(self, builds):
        if self.qref is None:
            raise RuntimeError('QREF_MISSING')
        active, q, gf = [], [], []
        for build in builds:
            frob = self.raw(build,frobenius=True)
            if frob == 0:
                continue
            raw = self.raw(build)
            if raw <= 0 or self.qref <= 0:
                raise NativeMetricBoundary('NATIVE_METRIC_UNRESOLVED')
            active.append(build); q.append(raw/self.qref)
            gf.append(frob/(raw/self.qref)/self.qfref)
        return active,torch.tensor(q,dtype=torch.float64),torch.diag(torch.tensor(gf,dtype=torch.float64))

    def net_action(self, deltas):
        # Exact factor-Gram norm of the cumulative physical block update;
        # never a sum of changing dictionary coefficients.
        native=frob=0.
        for i,a in enumerate(deltas):
            for b in deltas[i:]:
                if a.layer != b.layer:
                    continue
                scale = a.coefficient*b.coefficient*(1 if a is b else 2)
                native += scale*self.raw(a,b)
                frob += scale*self.raw(a,b,frobenius=True)
        return dict(native_net_raw=native,native_net_normalized=native/self.qref if self.qref else 0.,
                    frobenius_net_sq=frob)

    def actual_dense_action(self, parameters, entry):
        """Observation-only actual materialized endpoint, including FP32 rounding.

        This cost is accounted as endpoint observation, not controller solve.
        All comparators, including stock Official, use the same entry metric.
        """
        started=time.perf_counter();rows=[]
        for layer in (4,5,6,7,8):
            name=f'{self.family.hparams.rewrite_module_tmp.format(layer)}.weight'
            delta=parameters[name].detach().cpu().double()-entry[name].detach().cpu().double()
            q=float((delta.T*self.operator(layer)(delta.T)).sum())
            frob=float(delta.square().sum())
            rows.append(dict(layer=layer,native_raw=q,frobenius_sq=frob,
                materialized_nonzero=int((delta!=0).sum()),parameter_elements=delta.numel()))
        raw=sum(row['native_raw'] for row in rows)
        return dict(native_net_raw=raw,native_net_normalized=raw/self.qref if self.qref else 0.,
                    frobenius_net_sq=sum(row['frobenius_sq'] for row in rows),layers=rows,
                    observation_wall_seconds=time.perf_counter()-started,controller_influence_count=0)
