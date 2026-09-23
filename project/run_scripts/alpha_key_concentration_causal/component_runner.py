"""Actual E4 O-W/component-interchange/K-R diagnostics, with no additional z.

The caller supplies original native deltas and pre-write RAM transactions.
Every candidate is independently restored; this driver never appends history.
Only fixed geometry calibration512 fits means/bases. Observer N/H/R/P results
do not select, rank, tune, or suppress any mathematically defined branch.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .common import digest, rng_get, rng_preserved, save, tensor_file, tensor_sha
from .interventions import (
    InterventionError, PromptActionMean, action_component_hook,
    calibration_key_basis, interchange_keys, kr_cross_solve, weight_ablation,
)


class _PrefixDone(Exception):
    pass


def _device(rt):
    return rt.weights[4].device


def _module(rt, layer):
    return rt.model.model.layers[layer].mlp.down_proj


def _record_id(record):
    return int(record['case_id'])


def _calibration(rt, panels):
    if 'cohorts' not in panels and 'geometry-panels.json' in panels:
        panels = panels['geometry-panels.json']
    ids = [int(case) for cohort in ('early', 'middle', 'onset', 'late')
           for case in panels['cohorts'][cohort]['calibration_case_ids']]
    if len(ids) != 512 or len(set(ids)) != 512:
        raise InterventionError('component calibration must be exact unique512')
    return ids, [rt.byid[case] for case in ids]


def _canonical_text(record):
    request = record['requested_rewrite']
    return request['prompt'].format(request['subject'])


class InputBinding:
    """Nested-call-safe actual model input/mask provenance for all-token hooks."""
    def __init__(self, model):
        self.model = model
        self.frames, self.records, self.handles = [], [], []

    def _pre(self, module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        mask = kwargs.get('attention_mask')
        if ids is None or mask is None or ids.ndim != 2 or ids.shape != mask.shape:
            raise InterventionError('all-token intervention requires actual input_ids and attention_mask')
        if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
            raise InterventionError('non-binary actual attention mask')
        frame = {'mask': mask.bool(), 'input_ids': ids,
                 'input_sha256': tensor_sha(ids), 'mask_sha256': tensor_sha(mask),
                 'shape': list(ids.shape), 'valid_tokens': int(mask.sum()),
                 'position_ids_sha256': tensor_sha(kwargs['position_ids']) if kwargs.get('position_ids') is not None else 'MODEL_DEFAULT',
                 'depth': len(self.frames)}
        self.frames.append(frame)
        self.records.append({k: v for k, v in frame.items() if k not in ('mask', 'input_ids')})

    def _post(self, module, args, kwargs, output):
        if self.frames:
            self.frames.pop()

    def mask(self):
        if not self.frames:
            raise InterventionError('missing bound model input for component hook')
        return self.frames[-1]['mask']

    def __enter__(self):
        self.handles = [self.model.register_forward_pre_hook(self._pre, with_kwargs=True),
                        self.model.register_forward_hook(self._post, with_kwargs=True, always_call=True)]
        return self

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.frames.clear()


def physical_full_hook(stage_weight, delta, valid_mask):
    """F.linear(k,FP32(U+delta)): same materialization/GEMM as physical write.

    The different F.linear(k,U)+F.linear(k,delta) path is measured separately,
    never assumed bitwise equal. Padding receives no intervention.
    """
    endpoint = (stage_weight.detach() + delta.detach()).clone()
    def hook(module, inputs, output):
        key = inputs[0]
        mask = valid_mask()
        if mask.shape != key.shape[:-1]:
            raise InterventionError('full component input/mask mismatch')
        transformed = F.linear(key, endpoint)
        base = output[0] if isinstance(output, tuple) else output
        result = torch.where(mask.unsqueeze(-1), transformed, base)
        if not bool(torch.isfinite(result).all()):
            raise InterventionError('nonfinite full physical component')
        return (result,) + output[1:] if isinstance(output, tuple) else result
    return hook


@contextmanager
def component_context(rt, layer, base, delta, mean, mode):
    with InputBinding(rt.model) as binding:
        hook = physical_full_hook(base, delta, binding.mask) if mode == 'full' else action_component_hook(base, delta, mean, mode, binding.mask)
        handle = _module(rt, layer).register_forward_hook(hook)
        try:
            yield binding
        finally:
            handle.remove()


def fit_action_mean(rt, layer, delta, records, *, microbatch=16):
    """512 canonical prompts; within-prompt valid-token mean then prompt mean."""
    if len(records) != 512 or len({_record_id(r) for r in records}) != 512:
        raise InterventionError('action mean requires all fixed512 canonical requests')
    accumulator = PromptActionMean(panel_role='calibration512')
    started = time.monotonic()
    with rng_preserved(), torch.no_grad(), InputBinding(rt.model) as binding:
        def capture(module, args):
            actions = F.linear(args[0], delta)
            mask = binding.mask()
            for action, row_mask in zip(actions, mask, strict=True):
                accumulator.add(action, row_mask)
            raise _PrefixDone()
        handle = _module(rt, layer).register_forward_pre_hook(capture)
        try:
            for first in range(0, len(records), microbatch):
                pack = rt.tok([_canonical_text(r) for r in records[first:first + microbatch]], padding=True, return_tensors='pt')
                pack = {k: v.to(_device(rt)) for k, v in pack.items()}
                try:
                    rt.model(**pack)
                except _PrefixDone:
                    pass
        finally:
            handle.remove()
    mean, receipt = accumulator.finish()
    if receipt['prompts'] != 512:
        raise InterventionError('calibration action mean missing prompts')
    receipt.update(seconds=time.monotonic() - started, physical_layer=layer,
                   context_policy='one canonical rewrite per calibration request; no generated-context reweighting',
                   case_order_sha256=digest([_record_id(r) for r in records]),
                   microbatch=microbatch, inputs=binding.records)
    return mean, receipt


def _forward_logits(rt, pack):
    with torch.no_grad():
        value = rt.model(**pack).logits
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise InterventionError('component parity logits nonfinite/non-FP32')
    return value.detach().cpu()


def full_hook_parity(rt, layer, pre_state, delta, mean, records):
    """Same-token/GEMM full physical↔hook exact valid-logit comparison."""
    pack = rt.tok([_canonical_text(r) for r in records[:2]], padding=True, return_tensors='pt')
    pack = {k: v.to(_device(rt)) for k, v in pack.items()}
    mask = pack['attention_mask'].bool().cpu()
    rt.restore(pre_state)
    base = rt.weights[layer].detach().clone()
    with torch.no_grad():
        rt.weights[layer].copy_(base + delta)
    physical = _forward_logits(rt, pack)
    rt.restore(pre_state)
    with component_context(rt, layer, base, delta, mean, 'full'):
        hooked = _forward_logits(rt, pack)
    with InputBinding(rt.model) as binding:
        # Original additive activation decomposition is a numerical diagnostic.
        handle = _module(rt, layer).register_forward_hook(action_component_hook(base, delta, mean, 'full', binding.mask))
        try:
            summed = _forward_logits(rt, pack)
        finally:
            handle.remove()
    a, b, c = physical[mask], hooked[mask], summed[mask]
    difference = (a.double() - b.double()).abs()
    sum_difference = (a.double() - c.double()).abs()
    receipt = {'status': 'PASS' if torch.equal(a, b) else 'FAIL',
               'full_physical_valid_logits_bitwise_equal': bool(torch.equal(a, b)),
               'full_physical_max_abs': float(difference.max()),
               'full_physical_rms': float(difference.square().mean().sqrt()),
               'sum_route_max_abs': float(sum_difference.max()),
               'sum_route_rms': float(sum_difference.square().mean().sqrt()),
               'sum_route_not_bitwise_assumed': True,
               'tokens': int(mask.sum()), 'vocabulary': a.shape[-1],
               'input_ids_sha256': tensor_sha(pack['input_ids']),
               'attention_mask_sha256': tensor_sha(pack['attention_mask']),
               'threshold': 0, 'kernel': 'F.linear(input,FP32(U+delta))'}
    rt.restore(pre_state)
    return receipt


class KeyInterchange:
    """Bounded donor prefix pass for each exact receiving forward input.

    The donor differs only in one upstream weight. Capture stops before the
    downstream down_proj. Its full-token keys live for this one microbatch only;
    the receiving weight is restored in finally before the real forward begins.
    This extra prefix cost is explicit, not hidden as cache reuse.
    """
    def __init__(self, rt, upstream, downstream, donor_weight, basis):
        self.rt, self.upstream, self.downstream = rt, upstream, downstream
        self.donor_weight, self.basis = donor_weight, basis
        self.inside = False
        self.donor = None
        self.records, self.handles = [], []

    def _root_pre(self, module, args, kwargs):
        if self.inside:
            return
        if self.donor is not None:
            raise InterventionError('unconsumed donor prefix keys')
        current = self.rt.weights[self.upstream].detach().clone()
        started = time.monotonic()
        self.inside = True
        try:
            with torch.no_grad():
                self.rt.weights[self.upstream].copy_(self.donor_weight)
                try:
                    self.rt.model(*args, **kwargs)
                except _PrefixDone:
                    pass
            if self.donor is None:
                raise InterventionError('donor prefix did not reach target key')
        finally:
            with torch.no_grad():
                self.rt.weights[self.upstream].copy_(current)
            self.inside = False
        frame = self.binding.frames[-1]
        self.records.append({'input_sha256': frame['input_sha256'], 'mask_sha256': frame['mask_sha256'],
                             'input_shape': frame['shape'], 'seconds_donor_prefix': time.monotonic() - started,
                             'donor_shape': list(self.donor.shape), 'same_input': True,
                             'receiving_upstream_restored_before_forward': True})

    def _target_pre(self, module, args):
        if self.inside:
            self.donor = args[0].detach().clone()
            raise _PrefixDone()
        if self.donor is None:
            raise InterventionError('receiving key without donor')
        original = args[0]
        donor, self.donor = self.donor, None
        return (interchange_keys(original, donor, self.basis, self.binding.mask()),) + args[1:]

    def __enter__(self):
        self.binding = InputBinding(self.rt.model).__enter__()
        self.handles = [self.rt.model.register_forward_pre_hook(self._root_pre, with_kwargs=True),
                        _module(self.rt, self.downstream).register_forward_pre_hook(self._target_pre)]
        return self

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.binding.__exit__(*exc)
        self.donor = None


class ActionAudit:
    """Postseal all-valid-token action summaries, never fitting/selection."""
    def __init__(self, rt, layer, delta):
        self.rt, self.layer, self.delta, self.rows = rt, layer, delta, []

    def __enter__(self):
        self.binding = InputBinding(self.rt.model).__enter__()
        def pre(module, args):
            actions = F.linear(args[0], self.delta).double().square().sum(-1)
            mask = self.binding.mask()
            frame = self.binding.frames[-1]
            for index, (energy, valid) in enumerate(zip(actions, mask, strict=True)):
                positions = valid.nonzero().flatten()
                self.rows.append({'input_sha256': frame['input_sha256'], 'row': index,
                                  'valid_tokens': int(valid.sum()),
                                  'all_valid_action_energy': float(energy[valid].sum()),
                                  'last_valid_action_energy': float(energy[positions[-1]]),
                                  'last_valid_is_prediction_position': 'CALLER_OBSERVER_SCHEMA_DEPENDENT'})
        self.handle = _module(self.rt, self.layer).register_forward_pre_hook(pre)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        self.binding.__exit__(*exc)


def _seal_observe(rt, endpoint_id, out, observe, operation):
    before = rt.signature()
    def rng_identity():
        python_state, numpy_state, cpu_state, cuda_states = rng_get()
        return digest({'python': python_state, 'numpy': [numpy_state[0], numpy_state[1].tolist(),
                       numpy_state[2], numpy_state[3], numpy_state[4]],
                       'torch': tensor_sha(cpu_state), 'cuda': [tensor_sha(s) for s in cuda_states]})
    before_rng = rng_identity()
    seal = save(Path(out) / 'endpoint-seal.json', {'endpoint_id': endpoint_id, 'state': before,
                                                 'rng_sha256': before_rng, 'operation': operation, 'new_history_appends': 0})
    with rng_preserved():
        result = observe(endpoint_id, Path(out) / 'observer')
    after = rt.signature()
    after_rng = rng_identity()
    if before != after or before_rng != after_rng:
        raise InterventionError('component observer changed W/M/context/cursor or failed RNG restoration')
    save(Path(out) / 'observer-nonmutation.json', {'state_before': before, 'state_after': after,
                                                'same': True, 'endpoint_seal': seal,
                                                'rng_before_sha256': before_rng, 'rng_after_sha256': after_rng,
                                                'RNG': 'EXPLICIT_CALL_SCOPED_RESTORE_VERIFIED'})
    return result


def _native_operands(rt, layer, requests, z):
    with torch.no_grad():
        K = rt.native.compute_ks(rt.model, rt.tok, requests, rt.hp, layer, rt.contexts).T
        h8 = rt.native.get_module_input_output_at_words(
            rt.model, rt.tok, 8, context_templates=[r['prompt'] for r in requests],
            words=[r['subject'] for r in requests], module_template=rt.hp.layer_module_tmp,
            fact_token_strategy=rt.hp.fact_token)[1].T
        residual = z.to(h8.device) - h8
        if K.shape[1] % residual.shape[1]:
            raise InterventionError('native residual repeat-factor mismatch')
        residual = residual.repeat_interleave(K.shape[1] // residual.shape[1], dim=1)
        residual = residual / (9 - layer)
    if not bool(torch.isfinite(K).all()) or not bool(torch.isfinite(residual).all()):
        raise InterventionError('nonfinite K/R diagnostic operand')
    return K, residual


def run_components(rt, entry, pre_states, layer_deltas, z, requests, panels, observe, out):
    """Run 2 component + 2 matrix + 2 K/R families for one native entry.

    There are four approved entry states, hence 24 E4-W/E4-KR families total.
    No downstream suffix writes after each K/R receiving-layer update: this is
    explicitly the same receiving-state causal matrix/physical diagnostic.
    """
    if entry not in (50, 70, 80, 90) or len(requests) != 100:
        raise InterventionError('undeclared entry or current request cardinality')
    if set(pre_states) != {5, 6} or not {5, 6}.issubset(layer_deltas):
        raise InterventionError('native pre-L5/pre-L6 RAM states and deltas required')
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    ids, records = _calibration(rt, panels)
    entry_saved = rt.snapshot()
    rows, artifacts, timing = [], [], []
    started = time.monotonic()
    save(root / 'plan.json', {'entry': entry, 'calibration_ids': ids, 'calibration_order_sha256': digest(ids),
                              'additional_z': 0, 'current_requests': [r['case_id'] for r in requests],
                              'upstream_layers': [5, 6], 'downstream_layers': [6, 7],
                              'interchange_ranks': [1, 2, 4], 'N_P_selection': False,
                              'new_history_appends': 0, 'weight_snapshots_persisted': False,
                              'full_hook_route': 'FP32(U+delta) physical F.linear; exact valid-token parity'})
    try:
        for upstream in (5, 6):
            downstream = upstream + 1
            pre = pre_states[upstream]
            delta = layer_deltas[upstream].to(_device(rt)).detach()
            rt.restore(pre)
            base = rt.weights[upstream].detach().clone()
            step_start = time.monotonic()
            mean, mean_receipt = fit_action_mean(rt, upstream, delta, records)
            artifacts.append(save(root / f'L{upstream}' / 'calibration-mean.json', mean_receipt))
            artifacts.append(tensor_file(root / f'L{upstream}' / 'calibration-mean.pt', {'action_mean': mean.cpu()}))
            parity = full_hook_parity(rt, upstream, pre, delta, mean, records)
            from .numerical_policy import annotate
            parity = annotate(parity,getattr(rt,'numerical_comparison_policy','BLOCK_ON_NUMERICAL_MISMATCH'))
            artifacts.append(save(root / f'L{upstream}' / 'full-hook-physical-parity.json', parity))
            if parity['blocks_execution']:
                raise InterventionError('full component/physical valid-token parity failed')
            timing.append({'kind': 'mean_and_full_parity', 'layer': upstream, 'seconds': time.monotonic() - step_start})

            for mode in ('no', 'mean', 'centered', 'full'):
                rt.restore(pre)
                endpoint = f'W{entry}-L{upstream}-component-{mode}'
                target = root / f'L{upstream}' / 'components' / mode
                operation = {'family': 'O-W-activation', 'mode': mode, 'upstream': upstream,
                             'delta_sha256': tensor_sha(delta), 'mean_sha256': tensor_sha(mean),
                             'all_attention_mask_valid_tokens': True, 'not_weight_method': mode in ('mean', 'centered')}
                with component_context(rt, upstream, base, delta, mean, mode) as binding:
                    _seal_observe(rt, endpoint, target, observe, operation)
                artifacts.append(save(target / 'input-binding.json', binding.records))
                rows.append({'family': 'components', 'layer': upstream, 'endpoint': endpoint, 'status': 'COMPLETED'})

            ablation = weight_ablation(delta, mean)
            artifacts.append(save(root / f'L{upstream}' / 'matrix-definition.json', {'status': ablation.status,
                                   'reason': ablation.reason, 'receipt': ablation.receipt}))
            if ablation.status == 'DEFINED':
                for name, move in (('parallel', ablation.parallel), ('perpendicular', ablation.perpendicular), ('norm_control', ablation.norm_control)):
                    rt.restore(pre)
                    with torch.no_grad():
                        rt.weights[upstream].copy_(base + move)
                    endpoint = f'W{entry}-L{upstream}-matrix-{name}'
                    target = root / f'L{upstream}' / 'matrix' / name
                    _seal_observe(rt, endpoint, target, observe, {'family': 'O-W-weight', 'name': name,
                                   'direction_sha256': tensor_sha(ablation.direction), 'update_sha256': tensor_sha(move),
                                   'alpha_norm_control': ablation.alpha, 'quality_matching': 'NOT_ESTABLISHED'})
                    rows.append({'family': 'matrix', 'layer': upstream, 'endpoint': endpoint, 'status': 'COMPLETED'})
            else:
                rows.append({'family': 'matrix', 'layer': upstream, 'status': 'NOT_APPLICABLE', 'reason': ablation.reason})
            del ablation

            # b=full upstream physical native update; basis uses calibration only.
            rt.restore(pre)
            with torch.no_grad():
                rt.weights[upstream].copy_(base + delta)
            full_weight = rt.weights[upstream].detach().clone()
            calibrated = rt.capture(records, contexts=True, features=False, full=False)
            basis_keys = calibrated['means'][downstream].T.to(_device(rt))
            del calibrated
            for rank in (1, 2, 4):
                basis = calibration_key_basis(basis_keys, rank)
                artifacts.append(save(root / f'L{upstream}' / 'interchange' / f'rank{rank}-basis.json',
                                      {'status': basis.status, 'reason': basis.reason, 'receipt': basis.receipt}))
                if basis.status != 'DEFINED':
                    rows.append({'family': 'interchange', 'layer': upstream, 'rank': rank, 'status': 'NOT_APPLICABLE', 'reason': basis.reason})
                    continue
                artifacts.append(tensor_file(root / f'L{upstream}' / 'interchange' / f'rank{rank}-basis.pt', {'calibration_basis': basis.vectors.cpu()}))
                for receiving, donor in (('off', 'on'), ('on', 'off')):
                    rt.restore(pre)
                    with torch.no_grad():
                        rt.weights[upstream].copy_(base if receiving == 'off' else full_weight)
                    donor_weight = base if donor == 'off' else full_weight
                    endpoint = f'W{entry}-L{downstream}-interchange-r{rank}-{receiving}-from-{donor}'
                    target = root / f'L{upstream}' / 'interchange' / f'rank{rank}-{receiving}-from-{donor}'
                    with KeyInterchange(rt, upstream, downstream, donor_weight, basis.vectors) as interchange:
                        _seal_observe(rt, endpoint, target, observe, {'family': 'inference-key-interchange',
                                       'receiving': receiving, 'donor': donor, 'rank': rank,
                                       'basis_sha256': tensor_sha(basis.vectors), 'upstream': upstream,
                                       'downstream': downstream, 'writer_operand_intervention': False})
                    artifacts.append(save(target / 'same-input-donor-prefix.json', interchange.records))
                    rows.append({'family': 'interchange', 'layer': upstream, 'rank': rank, 'endpoint': endpoint, 'status': 'COMPLETED'})
            del basis_keys

            # K/R operands are recomputed at a and b; original entry z is shared.
            rt.restore(pre)
            ka, ra = _native_operands(rt, downstream, requests, z)
            with torch.no_grad():
                rt.weights[upstream].copy_(full_weight)
            kb, rb = _native_operands(rt, downstream, requests, z)
            receiving_id = digest(rt.signature())
            m = rt.M[downstream - 4].to(_device(rt))
            p = rt.P[downstream - 4].to(_device(rt))
            step_start = time.monotonic()
            updates, kr_receipt = kr_cross_solve(ka, kb, ra, rb, m, p, receiving_state_id=receiving_id)
            timing.append({'kind': 'KR_four_original_solves', 'layer': downstream, 'seconds': time.monotonic() - step_start})
            artifacts.append(tensor_file(root / f'L{upstream}' / 'KR' / 'operands.pt',
                              {'K_a': ka.cpu(), 'K_b': kb.cpu(), 'R_a': ra.cpu(), 'R_b': rb.cpu()}))
            artifacts.append(save(root / f'L{upstream}' / 'KR' / 'solve.json', kr_receipt))
            for name, update in updates.items():
                rt.restore(pre)
                with torch.no_grad():
                    rt.weights[upstream].copy_(full_weight)
                if digest(rt.signature()) != receiving_id:
                    raise InterventionError('K/R receiving state differs across matrix combinations')
                with torch.no_grad():
                    rt.weights[downstream].add_(update)
                endpoint = f'W{entry}-L{downstream}-KR-{name}'
                target = root / f'L{upstream}' / 'KR' / name
                artifacts.append(tensor_file(target / 'update.pt', {'actual_delta': update.cpu()}))
                actual_response = update @ kb
                artifacts.append(save(target / 'receiving-action.json', {'actual_response_energy': float(actual_response.double().square().sum()),
                                       'response_per_request': actual_response.double().square().sum(0).tolist(),
                                       'receiving_key_sha256': tensor_sha(kb), 'receiving_state_id': receiving_id,
                                       'suffix_writes': 'NOT_REQUESTED_NOT_EXECUTED', 'history_appends': 0}))
                with ActionAudit(rt, downstream, update) as action:
                    _seal_observe(rt, endpoint, target, observe, {'family': 'K/R-writer', 'operand_pair': name,
                                   'receiving_state_id': receiving_id, 'update_sha256': tensor_sha(update),
                                   'P_sha256': tensor_sha(p), 'M_sha256': tensor_sha(m), 'ridge': 10,
                                   'residual_divisor': 9 - downstream, 'extra_z': 0})
                artifacts.append(save(target / 'all-valid-observer-actions.json', action.rows))
                rows.append({'family': 'KR', 'layer': upstream, 'endpoint': endpoint, 'status': 'COMPLETED'})
            del ka, kb, ra, rb, p, m, updates, base, full_weight, delta, mean
        save(root / 'cell-receipts.json', rows)
        save(root / 'artifact-index.json', artifacts)
        save(root / 'cost.json', {'events': timing, 'seconds_inclusive': time.monotonic() - started,
                                 'observer_and_donor_prefix': 'see per-endpoint receipts; nested, do not sum twice'})
    except BaseException as exc:
        save(root / 'failure.json', {'status': 'TECHNICAL_FAILED', 'type': type(exc).__name__, 'message': str(exc),
                                     'completed_endpoints': rows, 'seconds': time.monotonic() - started})
        raise
    finally:
        rt.restore(entry_saved)
    terminal = {'status': 'COMPLETED', 'entry': entry, 'families': 6, 'endpoints': rows,
                'additional_z_requests': 0, 'history_appends': 0, 'caller_state_restored': True,
                'new_state_checkpoint_saved': False, 'seconds': time.monotonic() - started}
    save(root / 'terminal.json', terminal)
    return terminal
