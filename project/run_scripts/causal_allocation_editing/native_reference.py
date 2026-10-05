"""One unchanged native L8 compute_z endpoint per B1 request, never a write.

The only compatibility adapter is a process-local Tensor/tuple container view.
Native source, Adam, norm, clamp, KL direction and early exit remain untouched.
Returned delta (NOT reconstructed target-anchor) is the reference residual.
"""
import copy
import importlib
import time
import torch
from project.run_scripts.jlz_two_arm.baseline_pilot import (
    _import_native, _Proxy, _trace_dict_compat, _verify_hparams, _source_closure)
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_equal
from . import require, verify, member, state, tensor_sha, write

def reference_fit(adapter, bench, records, entry, history, config, out):
    before, rng, guard = state(adapter, history), rng_snapshot(), adapter.guard()
    hooks = adapter.hook_signature()
    anchor_layer = adapter.profile['anchor_layer']
    require(anchor_layer in adapter.sites and len(records) == entry['pack']['n_requests'], 'REFERENCE_SCOPE')
    for row in config['native_reference']:
        verify(row)
    flags = {name: p.requires_grad for name, p in adapter.model.named_parameters()}
    u = {l: torch.zeros(adapter.dims[l][0], len(records), device=adapter.device, dtype=torch.float32)
         for l in adapter.sites}
    Rref={l:torch.zeros_like(v) for l,v in u.items()}
    receipt = dict(reference='ORIGINAL_NATIVE_L8_RETURNED_DELTA', layer=anchor_layer,
                   requests=len(records), native_calls=[], reference_commits=0, history_appends=0,
                   checkpoint_saved=False, endpoint_expanded=False)
    start = time.monotonic()
    try:
        with _import_native(config['native_root']):
            module = importlib.import_module('memit.compute_z')
            hpmod = importlib.import_module('memit.memit_hparams')
            hp = hpmod.MEMITHyperParams.from_json(config['native_hparams'])
            _verify_hparams(hp, 'MEMIT-H')
            original_torch, original_hook = module.torch, module.nethook
            active = {}; totals = dict(forwards=0, updates=0)
            class CountAdam(original_torch.optim.Adam):
                def step(self, *args, **kwargs):
                    totals['updates'] += 1; active['updates'] += 1
                    return super().step(*args, **kwargs)
            def forward(_model, args, kwargs):
                totals['forwards'] += 1; active['evaluations'] += 1
                if active.get('input') is None:
                    expected = active['expected_tokens']
                    require(torch.equal(kwargs['input_ids'].cpu(), expected['input_ids']) and
                            torch.equal(kwargs['attention_mask'].cpu(), expected['attention_mask']),
                            'NATIVE_REFERENCE_INPUT_IDENTITY')
                    active['input'] = True
            handle = adapter.model.register_forward_pre_hook(forward, with_kwargs=True)
            module.torch = _Proxy(original_torch, optim=_Proxy(original_torch.optim, Adam=CountAdam))
            module.nethook = _Proxy(original_hook, TraceDict=_trace_dict_compat(original_hook.TraceDict))
            try:
                for owner, record in enumerate(records):
                    request = dict(copy.deepcopy(record['requested_rewrite']), case_id=record['case_id'])
                    if not request['target_new']['str'].startswith(' '):
                        request['target_new']['str'] = ' ' + request['target_new']['str']
                    from project.run_scripts.jlz_pilot.prompts import prepare as native_pack
                    exact=native_pack(bench.tokenizer,[request],bench.contexts,'cpu')
                    active.clear(); active.update(owner=owner,updates=0,evaluations=0,input=None,
                                                  expected_tokens=exact['tokens'])
                    t0 = time.monotonic()
                    target, delta = module.compute_z(adapter.model, bench.tokenizer, request, hp,
                                                     anchor_layer, bench.contexts, return_delta=True, verbose=False)
                    require(delta.shape == (adapter.dims[anchor_layer][0],) and delta.dtype == torch.float32
                            and bool(torch.isfinite(delta).all()), 'NATIVE_REFERENCE_RETURNED_DELTA')
                    anchor = entry['anchors'][anchor_layer][owner]
                    require(bool(torch.isfinite(anchor)) and bool(anchor > 0), 'NATIVE_REFERENCE_ANCHOR')
                    # Target's clean canonical anchor is captured by native itself.
                    canonical = target.detach() - delta.detach()
                    require(torch.isclose(canonical.norm(), anchor, atol=2e-5, rtol=2e-4), 'NATIVE_REFERENCE_ANCHOR_PARITY')
                    u[anchor_layer][:, owner] = delta.detach() / anchor
                    Rref[anchor_layer][:,owner]=delta.detach()
                    roundtrip=anchor*u[anchor_layer][:,owner]
                    rho = float(u[anchor_layer][:, owner].norm().double())
                    require(rho <= float(hp.clamp_norm_factor) + 1e-6, 'NATIVE_REFERENCE_CAP')
                    require(1 <= active['evaluations'] <= 25 and 0 <= active['updates'] <= 24
                            and active['updates'] == active['evaluations'] - 1, 'NATIVE_REFERENCE_COUNTERS')
                    receipt['native_calls'].append(dict(case_id=record['case_id'], owner=owner,
                        evaluations=active['evaluations'], updates=active['updates'], seconds=time.monotonic()-t0,
                        returned_delta=tensor_sha(delta), target=tensor_sha(target), anchor=float(anchor), rho=rho,
                        u_FP32_roundtrip_maxabs=float((roundtrip-delta.detach()).abs().max()),
                        actual_reference_R_is_returned_delta=True,
                        stop='NATIVE_EARLY_EXIT' if active['evaluations'] < 25 else 'NATIVE_25_EVALUATIONS',
                        cap_contact=abs(rho-float(hp.clamp_norm_factor)) <= 1e-6,
                        genuine_noop=active['evaluations'] == 1 and bool((delta == 0).all())))
            finally:
                handle.remove(); module.torch, module.nethook = original_torch, original_hook
            receipt.update(native_source=member(module.__file__), hparams=member(config['native_hparams']),
                           imported_native_closure=_source_closure(config['native_root']), **totals)
    finally:
        for name, parameter in adapter.model.named_parameters():
            parameter.requires_grad_(flags[name])
        require(state(adapter, history) == before and rng_equal(rng) and adapter.guard() == guard
                and adapter.hook_signature() == hooks, 'NATIVE_REFERENCE_MUTATION')
    receipt.update(seconds=time.monotonic()-start, state=before, pack=entry['pack']['identity'],
        u_hash={l: tensor_sha(v) for l,v in u.items()},
        returned_R_hash={l:tensor_sha(v) for l,v in Rref.items()},
        genuine_all_noop=all(r['genuine_noop'] for r in receipt['native_calls']),
        no_native_writer_or_history=True)
    write(out / 'native-reference.json', receipt)
    return u,receipt,Rref
