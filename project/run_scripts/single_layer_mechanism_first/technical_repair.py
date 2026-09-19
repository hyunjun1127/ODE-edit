"""Inline bounded hook repair; old valid native reference is read-only.

No new W/M/endpoint/checkpoint is persisted. The supplied four-request native
reference remains in RAM for the next T0 portion in the same process.
"""
import json
import math
from pathlib import Path
import torch
from .current import fixed_panel
from .config import NUMERIC
from .technical import _SavedTargetsFitter
from .z_hook import NATIVE_Z_SHA
from .z_hook_parity import compare_native_z_paths
from project.run_scripts.single_layer_edit_preserving_correction.common import member,sha,write,save_tensor,tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows


LENIENT_GATE_ID='USER_TRAJECTORY_GRADIENT_DIAGNOSTIC_20260920_V1'


def assess_hook(comparison,actual_write,policy):
    """Independent-trajectory drift is not a same-point derivative test.

    Keep the historical strict result. Only the explicit current user policy
    makes the relative trajectory-gradient ceiling diagnostic; actual write,
    loss, stop, finite and all subsequent method/FD criteria remain binding.
    No new, outcome-fitted numerical epsilon is introduced.
    """
    if policy.get('id')!=LENIENT_GATE_ID or not policy.get('user_quote'):
        raise ValueError('EXPLICIT_USER_GATE_POLICY_REQUIRED')
    rows=comparison['rows']
    if (len(rows)!=4 or comparison.get('finite_checked') is not True or
            any(not math.isfinite(float(r[k])) for r in rows for k in
                ('max_NLL_abs','max_total_loss_abs','max_gradient_relative','z_max_abs','z_l2','z_relative'))):
        raise ValueError('REPAIR_NONFINITE_OR_INCOMPLETE_PARITY')
    strict=comparison['pass_inherited_NLL_gradient_and_stop_gate']
    effective=all(r['loss_steps_equal'] and r['adam_steps_equal'] and
                  r['max_NLL_abs']<=NUMERIC['current_nll'] for r in rows) and actual_write['pass_']
    exceed=[r['request_index'] for r in rows if r['max_gradient_relative']>NUMERIC['gradient_relative']]
    return dict(policy_id=LENIENT_GATE_ID,pass_=bool(effective),
        historical_strict_gate_pass=bool(strict),trajectory_gradient_exceed_request_indices=exceed,
        trajectory_gradient='WARN_DIAGNOSTIC_ONLY' if exceed else 'WITHIN_ORIGINAL_CEILING',
        same_point_gradient_parity='NOT_ESTABLISHED_BY_TRAJECTORY_COMPARISON',
        effective_gate='finite AND native NLL/stop parity AND actual write NLL/strict/pair parity',
        method_acceptance_changed=False,full_T0_still_required=True)


def checked_json(item):
    p=Path(item['path'])
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('REPAIR_SOURCE_ARTIFACT_CHANGED')
    return json.loads(p.read_text())


def checked_tensor(item):
    p=Path(item['path'])
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('REPAIR_NATIVE_ARTIFACT_CHANGED')
    return torch.load(p,weights_only=True,mmap=True,map_location='cpu')


def validate_reference(rt,repair):
    if repair['kind']!='T0_NATIVE_Z_HOOK_PARITY_REPAIR':raise ValueError('REPAIR_KIND')
    failure=checked_json(repair['failure'])
    if failure['error']!="RuntimeError('T0_NATIVE_Z_HOOK_PARITY_FAILED')":raise ValueError('REPAIR_FAILURE_IDENTITY')
    original=checked_json(repair['original_lock'])
    if original['execution']['commit']!=repair['original_execution']:raise ValueError('REPAIR_ORIGINAL_SOURCE')
    fixed=('model_revision','model_config_sha256','model_weights_identity_sha256','tokenizer_identity_sha256',
           'torch','numpy','scipy','transformers','transformers_import','cold_capsule','config4','projector',
           'records_digest','sample_order','numerical','editor_sha256')
    for key in fixed:
        if original[key]!=rt.lock[key]:raise ValueError('REPAIR_COMPARABILITY:'+key)
    runtime=checked_json(repair['original_runtime'])
    if runtime['identity']!=rt.identity:raise ValueError('REPAIR_RUNTIME_W0_INPUT_BINDING')
    panel=checked_json(repair['panel'])
    indices=fixed_panel(rt.records[:100],lambda r:r['case_id']);records=[rt.records[i] for i in indices]
    if panel['case_ids']!=[r['case_id'] for r in records]:raise ValueError('REPAIR_FIXED_PANEL_CHANGED')
    vectors=checked_tensor(repair['native_trace']);native=checked_tensor(repair['native_write'])
    if native['weight'].shape!=rt.W0.shape or native['weight'].dtype!=torch.float32 or not torch.isfinite(native['weight']).all():
        raise ValueError('REPAIR_RETAINED_NATIVE_WEIGHT_INVALID')
    z=torch.stack(native['captures']['compute_z'],1)
    if not torch.equal(z,vectors['native_targets']):raise ValueError('REPAIR_TARGET_WRITE_BRIDGE')
    reuse=dict(native_targets=vectors['native_targets'],native_rows=vectors['native_rows'],
        binding=dict(native_source_sha256=NATIVE_Z_SHA,case_ids=panel['case_ids'],
            batch_input_identities=[r['input_identity'] for r in vectors['cache_batch1_receipts']],
            entry_identity_verified=True,artifacts=[repair['native_trace'],repair['native_write'],repair['original_runtime']]))
    return native,records,reuse


def run_hook_repair(rt,out):
    if rt.lock.get('save_checkpoints') is not False:raise ValueError('REPAIR_NO_CHECKPOINT_REQUIRED')
    out=Path(out)
    native,records,reuse=validate_reference(rt,rt.lock['hook_repair'])
    write(out/'reuse-binding.json',dict(native_source=rt.lock['hook_repair']['original_execution'],
        binding=reuse['binding'],new_unhooked_reference_fits=0,checkpoint_saved=False))
    receipt,vectors=compare_native_z_paths(rt.model,rt.tok,rt.module,rt.hp,rt.context,
                                          rt.requests(records),native_reuse=reuse)
    write(out/'z-comparison.json',receipt)
    # Small z vectors/gradients are diagnostic evidence, not model-weight state.
    save_tensor(out/'z-comparison-tensors.pt',vectors)
    endpoints={'native':native['weight']};fits={}
    for name,key in [('batch1','cache_batch1_targets'),('batched4','cache_batched_targets')]:
        rt.reset()
        fitter=_SavedTargetsFitter(rt.module,expected_source_sha256=rt.lock['editor_sha256'],
            contexts=rt.context,targets=vectors[key])
        result=fitter.fit(rt.model,rt.tok,rt.hp,rt.M,rt.P,rt.requests(records),layer=4,capture=True)
        endpoints[name]=result['weight'];fits[name]=result['receipt'];rt.guard()
    rt.reset();oracle,rows,K,meta=rt.protected_oracle(records)
    try:
        if any(w.dtype!=torch.float32 or not torch.isfinite(w).all() for w in endpoints.values()):
            raise FloatingPointError('REPAIR_ACTUAL_WRITE_NONFINITE_OR_DTYPE')
        observations={name:score_rows(oracle,w,rows) for name,w in endpoints.items()}
        if any(not math.isfinite(float(row['nll'])) for panel in observations.values() for row in panel.values()):
            raise FloatingPointError('REPAIR_ACTUAL_WRITE_NONFINITE_NLL')
        ref=observations['native'];comparison={}
        for name in ('batch1','batched4'):
            candidate=observations[name];delta=endpoints[name].double()-endpoints['native'].double()
            nll=max(abs(candidate[k]['nll']-ref[k]['nll']) for k in ref)
            strict_equal=all(candidate[k]['strict']==ref[k]['strict'] for k in ref)
            pair=lambda x:{k for k,r in x.items() if r['kind']=='canonical' and r['branch']=='new'
                and k[:-3]+'old' in x and r['nll']<x[k[:-3]+'old']['nll']}
            pair_equal=pair(candidate)==pair(ref)
            comparison[name]=dict(weight_sha256=tensor_sha(endpoints[name]),weight_bitwise_equal=torch.equal(endpoints[name],endpoints['native']),
                weight_max_abs=float(delta.abs().max()),weight_l2=float(delta.norm()),
                actual_current_max_NLL_abs=nll,strict_equal=strict_equal,pair_equal=pair_equal,
                pass_=nll<=NUMERIC['current_nll'] and strict_equal and pair_equal)
        assessment=assess_hook(receipt['batch1_comparison'],comparison['batch1'],rt.lock['hook_gate_policy'])
        summary=dict(hook=receipt,write=comparison,native_solve_receipts=fits,
            old_native_reference_fit_reused=True,new_unhooked_native_requests=0,new_z_fit_requests=8,
            additional_z_calls_for_write=0,history_appends=0,configured_science_batch_size=1,
            full16_NOT_TESTED=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
            gate_assessment=assessment,pass_=assessment['pass_'])
        write(out/'hook-summary.json',summary)
        if not summary['pass_']:raise RuntimeError('T0_NATIVE_Z_HOOK_PARITY_FAILED')
        return native,records
    finally:
        rt.reset();rt.oracles.remove(oracle)
