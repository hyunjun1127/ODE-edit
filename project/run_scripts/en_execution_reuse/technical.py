"""Bounded independent physical checks on this B1; no extra T campaign."""
import time
import torch
from .preparation import create_json
from project.run_scripts.single_layer_edit_preserving_correction.common import save_tensor, tensor_sha, digest
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows


def current_physical(rt,current,weight,rows):
    chosen=[r for r in rows if r['cache']<4]
    class Physical:
        def logits_at(self,index,w,positions):
            return current.logits_at(index,w,positions,route='physical')
    cached=score_rows(current,weight,chosen)
    physical=score_rows(Physical(),weight,chosen);rt.sync_oracles()
    maximum=max(abs(cached[k]['nll']-physical[k]['nll']) for k in cached)
    strict_equal=all(cached[k]['strict']==physical[k]['strict'] for k in cached)
    probes=[]
    for i in range(4):
        key=current.key_stationarity(i,weight);rt.sync_oracles()
        logits=current.compare_logits(i,weight,weight,left_route='physical',right_route='cached');rt.sync_oracles()
        probes.append(dict(index=i,key=key,logits=logits))
    value=dict(scope='FIRST4_CURRENT_INPUTS_NOT_WHOLE_PROTECTED_SET',weight_sha256=tensor_sha(weight),
        cached=cached,physical=physical,max_NLL_difference=maximum,strict_equal=strict_equal,
        key_stationarity=all(p['key']['byte_equal'] for p in probes),
        cached_physical_logit_noop=all(p['logits']['max_abs']<=1e-4 for p in probes),probes=probes)
    value['pass']=bool(value['key_stationarity'] and value['cached_physical_logit_noop'] and maximum<=1e-5 and strict_equal)
    return value


def check(rt, reference, current, WN, rows, root):
    start=time.monotonic();rng=capture_rng();entry=tensor_sha(rt.W.detach())
    result=dict(scope='SAME_B1_BOUNDED_PHYSICAL_CHECKS',reference_indices=[0,1],current_indices=list(range(4)),
        full_bank_physical_validation=False,extra_native_fit_calls=0,extra_scientific_batches=0,
        FD_campaign='NOT_REQUESTED_BY_EXECUTION_DEDUP_DESIGN',Past='B1_EMPTY_ACTUAL; NONEMPTY_CPU_FIXTURES_ONLY')
    try:
        comparison=reference.check_documents(WN,[0,1],gradient=True)
        rt.sync_oracles()
        for route in ('cached','physical'):
            comparison[route]['gradient']=save_tensor(root/f'{route}-bounded-gradient.pt',
                dict(gradient=comparison[route]['gradient']))
        comparison['pass']=(comparison['loss_absolute_difference']<=1e-6 and
                            comparison['gradient_relative_l2']<=1e-4)
        create_json(root/'reference-physical-AD.json',comparison)
        current_result=current_physical(rt,current,WN,rows)
        result.update(reference_AD=comparison,current=current_result,pass_=comparison['pass'] and current_result['pass'])
        rt.guard()
        if tensor_sha(rt.W.detach())!=entry:raise ValueError('PHYSICAL_CHECK_ENTRY_RESTORE')
        if digest(capture_rng())!=digest(rng):raise ValueError('PHYSICAL_CHECK_RNG_MUTATION')
        result.update(entry_weight_exact_restore=True,RNG_exact=True,seconds=time.monotonic()-start)
        create_json(root/'checks.json',result)
        if not result['pass_']:raise ValueError('BOUNDED_ACTUAL_PHYSICAL_PARITY_FAILED')
        return result
    finally:
        restore_rng(rng)
