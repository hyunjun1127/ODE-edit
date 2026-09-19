"""Prospectively fixed T0, four Current and four reference documents only."""
import copy
import hashlib
import json
import time
import types
from pathlib import Path
import numpy as np
import torch
from .current import fixed_panel
from .config import NUMERIC
from project.run_scripts.single_layer_edit_preserving_correction.common import write, save_tensor, tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows
from project.run_scripts.single_layer_edit_preserving_correction.geometry import RightSpace, edit_null_space
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter


class _SavedTargetsFitter(NativeSingletonFitter):
    """Technical write parity: native body, shared entry K, precomputed T0 z."""
    def __init__(self,*args,targets,**kwargs):
        super().__init__(*args,**kwargs);self.targets=targets;self.position=0

    def _functions(self,counts,capture):
        fit,final=super()._functions(counts,capture);ns=dict(fit.__globals__)
        def z(*args,**kwargs):
            value=self.targets[:,self.position].to(next(args[0].parameters()).device);self.position+=1
            counts['compute_z']=counts.get('compute_z',0)+1
            if capture is not None:capture.setdefault('compute_z',[]).append(value.cpu().clone())
            return value
        ns['compute_z']=z
        return types.FunctionType(fit.__code__,ns,fit.__name__,fit.__defaults__,fit.__closure__),final


def hook_checks(rt,out):
    # Historical frozen 50974 retains its original source. New submissions
    # cannot silently use that checkpoint-writing route under the new default.
    if rt.lock.get('save_checkpoints') is not True or not rt.lock.get('checkpoint_exception_authority'):
        raise ValueError('LEGACY_HOOK_WEIGHT_STORAGE_REQUIRES_EXPLICIT_EXCEPTION')
    from .z_hook_parity import compare_native_z_paths
    indices=fixed_panel(rt.records[:100],lambda r:r['case_id'])
    records=[rt.records[i] for i in indices]
    write(out/'panel.json',dict(current_prefix_indices=indices,case_ids=[r['case_id'] for r in records],
        selection='SHA256(SL-MECHANISM-T0|20260919|case_id), first4; no scores',numerical=NUMERIC))
    receipt,vectors=compare_native_z_paths(rt.model,rt.tok,rt.module,rt.hp,rt.context,rt.requests(records))
    save_tensor(out/'z-comparison-tensors.pt',vectors)
    write(out/'z-comparison.json',receipt)
    # Apply actual native source solve/FP32 addition, with the 12 already
    # observed z targets. This does NOT refit the requests again.
    endpoints={};fits={}
    for name,key in [('native','native_targets'),('batch1','cache_batch1_targets'),('batched4','cache_batched_targets')]:
        rt.reset()
        fitter=_SavedTargetsFitter(rt.module,expected_source_sha256=rt.lock['editor_sha256'],
            contexts=rt.context,targets=vectors[key])
        result=fitter.fit(rt.model,rt.tok,rt.hp,rt.M,rt.P,rt.requests(records),layer=4,capture=True)
        endpoints[name]=result['weight'];fits[name]=result['receipt'];rt.guard()
        save_tensor(out/f'actual-write-{name}.pt',result)
    rt.reset();current,rows,K,meta=rt.protected_oracle(records)
    observations={name:score_rows(current,w,rows) for name,w in endpoints.items()}
    ref=observations['native'];write_comparisons={}
    for name in ('batch1','batched4'):
        candidate=observations[name]
        delta=endpoints[name].double()-endpoints['native'].double()
        nll=max(abs(candidate[k]['nll']-ref[k]['nll']) for k in ref)
        strict_equal=all(candidate[k]['strict']==ref[k]['strict'] for k in ref)
        pair=lambda x:{k for k,r in x.items() if r['kind']=='canonical' and r['branch']=='new'
            and k[:-3]+'old' in x and r['nll']<x[k[:-3]+'old']['nll']}
        pair_equal=pair(candidate)==pair(ref)
        write_comparisons[name]=dict(weight_bitwise_equal=torch.equal(endpoints[name],endpoints['native']),
            weight_max_abs=float(delta.abs().max()),weight_l2=float(delta.norm()),
            actual_current_max_NLL_abs=nll,strict_equal=strict_equal,pair_equal=pair_equal,
            pass_=nll<=NUMERIC['current_nll'] and strict_equal and pair_equal)
    summary=dict(hook=receipt,write=write_comparisons,native_solve_receipts=fits,
        new_z_fit_requests=12,additional_z_calls_for_write=0,history_appends=0,
        configured_science_batch_size=1,full16_NOT_TESTED=True,
        batch_size_choice_rule='PREDECLARED_BATCH1; actual4-tail only diagnostic, no full16 memory qualification',
        pass_=receipt['batch1_comparison']['pass_inherited_NLL_gradient_and_stop_gate'] and write_comparisons['batch1']['pass_'])
    write(out/'hook-summary.json',summary)
    rt.reset()
    if not summary['pass_']:raise RuntimeError('T0_NATIVE_Z_HOOK_PARITY_FAILED')
    return summary,records


# Full T0 uses technical_decision.run_checks from program.py after exact reuse
# of this already submitted hook-only job. It does not repeat these z fits.
