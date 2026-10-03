"""Pinned original MEMIT-H with only process-local Tensor/tuple compatibility."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import torch
from project.run_scripts.jlz_two_arm.baseline_pilot import _import_native, _bound_native, _observe_native, _source_closure
from .common import require, write

@contextmanager
def binding(adapter, bench, config, out):
    options=dict(native_root=config['native_root'],stats_root=config['stats_root'],
        configs={'MEMIT-H':config['native_hparams']},source_sha256={'memit.memit_seq_main':config['native_writer_sha']})
    # Native compute_z and repr_tools must see the SAME tokenizer as ours.
    with _import_native(options['native_root']):
        with _bound_native('MEMIT-H',options,adapter.model,bench.tokenizer,bench.contexts) as (module,hp,H,apply,receipt):
            write(out/'native-binding.json',dict(binding=receipt,actual_closure=_source_closure(options['native_root']),
                container_adapter='Tensor/tuple value-and-gradient preserving; not source rewrite',fresh_same_runtime=True))
            yield module,hp,H,apply

def batch(adapter,bench,records,bound,out,parity=False,history_before=None):
    module,hp,H,apply=bound
    requests=[dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]
    original_z=module.compute_z;checks=[];final_keys={};key_calls=[0]
    if parity:
        def aligned_z(model,tok,request,*args,**kwargs):
            normalized=copy.deepcopy(request)
            if not normalized['target_new']['str'].startswith(' '):normalized['target_new']['str']=' '+normalized['target_new']['str']
            from project.run_scripts.jlz_pilot.prompts import prepare
            pack=prepare(tok,[normalized],bench.contexts,'cpu');tokens={k:v.to(adapter.device) for k,v in pack['tokens'].items()}
            with torch.no_grad():
                nh,_=adapter.full(tokens);selected=[];targets=[]
                for row in pack['rw_rows']:
                    pos=torch.nonzero(pack['targets'][row]!=-100).flatten().to(adapter.device)
                    selected.append(nh[row,pos]);targets.append(pack['targets'][row,pos.cpu()].to(adapter.device))
                logp=adapter.head(torch.cat(selected)).log_softmax(-1)
                cursor=0;means=[]
                for target in targets:
                    lp=logp[cursor:cursor+len(target)];means.append(-lp.gather(1,target[:,None]).mean());cursor+=len(target)
                expected=float(torch.stack(means).mean())
            observed=[]
            def first_forward(m,inputs,kw,output):
                if observed:return
                require(torch.equal(kw['input_ids'],tokens['input_ids']) and torch.equal(kw['attention_mask'],tokens['attention_mask']),'NATIVE_ACTUAL_TOKEN_IDENTITY')
                values=[]
                with torch.no_grad():
                    for row in pack['rw_rows']:
                        pos=torch.nonzero(pack['targets'][row]!=-100).flatten().to(adapter.device)
                        lp=output.logits[row,pos].float().log_softmax(-1)
                        target=pack['targets'][row,pos.cpu()].to(adapter.device)
                        values.append(-lp.gather(1,target[:,None]).mean())
                observed.append(float(torch.stack(values).mean()))
            handle=model.register_forward_hook(first_forward,with_kwargs=True)
            try:result=observed_z(model,tok,normalized,*args,**kwargs)
            finally:handle.remove()
            require(len(observed)==1,'NATIVE_INITIAL_FORWARD_MISSING')
            check=dict(case_id=request['case_id'],reference_NLL=observed[0],selected_head_NLL=expected,
                error=abs(expected-observed[0]),limit=2e-5+2e-4*abs(observed[0]),input_exact=True,extra_qualification_forward=1)
            checks.append(check);write(out/('native-alignment-'+str(request['case_id'])+'.json'),check)
            require(check['error']<=check['limit'],'NATIVE_INITIAL_NLL_PARITY')
            return result
    try:
        with _observe_native(module,hp,'MEMIT-H',adapter.model) as counters:
            observed_z=module.compute_z
            observed_keys=module.compute_ks
            def capture_final_keys(*args,**kwargs):
                value=observed_keys(*args,**kwargs);index=key_calls[0];key_calls[0]+=1
                if index>=len(hp.layers):final_keys[args[4]]=value.T.detach().cpu()
                return value
            module.compute_ks=capture_final_keys
            if parity:module.compute_z=aligned_z
            apply(requests)
    finally:
        module.compute_z=original_z
        if 'counters' in locals():write(out/'native-calls.json',counters)
    require(len(counters['z_calls'])==len(records) and len(counters['solves'])==5 and len(counters['keys'])==10,'NATIVE_CALL_COVERAGE')
    require(torch.isfinite(H).all() and all(torch.isfinite(w).all() for w in adapter.weights.values()),'NATIVE_FINITE')
    require(history_before is not None and set(final_keys)==set(adapter.sites),'NATIVE_HISTORY_EVIDENCE')
    for i,l in enumerate(adapter.sites):
        k=final_keys[l];expected=history_before[l]+k@k.T
        require(torch.equal(H[i],expected),'NATIVE_HISTORY_EXACTLY_ONCE')
    return dict(candidates=None,updates=counters['target_adam_steps'],native_target_calls=len(records),
                history_appends=5,native_residual_divisor_preserved=True,checkpoint_saved=False,
                history_append_exact_checked=True,native_alignment_extra_forwards=len(checks))
