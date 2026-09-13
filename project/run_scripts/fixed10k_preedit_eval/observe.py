"""Thin observation adapter around byte-identical SH4 PRE_EDIT evaluator.

No altered arguments, batching, numerical operation, metric or model method.
Retain the exact returned prompt rows and count existing model forwards only.
"""
import argparse,json,resource,time
from pathlib import Path

def expected_calls(pair_count,microbatch=16):return (pair_count+microbatch-1)//microbatch

def run(lock_path,output):
    import torch
    from native import preedit
    from project.run_scripts.blue_alphaedit_sequential_comparison import evaluation as shared
    shared.bind_observation_only_package()
    from project.run_scripts.alphaedit_strength_neutral_barrier import evaluator as kernel
    original_pairs=kernel.evaluate_pairs;original_eval=preedit.evaluate;original_save=preedit.save
    meter={'part':0,'calls':0,'forward_calls':0,'input_tokens_including_padding':0,'nonpadding_input_tokens':0,'target_tokens':0,'raw_rows':0}
    out=Path(output)
    def pairs(model,tok,prompt_pairs,**kwargs):
        assert kwargs['microbatch_size']==16
        started=time.monotonic();counter={'forward_calls':0,'input_tokens_including_padding':0,'nonpadding_input_tokens':0}
        def before_forward(module,args,kw):
            assert module.training is False and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
            assert kw.get('use_cache') is False
            counter['forward_calls']+=1
            counter['input_tokens_including_padding']+=kw['input_ids'].numel()
            counter['nonpadding_input_tokens']+=int(kw['attention_mask'].sum().item())
        hook=model.register_forward_pre_hook(before_forward,with_kwargs=True)
        try:result=original_pairs(model,tok,prompt_pairs,**kwargs)
        finally:hook.remove()
        assert counter['forward_calls']==expected_calls(len(prompt_pairs))
        assert len(result)==len(prompt_pairs) and {p.kind for p in prompt_pairs}=={r['kind'] for r in result}
        kind=prompt_pairs[0].kind
        original_save(out/'raw-prompt-pairs'/f"part-{meter['part']:03d}-{kind}.json",result)
        meter['calls']+=1;meter['raw_rows']+=len(result);meter['target_tokens']+=sum(len(x['target_token_ids']) for x in result)
        for k,v in counter.items():meter[k]+=v
        original_save(out/'compute'/f"part-{meter['part']:03d}-{kind}.json",dict(**counter,kind=kind,raw_rows=len(result),seconds=time.monotonic()-started,extra_forward=0,backward=0,metric_change=0))
        return result
    def evaluate(*args,**kwargs):
        meter['part']+=1
        return original_eval(*args,**kwargs)
    def save(path,value):
        result=original_save(path,value)
        name=Path(path).name
        if name.startswith('part-'):
            assert meter['calls']==meter['part']*6 and meter['raw_rows']==meter['part']*2600
            assert meter['forward_calls']==meter['part']*166
            counts={k:value['metrics'][k]['denominator'] for k in ('RS','PS','NS')}
            assert counts==dict(RS=100,PS=200,NS=1000)
            rec=dict(meter,part_receipt=result,denominators=counts,evaluation_type='PRE_EDIT_W0_FULL10000_PROGRESS',
                     original_state_guard_pass=value['before_after_exact'],peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                     edit=0,compute_z=0,key=0,covariance=0,projector=0,history_append=0,backward=0,extra_forward=0)
            original_save(out/'compute'/f"part-{meter['part']:03d}-cumulative.json",rec)
            if meter['part']==1:original_save(out/'first-valid.json',dict(rec,status='INITIAL_VALID_MONITORING_PAUSE_READY'))
        if name=='runtime.json':
            original_save(out/'observer-source-runtime.json',dict(lock_sha256=preedit.file_sha(lock_path),torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),kernel_sha256=preedit.file_sha(kernel.__file__),
                shared_evaluator_sha256=preedit.file_sha(shared.__file__),native_preedit_sha256=preedit.file_sha(preedit.__file__),
                native_model_forward_policy_unchanged=True,actual_microbatch=16,host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        return result
    kernel.evaluate_pairs=pairs;preedit.evaluate=evaluate;preedit.save=save
    try:preedit.run(lock_path,output)
    finally:kernel.evaluate_pairs=original_pairs;preedit.evaluate=original_eval;preedit.save=original_save

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.lock,a.output)
