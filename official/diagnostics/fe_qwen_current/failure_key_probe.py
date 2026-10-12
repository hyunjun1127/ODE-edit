"""Passive original B1..B3 key/residual consistency probe, no repaired writes.

CLI: python -m official.diagnostics.fe_qwen_current.failure_key_probe BINDINGS NEW_OUTPUT
Old b255 bindings + control_result (exact B5 proof), control_targets and
control_writers (B01..03 immutable members). No extra model forward/solve,
evaluation, checkpoint or W&B. cfg.num_edits stays2000; dedicated stop at B3.
"""
import argparse
import copy
import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import failure_prefix_probe as prefix
from official.diagnostics.fe_qwen_current import failure_case_probe as case
from official.diagnostics.fe_qwen_current import observer
from official.runners.server1.common import read,verify,require,member
from official.experiments.prepare import write_new


def capture_keys(function,model,tok,requests,cfg,layer,contexts,lookup):
    """Observe existing extracted key rows, before native group means; return K unchanged."""
    require(not cfg.negetive_prompt_test and [len(g) for g in contexts]==[1,5] and contexts[0]==['{}'],'NATIVE_SIX_CONTEXTS')
    name='get_module_input_output_at_words';original=function.__globals__[name];saved={}
    expected=[ctx.format(r['prompt']) for r in requests for group in contexts for ctx in group]
    words=[r['subject'] for r in requests for group in contexts for _ in group]
    def extract(m,t,l,**kw):
        require(sys._getframe(1).f_code is function.__code__ and not saved,'EXACT_SINGLE_KEY_EXTRACTION')
        require(m is model and t is tok and l==layer and kw['context_templates']==expected and kw['words']==words
            and kw['module_template']==cfg.llms.rewrite_module_tmp and kw['fact_token_strategy']==cfg.llms.fact_token,'NATIVE_KEY_ROW_ORDER')
        value=original(m,t,l,**kw)
        raw=value[0];require(raw.ndim==2 and raw.shape[0]==6*len(requests),'RAW_KEY_SHAPE')
        # Detached selected representation rows only, NOT full forward activations.
        saved['raw']=raw.detach()
        return value
    with patch.dict(function.__globals__,{name:extract}):returned=function(model,tok,requests,cfg,layer,contexts)
    raw=saved['raw'];require(returned.shape==(len(requests),raw.shape[1]),'RETURNED_KEY_SHAPE')
    # Reproduce exact author's reduction order in original dtype, never change returned keys.
    rebuilt=torch.stack([torch.stack([raw[i:i+1].mean(0),raw[i+1:i+6].mean(0)],0).mean(0) for i in range(0,len(raw),6)])
    require(torch.equal(returned,rebuilt),'NATIVE_KBAR_BITWISE_AGGREGATION')
    bare=raw[::6].clone();positions=[int(lookup(prompt,word)) for prompt,word in zip(expected,words)]
    return returned,dict(bare=bare,positions=[positions[i:i+6] for i in range(0,len(positions),6)],
        native_dtype=str(raw.dtype),aggregation_exact=True)


@torch.no_grad()
def vector_metrics(x,y):
    x=x.double();y=y.double();nx=x.norm();ny=y.norm();err=(x-y).norm()
    return dict(norm=float(nx),reference_norm=float(ny),cosine=float(torch.dot(x,y)/(nx*ny).clamp_min(1e-30)),
        error_norm=float(err),relative_error=float(err/ny.clamp_min(1e-30)))


@torch.no_grad()
def transfer_rows(D,Kbar,Kbare,R,requests,positions,channels):
    require(D.ndim==2 and D.shape[1]==Kbar.shape[0]==Kbare.shape[1]
        and D.shape[0]==R.shape[0] and Kbar.shape[1]==len(requests)==Kbare.shape[0]==R.shape[1],'WEIGHT_ORIENTATION')
    rows=[]
    # Four-request chunks bound temporary predictions; no D/H clone, inverse or solve.
    for start in range(0,len(requests),4):
        stop=min(start+4,len(requests));kb=Kbare[start:stop].double().T;ka=Kbar[:,start:stop]
        pa=D@ka;pb=D@kb
        for j in range(stop-start):
            i=start+j
            rows.append(dict(case_id=requests[i]['case_id'],local_index=i,lookup_positions=positions[i],
                bare_vs_bar=vector_metrics(kb[:,j],ka[:,j]),
                selected_channels={str(ch):dict(bare=float(kb[ch,j]),bar=float(ka[ch,j])) for ch in channels},
                D_Kbar_vs_R=vector_metrics(pa[:,j],R[:,i]),D_Kbare_vs_R=vector_metrics(pb[:,j],R[:,i]),
                prediction_kind='RAW_FP64_LINEAR_NOT_BF16_POST_FORWARD'))
    return rows


def passive_write(writer,cfg,model,tok,requests,device,H,targets,batch,lookup,channels=(10654,3267)):
    require(sys.gettrace() is None,'TRACE_BUSY');original=writer.__globals__['compute_ks'];captures={};rows=[];seen=set()
    def keys(m,t,req,c,l,contexts):
        require(sys._getframe(1).f_code is writer.__code__ and m is model and req is requests and c is cfg,'NATIVE_WRITER_KEY_CALL')
        require(l in cfg.llms.layers and not captures,'SINGLE_LAYER_CAPTURE')
        K,record=capture_keys(original,m,t,req,c,l,contexts,lookup);captures[l]=record;return K
    def trace(frame,event,arg):
        if frame.f_code is not writer.__code__:return None
        v=frame.f_locals
        if event=='line' and all(k in v for k in ('layer','weight_name','weights','upd_matrix','resid')):
            l=int(v['layer']);name=cfg.llms.rewrite_module_tmp.format(l)+'.weight'
            if v['weight_name']==name and l not in seen and v['upd_matrix'].shape==v['weights'][name].shape:
                require(l in captures,'EXISTING_KEY_CAPTURE');seen.add(l);r=captures.pop(l)
                rows.append(dict(layer=l,weight_norm=prefix.norm(v['weights'][name]),update_norm=prefix.norm(v['upd_matrix']),
                    residual_norm=prefix.norm(v['resid']),target_norm=float(targets[l].detach().float().norm()),
                    aggregation_exact=r['aggregation_exact'],native_key_dtype=r['native_dtype'],
                    requests=transfer_rows(v['upd_matrix'],v['layer_ks'],r['bare'],v['resid'],requests,r['positions'],channels)))
        return trace
    with patch.dict(writer.__globals__,compute_ks=keys):
        sys.settrace(trace)
        try:writer(cfg,model,tok,requests,device,H,targets)
        finally:sys.settrace(None)
    require(seen==set(cfg.llms.layers) and not captures,'ALL_LAYERS_CAPTURED')
    return dict(batch=batch,case_ids=[r['case_id'] for r in requests],layers=rows,extra_forward=0,extra_solve=0,actual_keys_unchanged=True)


def scalar_check(actual,control):
    # Same chunked FP64 observer reductions permit ONLY roundoff, not metric drift.
    error=abs(actual-control);bound=1e-12+1e-12*abs(control)
    require(error<=bound,'NATIVE_SCALAR_MISMATCH')
    return dict(actual=actual,control=control,absolute_error=error,bound=bound,exact=actual==control)


class KeyObserver(prefix.Observer):
    def __init__(self,*args,ct,cw,**kw):super().__init__(*args,**kw);self.ct=ct;self.cw=cw;self.checks=[]
    def targets(self,*args,**kw):
        require(self.batch<3,'THREE_BATCHES_ONLY');z=super().targets(*args,**kw)
        current=self.target_rows[-1];old=self.ct[self.batch];require(len(current)==len(old)==100,'FIT100')
        checks=[]
        for r,o in zip(current,old):
            require((r['case_id'],r['bare_lookup'])==(o['case_id'],o['bare_lookup']),'FIT_ORDER_LOOKUP')
            checks.append(dict(case_id=r['case_id'],fit={k:case.norm_check(r['fit'][k],o['fit'][k]) for k in ('init','delta','target')},
                targets={l:case.norm_check(r['target_norms'][l],o['target_norms'][l]) for l in r['target_norms']}))
        require(all(c['within_reduction_roundoff'] for r in checks for group in ('fit','targets') for c in r[group].values()),'FIT_REDUCTION_PARITY')
        self.checks.append(dict(batch=self.batch+1,fit=checks));return z
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1 and batch<=3,'WRITE_ORDER')
        lookup=lambda p,s:self.lookup(p,s,tok,cfg.llms.fact_token,verbose=False)
        result=passive_write(writer,cfg,model,tok,requests,device,H,targets,batch,lookup)
        old=self.cw[batch-1];require(old['case_ids']==result['case_ids'] and len(old['layers'])==len(result['layers']),'CONTROL_WRITER')
        checks=[]
        for r,o in zip(result['layers'],old['layers']):
            require(r['layer']==o['layer'],'LAYER_ORDER');checks.append({k:scalar_check(r[k],o[k]) for k in ('weight_norm','update_norm','target_norm','residual_norm')})
        self.checks[-1]['writer']=checks;self.writers.append(result);self.batch=batch;write_new(self.out/f'B{batch:02d}-keys.json',result)
    def snapshot(self,cfg,model,tok,H,contexts,batch,requests):
        if batch==0:return super().snapshot(cfg,model,tok,H,contexts,batch,requests)
        require(batch==self.batch,'SNAPSHOT_ORDER')
        if batch==3:self.final=True;raise prefix.PrefixComplete()


def run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    proof=read(verify(b['control_result']));require(proof['status']=='PREFIX_STATE_EXACT' and all(proof['final'][k] for k in ('weights_exact','history_exact','contexts_exact','rng_exact')),'EXACT_B5_CONTROL')
    binding=read(verify(proof['bindings']));require(binding['checkpoint']==b['checkpoint'] and binding['config']==b['config'],'CONTROL_IDENTITY')
    require(len(b['control_targets'])==len(b['control_writers'])==3,'THREE_CONTROL_BATCHES')
    ct=[read(verify(m)) for m in b['control_targets']];cw=[read(verify(m)) for m in b['control_writers']]
    require(not out.exists(),'NEW_OUTPUT_ONLY')
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    ref=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(ref['batch']==5 and ref['cursor']['edits']==500 and ref['identity']==v['pointer']['identity'] and ref['cursor']['native_config']==v['runtime']['resolved'],'B5_REFERENCE_ONLY')
    cfg=OmegaConf.create(ref['cursor']['native_config']);require(cfg.num_edits==2000 and cfg.bs==100 and cfg.algs.L2==10 and cfg.llms.clamp_norm_factor==1,'UNCHANGED_CONFIG')
    for m in c['C0']:
        verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else ''
        p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    data=read(verify(c['native_data']));stream=read(verify(c['stream']))
    require(len(data)==len(stream)==2000 and [r['case_id'] for r in data[:500]]==ref['cursor']['case_ids']==[r['case_id'] for r in stream[:500]],'ORIGINAL_ORDER')
    for i in range(3):require([r['case_id'] for r in ct[i]]==[r['case_id'] for r in data[i*100:(i+1)*100]]==cw[i]['case_ids'] and cw[i]['batch']==i+1,'CONTROL_MEMBERS')
    snap=read(verify(c['model_assets']))['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'ORIGINAL_BACKEND')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main')
    require(Path(native.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE_IMPORT')
    prefix.check_callback_binding(inspect.getsource(native));require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[] and observer._STATE is None,'FRESH_GLOBALS')
    out.mkdir(parents=True);obs=KeyObserver(out,ref,native.find_fact_lookup_idx,tok,cfg,ct=ct,cw=cw)
    with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
        try:native.apply_memit_to_model(model,tok,copy.deepcopy(data[:300]),cfg)
        except prefix.PrefixComplete:pass
        else:raise ValueError('DEDICATED_STOP_REQUIRED')
    require(obs.final is True and obs.batch==3,'THREE_WRITES');verify(b['checkpoint'])
    result=dict(status='PASSIVE_KEY_PROBE_COMPLETE',bindings=member(bindings),source=member(__file__),checks=obs.checks,
        batches=3,fits=300,new_forward=0,new_solve=0,evaluation=0,checkpoint_save=0,weights_history_restored=False,
        control_match='SAVED_SCALARS_REDUCTION_TOLERANCE_ONLY_NOT_B3_FULL_STATE',representation_intervention=False,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,backend=model.config._attn_implementation,
            gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,pad=tok.pad_token_id,padding_side=tok.padding_side))
    write_new(out/'key-probe.json',result);return member(out/'key-probe.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args();print(run(a.bindings,a.output))
