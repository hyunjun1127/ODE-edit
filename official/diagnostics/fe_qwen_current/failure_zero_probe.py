"""Single conditional causal diagnostic, NOT repaired FE or a benchmark.

CLI: python -m official.diagnostics.fe_qwen_current.failure_zero_probe BINDINGS NEW_OUTPUT
Old b255 bindings plus control_result, control_targets/control_writers (five
ordered members each), reference_case_probe. All five requests are fitted;
only their native writer K rows are zeroed before transpose/Gram/RHS/history.
No extra solve, evaluation, checkpoint, W&B or B6 write. Cold first500 then
three fixed B6 target probes. Original frozen source and checkpoint untouched.
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
from official.diagnostics.fe_qwen_current import failure_case_probe as cases
from official.diagnostics.fe_qwen_current import observer
from official.runners.server1.common import read,verify,require,member
from official.experiments.prepare import write_new
from official.diagnostics.fe_qwen_current.storage import tensor_sha

MASK={2:{21007},3:{14707,15452,6263},4:{11829}}
CASE_SHA='2ab87606b4136d3dfb9f2d55ceaa4d141c7131e32e6a69748a5c35ef0d4fa90e'


def masked_write(writer,cfg,model,tok,requests,device,H,targets,batch,*,channels=(10654,3267),lookup=None):
    """Wrap only this writer's compute_ks binding; return original K when unmasked."""
    require(sys.gettrace() is None,'TRACE_BUSY');original=writer.__globals__['compute_ks']
    expected=MASK.get(batch,set());ids=[r['case_id'] for r in requests]
    require(len(set(ids))==len(ids) and set(ids)&set().union(*MASK.values())==expected,'EXACT_BATCH_MASK_MEMBERS')
    seen=[];rows=[];captured=set();before=None
    def keys(m,t,req,c,l,contexts):
        require(sys._getframe(1).f_code is writer.__code__ and m is model and req is requests and c is cfg,'KEY_CALL_BINDING')
        require(l in cfg.llms.layers and l not in seen,'KEY_LAYER');seen.append(l)
        K=original(m,t,req,c,l,contexts)
        W=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(l)+'.weight']
        require(K.ndim==2 and K.shape==(len(req),W.shape[1]),'NATIVE_CONTEXT_AVERAGED_KEYS')
        indices=[i for i,r in enumerate(req) if r['case_id'] in expected]
        if not indices:return K
        require(lookup is not None and all(lookup(r)==0 for r in req if r['case_id'] in expected),'EXACT_NATIVE_LOOKUP_ZERO')
        value=K.clone();value[indices]=0 # small K only, never W/H clone
        return value
    def trace(frame,event,arg):
        nonlocal before
        if frame.f_code is not writer.__code__:return None
        v=frame.f_locals
        if event=='line' and all(k in v for k in ('layer','weight_name','weights','upd_matrix','targets')):
            l=int(v['layer']);name=cfg.llms.rewrite_module_tmp.format(l)+'.weight'
            if v['weight_name']==name and l not in captured and v['upd_matrix'].shape==v['weights'][name].shape:
                captured.add(l);W=v['weights'][name];D=v['upd_matrix'];K=v['layer_ks'];R=v['resid']
                row=dict(layer=l,weight_norm=prefix.norm(W),update_norm=prefix.norm(D),residual_norm=prefix.norm(R),
                    target_norm=float(targets[l].detach().float().norm()),key_norms=K.detach().norm(dim=0).cpu().tolist(),
                    residual_norms=R.detach().norm(dim=0).cpu().tolist())
                if l==5:
                    before=W[:,list(channels)].detach().clone()
                    row['columns']=[dict(channel=ch,pre_column_norm=prefix.norm(W[:,ch]),delta_column_norm=prefix.norm(D[:,ch]),
                        selected_K_channel=K[ch].detach().cpu().tolist()) for ch in channels]
                rows.append(row)
        return trace
    writer.__globals__['compute_ks']=keys;sys.settrace(trace)
    try:writer(cfg,model,tok,requests,device,H,targets)
    finally:sys.settrace(None);writer.__globals__['compute_ks']=original
    require(seen==list(cfg.llms.layers) and captured==set(cfg.llms.layers),'ALL_NATIVE_LAYERS')
    W=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(5)+'.weight'];l5=next(r for r in rows if r['layer']==5)
    for i,col in enumerate(l5['columns']):
        col.update(post_column_norm=prefix.norm(W[:,col['channel']]),
            actual_delta_column_norm=prefix.norm(W[:,col['channel']].detach().double()-before[:,i].double()))
    return dict(batch=batch,case_ids=ids,masked_case_ids=sorted(expected),layers=rows,additional_solves=0)


class ZeroObserver(prefix.Observer):
    def __init__(self,*args,control_targets,control_writers,**kw):
        super().__init__(*args,**kw);self.ct=control_targets;self.cw=control_writers;self.contexts=None;self.H=None
    def targets(self,*args,**kw):
        z=super().targets(*args,**kw)
        if self.batch<2:
            require(self.target_rows[-1]==self.ct[self.batch],'PRE_INTERVENTION_TARGET_SCALARS_MISMATCH')
        return z
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1 and batch<=5,'PREFIX_ORDER')
        lookup=lambda r:self.lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)
        r=masked_write(writer,cfg,model,tok,requests,device,H,targets,batch,lookup=lookup)
        if batch==1:
            for actual,control in zip(r['layers'],self.cw[0]['layers']):
                require(all(actual[k]==control[k] for k in ('layer','weight_norm','update_norm','residual_norm','target_norm')),'B1_NATIVE_SCALAR_MISMATCH')
            r['state_sha']=dict(weights={n:tensor_sha(p) for n,p in model.named_parameters() if n in self.reference['weights']},history=tensor_sha(H))
            # Control prefix did not serialize B1 W/H hashes; never invent them.
            r['control_B1_fullhash_available']=False
        self.writers.append(r);self.batch=batch;write_new(self.out/f'B{batch:02d}-writer.json',r)
    def snapshot(self,cfg,model,tok,H,contexts,batch,requests):
        if batch==0:return super().snapshot(cfg,model,tok,H,contexts,batch,requests)
        require(batch==self.batch,'SNAPSHOT_ORDER')
        if batch==5:
            self.contexts=contexts;self.H=H
            self.final=dict(weights={n:tensor_sha(p) for n,p in model.named_parameters() if n in self.reference['weights']},history=tensor_sha(H))
            raise prefix.PrefixComplete()


def run(bindings,output):
    b=read(bindings);v=prefix.base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    proof=read(verify(b['control_result']));require(proof['status']=='PREFIX_STATE_EXACT' and all(proof['final'][k] for k in ('weights_exact','history_exact','contexts_exact','rng_exact')),'EXACT_CONTROL_REQUIRED')
    control_binding=read(verify(proof['bindings']));require(control_binding['checkpoint']==b['checkpoint'] and control_binding['config']==b['config'],'CONTROL_IDENTITY')
    require(len(b['control_targets'])==len(b['control_writers'])==5,'FIVE_CONTROL_BATCHES')
    ct=[read(verify(m)) for m in b['control_targets']];cw=[read(verify(m)) for m in b['control_writers']]
    require(b['reference_case_probe']['sha256']==CASE_SHA and b['reference_case_probe']['bytes']==187753,'EXACT_CASE_REFERENCE')
    oldcases=read(verify(b['reference_case_probe']));require(oldcases['status']=='FIXED_CASE_PROBE_COMPLETE','CASE_REFERENCE_COMPLETE')
    require(not out.exists(),'NEW_OUTPUT_ONLY')
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    ref=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(ref['batch']==5 and ref['cursor']['edits']==500 and ref['identity']==v['pointer']['identity'] and ref['cursor']['native_config']==v['runtime']['resolved'],'OLD_B5_REFERENCE')
    cfg=OmegaConf.create(ref['cursor']['native_config']);require(cfg.num_edits==2000 and cfg.bs==100 and cfg.algs.L2==10 and cfg.llms.clamp_norm_factor==1,'ORIGINAL_SETTINGS')
    for m in c['C0']:
        verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else ''
        p=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(p.is_file() and any(p.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    data=read(verify(c['native_data']));stream=read(verify(c['stream']))
    require(len(data)==len(stream)==2000 and [r['case_id'] for r in data[:500]]==ref['cursor']['case_ids']==[r['case_id'] for r in stream[:500]],'ORDER')
    for i in range(5):require([r['case_id'] for r in ct[i]]==[r['case_id'] for r in data[i*100:(i+1)*100]]==cw[i]['case_ids'] and cw[i]['batch']==i+1,'CONTROL_MEMBERS')
    snap=read(verify(c['model_assets']))['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'BACKEND')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    w0=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(5)+'.weight'].detach().to('cpu',copy=True)
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main')
    z=importlib.import_module('algs.memit.compute_z');supplier=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,supplier):require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE')
    prefix.check_callback_binding(inspect.getsource(native));require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[] and observer._STATE is None,'FRESH_GLOBALS')
    out.mkdir(parents=True);obs=ZeroObserver(out,ref,native.find_fact_lookup_idx,tok,cfg,control_targets=ct,control_writers=cw)
    with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
        try:native.apply_memit_to_model(model,tok,copy.deepcopy(data[:500]),cfg)
        except prefix.PrefixComplete:pass
        else:raise ValueError('DEDICATED_STOP_REQUIRED')
    require(obs.final is not None and obs.batch==5,'FIVE_WRITES')
    requests=[copy.deepcopy(data[500+i]) for i in cases.INDICES];require([r['case_id'] for r in requests]==cases.CASES,'FIXED_THREE')
    for i,r in enumerate(requests):r['sample_idx']=i;r['target_new']=' '+r['target_new']
    targets,result=cases.observe(model,tok,requests,cfg,obs.contexts,supplier=supplier.current_batch_targets,
        compute_z=z.compute_z,lookup=z.find_fact_lookup_idx,TraceDict=z.nethook.TraceDict,w0=w0)
    result.update(status='FIXED_ZERO_COLUMN_COUNTERFACTUAL_COMPLETE',bindings=member(bindings),source=member(__file__),
        control_result=b['control_result'],reference_case_probe=b['reference_case_probe'],B5_state=obs.final,
        control_B1_fullhash_available=False,pre_intervention_check_scope='SCALARS_NOT_B1_FULL_STATE_PARITY',
        B1_unmasked_native_path=True,B1_fit_writer_scalars_exact=True,B2_prewrite_target_scalars_exact=True,
        masked_cases=sorted(set().union(*MASK.values())),prefix_fits=500,postprefix_fits=3,writer_batches=5,
        benchmark=False,recovery_claim=False,additional_solves=0,
        control_B5_columns=next(r for r in cw[4]['layers'] if r['layer']==5)['columns'],
        counterfactual_B5_columns=next(r for r in obs.writers[4]['layers'] if r['layer']==5)['columns'],
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,attention_backend=model.config._attn_implementation,
            gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            pad_token_id=tok.pad_token_id,padding_side=tok.padding_side),
        control_comparison=[dict(case_id=r['case_id'],L5_target=r['L5_target']['norm'],control_L5_target=o['L5_target']['norm'],
            replay_key=r['replay']['key']['norm'],control_replay_key=o['replay']['key']['norm']) for r,o in zip(result['rows'],oldcases['rows'])])
    verify(b['checkpoint']);write_new(out/'zero-probe.json',result);return member(out/'zero-probe.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args();print(run(a.bindings,a.output))
