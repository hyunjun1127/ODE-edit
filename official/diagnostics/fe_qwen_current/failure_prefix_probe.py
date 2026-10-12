"""Exact old native cold B1..B5 reproduction, never a benchmark or sweep.

CLI: python -m official.diagnostics.fe_qwen_current.failure_prefix_probe BINDINGS NEW_OUTPUT
Same b255 bindings plus prefix_fits/prefix_writers: five immutable members each.
Original cfg.num_edits=2000 is unchanged; a dedicated stop follows native B5.
Only task observer callbacks are replaced in this new process. No old paths written.
One diagnostic transposed solve with two RHS per L5 batch reconstructs selected
update columns. Five extra solves total, no change to native returned update.
"""
import argparse
import ast
import copy
import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import failure_probe as base
from official.diagnostics.fe_qwen_current import observer
from official.experiments.checkpoint import rng_snapshot
from official.experiments.prepare import write_new
from official.runners.server1.common import read,verify,require,member
from official.diagnostics.fe_qwen_current.storage import tensor_sha,state_digest


class PrefixComplete(Exception):pass


class Proxy:
    def __init__(self,base,**overrides):self.base=base;self.overrides=overrides
    def __getattr__(self,name):return self.overrides.get(name,getattr(self.base,name))


@torch.no_grad()
def dual_columns(A,K,R,U,channels,solve):
    require(A.dtype==K.dtype==R.dtype==U.dtype==torch.float64,'FP64_NATIVE_SYSTEM')
    require(A.ndim==2 and A.shape[0]==A.shape[1]==K.shape[0] and U.shape==(K.shape[0],R.shape[0]),'DUAL_SHAPE')
    E=torch.zeros((A.shape[0],2),device=A.device,dtype=A.dtype)
    require(len(channels)==2 and len(set(channels))==2,'TWO_COLUMNS')
    E[list(channels),torch.arange(2,device=A.device)]=1
    Y=solve(A.T,E);coeff=Y.T@K
    dual_error=float((A.T@Y-E).norm()/E.norm());rows=[]
    for i,ch in enumerate(channels):
        pieces=R*coeff[i].unsqueeze(0);reconstruction=pieces.sum(1);actual=U[ch]
        absolute=float((reconstruction-actual).norm());relative=absolute/max(float(actual.norm()),1e-30)
        valid=bool(torch.isfinite(reconstruction).all()) and relative<=1e-8 and dual_error<=1e-8
        den=actual.square().sum().clamp_min(1e-30)
        rows.append(dict(channel=ch,dual_relative_residual=dual_error,reconstruction_absolute_error=absolute,
            reconstruction_relative_error=relative,relative_tolerance=1e-8,
            interpretation='NUMERICALLY_VERIFIED_UPDATE_CONTRIBUTION' if valid else 'UNVERIFIED_NO_UPDATE_ATTRIBUTION',
            per_request_norm=pieces.norm(dim=0).cpu().tolist(),
            per_request_signed_projection=((actual[:,None]*pieces).sum(0)/den).cpu().tolist(),
            coefficients=coeff[i].cpu().tolist(),causal_deletion=False))
    return rows


def norm(v):
    v=v.detach();total=0.
    for start in range(0,v.shape[0],64):total+=float(v[start:start+64].double().square().sum())
    return total**.5


@torch.no_grad()
def rhs_columns(K,R,delta,channels):
    answer=[]
    for channel in channels:
        pieces=R*K[channel].unsqueeze(0)
        rhs=K[channel]@R.T;reconstructed=pieces.sum(1);den=rhs.square().sum().clamp_min(1e-30)
        error=float((reconstructed-rhs).norm()/rhs.norm().clamp_min(1e-30))
        require(error<=1e-10,'RHS_RECONSTRUCTION_FP64')
        answer.append(dict(channel=channel,delta_column_norm=norm(delta[:,channel]),
            rhs_column_norm=norm(rhs),rhs_relative_reconstruction_error=error,
            per_request_rhs_norm=pieces.norm(dim=0).cpu().tolist(),
            per_request_signed_rhs_projection=((rhs[:,None]*pieces).sum(0)/den).cpu().tolist(),
            selected_K_channel=K[channel].cpu().tolist(),
            per_request_delta_attribution='UNAVAILABLE_NATIVE_SOLVE_RETURNS_UPD_NOT_SOLVED_K',
            projection_target='RHS_NOT_WEIGHT_DELTA',additional_solves=0))
    return answer


def observe_write(writer,cfg,model,tok,requests,device,H,targets,batch,channels=(10654,3267)):
    require(sys.gettrace() is None,'EXISTING_TRACE')
    rows=[];seen=set();before_columns=None;delta_columns=None;dual=[];solve_layers=[]
    original_torch=writer.__globals__['torch'];original_solve=original_torch.linalg.solve
    def diagnostic_solve(A,B):
        frame=sys._getframe(1)
        require(frame.f_code is writer.__code__,'UNEXPECTED_SOLVE_CALLER')
        v=frame.f_locals;layer=int(v['layer'])
        require(layer in cfg.llms.layers and layer not in solve_layers,'UNEXPECTED_SOLVE_LAYER')
        solve_layers.append(layer);U=original_solve(A,B)
        if layer==5:dual.extend(dual_columns(A,v['layer_ks'],v['resid'],U,channels,original_solve))
        return U
    def trace(frame,event,arg):
        nonlocal before_columns,delta_columns
        if frame.f_code is not writer.__code__:return None
        v=frame.f_locals
        if event=='line' and all(k in v for k in ('layer','upd_matrix','weight_name','weights','targets')):
            layer=int(v['layer']);name=cfg.llms.rewrite_module_tmp.format(layer)+'.weight'
            if v['weight_name']==name and layer not in seen and v['upd_matrix'].shape==v['weights'][name].shape:
                seen.add(layer);W=v['weights'][name];D=v['upd_matrix']
                row=dict(layer=layer,weight_norm=norm(W),update_norm=norm(D),residual_norm=norm(v['targets']),
                    target_norm=float(targets[layer].detach().float().norm()))
                if layer==5:
                    K=v['layer_ks'];R=v['resid'];require(K.shape[1]==R.shape[1]==len(requests),'NATIVE_REQUEST_COLUMNS')
                    before_columns=W[:,list(channels)].detach().clone();delta_columns=D[:,list(channels)].detach().clone()
                    row.update(update_contributions=dual,columns=rhs_columns(K,R,D,channels),key_norms=K.norm(dim=0).cpu().tolist(),
                        residual_norms=R.norm(dim=0).cpu().tolist())
                    for i,col in enumerate(row['columns']):
                        w=before_columns[:,i].double();d=delta_columns[:,i].double()
                        col.update(pre_column_norm=norm(w),delta_dot_pre=float(torch.dot(w,d)),
                            delta_cosine_pre=float(torch.dot(w,d)/(w.norm()*d.norm()).clamp_min(1e-30)))
                rows.append(row)
        return trace
    writer.__globals__['torch']=Proxy(original_torch,linalg=Proxy(original_torch.linalg,solve=diagnostic_solve))
    sys.settrace(trace)
    try:writer(cfg,model,tok,requests,device,H,targets)
    finally:
        sys.settrace(None);writer.__globals__['torch']=original_torch
    require(seen==set(cfg.llms.layers),'NATIVE_LAYER_COVERAGE')
    W=dict(model.named_parameters())[cfg.llms.rewrite_module_tmp.format(5)+'.weight']
    layer5=next(r for r in rows if r['layer']==5)
    for i,col in enumerate(layer5['columns']):
        post=W[:,col['channel']].detach().double();actual=post-before_columns[:,i].double()
        col.update(post_column_norm=norm(post),actual_BF16_delta_norm=norm(actual),
            raw_delta_vs_actual_l2=norm(actual-delta_columns[:,i].double()))
    layer5['post_weight_norm']=norm(W)
    require(solve_layers==list(cfg.llms.layers) and len(dual)==2,'SOLVE_COVERAGE')
    return dict(batch=batch,layers=rows,additional_diagnostic_solves=1,diagnostic_rhs=2,no_causal_deletion_claim=True)


class Observer:
    def __init__(self,out,reference,lookup,tok,cfg):
        self.out=out;self.reference=reference;self.lookup=lookup;self.tok=tok;self.cfg=cfg
        self.batch=0;self.fits=[];self.writers=[];self.target_rows=[];self.final=None

    def snapshot(self,cfg,model,tok,H,contexts,batch,requests):
        require(batch==self.batch,'PREFIX_ORDER')
        if batch==0:
            require(contexts==self.reference['contexts'],'FRESH_CONTEXT_MISMATCH')
            require(not bool(torch.count_nonzero(H)),'COLD_HISTORY_NOT_ZERO')
            return
        if batch==5:
            weights={n:tensor_sha(p) for n,p in model.named_parameters() if n in self.reference['weights']}
            expected={n:tensor_sha(p) for n,p in self.reference['weights'].items()}
            h=tensor_sha(H);oldh=tensor_sha(self.reference['cache_c'])
            ctx=state_digest(contexts);oldctx=state_digest(self.reference['contexts'])
            rng=state_digest(rng_snapshot());oldrng=state_digest(self.reference['rng'])
            self.final=dict(weights_exact=weights==expected,actual_weight_sha=weights,reference_weight_sha=expected,
                history_exact=h==oldh,history_sha=h,reference_history_sha=oldh,
                contexts_exact=ctx==oldctx,rng_exact=rng==oldrng,rng_sha=rng,reference_rng_sha=oldrng,
                no_history_restore_or_clone=True)
            raise PrefixComplete()

    def targets(self,supplier,model,tok,requests,cfg,contexts,**kwargs):
        require(self.batch<5 and len(requests)==100,'FIRST500_ONLY')
        fit=kwargs['compute_z'];receipts=[]
        def wrapped(*args,**kw):
            require(sys.getprofile() is None,'EXISTING_PROFILE');receipt={}
            def profile(frame,event,value):
                if frame.f_code is fit.__code__ and event=='return':
                    v=frame.f_locals
                    receipt.update(init=float(v['target_init'].norm()),delta=float(v['delta'].norm()),
                        target=float(value.detach().float().norm()),nll_each=v['nll_loss_each'].detach().float().cpu().tolist(),
                        probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(),iterations=int(v['it'])+1)
            sys.setprofile(profile)
            try:value=fit(*args,**kw)
            finally:sys.setprofile(None)
            receipts.append(receipt);return value
        targets=supplier(model,tok,requests,cfg,contexts,**dict(kwargs,compute_z=wrapped))
        self.fits.append(receipts);joined=[]
        for i,r in enumerate(requests):
            lookup=self.lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)
            joined.append(dict(case_id=r['case_id'],local_index=i,bare_lookup=int(lookup),lookup0=int(lookup)==0,
                fit=receipts[i],target_norms={str(l):float(z[:,i].detach().float().norm()) for l,z in targets.items()}))
        self.target_rows.append(joined)
        write_new(self.out/f'B{self.batch+1:02d}-targets.json',joined)
        return targets

    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1 and batch<=5,'PREFIX_WRITE_ORDER')
        result=observe_write(writer,cfg,model,tok,requests,device,H,targets,batch)
        result['case_ids']=[r['case_id'] for r in requests]
        self.writers.append(result);self.batch=batch
        write_new(self.out/f'B{batch:02d}-writer.json',result)


def check_callback_binding(source):
    tree=ast.parse(source)
    apply=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_memit_to_model')
    imports=[n for n in ast.walk(apply) if isinstance(n,ast.ImportFrom) and n.module=='official.diagnostics.fe_qwen_current.observer']
    require(len(imports)==1 and {n.name for n in imports[0].names}=={'snapshot','observed_write','observed_targets'},'LOCAL_OBSERVER_IMPORT_REQUIRED')
    calls={n.func.id for n in ast.walk(apply) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    require({'snapshot','observed_write','observed_targets'}<=calls and 'batch_edit' not in calls,'OBSERVER_CALL_BINDING')
    return True


def run(bindings,output):
    b=read(bindings);v=base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    require(not out.exists(),'NEW_OUTPUT_ONLY')
    require(len(b['prefix_fits'])==len(b['prefix_writers'])==5,'FIVE_HISTORICAL_MEMBERS')
    oldfits=[read(verify(m)) for m in b['prefix_fits']];oldwriters=[read(verify(m)) for m in b['prefix_writers']]
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    reference=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(reference['batch']==5 and reference['cursor']['edits']==500 and reference['identity']==v['pointer']['identity']
        and reference['cursor']['native_config']==v['runtime']['resolved'],'B5_REFERENCE')
    cfg=OmegaConf.create(reference['cursor']['native_config'])
    require(cfg.cov_mode=='default' and cfg.num_edits==2000 and cfg.bs==100,'UNCHANGED_NATIVE_SETTINGS')
    for m in c['C0']:
        verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_ALIAS')
    # Native apply's existence check may not fall through to statistics generation.
    for layer in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else ''
        path=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{layer}{suffix}.npz'
        require(path.is_file() and any(path.resolve()==Path(m['original']['path']).resolve() for m in c['C0']),'NO_C0_REBUILD')
    data=read(verify(c['native_data']));stream=read(verify(c['stream']))
    require(len(data)==len(stream)==2000 and [r['case_id'] for r in data[:500]]==reference['cursor']['case_ids']
        ==[r['case_id'] for r in stream[:500]],'FIRST500_ORDER')
    assets=read(verify(c['model_assets']));snap=assets['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'ORIGINAL_BACKEND')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main')
    require(Path(native.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'OLD_NATIVE_IMPORT')
    check_callback_binding(inspect.getsource(native))
    require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[],'FRESH_NATIVE_GLOBALS')
    out.mkdir(parents=True);obs=Observer(out,reference,native.find_fact_lookup_idx,tok,cfg)
    with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
        require(observer._STATE is None,'OLD_OBSERVER_ALREADY_ACTIVE')
        try:native.apply_memit_to_model(model,tok,copy.deepcopy(data[:500]),cfg)
        except PrefixComplete:pass
        else:raise ValueError('DEDICATED_PREFIX_STOP_MISSING')
    require(obs.final is not None and obs.batch==5,'PREFIX_INCOMPLETE')
    comparisons=[]
    for i in range(5):
        require(oldwriters[i]['batch']==i+1 and len(oldfits[i]['rows'])==100,'HISTORICAL_PREFIX_ORDER')
        comparisons.append(dict(batch=i+1,fit_norm_differences=[r['target']-o['target_norm'] for r,o in zip(obs.fits[i],oldfits[i]['rows'])],
            writer_norm_differences=[dict(layer=r['layer'],**{k:r[k]-o[k] for k in ('weight_norm','update_norm','target_norm','residual_norm')})
                for r,o in zip(obs.writers[i]['layers'],oldwriters[i]['layers'])]))
    exact=all(obs.final[k] for k in ('weights_exact','history_exact','contexts_exact','rng_exact'))
    verify(b['checkpoint'])
    result=dict(status='PREFIX_STATE_EXACT' if exact else 'REPRODUCTION_MISMATCH_NO_HISTORICAL_ATTRIBUTION',
        final=obs.final,comparisons=comparisons,bindings=member(bindings),source=member(__file__),
        batches=5,edits=500,benchmark=False,new_checkpoint=False,additional_diagnostic_solves=5,
        runtime=dict(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,attention_backend=model.config._attn_implementation,
            pad_token_id=tok.pad_token_id,padding_side=tok.padding_side),historical_zero_causality=False)
    write_new(out/'prefix-result.json',result);return member(out/'prefix-result.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args()
    print(run(a.bindings,a.output))
