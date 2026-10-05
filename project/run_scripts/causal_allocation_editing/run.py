"""Single frozen actual-loss chain; resource-pending DAG needs no callbacks."""
import argparse
import gc
import json
import os
import random
import resource
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal,rng_restore
from .entry import prepare_entry as capture_entry
from project.run_scripts.jlz_realized_subject.geometry import prior,mean_keys
from project.run_scripts.jlz_realization.profile import move
from project.run_scripts.jlz_realization.observe import observe,active_flags
from . import *
from .engine import Adapter,CausalObjective,native_terms
from .solver import solve
from .calibration import calibrate,PriceLock
from .native_reference import reference_fit
from .profile import execution, check_horizon

SOURCE_ENV='CAUSAL_ALLOCATION_EDITING_SOURCE_COMMIT'

class Events:
    def __init__(self,path,batch,task=TASK): self.path,self.batch,self.task=path,batch,task
    def emit(self,event,payload,**kw):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        row=dict(experiment=self.task,batch=self.batch,event=event,payload=payload,**kw)
        with self.path.open('a') as f:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    def __call__(self,event,payload=None,**kw):
        if isinstance(event,dict):self.emit('solver',event,**kw)
        else:self.emit(event,payload,**kw)

class BatchTransaction(Transaction):
    """Snapshots RAM state only; additionally restore context and cursor/cache."""
    def __init__(self,a,H,bench,cursor):
        super().__init__(a,H);self.bench,self.cursor=bench,cursor
    def __enter__(self):
        super().__enter__();self.context=json.loads(json.dumps(self.bench.contexts))
        self.cursor_before=list(self.cursor);self.hooks=self.a.hook_signature()
        self.cache={k:getattr(self.a,k) for k in ('last_virtual','capture_virtual') if hasattr(self.a,k)}
        return self
    def finish(self):
        require(self.bench.contexts==self.context and self.a.hook_signature()==self.hooks,'TRANSACTION_CONTEXT_HOOKS')
        super().finish()
    def __exit__(self,*args):
        super().__exit__(*args)
        if not self.done:
            self.bench.contexts=self.context;self.cursor[:]=self.cursor_before
            for k,v in self.cache.items():setattr(self.a,k,v)
        require(self.bench.contexts==self.context and self.a.hook_signature()==self.hooks,'TRANSACTION_RESTORE_CONTEXT_HOOKS')

def rng_identity():
    r=rng_snapshot()
    return digest([repr(r[0]),r[1][0],r[1][1].tolist(),repr(r[1][2:]),tensor_sha(r[2]),[tensor_sha(x) for x in r[3]]])

def locked(attempt):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    profile=check_horizon(c)
    require(c['instruction_id']==lock['instruction_id']==profile['nonce'] and c['task_id']==profile['task'],'AUTHORITY')
    require(os.environ.get(SOURCE_ENV)==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_SOURCE_CONFIG')
    for key in ('source_members','runtime_sources','native_reference','dependency_sources','launchers'):
        for row in lock.get(key,[]):verify(row)
    for key in ('archive','native_hparams'):
        if key in lock:verify(lock[key])
    for row in c['authority_members']+[c['native_input_alignment'],c['native_full_input_binding'],c['observer_identity'],c['cpu_preflight']]:verify(row)
    for row in c['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    s=c['settings']
    require(s['main_arms']==1 and s['shared_sum_cap'] is None and s['per_layer_cap']==.75
            and (s['logical_candidate_cap'],s['accepted_update_cap'],s['full_gradient_cap'])==(50,24,25)
            and not s['save_checkpoints'],'FIXED_METHOD_BUDGET')
    return c,lock

def setup(c,out):
    require(shutil.disk_usage(out).free>=c['resources']['startup_free_bytes_min'],'RESOURCE_BLOCKED_STORAGE')
    require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
    torch.set_num_threads(c['resources'].get('cpu',8));random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    a=Adapter(model,c['profile']);bench=CounterFactAdapter(tok,json.loads(Path(c['contexts']).read_text()))
    H={l:torch.zeros(dim[1],dim[1],dtype=torch.float32) for l,dim in a.dims.items()}
    records=load_prefix(Path(c['stream']).parent,execution(c)['requests'])
    require(digest([r['case_id'] for r in records])==c['ordered_ids_sha256'],'ORDERED_FIRST2000')
    require(state(a,H)==c['cold_W0_H0'],'ACTUAL_COLD_W0_H0')
    write(out/'runtime.json',dict(experiment=c['task_id'],source=os.environ[SOURCE_ENV],config=digest(c),
        job=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),cold_W0_H0=state(a,H),
        torch=torch.__version__,transformers=transformers.__version__,FP32=True,geometry_FP64=True,
        eager=True,TF32=False,autocast=False,checkpoint_saved=False))
    return a,bench,records,H

def entry_for(a,bench,pack,H,c):
    # A complete owner graph exactly preserves native compute_z's input shape.
    entry=capture_entry(a,bench,pack,H,c['stats'],requests_per_group=1)
    require(all(len({r['request'] for r in g['rows']})==1 for g in entry['groups']),'ORIGINAL_OWNER_GRAPH_ONLY')
    entry['factors']={};entry['geometry']={};entry['first_geometry']={}
    entry['history_entry']={l:H[l] for l in a.sites}
    for l in a.sites:
        entry['factors'][l],entry['geometry'][l]=prior(c['stats'][str(l)],H[l],a.device,c['profile']['lambda_C'])
    return entry

def observer(a,bench,seen,selected,H,name,out,c,identities,current):
    before=state(a,H);rng=rng_snapshot()
    result=observe(a,bench,seen,selected,H,name,out,c['settings']['observer_microbatch'],current)
    rows=rows_from(out,before)
    require(validate_rows(rows,identities,[r['case_id'] for r in selected],name)==result['summary'],'OBSERVER_REDUCTION')
    flags=active_flags(seen)
    require(all(r['active_at_endpoint']==flags[r['case_id']] for r in rows),'OBSERVED_SEEN_PREFIX_ONLY')
    require(state(a,H)==before and rng_equal(rng),'OBSERVER_NONMUTATION')
    return result

def subset_w0(out,current,seen,folder,cold,identities):
    ids=[r['case_id'] for r in current];flags=active_flags(seen)
    rows=[dict(r,endpoint='B1_PRE',active_at_endpoint=flags[r['case_id']]) for r in rows_from(out/'W0',cold)
          if r['case_id'] in set(ids)]
    summary=validate_rows(rows,identities,ids,'B1_PRE')
    write(folder/'chunk-0000.json',dict(state=cold,rows=rows,optimizer_feedback=False))
    result=dict(endpoint='B1_PRE',state=cold,requests=len(current),summary=summary,current=summary,
        seconds=0,new_forwards=0,no_mutation=True,optimizer_feedback=False,reused_from=str(out/'W0'))
    write(folder/'summary.json',result);return result

def w0(a,bench,records,H,c,out,identities):
    reuse=c.get('W0_reuse',{});cold=state(a,H)
    if reuse.get('status')!='NOT_AVAILABLE' and reuse.get('path'):
        for r in reuse['chunks']+[reuse['summary'],reuse['runtime'],reuse['prior_config']]:verify(r)
        require(reuse['state']==cold,'W0_EXACT_COLD_REUSE')
        rows=rows_from(reuse['path'],cold);summary=validate_rows(rows,identities,[r['case_id'] for r in records],'W0')
        for index,start in enumerate(range(0,len(rows),1000)):
            write(out/'W0'/f'chunk-{index:04d}.json',dict(state=cold,rows=rows[start:start+1000],optimizer_feedback=False))
        write(out/'W0/summary.json',dict(endpoint='W0',state=cold,requests=len(records),summary=summary,current=summary,
            new_forwards=0,seconds=0,no_mutation=True,optimizer_feedback=False,reused_from=reuse['path']))
    else:observer(a,bench,records,records,H,'W0',out/'W0',c,identities,[r['case_id'] for r in records])

@torch.no_grad()
def commit_payload(a,entry,H,payload,out):
    """Exact accepted copy, one final actual native capture, one CPUFP32 Gram."""
    require(set(payload['weights'])==set(a.weights),'WHOLE_PAYLOAD_COMMIT')
    for l,w in a.weights.items():
        value=payload['weights'][l]
        require(value.shape==w.shape and value.dtype==w.dtype and bool(torch.isfinite(value).all()),'ACCEPTED_WEIGHT_SCHEMA')
        w.copy_(value);require(torch.equal(w.detach().cpu(),value.cpu()),'ACCEPTED_PAYLOAD_EXACT_COPY')
    keys={l:[] for l in a.sites};rows=[];total_task=0.;t0=time.monotonic()
    for group in entry['groups']:
        found={};handles=[]
        for l in a.sites:
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:found.update({l:args[0]})))
        try:
            nh,fh=a.full(move(group['tokens'],a.device))
            terms,_=native_terms(a,entry,group,nh,fh);total_task+=float(terms)
            for index,row in enumerate(group['rows']):
                if row['kind']=='rewrite':
                    rows.append(row)
                    for l in a.sites:keys[l].append(found[l][index,row['lookup']].detach().cpu().clone())
        finally:
            for handle in handles:handle.remove()
    history={};key_checks={}
    for l in a.sites:
        K=mean_keys(torch.stack(keys[l]).T,rows,entry['pack']).cpu().float().contiguous()
        ref=payload['K'][l].cpu().float()
        limit=2e-5+2e-4*ref.abs();error=(K-ref).abs()
        require(bool((error<=limit).all()),'ACCEPTED_FINAL_NATIVE_KEY_PARITY')
        before=tensor_sha(H[l]);H[l].add_(K@K.T)
        require(H[l].dtype==torch.float32 and H[l].device.type=='cpu' and bool(torch.isfinite(H[l]).all()),'HISTORY_NATIVE_FP32_ONCE')
        history[str(l)]=dict(before=before,after=tensor_sha(H[l]),key=tensor_sha(K),appends=1,
            rows='native_rewrite_only_nestedmean',requests=entry['pack']['n_requests'])
        key_checks[str(l)]=dict(maxabs=float(error.max()),elementwise_limit_min=float(limit.min()),
                              component_failures=int((error>limit).sum()))
    loss_error=abs(total_task-payload['task_sum']);loss_limit=2e-5+2e-4*abs(payload['task_sum'])
    require(loss_error<=loss_limit,'ACCEPTED_COMMITTED_NATIVE_TASK_PARITY')
    receipt=dict(candidate=payload['candidate'],accepted_weight_copy_exact=True,
        weight_hashes={l:tensor_sha(w) for l,w in a.weights.items()},history_appends=len(H),history=history,
        native_loss=total_task,accepted_native_loss=payload['task_sum'],loss_error=loss_error,loss_limit=loss_limit,
        final_key_checks=key_checks,seconds=time.monotonic()-t0,checkpoint_saved=False)
    write(out/'commit-payload.json',receipt);return receipt

def qualification(a,bench,records,H,c,lock,attempt):
    from .qualification import qualify
    before=state(a,H);rng=rng_snapshot();guard=a.guard();hooks=a.hook_signature()
    pack=bench.prepare(records[:c['qualification']['native_requests']]);entry=entry_for(a,bench,pack,H,c)
    result=qualify(a,entry,attempt/'qualification')
    require(state(a,H)==before and rng_equal(rng) and a.guard()==guard and a.hook_signature()==hooks,'QUALIFICATION_MUTATION')
    del entry;gc.collect();torch.cuda.empty_cache()
    write(attempt/'qualification/READY.json',dict(status='TECHNICAL_READY',experiment=c['task_id'],
        source=lock['source_commit'],config_sha256=lock['config_sha256'],cold_W0_H0=before,
        result=result,extra_fits=0,updates=0,no_state_seed_main=True))

def drive(a,bench,records,H,c,out,lock,attempt):
    profile=check_horizon(c)
    identities=json.loads(verify(c['observer_identity']).read_text())['rows'];cursor=[];commits=[]
    previous=state(a,H);previous_rng=rng_identity();price=None;price_hash=None
    w0(a,bench,records,H,c,out,identities)
    for number,current,seen in batches(records):
        require(number<=profile['batches'],'NO_NEXT_BATCH');folder=out/f'batch-{number:02d}';folder.mkdir(exist_ok=False)
        require(state(a,H)==previous and rng_identity()==previous_rng,'OWN_W_H_RNG_NEXT_ENTRY_JOIN')
        pack=bench.prepare(current);expected=c['packs'][number-1]
        require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],'SEALED_NATIVE_PACK')
        write(folder/'entry.json',dict(experiment=c['task_id'],source=lock['source_commit'],config=digest(c),batch=number,
            ids=pack['record_ids'],native_pack=pack['identity'],state=previous,RNG=previous_rng,
            context=digest(bench.contexts),ledger=digest(cursor),seen_ids=[r['case_id'] for r in seen]))
        if number==2:write(out/'initial.json',dict(status='MAIN_B1_COMMIT_B2_OWN_ENTRY',
            B1_commit=member(out/'batch-01/commit.json'),next_entry=previous,source=lock['source_commit'],
            config=digest(c),observer_restored=True,own_W_H_RNG_join=True))
        tx=BatchTransaction(a,H,bench,cursor);t0=time.monotonic()
        try:
            with tx:
                pre=subset_w0(out,current,seen,folder/'pre',previous,identities) if number==1 else observer(
                    a,bench,seen,current,H,f'B{number}_PRE',folder/'pre',c,identities,pack['record_ids'])
                entry=entry_for(a,bench,pack,H,c);events=Events(folder/'solver-events.jsonl',number,c['task_id'])
                ready=json.loads((attempt/'qualification/READY.json').read_text())
                engine=CausalObjective(a,entry,events,route=ready['result']['selected_gradient_route']);zero=engine.zeros()
                write(folder/'entry-capture.json',dict(pack=pack['identity'],teacher_hash=entry['teacher_hash'],
                    anchors={l:tensor_sha(v) for l,v in entry['anchors'].items()},H_entry=previous['H'],
                    fresh_capture=True,factor_lifetime='this own batch only',seconds=entry['seconds']))
                if price is None:
                    uref,reference,Rref=reference_fit(a,bench,current,entry,H,c,folder/'calibration')
                    channels=engine.evaluate(uref,gradient=True,lambda_Q=0.,channels=True,
                                             candidate_id='native-reference',reference_R=Rref)
                    identity=dict(source=lock['source_commit'],config_sha256=lock['config_sha256'],
                        native_pack=pack['identity'],entry=previous,reference=member(folder/'calibration/native-reference.json'))
                    if reference['genuine_all_noop']:
                        require(all(not bool(torch.count_nonzero(v)) for v in uref.values()),'GENUINE_ALL_NOOP_IDENTITY')
                        write(folder/'calibration/noop.json',dict(status='GENUINE_NATIVE_ALL_NOOP_PENDING_PRICE',identity=identity))
                        fitted=None;payload=channels['payload'];fit_receipt=dict(status='GENUINE_NATIVE_ALL_NOOP_PENDING_PRICE',
                            logical_candidate_evaluations=0,accepted_updates=0,full_gradient_evaluations=0,
                            accepted_candidate=payload['candidate'])
                    else:
                        # The radial endpoint is the unchanged native returned
                        # delta, not its rounded u32 -> a32*u32 reconstruction.
                        radial_u={l:Rref[l].double()/entry['anchors'][l].double()[None,:] for l in a.sites}
                        calibration=calibrate(radial_u,entry['anchors'],channels['task_gradient'],channels['Q_gradient'],
                            channels['Q'],identity,metadata=dict(native=reference,actual_calls=dict(engine.calls),
                            radial_coordinate='FP64(native_returned_R)/FP64(native_FP32_anchor)',
                            reference_forward='EXACT_NATIVE_RETURNED_R_OVERRIDE_ONLY'))
                        PriceLock(attempt/'price.json').write(calibration.receipt)
                        price=calibration.lambda_Q;price_hash=sha(attempt/'price.json');fitted=None
                        del radial_u
                    del uref,channels,Rref
                else:fitted=None
                if price is not None:
                    require(sha(attempt/'price.json')==price_hash,'IMMUTABLE_PRICE')
                    calls_before=dict(engine.calls)
                    fitted=solve(engine,zero,entry['anchors'],price,emit=events)
                    payload=fitted.payload;fit_receipt=fitted.receipt
                    fit_receipt['engine_calls']={k:engine.calls[k]-calls_before[k] for k in engine.calls}
                    write(folder/'fit.json',fit_receipt)
                require(state(a,H)==previous and rng_identity()==previous_rng,'FIT_REFERENCE_ENTRY_NONMUTATION')
                write(folder/'realization.json',engine.terminal_telemetry(payload))
                writer=commit_payload(a,entry,H,payload,folder/'writer')
                selected=seen if number in profile['milestones'] else current
                # Milestone current/fixed500/cohorts are CPU subsets of these exact rows.
                del payload,zero,entry,engine,fitted;gc.collect();torch.cuda.empty_cache()
                post=observer(a,bench,seen,selected,H,f'W{number}',folder/'post',c,identities,pack['record_ids'])
                cursor.extend(pack['record_ids']);after=state(a,H);after_rng=rng_identity()
                require(after_rng==previous_rng,'BATCH_RNG_DRIFT')
                receipt=dict(experiment=c['task_id'],batch=number,source=lock['source_commit'],config=digest(c),
                    ids=pack['record_ids'],native_pack=pack['identity'],before=previous,after=after,
                    RNG_before=previous_rng,RNG_after=after_rng,context=digest(bench.contexts),ledger=digest(cursor),
                    price=price,price_sha256=price_hash,fit_count=1,fit=fit_receipt,
                    accepted_candidate=writer['candidate'],history_appends=writer['history_appends'],history=writer['history'],
                    writer=member(folder/'writer/commit-payload.json'),observer_no_mutation=True,
                    pre=pre['summary'],post=post['summary'],post_current=post['current'],
                    post_scope='ALL_SEEN' if number in profile['milestones'] else 'CURRENT',seen_requests=len(seen),
                    fresh_teacher_anchor_factor_next_batch=True,seconds=time.monotonic()-t0,
                    checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
                write(folder/'prepared-commit.json',receipt);tx.finish()
                try:write(folder/'commit.json',receipt)
                except BaseException:tx.done=False;raise
            commits.append(receipt);previous=after;previous_rng=after_rng
            print(json.dumps(dict(event='CAUSAL_ALLOCATION_COMMIT',experiment=c['task_id'],batch=number,requests=len(seen))),flush=True)
        except BaseException as error:
            write(folder/'rollback.json',dict(verified=tx.rollback_verified,logical_commit=False,
                committed_prefix=len(commits),entry=tx.before,error_type=type(error).__name__,
                rollback_after_SIGTERM='NOT_VERIFIED',original_KEEP=True));raise
    require(len(commits)==profile['batches'] and len(cursor)==profile['requests'] and
            sum(r['history_appends'] for r in commits)==5*profile['batches'],'PROFILE_CHAIN_COVERAGE')

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--qualification-only',action='store_true');modes.add_argument('--main-only',action='store_true')
    args=p.parse_args();attempt=args.attempt.resolve();out=attempt/('qualification-runtime' if args.qualification_only else 'main')
    out.mkdir(exist_ok=False);t0=time.monotonic();status='TECHNICAL_BLOCKED';a=None;c=None
    try:
        c,lock=locked(attempt)
        if args.main_only:
            # afterany permits failure collection without waiting forever on a
            # failed predecessor. Never load a model if exact READY is absent.
            ready_path=attempt/'qualification/READY.json'
            require(ready_path.is_file(),'TECHNICAL_READY_NOT_AVAILABLE')
            ready=json.loads(ready_path.read_text())
            require(ready['status']=='TECHNICAL_READY' and ready['source']==lock['source_commit']
                    and ready['config_sha256']==lock['config_sha256'],'EXACT_TECHNICAL_READY_BEFORE_MODEL_LOAD')
        a,bench,records,H=setup(c,out)
        if args.main_only:
            ready=json.loads((attempt/'qualification/READY.json').read_text())
            require(ready['status']=='TECHNICAL_READY' and ready['source']==lock['source_commit']
                    and ready['config_sha256']==lock['config_sha256'] and ready['cold_W0_H0']==state(a,H),'EXACT_TECHNICAL_READY')
        else:qualification(a,bench,records,H,c,lock,attempt)
        if args.qualification_only:status='TECHNICAL_READY';return
        drive(a,bench,records,H,c,out,lock,attempt);status=execution(c)['status']
        write(out/'ready.json',dict(status='CHAIN_COMPLETE',source=lock['source_commit'],config=digest(c),
            commits=execution(c)['batches'],requests=execution(c)['requests'],
            history_appends=5*execution(c)['batches'],**{execution(c)['no_next']:True}))
    except BaseException as error:
        raw=getattr(error,'receipt',None)
        write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            operands=raw,original_KEEP=True,automatic_retry=False));raise
    finally:
        write(out/'terminal.json',dict(experiment=c['task_id'] if c else None,status=status,source=os.environ.get(SOURCE_ENV),
            commits=len(list(out.glob('batch-*/commit.json'))),seconds=time.monotonic()-t0,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'),checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
            **({execution(c)['no_next']:True} if c else {})))

if __name__=='__main__':main()
