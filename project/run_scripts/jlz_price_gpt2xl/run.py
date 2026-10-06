"""One cold arm per GPU process; own persistent W/H/RNG chain and no B21."""
import argparse,copy,gc,json,os,random,resource,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_realization.observe import active_flags
from project.run_scripts.jlz_realized_subject.geometry import mean_keys
from project.run_scripts.jlz_realized_subject.subject import row_logprobs
from .common import *
from .common import ARMS,history_expected
from .storage import Events,write,guard,storage_plan,ERROR_RESERVE_BYTES,CHUNK_ROWS,ConsoleBudget,console_files_guard,error_operands
from .observer import observe

SOURCE_ENV='JLZ_GPT2XL_SOURCE_COMMIT'

class BatchTransaction(Transaction):
    def __init__(self,a,H,bench,cursor):super().__init__(a,H);self.bench,self.cursor=bench,cursor
    def __enter__(self):
        super().__enter__();self.context=json.loads(json.dumps(self.bench.contexts));self.cursor_before=list(self.cursor)
        self.hooks=self.a.hook_signature();self.cache={k:ram_copy(getattr(self.a,k)) for k in ('last_virtual','capture_virtual') if hasattr(self.a,k)}
        return self
    def finish(self):
        require(self.bench.contexts==self.context and self.a.hook_signature()==self.hooks,'TRANSACTION_CONTEXT_HOOKS')
        super().finish()
    def __exit__(self,*args):
        super().__exit__(*args)
        if not self.done:
            self.bench.contexts=self.context;self.cursor[:]=self.cursor_before
            for k,v in self.cache.items():setattr(self.a,k,ram_copy(v))
        complete=self.bench.contexts==self.context and self.a.hook_signature()==self.hooks
        if not self.done:
            complete=complete and self.cursor==self.cursor_before and all(
                ram_identity(getattr(self.a,k))==ram_identity(v) for k,v in self.cache.items())
        if not complete:self.rollback_verified=False
        require(complete,'TRANSACTION_RESTORE_CONTEXT_HOOKS')

def ram_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:ram_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [ram_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(ram_copy(v) for v in value)
    return copy.deepcopy(value)

def ram_identity(value):
    if isinstance(value,torch.Tensor):return dict(tensor=tensor_sha(value),shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):return {str(k):ram_identity(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [ram_identity(v) for v in value]
    return value

def rng_identity():
    r=rng_snapshot();return digest([repr(r[0]),r[1][0],r[1][1].tolist(),repr(r[1][2:]),tensor_sha(r[2]),[tensor_sha(x) for x in r[3]]])

def locked(attempt,cell):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==lock['task_id']==TASK,'AUTHORITY')
    require(os.environ.get(SOURCE_ENV)==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'SOURCE_CONFIG')
    for key in ('source_members','runtime_sources','native_reference','dependency_sources','launchers'):
        for row in lock[key]:verify(row)
    c=cell_config(c,cell)
    for row in c['authority_members']+[c['cpu_preflight']]:verify(row)
    verify(lock['archive']);verify(lock['native_hparams'])
    verify(lock['tracking_env'])
    if lock.get('input_ready_member'):verify(lock['input_ready_member'])
    for row in c['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    s=c['settings'];require((s['B'],s['batches'],s['requests'])==(100,20,2000) and tuple(s['arms'])==ARMS
        and (s['build_cap'],s['subject_forward_cap'],s['subject_backward_cap'])==(20,20,19)
        and s['no_B21'] and not s['save_checkpoints'],'METHOD_HORIZON_THREE_ARMS')
    require(c['storage']==storage_plan(),'SEALED_SERIALIZER_STORAGE_CONTRACT')
    return c,lock

def shared_source_guard(attempt):
    require(not (attempt/'SHARED_TECHNICAL_BLOCK.json').exists(),'SHARED_SOURCE_TECHNICAL_BLOCK')

def setup(c,out,arm):
    from .adapter import Adapter
    guard(out,c['storage']['reserve_bytes'])
    require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
    torch.set_num_threads(c['resources']['cpu']);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    a=Adapter(model,c['arm_profiles'][arm]);a.alpha_projector=c['projector']
    records=load_prefix(Path(c['stream']).parent,2000)
    from .inputs import bind
    bind(c,Path(c['attempt']),model,tok,records)
    # New sources log the comparison schema directly into their unique science
    # run. No duplicate comparison writer/daemon is needed for these future jobs.
    bench=CounterFactAdapter(tok,json.loads(Path(c['contexts']).read_text()))
    H={l:torch.zeros(d[1],d[1],dtype=torch.float32) for l,d in a.dims.items()}
    expected={key:{str(l):c['cold_W0_H0'][key][str(l)] for l in a.sites} for key in ('W','H')}
    require(state(a,H)==expected,'ACTUAL_COLD_W0_H0_ARM')
    require(digest([r['case_id'] for r in records[:2000]])==c['ordered_ids_sha256'],'FIRST2000_ORDER')
    write(out/'runtime.json',dict(task=TASK,arm=arm,source=os.environ[SOURCE_ENV],config=digest(c),
        writer=c['writer'],model_profile=c['model_profile'],projector_sha256=c['projector']['sha256'],
        profile=digest(c['arm_profiles'][arm]),job=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),
        cold_W0_H0=state(a,H),torch=torch.__version__,transformers=transformers.__version__,
        model=c['model'],FP32=True,geometry_FP64=True,eager=True,TF32=False,autocast=False,
        CPU_threads=c['resources']['cpu'],checkpoint_saved=False))
    return a,bench,records,H

def entry_for(a,bench,pack,H,c):
    from .entry import prepare_entry
    e=prepare_entry(a,bench,pack,H,c['stats'],requests_per_group=1)
    require(e['pack']['identity']==pack['identity'] and e['anchor_star'].shape==(pack['n_requests'],),'FRESH_OWN_ENTRY')
    e['history_entry']={l:H[l] for l in a.sites}
    return e

def observer(a,bench,seen,selected,H,name,out,c,identities,current):
    before=state(a,H);rng=rng_snapshot()
    result=observe(a,bench,seen,selected,H,name,out,c['settings']['observer_microbatch'],current)
    rows=rows_from(out,before);require(validate_rows(rows,identities,[r['case_id'] for r in selected],name)==result['summary'],'OBSERVER_STRICT_REDUCER')
    flags=active_flags(seen);require(all(r['active_at_endpoint']==flags[r['case_id']] for r in rows),'SEEN_PREFIX_METADATA')
    require(state(a,H)==before and rng_equal(rng),'OBSERVER_NONMUTATION');return result

def pre_from_w0(out,current,seen,folder,cold,identities):
    ids=[r['case_id'] for r in current];flags=active_flags(seen)
    rows=[dict(r,endpoint='B1_PRE',active_at_endpoint=flags[r['case_id']]) for r in rows_from(out/'W0',cold) if r['case_id'] in set(ids)]
    summary=validate_rows(rows,identities,ids,'B1_PRE')
    for start in range(0,len(rows),CHUNK_ROWS):
        write(folder/f'chunk-{start:04d}.json',dict(state=cold,rows=rows[start:start+CHUNK_ROWS],optimizer_feedback=False))
    result=dict(endpoint='B1_PRE',state=cold,requests=100,summary=summary,current=summary,seconds=0,new_forwards=0,
        no_mutation=True,optimizer_feedback=False,reused_from=str(out/'W0'))
    write(folder/'summary.json',result);return result


@torch.no_grad()
def commit_measure(a,entry,H,plan,out):
    from .dispatch import terminal
    started=time.monotonic();built=plan['built'];R=plan['R'];weight_hashes={};before=state(a,H)
    require(set(built['weights'])==set(a.weights)==set(H),'SELECTED_WHOLE_PAYLOAD')
    telemetry=terminal(a,entry,R,built,plan['observed'],plan['controller']);write(out/'realization.json',telemetry)
    for l,w in a.weights.items():
        value=built['weights'][l]
        require(value.shape==w.shape and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'TERMINAL_WEIGHT_SCHEMA')
        w.copy_(value);require(torch.equal(w,value.to(w.device)),'TERMINAL_WEIGHT_EXACT_COPY')
        weight_hashes[str(l)]=tensor_sha(w)
    raw={l:[] for l in a.sites};rows=[];B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    for group in entry['groups']:
        captured={};handles=[]
        for l in a.sites:
            handles.append(a.projection(l).register_forward_pre_hook(lambda m,args,l=l:captured.update({l:args[0].detach()})))
        try:
            nh,fh=a.full({k:v.to(a.device) for k,v in group['tokens'].items()})
            probs=row_logprobs(a,group['rows'],nh,fh)
            for j,(row,lp) in enumerate(zip(group['rows'],probs)):
                rows.append(row);owner=row['request'];column=row['reduction_index']
                if row['kind']=='rewrite':
                    labels=row['target'][row['target']!=-100].to(a.device)
                    nll[owner,column]=float(-lp.gather(1,labels[:,None]).mean())
                else:kl[owner]=float((lp.exp()*(lp-entry['teachers'][owner].to(a.device))).sum())
                for l in a.sites:raw[l].append(captured[l][j,row['lookup']].cpu().clone())
        finally:
            for h in handles:h.remove()
    rw=[i for i,row in enumerate(rows) if row['kind']=='rewrite'];rwrows=[rows[i] for i in rw];history={};keychecks={}
    for l in a.sites:
        keys=torch.stack(raw[l]).T;reference=built['raw'][l].T.cpu()
        error=(keys-reference).abs();limit=2e-5+2e-4*reference.abs()
        require(bool((error<=limit).all()),'FINAL_NATIVE_ALL_ROW_KEY_PARITY')
        K=mean_keys(keys[:,rw],rwrows,entry['pack']).float().cpu().contiguous()
        ref=built['K'][l].float().cpu();e=(K-ref).abs();tol=2e-5+2e-4*ref.abs()
        require(bool((e<=tol).all()) and H[l].dtype==torch.float32 and H[l].device.type=='cpu','FINAL_NATIVE_MEAN_KEY_PARITY')
        hbefore=tensor_sha(H[l]);H[l].add_(K@K.T);require(bool(torch.isfinite(H[l]).all()),'NATIVE_HISTORY_FINITE_ONCE')
        history[str(l)]=dict(before=hbefore,after=tensor_sha(H[l]),appends=1,requests=B,key=tensor_sha(K),
            rows='native_rewrite_only_nestedmean',zero_and_unsatisfied_requests_included=True)
        keychecks[str(l)]=dict(max_absolute=float(error.max()),failed_elements=int((error>limit).sum()),
            mean_max_absolute=float(e.max()),atol=2e-5,rtol=2e-4)
    actualF=nll.mean(1)+.0625*kl;subjectF=plan['F'].double().cpu()
    gap=dict(actual_alltoken_F=actualF.tolist(),subject_F=subjectF.tolist(),
        actual_minus_subject=(actualF-subjectF).tolist(),actual_NLL=nll.mean(1).tolist(),actual_KL=kl.tolist(),
        record_only=True,not_a_payload_or_gradient_exactness_claim=True,extra_actual_native_forward_groups=len(entry['groups']))
    write(out/'subject-alltoken-gap.json',gap)
    receipt=dict(candidate=built['candidate'],accepted_weight_copy_exact=True,terminal_last_evaluated_not_best=True,
        writer=a.profile['writer'],projector_sha256=a.alpha_projector['sha256'] if a.profile['writer']=='alphaedit' else None,lambda_alpha=10. if a.profile['writer']=='alphaedit' else None,
        weight_hashes=weight_hashes,before=before,after=state(a,H),history_appends=len(a.sites),history=history,
        keychecks=keychecks,realization=member(out/'realization.json'),actual_gap=member(out/'subject-alltoken-gap.json'),
        no_resolve=True,no_double_add=True,seconds=time.monotonic()-started,
        timer_policy='inclusive telemetry/exactcopy/finalnativecapture/H; nested telemetry timer not additive',checkpoint_saved=False)
    write(out/'commit.json',receipt);return receipt

def drive(a,bench,records,H,c,out,lock,attempt,arm):
    from .fit import fit
    identities=json.loads(verify(c['observer_identity']).read_text())['rows'];cursor=[];commits=[]
    previous=state(a,H);previous_rng=rng_identity()
    guard(out,c['storage']['first_W0_write_bytes']+c['storage']['error_reserve_bytes'])
    from .w0 import choose_reuse
    c_reuse=choose_reuse(c,attempt)
    if c_reuse is not None:
        from .w0 import install
        rng=rng_snapshot();before=state(a,H);hooks=a.hook_signature();context=digest(bench.contexts)
        install(out/'W0',c,before,c_reuse)
        require(state(a,H)==before and rng_equal(rng) and a.hook_signature()==hooks
            and digest(bench.contexts)==context,'W0_REUSE_NONMUTATION')
    else:
        observer(a,bench,records,records,H,'W0',out/'W0',c,identities,[r['case_id'] for r in records])
    from .tracking import w0_rows,log_w0,log_batch
    w0_raw,w0_summary=w0_rows(out,identities,[r['case_id'] for r in records],previous)
    log_w0(a.tracker,w0_summary)
    for number,current,seen in batches(records):
        shared_source_guard(attempt)
        require(number<=20,'NO_B21');folder=out/f'batch-{number:02d}';folder.mkdir(exist_ok=False)
        require(state(a,H)==previous and rng_identity()==previous_rng,'OWN_W_H_RNG_NEXT_ENTRY_JOIN')
        pack=bench.prepare(current);expected=c['packs'][number-1]
        require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],'SEALED_NATIVE_PACK')
        write(folder/'entry.json',dict(task=TASK,arm=arm,source=lock['source_commit'],config=digest(c),profile=digest(c['arm_profiles'][arm]),
            batch=number,ids=pack['record_ids'],native_pack=pack['identity'],state=previous,RNG=previous_rng,
            context=digest(bench.contexts),ledger=digest(cursor),seen_ids=[r['case_id'] for r in seen]))
        if number==2:write(out/'initial.json',dict(status='B1_COMMIT_OBSERVER_B2_OWN_ENTRY',arm=arm,
            B1_commit=member(out/'batch-01/commit.json'),next_entry=previous,source=lock['source_commit'],config=digest(c)))
        tx=BatchTransaction(a,H,bench,cursor);t0=time.monotonic()
        try:
            with tx:
                console_files_guard(attempt,arm)
                capacity=guard(out,c['storage']['next_batch_bytes']+c['storage']['error_reserve_bytes'])
                write(folder/'storage-prefit.json',dict(capacity=capacity,batch=number,
                    max_next_batch_write_bytes=c['storage']['next_batch_bytes'],fit_not_started=True,
                    error_terminal_reserve_bytes=c['storage']['error_reserve_bytes'],
                    original_artifacts_KEEP=True,no_log_omission=True,no_evaluation_reduction=True))
                pre=pre_from_w0(out,current,seen,folder/'pre',previous,identities) if number==1 else observer(
                    a,bench,seen,current,H,f'B{number}_PRE',folder/'pre',c,identities,pack['record_ids'])
                entry=entry_for(a,bench,pack,H,c);events=Events(folder/'events.jsonl',arm,number)
                from .tracking import TrackedEvents
                events=TrackedEvents(events,a.tracker)
                entry['batch']=number;entry['entry_state']=previous;entry['entry_state_sha256']=digest(previous)
                entry['source_binding']=dict(source=lock['source_commit'],config_sha256=lock['config_sha256'],
                    profile=digest(c['arm_profiles'][arm]),batch=number,arm=arm,pack=pack['identity'],
                    state_sha256=digest(previous))
                write(folder/'entry-capture.json',dict(pack=pack['identity'],teacher_hash={str(k):v for k,v in entry['teacher_hash'].items()},
                    anchors={str(l):tensor_sha(v) for l,v in entry['anchors'].items()},H_entry=previous['H'],
                    anchor_layer=17,eligible_layers=list(a.sites),fresh_capture=True,seconds=entry['seconds']))
                plan=fit(a,entry,c['arm_profiles'][arm],events=events)
                require(state(a,H)==previous and rng_identity()==previous_rng,'FIT_ENTRY_NONMUTATION')
                require('events' not in plan['receipt'],'FIT_RECEIPT_STREAM_NOT_DUPLICATE_ARRAY')
                write(folder/'fit.json',plan['receipt'])
                writer=commit_measure(a,entry,H,plan,folder/'writer')
                require(writer['history_appends']==len(a.sites),'ELIGIBLE_HISTORY_ONCE')
                del plan,entry;gc.collect();torch.cuda.empty_cache()
                selected=seen if number in MILESTONES else current
                post=observer(a,bench,seen,selected,H,f'W{number}',folder/'post',c,identities,pack['record_ids'])
                cursor.extend(pack['record_ids']);after=state(a,H);after_rng=rng_identity()
                require(after_rng==previous_rng,'OWN_BATCH_RNG_DRIFT')
                receipt=dict(task=TASK,arm=arm,batch=number,source=lock['source_commit'],config=digest(c),profile=digest(c['arm_profiles'][arm]),
                    ids=pack['record_ids'],native_pack=pack['identity'],before=previous,after=after,RNG_before=previous_rng,RNG_after=after_rng,
                    context=digest(bench.contexts),ledger=digest(cursor),fit_count=1,fit=member(folder/'fit.json'),
                    accepted_candidate=writer['candidate'],history_appends=writer['history_appends'],history=writer['history'],
                    writer=member(folder/'writer/commit.json'),observer_no_mutation=True,pre=pre['summary'],post=post['summary'],
                    post_current=post['current'],post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',seen_requests=len(seen),
                    fresh_teacher_anchor_factor_next_batch=True,seconds=time.monotonic()-t0,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
                shared_source_guard(attempt)
                write(folder/'prepared-commit.json',receipt);tx.finish()
                try:write(folder/'commit.json',receipt)
                except BaseException:tx.done=False;raise
            commits.append(receipt);previous=after;previous_rng=after_rng
            log_batch(a.tracker,receipt,w0_raw,pack['record_ids'],[r['case_id'] for r in seen])
            print(json.dumps(dict(event='BATCH_COMMIT',arm=arm,batch=number,requests=len(seen))),flush=True)
        except BaseException as error:
            try:
                write(folder/'rollback.json',dict(verified=tx.rollback_verified,logical_commit=False,committed_prefix=len(commits),
                    error_type=type(error).__name__,original_KEEP=True))
            except BaseException as receipt_error:
                print(json.dumps(dict(event='ROLLBACK_RECEIPT_WRITE_FAILED',verified='NOT_VERIFIED',
                    original_error=type(error).__name__,receipt_error=str(receipt_error))),flush=True)
            raise
    require(len(commits)==20 and len(cursor)==2000 and sum(r['history_appends'] for r in commits)==history_expected(c['arm']),'ARM_W20_COVERAGE')

def main():
    require_execution_authority()
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--cell',choices=CELLS,required=True)
    args=p.parse_args();attempt=args.attempt.resolve();arm=args.cell
    out=attempt/arm;out.mkdir(exist_ok=False)
    ConsoleBudget(out/'console-bound-failure.json').install()
    tracker=None
    started=time.monotonic();status='TECHNICAL_BLOCKED'
    try:
        c,lock=locked(attempt,arm)
        shared_source_guard(attempt)
        from .tracking import start
        tracker=start(c,lock,out,arm)
        a,bench,records,H=setup(c,out,arm)
        a.tracker=tracker
        drive(a,bench,records[:2000],H,c,out,lock,attempt,arm);status='W20_COMPLETE'
    except BaseException as error:
        # Afterany ordering is only a resource barrier, never a quality gate.
        # Only confirmed common identity/source faults block later processes.
        # Own numeric/storage failures remain arm-local; no scheduler mutation.
        common_identity_failures={'AUTHORITY','SOURCE_CONFIG','METHOD_HORIZON_THREE_ARMS','SEALED_SERIALIZER_STORAGE_CONTRACT',
            'RUNTIME','FIRST2000_ORDER','ASSET_CHANGED'}
        if isinstance(error,RuntimeError) and str(error) in common_identity_failures:
            block=attempt/'SHARED_TECHNICAL_BLOCK.json'
            if not block.exists():
                try:write(block,dict(status='SHARED_SOURCE_TECHNICAL_BLOCK',source=os.environ.get(SOURCE_ENV),
                    config_sha256=sha(attempt/'config.json'),trigger_arm=args.cell,error_type=type(error).__name__,error=str(error),
                    scope='confirmed common authority/source/input/runtime identity only; own trajectory numeric/storage failures remain arm-local',
                    automatic_retry=False))
                except BaseException as receipt_error:
                    print(json.dumps(dict(event='SHARED_BLOCK_RECEIPT_NOT_VERIFIED',error=str(error),
                        receipt_write_error=str(receipt_error))),flush=True)
        try:
            write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                operands=error_operands(getattr(error,'receipt',None)),
                nonfinite_operand_representation='Explicit IEEE tags in error evidence only; no metric normalization',
                original_KEEP=True,automatic_retry=False))
        except BaseException as receipt_error:
            print(json.dumps(dict(event='FIRST_ERROR_RECEIPT_NOT_VERIFIED',error=str(error),
                receipt_write_error=str(receipt_error),original_KEEP=True)),flush=True)
        raise
    finally:
        if tracker is not None:
            try:tracker.finish(exit_code=0 if status=='W20_COMPLETE' else 1)
            except Exception as logging_error:
                print(json.dumps(dict(event='LOGGING_DEGRADED_FINISH',error_type=type(logging_error).__name__)),flush=True)
            # The shared helper already performs bounded finish readback.
        try:
            write(out/'terminal.json',dict(task=TASK,arm=arm,status=status,source=os.environ.get(SOURCE_ENV),
                commits=len(list(out.glob('batch-*/commit.json'))),seconds=time.monotonic()-started,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                job=os.environ.get('SLURM_JOB_ID'),no_B21=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE'))
        except BaseException as receipt_error:
            print(json.dumps(dict(event='TERMINAL_RECEIPT_NOT_VERIFIED',status='NOT_VERIFIED',
                receipt_write_error=str(receipt_error),original_KEEP=True)),flush=True)

if __name__=='__main__':main()
