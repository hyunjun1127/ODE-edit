"""Four independent native GPT-J cold chains; no checkpoint or extra fit."""
import argparse
import gc
import json
import os
import random
import resource
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import *
from .metrics import ObservationView,state,observe,rows,install_W0
from .native_cake_blue import normalize_requests
from .logic import expected_counts, writer_identity, fit_payload

def prepare_native(c, model, tok, arm, **kwargs):
    if arm in ('CAKE','ALPHAEDIT_BLUE'):
        from .native_cake_blue import prepare_native as prepare
    else:
        from .native_prune_rect import prepare_native as prepare
    return prepare(c, model, tok, arm, **kwargs)
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_realization.writer import rng_snapshot,rng_restore,rng_equal
from project.run_scripts.experiment_tracking import init
from project.run_scripts.jlz_interference_l1.cap_tracking import SCHEMA,log_w0,log_batch
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
from scripts.fixed_counterfact import load_prefix


def locked(attempt):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==TASK,'TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV)==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'SOURCE_CONFIG_IDENTITY')
    for row in lock['source_members']+lock['runtime_sources']+lock['launchers']+[lock['archive'],lock['tracking_env'],c['authority']]:verify(row)
    from .prepare import stat_seal
    for row in c['assets']:stat_seal(row)
    require(not c['z_disk_cache'] and c['noCP'] and c['exact_resume']=='NOT_AVAILABLE','NOCP_ZCACHE')
    return c,lock

def nonselected(view):
    selected={id(v) for v in view.weights.values()}
    return tuple((n,id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in view.model.named_parameters() if id(p) not in selected)

class NativeTransaction:
    """Rollback edited projections and native cache only in RAM; stop on errors."""
    def __init__(self,view,engine,bench):self.view,self.engine,self.bench=view,engine,bench;self.done=False;self.rollback_verified=False
    def __enter__(self):
        self.before=state(self.view,self.engine.history());self.guard=nonselected(self.view);self.hooks=self.view.hook_signature()
        self.rng=rng_snapshot();self.context=json.loads(json.dumps(self.bench.contexts))
        self.W={l:w.detach().cpu().clone() for l,w in self.view.weights.items()}
        self.H={l:h.detach().cpu().clone() for l,h in self.engine.history().items()}
        return self
    def finish(self):
        require(nonselected(self.view)==self.guard and self.view.hook_signature()==self.hooks and self.bench.contexts==self.context,'NATIVE_COMMIT_GUARD')
        self.done=True
    def __exit__(self,kind,value,tb):
        if not self.done:
            with torch.no_grad():
                for l,w in self.view.weights.items():w.copy_(self.W[l])
                history=self.engine.history()
                if self.H:
                    require(set(history)==set(self.H),'NATIVE_ROLLBACK_HISTORY_LAYOUT')
                    for l,h in history.items():h.copy_(self.H[l])
                elif history:
                    # Before Alpha's first apply native cache is genuinely uninitialized.
                    self.engine.reset_history_to_uninitialized()
            rng_restore(self.rng)
            require(state(self.view,self.engine.history())==self.before and nonselected(self.view)==self.guard and self.view.hook_signature()==self.hooks and self.bench.contexts==self.context and rng_equal(self.rng),'NATIVE_ROLLBACK_MISMATCH')
            self.rollback_verified=True
        self.W.clear();self.H.clear()

def start_tracking(c,lock,out,arm):
    cfg=dict(server='server2',task_id=TASK,arm=arm,attempt='attempt-r1',
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=os.environ['SLURM_JOB_ID'],
        model='gptj',model_family='gptj',writer=writer_identity(arm),
        role='scientific',metric_schema=SCHEMA)
    tracker=init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,source_sha=lock['source_commit'],config_sha=lock['config_sha256'],
        startup_readback=tracker.startup,scientific_complete=False))
    return tracker

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);args=p.parse_args()
    attempt=args.attempt.resolve();out=attempt/args.arm;require(not out.exists(),'CREATE_ONCE_ARM');out.mkdir()
    started=time.monotonic();stage='LOCK';commits=[];engine=tracker=None;tx=None;terminal={}
    try:
        c,lock=locked(attempt)
        require(str(torch.__version__)==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'PINNED_RUNTIME')
        torch.set_num_threads(c['resources']['cpu']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        stage='ONLINE_STARTUP';tracker=start_tracking(c,lock,out,args.arm)
        stage='LOAD_W0';t0=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True,use_safetensors=False).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        view=ObservationView(model,c['arm_layers'][args.arm]);require(state(view,{})['W']=={str(l):c['cold_W'][str(l)] for l in view.sites},'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr()!=model.transformer.wte.weight.data_ptr(),'GPTJ_UNTIED_HEAD')
        model.config.use_cache=False
        engine=prepare_native(c,model,tok,args.arm,attempt=out)
        require(set(engine.history()) == (set(view.sites) if args.arm in ('CAKE','ALPHAEDIT_BLUE') else set()),'INITIAL_NATIVE_HISTORY_LAYOUT')
        require(all(bool((h == 0).all()) for h in engine.history().values()),'COLD_ZERO_HISTORY')
        stage='NATIVE_CONTEXT_PREPARATION';context_started=time.monotonic();engine.prepare_contexts()
        native_contexts=engine.contexts()
        write(out/'native-contexts.json',dict(contexts=native_contexts,context_sha256=digest(native_contexts),
            seed=c['seed'],native_generator=True,source=lock['source_commit'],preparation_seconds=time.monotonic()-context_started,
            training_context_not_evaluator_input=True))
        bench=CounterFactAdapter(tok,[]) # R/P/N does not consume generated training context.
        records=load_prefix(Path(c['stream']).parent,2000);list(batches(records))
        write(out/'runtime.json',dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),transformers=transformers.__version__,model=c['model'],FP32=True,eager=True,TF32=False,autocast=False,CPU_threads=c['resources']['cpu'],
            source=lock['source_commit'],config=digest(c),arm=args.arm,job=os.environ['SLURM_JOB_ID'],checkpoint_saved=False,
            initial_history_zero=True,initial_state=state(view,engine.history()),
            native_solve_dtype='FP64' if args.arm in ('PRUNE','RECT') else 'native FP32',load_seconds=time.monotonic()-t0))
        stage='W0_REUSE';w0=install_W0(c,view,engine.history(),out,bench,records);w0raw=rows(out/'W0',state(view,engine.history()));log_w0(tracker,w0['summary'])
        previous=state(view,engine.history());cursor=[]
        def progress(value):
            # Native hooks expose already-computed scalars; this adds no forward/sync.
            trace=engine.current.get('fit_trace',[])
            payload=fit_payload(value,trace[-1] if trace else None)
            safe_log(tracker,lambda:payload,'native_fit_forward_axis')
        engine.progress=progress
        for number,current,seen in batches(records):
            folder=out/f'batch-{number:02d}';ids=[r['case_id'] for r in current];t0=time.monotonic()
            require(state(view,engine.history())==previous,'BATCH_ENTRY_LINK')
            require(__import__('shutil').disk_usage(out).free>=c['resources']['reserve_bytes'],'DISK_RESERVE')
            pack=dict(identity=digest(normalize_requests(current)))
            stage=f'B{number}_PRE';pre=observe(view,bench,seen,current,engine.history(),f'B{number}_PRE',folder/'pre',current_ids=ids)
            stage=f'B{number}_NATIVE_APPLY'
            with NativeTransaction(view,engine,bench) as tx:
                returned,native=engine.apply(normalize_requests(current),number)
                require(returned is model,'SAME_ACCUMULATED_NATIVE_MODEL')
                require(all(bool(torch.isfinite(w).all()) for w in view.weights.values()) and all(bool(torch.isfinite(h).all()) for h in engine.history().values()),'NATIVE_NONFINITE_COMMIT')
                counts={key:native['delta'][key] for key in expected_counts(args.arm)}
                require(counts==expected_counts(args.arm),'NATIVE_CALL_HISTORY_COUNTS')
                require(set(engine.history())==(set(view.sites) if args.arm in ('CAKE','ALPHAEDIT_BLUE') else set()),'NATIVE_HISTORY_BOUNDARY')
                if number==20 and args.arm=='PRUNE':
                    stage='TERMINAL_PRUNE_BASE_FIX';native['terminal_prune']=engine.terminal_prune()
                    require(native['terminal_prune']['repair_label']=='PRUNE_TERMINAL_BASE_FIX','PRUNE_EXPLICIT_BASE_FIX')
                    native['prune_applied']=True
                    native['explicit_repair']='PRUNE_TERMINAL_BASE_FIX'
                stage=f'B{number}_POST';selected=seen if number in MILESTONES else current
                post=observe(view,bench,seen,selected,engine.history(),f'W{number}',folder/'post',current_ids=ids)
                after=state(view,engine.history());cursor.extend(ids)
                receipt=dict(task=TASK,arm=args.arm,batch=number,case_ids=ids,source=lock['source_commit'],config=digest(c),before=previous,after=after,
                    native=native,native_counts=counts,pre=pre['summary'],post=post['summary'],post_current=post['current'],
                    post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',seen_requests=len(seen),
                    native_pack=pack['identity'],ledger=digest(cursor),observer_no_mutation=True,
                    seconds=time.monotonic()-t0,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
                write(folder/'native-context-identity.json',dict(context_sha256=digest(engine.contexts()),
                    groups=[len(g) for g in engine.contexts()],generated_inside_native=True))
                tx.finish()
                try:write(folder/'commit.json',receipt)
                except BaseException:tx.done=False;raise
            commits.append(receipt);previous=after
            log_batch(tracker,receipt,w0raw,ids,[r['case_id'] for r in seen])
            print(json.dumps(dict(event='native_batch_committed',arm=args.arm,batch=number,edits=len(cursor),native_counts=counts)),flush=True)
            gc.collect();torch.cuda.empty_cache()
        require(len(commits)==20 and len(cursor)==2000,'FULL_20_BATCH_COMPLETION')
        terminal=dict(status='COMPLETED',completed_batches=20,commits=20,edits=2000,state=previous,native_counts=engine.counts,
            source=lock['source_commit'],config=digest(c),checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    except BaseException as error:
        terminal=dict(status='FAILED',stage=stage,error_type=type(error).__name__,error=str(error)[:1800],
            completed_batches=len(commits),commits=len(commits),source=locals().get('lock',{}).get('source_commit',os.environ.get(SOURCE_ENV)),
            config=digest(c) if 'c' in locals() else None,
            native_counts=engine.counts if engine else dict(native_z=0,write_keys=0,history_keys=0,solves=0,history_appends=0),
            rollback_verified=bool(tx and tx.rollback_verified),checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        write(out/'failure.json',terminal)
        traceback.print_exc(limit=8)
    finally:
        if tracker:
            try:write(out/'tracking-finish.json',tracker.finish(exit_code=0 if terminal.get('status')=='COMPLETED' else 1,timeout=45))
            except Exception as error:write(out/'tracking-finish-error.json',dict(type=type(error).__name__,science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic()-started,peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(out/'terminal.json',terminal)
    if terminal['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
