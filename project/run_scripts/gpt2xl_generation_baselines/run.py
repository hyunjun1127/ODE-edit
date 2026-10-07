"""Six cold native baselines with observation-only generation metrics.

Native fitting/writing is imported read-only. Generation has isolated RNG and
same-state occurrence cache; one observation supplies both lexical metrics.
"""
import argparse
import copy
import gc
import json
import os
import random
import resource
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import TASK, NONCE, ARMS, MILESTONES, SOURCE_ENV, digest, require, sha, verify, write, batches, member
from project.run_scripts.gpt2xl_cake_blue.common import stat_seal
from .metrics import ObservationView, state, observe, rows, install_W0, generation_payload
from project.run_scripts.gpt2xl_native_baselines import native as stock_native
from project.run_scripts.gpt2xl_cake_blue import native as cake_native
from project.run_scripts.gpt2xl_prune_rect import native as prune_native
from project.run_scripts.gpt2xl_prune_rect.run import finish_native_batch as prune_finish
from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
from project.run_scripts.experiment_generation_eval.kv_qualification import run_qualification,verify_actual_receipt
from project.run_scripts.experiment_generation_eval.assets import load_assets
from project.run_scripts.experiment_tracking import init
from project.run_scripts.jlz_price_gpt2xl.tracking import SCHEMA, log_w0, log_batch
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_restore, rng_equal
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.base_model_eval.gpt2xl_server1_common import token_identity

ARM_LAYERS = {a:(13,14,15,16,17) for a in ARMS}
ARM_LAYERS['ALPHAEDIT_BLUE'] = (13,17)
HISTORY_LAYERS = dict(BASE_MEMIT=(), BASE_ALPHAEDIT=(13,14,15,16,17),
    CAKE=(13,14,15,16,17), ALPHAEDIT_BLUE=(13,17), PRUNE=(), RECT=())
WRITERS = dict(BASE_MEMIT='memit', BASE_ALPHAEDIT='alphaedit', CAKE='cake',
               ALPHAEDIT_BLUE='alphaedit_blue', PRUNE='prune', RECT='rect')
EXPECTED = {a:dict(native_z=200 if a=='ALPHAEDIT_BLUE' else 100,
    write_keys=len(ARM_LAYERS[a]), history_keys=len(HISTORY_LAYERS[a]),
    solves=len(ARM_LAYERS[a]), history_appends=len(HISTORY_LAYERS[a])) for a in ARMS}


def locked(attempt):
    c=json.loads((attempt/'config.json').read_text())
    lock=json.loads((attempt/'execution.lock.json').read_text())
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==TASK,'TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV)==lock['source_commit']
        and sha(attempt/'config.json')==lock['config_sha256'],'SOURCE_CONFIG_IDENTITY')
    for field in ('source_members','runtime_sources','launchers','native_closure','source_config_members'):
        for item in lock.get(field,[]):verify(item)
    require(lock.get('native_closure') and set(c['source_configs'])==set(ARMS),'NATIVE_SOURCE_CLOSURE_REQUIRED')
    for item in c['assets']:stat_seal(item)
    for item in c['runtime'].get('source_members',[])+c.get('evaluator_sources',[]):verify(item)
    for key in ('archive','tracking_env'):
        if key in lock:verify(lock[key])
    for key in ('authority','contract','observer_identity','generation_policy'):
        if key in c:verify(c[key])
    verify(c['generation']['assets_manifest_member'])
    verify(c['generation']['qualification_plan_member'])
    require(digest(json.loads(verify(c['generation']['qualification_plan_member']).read_text()))
        ==c['generation']['qualification_plan_sha256'],'QUALIFICATION_PLAN_IMMUTABLE')
    for item in c['generation'].get('source_members',[]):verify(item)
    require(subprocess.check_output(['git','-C',c['model'],'rev-parse','HEAD'],text=True).strip()
        ==c['model_revision'],'RUNTIME_MODEL_CHECKOUT_REVISION')
    require(c.get('noCP') is True and not c.get('z_disk_cache')
        and c.get('exact_resume')=='NOT_AVAILABLE','NOCP_ZCACHE')
    return c,lock


class EngineAdapter:
    """API-only adaptation; never substitutes a native solve, fit or apply."""
    def __init__(self,c,model,tok,arm,attempt=None):
        self.arm=arm
        old=c['source_configs'][arm]
        if arm in ('BASE_MEMIT','BASE_ALPHAEDIT'):
            self.inner=stock_native.prepare_native(old,model,tok,arm)
        elif arm in ('CAKE','ALPHAEDIT_BLUE'):
            self.inner=cake_native.prepare_native(old,model,tok,arm,attempt=attempt)
        else:
            self.inner=prune_native.prepare_native(old,model,tok,arm,attempt=attempt)
    def __getattr__(self,name):return getattr(self.inner,name)
    @property
    def progress(self):return self.inner.progress
    @progress.setter
    def progress(self,value):self.inner.progress=value
    def prepare_contexts(self):
        if hasattr(self.inner,'prepare_contexts'):return self.inner.prepare_contexts()
        saved=rng_snapshot()
        try:
            value=self.inner.module.get_context_templates(self.inner.model,self.inner.tokenizer)
        finally:rng_restore(saved)
        require(rng_equal(saved),'NATIVE_INPUT_CONTEXT_RNG_RESTORE')
        return copy.deepcopy(value)
    def context_snapshot(self):
        if hasattr(self.inner,'context_snapshot'):return self.inner.context_snapshot()
        return copy.deepcopy(self.inner.module.CONTEXT_TEMPLATES_CACHE)
    def restore_context(self,value):
        if hasattr(self.inner,'restore_context'):return self.inner.restore_context(value)
        self.inner.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(value)
    def snapshot_native_state(self):
        if hasattr(self.inner,'snapshot_native_state'):return self.inner.snapshot_native_state()
        return dict(context=self.context_snapshot(),next_batch=self.inner.next_batch)
    def restore_native_state(self,value):
        if hasattr(self.inner,'restore_native_state'):return self.inner.restore_native_state(value)
        self.restore_context(value['context']);self.inner.next_batch=value['next_batch']
    def native_state_signature(self):
        if hasattr(self.inner,'native_state_signature'):return self.inner.native_state_signature()
        return dict(context=self.context_snapshot(),next_batch=self.inner.next_batch)
    def restore_history(self,value):
        current=self.history()
        if value:
            require(set(current)==set(value),'ROLLBACK_HISTORY_LAYOUT')
            with torch.no_grad():
                for l,h in current.items():h.copy_(value[l])
        elif current:
            require(self.arm=='BASE_ALPHAEDIT','COLD_HISTORY_UNINITIALIZED')
            self.inner.reset_history_to_uninitialized()
    def apply(self,records,batch):
        # Stock normalizes dict->string itself; all other native engines retain
        # requested_rewrite target_new dict. Never transplant the stock schema.
        return self.inner.apply(records,batch)


def nonselected(view,arm):
    selected={id(view.weights[l]) for l in ARM_LAYERS[arm]}
    return tuple((n,id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype))
        for n,p in view.model.named_parameters() if id(p) not in selected)


def generation_guard(view,engine,bench):
    return (view.guard(),view.hook_signature(),copy.deepcopy(bench.contexts),
        engine.context_snapshot(),tuple((l,h.data_ptr(),h._version,tuple(h.shape),str(h.dtype))
        for l,h in engine.history().items()))


class NativeTransaction:
    """Selected W/native H and engine metadata rollback in RAM only."""
    def __init__(self,view,engine,bench,arm):
        self.view,self.engine,self.bench,self.arm=view,engine,bench,arm
        self.done=self.rollback_verified=False
    def __enter__(self):
        self.before=state(self.view,self.engine.history());self.guard=nonselected(self.view,self.arm)
        self.hooks=self.view.hook_signature();self.rng=rng_snapshot()
        self.context=copy.deepcopy(self.bench.contexts)
        self.W={l:self.view.weights[l].detach().cpu().clone() for l in ARM_LAYERS[self.arm]}
        self.H={l:h.detach().clone() for l,h in self.engine.history().items()}
        self.native_context=self.engine.context_snapshot()
        self.native_state=self.engine.snapshot_native_state()
        self.native_signature=self.engine.native_state_signature()
        return self
    def finish(self):
        require(nonselected(self.view,self.arm)==self.guard
            and self.view.hook_signature()==self.hooks and self.bench.contexts==self.context,
            'NATIVE_COMMIT_GUARD');self.done=True
    def __exit__(self,kind,value,tb):
        try:
            if not self.done:
                with torch.no_grad():
                    for l,w in self.W.items():self.view.weights[l].copy_(w)
                self.engine.restore_history(self.H)
                self.engine.restore_native_state(self.native_state)
                rng_restore(self.rng)
                require(state(self.view,self.engine.history())==self.before
                    and nonselected(self.view,self.arm)==self.guard and self.view.hook_signature()==self.hooks
                    and self.bench.contexts==self.context
                    and self.engine.context_snapshot()==self.native_context
                    and self.engine.native_state_signature()==self.native_signature
                    and rng_equal(self.rng),'NATIVE_ROLLBACK_MISMATCH')
                self.rollback_verified=True
        finally:self.W.clear();self.H.clear();self.native_state.clear()


def check_native(arm,native,engine,batch):
    require(native['delta']==EXPECTED[arm],'NATIVE_CALL_HISTORY_COUNTS')
    require(native['arm']==arm and native['batch']==batch and native['requests']==100,
        'NATIVE_RECEIPT_IDENTITY')
    require(set(engine.history())==set(HISTORY_LAYERS[arm]),'NATIVE_PHYSICAL_HISTORY_BOUNDARY')
    require(native['same_model_returned'] is True and native['native_has_history'] is bool(HISTORY_LAYERS[arm])
        and native['caller_history_appends']==0 and native['cache_template'] is None
        and native['native_z_disk_cache'] is False and native['checkpoint_saved'] is False,
        'NATIVE_NO_EXTRA_APPLY_CACHE_OR_HISTORY')


def finish_native_batch(engine,arm,batch,view):
    if arm in ('PRUNE','RECT'):return prune_finish(engine,arm,batch,view)
    return state(view,engine.history()),dict(prune_applied=False,terminal_transforms=0,no_model_mutation=True)


def generation_records(records):
    return [dict(record,occurrence_index=i+1) for i,record in enumerate(records)]


def generation_receipt(observed):
    result=dict(rows=member(Path(observed['rows_path'])),identity=observed['identity'],
        identity_sha256=observed['identity_sha256'],summary=observed['summary'],work=observed['work'],
        RNG_restored=observed['RNG_restored'],observer_no_mutation=observed['observer_no_mutation'])
    for key in ('compatibility_member','qualification_receipt_member','provenance'):
        if key in observed:result[key]=observed[key]
    return result


def generation_phase(out,phase,observed,batch=0):
    """Preserve completed observer work even if a later commit/IO fails.

    Only returned scalar diagnostics and immutable row-path binding are stored;
    generated strings/tokens are not duplicated or uploaded.
    """
    write(Path(out)/'generation-work'/(phase+'.json'),dict(phase=phase,batch=batch,
        observation=generation_receipt(observed),scientific_commit_not_asserted=True,
        cost_scope='returned_observer_work_not_additive_to_program_or_allocation',
        raw_local_only=True,checkpoint_saved=False))
    return observed


def generation_state(value):
    """Observation depends on actual W, not method-native H layout."""
    return dict(W=value['W'],H={})


def install_generation_W0(c,arm,observer,records,cold,out):
    root=Path(c['generation']['shared_W0_root'])
    ready=root/'READY.json'
    state_identity=generation_state(cold) # no native history resume/projection
    expected=dict(runtime=observer.runtime_sha,model_W=state_identity['W'],
        occurrences=[r['occurrence_index'] for r in records],case_ids=[r['case_id'] for r in records])
    if arm==c['generation'].get('primary_arm','BASE_MEMIT'):
        require(not ready.exists(),'FRESH_GENERATION_W0_PRIMARY_CREATE_ONCE')
        observed=generation_phase(out,'W0',observer.observe(records,'W0',cohort='FIRST2000',state_identity=state_identity))
        qualification=c['generation']['qualification_receipt_member']
        require(observed['summary']['planned_count']==2000
            and len(observed['rows'])==2000,'COMPLETE_W0_REQUIRED_BEFORE_READY')
        compatibility=observed.get('compatibility_member')
        require(compatibility is not None,'W0_COMPATIBILITY_REQUIRED')
        reused_old=sum(row.get('provenance',{}).get('origin')=='COMPATIBLE_ORIGINAL_W0' for row in observed['rows'])
        write(ready,dict(identity=expected,observation=member(Path(observed['rows_path'])),
            generation_receipt=generation_receipt(observed),fresh_actual_W0=reused_old==0,
            cold_W0_completed_observation=True,mixed_provenance=reused_old>0,
            raw_local_only=True,no_checkpoint=True,qualification_receipt=qualification,
            qualification_receipt_sha256=qualification['sha256'],compatibility=compatibility,
            compatibility_sha256=compatibility['sha256'],completion_verified=True,
            generation_repair_nonce=NONCE,old_provenance_preserved=True))
        reused=False
    else:
        require(ready.is_file() and not ready.is_symlink(),'INPUT_GENERATION_W0_READY_MISSING')
        value=json.loads(ready.read_text())
        require(value['identity']==expected and value.get('cold_W0_completed_observation') is True
            and value['raw_local_only'] is True and value['no_checkpoint'] is True,
            'GENERATION_SHARED_W0_IDENTITY')
        qualification=c['generation']['qualification_receipt_member']
        require(value.get('completion_verified') is True and value.get('generation_repair_nonce')==NONCE
            and value['qualification_receipt_sha256']==qualification['sha256']
            and value['qualification_receipt']==qualification
            and value['compatibility_sha256']==value['compatibility']['sha256'],
            'GENERATION_W0_ACTUAL_QUALIFICATION_COMPATIBILITY_REQUIRED')
        verify(value['qualification_receipt']);verify(value['compatibility'])
        observed=observer.read_observed(verify(value['observation']))
        require(observed['identity']['state_sha256']==digest(state_identity),'GENERATION_SHARED_W0_STATE')
        observed=observer.subset(observed,records,'W0',cohort='FIRST2000')
        generation_phase(out,'W0',observed)
        reused=True
    write(Path(out)/'generation-W0.json',dict(shared_ready=member(ready),
        reused=reused,model_weights_only=True,method_state_reused=False,
        observation=generation_receipt(observed)))
    return observed


def start_tracking(c,lock,out,arm):
    generation=c['generation']
    cfg=dict(server='server1',task_id=TASK,arm=arm,attempt=c['run_instance']['attempt'],
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=os.environ['SLURM_JOB_ID'],
        model='gpt2xl',model_family='gpt2',writer=WRITERS[arm],role='scientific',metric_schema=SCHEMA,
        generation_metric_schema=generation['schema'],generation_profile=generation['profile'],
        generation_eval_seed=generation['eval_seed'],reference_assets_sha256=generation['reference_assets_sha256'],
        generation_source_sha=generation['generation_source_sha'],baseline=arm,
        generation_qualification_plan_sha256=generation['qualification_plan_sha256'],
        generation_repair_instruction=NONCE)
    tracker=init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,source_sha=lock['source_commit'],config_sha=lock['config_sha256'],
        startup_readback=tracker.startup,scientific_complete=False))
    return tracker


def log_generation(tracker,items,batch,post_edits):
    values=dict(edits=post_edits,pre_state_edits=max(0,post_edits-100),post_state_edits=post_edits)
    if batch:values['batch']=batch
    for prefix,summary in items:values.update(generation_payload(prefix,summary))
    return safe_log(tracker,lambda:values,'generation_scalar_axes')


def generation_assets(c):
    item=c['generation']['assets_manifest_member']
    path=verify(item)
    require(path.resolve()==Path(c['generation']['assets_manifest']).resolve(),
        'GENERATION_REFERENCE_MANIFEST_POINTER')
    assets=load_assets(path)
    require(assets.sha==c['generation']['reference_assets_sha256'],'GENERATION_REFERENCE_ASSET_IDENTITY')
    return assets


def bind_runtime_generation(c,arm,model,tok,assets,view,engine,bench,attempt,out):
    """Actual GPU qualification belongs to the first cold replacement only.

    An afterany dependency is not success. Other arms verify both the immutable
    actual qualification and complete compatible W0 READY before proceeding.
    """
    generation=copy.deepcopy(c['generation'])
    plan=json.loads(verify(generation['qualification_plan_member']).read_text())
    require(digest(plan)==generation['qualification_plan_sha256'],'QUALIFICATION_PLAN_HASH')
    receipt_path=Path(generation['qualification_receipt'])
    if arm==generation['primary_arm']:
        require(not receipt_path.exists(),'CREATE_ONCE_ACTUAL_QUALIFICATION')
        actual=run_qualification(model,tok,assets,plan,out=receipt_path.parent,
            state_callback=lambda:generation_guard(view,engine,bench),
            source_identity=generation['source_identity'])
        actual_member=actual['member']
    else:
        ready_path=Path(generation['shared_W0_root'])/'READY.json'
        require(ready_path.is_file() and not ready_path.is_symlink(),'INPUT_GENERATION_W0_READY_MISSING')
        ready=json.loads(ready_path.read_text())
        require(ready.get('completion_verified') is True and ready.get('generation_repair_nonce')==NONCE,
            'INPUT_GENERATION_W0_FAILED_OR_INCOMPLETE')
        actual_member=ready['qualification_receipt']
        require(actual_member['path']==str(receipt_path) and ready['qualification_receipt_sha256']==actual_member['sha256'],
            'INPUT_W0_QUALIFICATION_POINTER')
        verify(ready['compatibility'])
    actual=verify_actual_receipt(actual_member,expected_plan_sha256=generation['qualification_plan_sha256'],
        expected_model_identity=generation['model_identity'])
    require(actual.get('qualification_pass') is True,'SELECTED_GENERATION_ROUTE_NOT_QUALIFIED')
    generation.update(generation_route=actual['selected_route'],declared_route=actual['selected_route'],
        generation_microbatch=actual['fixed_microbatch'],qualification_receipt_member=actual_member)
    write(out/'generation-qualification-binding.json',dict(plan=generation['qualification_plan_member'],
        actual=actual_member,selected_route=actual['selected_route'],fixed_microbatch=actual['fixed_microbatch'],
        qualification_reused=arm!=generation['primary_arm'],qualification_plan_not_actual_GPU_PASS=True,
        scientific_source_unchanged=True,native_extra_fits=0))
    return dict(c,generation=generation)


def log_generation_progress(tracker,payload,out=None):
    """Strict public progress only, never an incomplete endpoint score."""
    require(payload.get('phase') in ('W0_generation','generation_evaluation'),'PUBLIC_GENERATION_PHASE')
    keys={'phase'}|{'generation_progress/'+k for k in
        ('completed_cases','total_cases','completed_prompts','total_prompts','generated_tokens','new_cases',
         'reused_cases','elapsed_sec','cases_per_sec','prompts_per_sec','tokens_per_sec',
         'physical_forward_calls','prefill_query_tokens','decode_query_tokens','step')}
    require(set(payload)<=keys and 'generation_progress/step' in payload,'GENERATION_PROGRESS_ONLY')
    from project.run_scripts.experiment_tracking.schema import metrics as validate_metrics
    validate_metrics(payload,scientific=True) # schema failures are not silent transport loss
    accepted=tracker.log(payload)
    if out is not None:
        write(Path(out)/'generation-progress-transport'/
            ('step-'+str(payload['generation_progress/step'])+'.json'),dict(accepted=accepted,
            status='SDK_QUEUE_ACCEPTED_NOT_REMOTE_ACK' if accepted else 'LOGGING_DEGRADED',
            payload=payload,scientific_result_not_restarted=True))
    return accepted


def native_loop(c,lock,out,arm,records,model,view,engine,bench,tracker,commits,generation,w0generation):
    previous=state(view,engine.history());cursor=[];w0raw=rows(out/'W0',previous)
    gen_records=generation_records(records)
    for number,current,seen in batches(records):
        folder=out/f'batch-{number:02d}';ids=[r['case_id'] for r in current];started=time.monotonic()
        require(state(view,engine.history())==previous,'BATCH_ENTRY_LINK')
        gen_current=gen_records[(number-1)*100:number*100];gen_seen=gen_records[:number*100]
        pack=dict(record_ids=ids,identity=digest(dict(requests=current,contexts=engine.context_snapshot(),
            hparams=asdict(engine.hp))))
        require(ids==c['packs'][number-1]['ids'],'NATIVE_TOKEN_ORDER_CONTEXT')
        pre=observe(view,bench,seen,current,engine.history(),f'B{number}_PRE',folder/'pre',current_ids=ids)
        gen_pre=(generation.subset(w0generation,gen_current,f'B{number}_PRE',cohort='CURRENT')
            if number==1 else generation.observe(gen_current,f'B{number}_PRE',cohort='CURRENT',
                state_identity=generation_state(previous)))
        generation_phase(out,f'B{number}_PRE',gen_pre,number)
        with NativeTransaction(view,engine,bench,arm) as tx:
            returned,native=engine.apply(current,number)
            require(returned is model,'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(w).all()) for w in view.weights.values())
                and all(bool(torch.isfinite(h).all()) for h in engine.history().values()),'NATIVE_NONFINITE_COMMIT')
            check_native(arm,native,engine,number)
            native_after,transform=finish_native_batch(engine,arm,number,view)
            selected=seen if number in MILESTONES else current
            post=observe(view,bench,seen,selected,engine.history(),f'W{number}',folder/'post',current_ids=ids)
            after=state(view,engine.history())
            gen_post=generation.observe(gen_seen if number in MILESTONES else gen_current,
                f'W{number}',cohort='ALL_SEEN' if number in MILESTONES else 'CURRENT',state_identity=generation_state(after))
            generation_phase(out,f'W{number}_POST',gen_post,number)
            gen_post_current=generation.subset(gen_post,gen_current,f'W{number}_CURRENT',cohort='CURRENT')
            generation_phase(out,f'W{number}_CURRENT',gen_post_current,number)
            w0current=generation.subset(w0generation,gen_current,f'W0_CURRENT_B{number}',cohort='CURRENT')
            generation_phase(out,f'W0_CURRENT_B{number}',w0current,number)
            w0seen=generation.subset(w0generation,gen_seen,f'W0_ALL_SEEN_B{number}',cohort='ALL_SEEN') if number in MILESTONES else None
            if w0seen:generation_phase(out,f'W0_ALL_SEEN_B{number}',w0seen,number)
            receipt=dict(task=TASK,arm=arm,batch=number,case_ids=ids,source=lock['source_commit'],config=digest(c),
                before=previous,after=after,native=native,native_counts=native['delta'],pre=pre['summary'],post=post['summary'],
                native_after=native_after,terminal_transform=transform,prune_applied=transform['prune_applied'],
                post_current=post['current'],post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',
                seen_requests=len(seen),native_pack=pack['identity'],ledger=digest(cursor+ids),observer_no_mutation=True,
                generation=dict(pre=generation_receipt(gen_pre),post=generation_receipt(gen_post),
                    current=generation_receipt(gen_post_current),w0_current=generation_receipt(w0current),
                    w0_all_seen=generation_receipt(w0seen) if w0seen else None),
                seconds=time.monotonic()-started,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
            tx.finish()
            try:write(folder/'commit.json',receipt)
            except BaseException:tx.done=False;raise
        commits.append(receipt);cursor.extend(ids);previous=after
        rp=log_batch(tracker,receipt,w0raw,ids,[r['case_id'] for r in seen])
        items=[('current/pre',gen_pre['summary']),('current/post',gen_post_current['summary']),
            ('w0/current',w0current['summary'])]
        if number in MILESTONES:items.extend([('all_seen/post',gen_post['summary']),('w0/all_seen',w0seen['summary'])])
        gp=log_generation(tracker,items,number,len(cursor))
        write(folder/'logging.json',dict(RPN_accepted=rp,generation_accepted=gp,
            status='LOGGING_ACCEPTED' if rp and gp else 'LOGGING_DEGRADED',
            SDK_acceptance_is_not_remote_readback=True,scientific_state_unchanged=True))
        print(json.dumps(dict(event='native_batch_committed',arm=arm,batch=number,edits=len(cursor),
            generation_current=gen_post_current['summary'],native_counts=native['delta'])),flush=True)
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    require(len(commits)==20 and len(cursor)==2000,'FULL_20_BATCH_COMPLETION')
    return previous


def run(attempt,arm):
    attempt=Path(attempt).resolve();require(arm in ARMS,'ARM_IDENTITY')
    out=attempt/arm;require(not out.exists(),'CREATE_ONCE_ARM');out.mkdir()
    started=time.monotonic();stage='LOCK';commits=[];engine=tracker=None;terminal={}
    try:
        c,lock=locked(attempt)
        require(str(torch.__version__)==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'PINNED_RUNTIME')
        torch.set_num_threads(c['resources']['cpu']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        stage='ONLINE_STARTUP';tracker=start_tracking(c,lock,out,arm)
        if arm!=c['generation']['primary_arm']:
            stage='INPUT_GENERATION_W0_READY_VERIFY'
            ready_path=Path(c['generation']['shared_W0_root'])/'READY.json'
            require(ready_path.is_file() and not ready_path.is_symlink(),'INPUT_GENERATION_W0_READY_MISSING')
            ready=json.loads(ready_path.read_text())
            require(ready.get('completion_verified') is True and ready.get('generation_repair_nonce')==NONCE
                and ready.get('cold_W0_completed_observation') is True,'INPUT_GENERATION_W0_FAILED_OR_INCOMPLETE')
            verify_actual_receipt(ready['qualification_receipt'],
                expected_plan_sha256=c['generation']['qualification_plan_sha256'],
                expected_model_identity=c['generation']['model_identity'])
            verify(ready['compatibility']);verify(ready['observation'])
        stage='LOAD_W0';loaded=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True,use_safetensors=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        view=ObservationView(model);require(state(view,{})['W']==c['cold_W'],'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr()==model.transformer.wte.weight.data_ptr(),'GPT2_TIED_HEAD')
        engine=EngineAdapter(c,model,tok,arm,attempt)
        require(all(bool(torch.count_nonzero(h)==0) for h in engine.history().values()),'COLD_NATIVE_HISTORY_ZERO')
        stage='NATIVE_COLD_CONTEXT_PREPARATION';contexts=engine.prepare_contexts()
        require(isinstance(contexts,list) and contexts,'NATIVE_CONTEXT_READY')
        write(out/'native-contexts.json',dict(contexts=contexts,identity=digest(contexts),generator=engine.context_receipt,
            local_only=True,checkpoint_saved=False,raw_Git_W_B=False))
        bench=CounterFactAdapter(tok,contexts);records=load_prefix(Path(c['stream']).parent,2000);list(batches(records))
        expected=json.loads(verify(c['observer_identity']).read_text())['rows'];actual=token_identity(records,bench)
        require(len(actual)==26000 and actual==expected,'ACTUAL_OBSERVER_TOKEN_ROW_ORDER')
        cold=state(view,engine.history())
        write(out/'runtime.json',dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),
            transformers=transformers.__version__,model=c['model'],FP32=True,eager=True,TF32=False,autocast=False,
            CPU_threads=c['resources']['cpu'],source=lock['source_commit'],config=digest(c),arm=arm,
            job=os.environ['SLURM_JOB_ID'],checkpoint_saved=False,native_solve_dtype='unchanged actual native source',
            cold_state=cold,cold_history_zero_verified=True,native_context_identity=digest(contexts),
            observer_token_order_identity=digest(actual),load_seconds=time.monotonic()-loaded))
        stage='W0_RPN_OBSERVER'
        # Qualification belongs to its own native-context/source bundle. A
        # stock qualified raw is not automatically transplanted to other arms.
        observation_config=dict(c,W0_reuse=c['source_configs'][arm].get('W0_reuse'),
            observation_identity=c['source_configs'][arm].get('observation_identity',c['observation_identity']))
        w0=install_W0(observation_config,view,engine.history(),out,bench,records)
        stage='GENERATION_ROUTE_QUALIFICATION';assets=generation_assets(c)
        observer_c=bind_runtime_generation(c,arm,model,tok,assets,view,engine,bench,attempt,out)
        write(out/'runtime-generation-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
            job_id=os.environ['SLURM_JOB_ID'],source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],
            qualification_plan=observer_c['generation']['qualification_plan_member'],
            qualification_plan_sha256=observer_c['generation']['qualification_plan_sha256'],
            qualification_actual=observer_c['generation']['qualification_receipt_member'],
            selected_route=observer_c['generation']['generation_route'],
            fixed_microbatch=observer_c['generation']['generation_microbatch'],
            source_native_hparams_unchanged=True,immutable=True,
            remote_startup_evidence='PLAN identity verified; actual qualification evidence local, not remotely asserted'))
        stage='W0_GENERATION_OBSERVER'
        generation=GenerationObserver(model,tok,assets,observer_c['generation'],out/'generation',
            state_callback=lambda:generation_guard(view,engine,bench),
            progress_callback=lambda payload:log_generation_progress(tracker,payload,out))
        wg=install_generation_W0(observer_c,arm,generation,generation_records(records),cold,out)
        write(out/'W0/logging.json',dict(RPN_accepted=log_w0(tracker,w0['summary']),
            generation_accepted=log_generation(tracker,[('W0_first2000',wg['summary'])],0,0),
            SDK_acceptance_is_not_remote_readback=True))
        def progress(value):
            candidate=value.get('fit_global_candidate',value.get('native_z_completed',value.get('native_z')))
            require(type(candidate) is int and candidate>=0,'ACTUAL_NATIVE_FIT_AXIS')
            payload=dict(batch=value['batch'],candidate=candidate,**{'fit/global_candidate':candidate})
            if 'fit_updates' in value:payload['optimizer/calls']=value['fit_updates']
            safe_log(tracker,lambda:payload,'actual_native_candidate_axis')
        engine.progress=progress;stage='NATIVE_20_BATCH_LOOP'
        final=native_loop(c,lock,out,arm,records,model,view,engine,bench,tracker,commits,generation,wg)
        terminal=dict(status='COMPLETED',completed_batches=20,commits=20,edits=2000,state=final,
            native_counts=engine.counts,source=lock['source_commit'],config=digest(c),
            terminal_transforms=1 if arm=='PRUNE' else 0,prune_applied=arm=='PRUNE',
            repair='PRUNE_TERMINAL_BASE_FIX' if arm=='PRUNE' else None,
            generation_schema=c['generation']['schema'],checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    except BaseException as error:
        terminal=dict(status='FAILED',stage=stage,error_type=type(error).__name__,error=str(error)[:1800],
            completed_batches=len(commits),commits=len(commits),source=locals().get('lock',{}).get('source_commit',os.environ.get(SOURCE_ENV)),
            config=digest(c) if 'c' in locals() else None,native_counts=engine.counts if engine else {k:0 for k in EXPECTED[arm]},
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        write(out/'failure.json',terminal);traceback.print_exc(limit=8)
    finally:
        if tracker:
            try:write(out/'tracking-finish.json',tracker.finish(exit_code=0 if terminal.get('status')=='COMPLETED' else 1,timeout=45))
            except Exception as error:write(out/'tracking-finish-error.json',dict(type=type(error).__name__,science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic()-started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(out/'terminal.json',terminal)
    return terminal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args()
    if run(a.attempt,a.arm)['status']!='COMPLETED':raise SystemExit(1)
