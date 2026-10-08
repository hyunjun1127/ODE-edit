"""New isolated process: existing PRICE trajectory with reuse-only W0.

Imported runner functions are rebound only in this new process. No running job,
archive, shared tracking helper or original config is edited.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import time

from .profile import forbid_w0
from .generation_schedule import generation_due,first_generation_batch

TASK='price-ridge-m1-m3-2k-20261008'
NONCE='USER-SH4-PRICE-RIDGE-M1-M3-20261008-R1'


def backend(model):
    names={'llama3':'jlz_interference_l1.cap_run','gptj':'jlz_price_gptj.run',
           'gpt2xl':'jlz_price_gpt2xl.run'}
    return importlib.import_module('project.run_scripts.'+names[model])


def validate_lock(attempt,cell):
    from project.run_scripts.jlz_interference_l1.cap_common import sha,verify,require
    lock=json.loads((attempt/'execution.lock.json').read_text())
    require(lock['instruction_id']==NONCE and lock['task_id']==TASK,'TASK_AUTHORITY')
    require(sha(attempt/'config.json')==lock['config_sha256'],'TASK_CONFIG_SHA')
    require(os.environ.get('PRICE_M1_M3_SOURCE')==lock['source_commit'],'TASK_SOURCE_SHA')
    for row in lock['source_members']+lock['input_members']:verify(row)
    config=json.loads((attempt/'config.json').read_text());c=config['cells'][cell]
    require(c['phase0_status']=='PASS' and c['cell']==cell,'PHASE0_REQUIRED')
    forbid_w0(c)
    for row in c['assets']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']), 'ASSET_STAT_CHANGED')
    return c,lock


def attach_post_generation(parent,a,bench,records,H,c,out,tracker,lock):
    from project.run_scripts.experiment_generation_eval import load_assets
    from .generation_adapter import GenerationObserver,run_qualification
    from project.run_scripts.experiment_generation_eval.kv_qualification import verify_actual_receipt
    from project.run_scripts.llama3_native_baselines.generation import normalize_summary,endpoint_ref
    from project.run_scripts.llama3_native_baselines.producer import generation_values
    from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
    g=c['generation'];assets=load_assets(dict(reference_assets=g['reference_manifest'],asset_paths=g['asset_paths']))
    parent.require(assets.sha==g['assets_sha256'],'GENERATION_ASSET_IDENTITY')
    identity=dict(model=c['model_alias'],revision=c['model_revision'],observation_identity=c['observation_identity'])
    gen_config=dict(model_identity=identity,
        profile=g['profile'],eval_seed=20261007,generation_source_sha=lock['source_commit'],
        occurrence_by_case_id={str(r['case_id']):i for i,r in enumerate(records)})
    guard=lambda:dict(editor=parent.state(a,H),contexts=parent.digest(bench.contexts),hooks=a.hook_signature())
    original=parent.observer;pending={};gen=None
    def observer(a,bench,seen,selected,H,name,folder,config,identities,current):
        nonlocal gen
        if name.startswith('W0'):
            raise RuntimeError('NEW_W0_FORWARD_FORBIDDEN')
        result=original(a,bench,seen,selected,H,name,folder,config,identities,current)
        if name.startswith('W'):
            b=int(name[1:])
            if not generation_due(c,b):
                parent.write(folder/'generation-skipped.json',dict(schedule=g['schedule'],batch=b,
                    status='NOT_SCHEDULED',new_generation_forwards=0,metric_values_omitted=True))
                return result
            if g.get('schedule')=='W20_ONLY':
                parent.require(b==20 and len(selected)==2000,'GENERATION_W20_FULL_COHORT')
            state=parent.state(a,H);rng=parent.rng_snapshot()
            if gen is None:
                parent.require(b==first_generation_batch(c),'GENERATION_FIRST_SCHEDULED_POST_ONLY_QUALIFICATION')
                plan=json.loads(parent.verify(g['qualification_plan']).read_text())
                parent.require(plan['model_identity']==identity,'GENERATION_PLAN_MODEL_IDENTITY')
                actual=run_qualification(a.model,bench.tokenizer,assets,plan,
                    out=out/'generation-qualification',state_callback=guard,source_identity=lock['source_commit'])
                actual_member=actual['member']
                qualified=verify_actual_receipt(actual_member,expected_plan_sha256=parent.digest(plan),
                    expected_model_identity=identity)
                parent.write(out/'generation-route.json',dict(actual=actual_member,
                    route=qualified['selected_route'],microbatch=qualified['fixed_microbatch'],
                    qualification_state=state,new_W0=False))
                parent.require(qualified['selected_route']!='UNPADDED_FULL_PREFIX_NO_CACHE',
                    'GENERATION_KV_PARITY_FAILED_NO_SILENT_SLOW_FALLBACK')
                gen=GenerationObserver(a.model,bench.tokenizer,assets,dict(gen_config,
                    generation_route=qualified['selected_route'],generation_microbatch=qualified['fixed_microbatch'],
                    qualification_receipt_member=actual_member,qualification_plan_sha256=parent.digest(plan)),
                    out/'generation',state_callback=guard,
                    progress_callback=lambda payload:safe_log(tracker,lambda:payload,'post_generation_progress'))
            parent.guard(out,c['storage']['next_batch_bytes']+c['generation_reserve_bytes'])
            observed=gen.observe(selected,name,state_identity=dict(W=state['W'],model_identity=identity))
            selected_ids=set(current)
            current_records=[r for r in selected if r['case_id'] in selected_ids]
            current_observation=gen.subset(observed,current_records,name+'_CURRENT')
            values=dict(edits=100*b,batch=b,pre_state_edits=100*(b-1),post_state_edits=100*b)
            values.update(generation_values('current/post',normalize_summary(current_observation['summary']),100))
            if b in (5,10,15,20):
                values.update(generation_values('all_seen/post',normalize_summary(observed['summary']),100*b))
                first500=gen.subset(observed,records[:500],name+'_FIRST500')
                parent.write(folder/'generation-first500-reference.json',endpoint_ref(first500))
            parent.write(folder/'generation-reference.json',endpoint_ref(observed))
            parent.write(folder/'generation-current-reference.json',endpoint_ref(current_observation))
            parent.require(parent.state(a,H)==state and parent.rng_equal(rng),'POST_GENERATION_STATE_RNG')
            pending[b]=values
        return result
    parent.observer=observer
    tracking=importlib.import_module('project.run_scripts.'+('jlz_interference_l1.cap_tracking' if c['model_alias']=='llama3'
        else 'jlz_price_gptj.tracking' if c['model_alias']=='gptj' else 'jlz_price_gpt2xl.tracking'))
    old_log=tracking.log_batch
    def log_batch(tracker,receipt,*args,**kwargs):
        old_log(tracker,receipt,*args,**kwargs)
        values=pending.pop(receipt['batch'],None)
        if values is not None:
            safe_log(tracker,lambda:values,'committed_post_generation')
    tracking.log_batch=log_batch


def execute(attempt,cell):
    from project.run_scripts.experiment_tracking import init
    c,lock=validate_lock(attempt,cell);parent=backend(c['model_alias'])
    out=attempt/cell;out.mkdir(exist_ok=False)
    parent.TASK=TASK
    # Scalar provenance in inherited Events/ConsoleBudget belongs to this task.
    storage=importlib.import_module(parent.Events.__module__);storage.TASK=TASK
    os.environ[parent.SOURCE_ENV]=lock['source_commit']
    parent.ConsoleBudget(out/'console-bound-failure.json').install()
    tracker=None;status='TECHNICAL_BLOCKED';started=time.monotonic()
    try:
        cfg=dict(server='server4',task_id=TASK,arm=cell,attempt=c['run_instance']['attempt'],
            source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=os.environ['SLURM_JOB_ID'],
            model=c['model_alias'],model_family=c['model_profile'],writer='memit',role='scientific',
            metric_schema='price-first2k-scalar-v1',baseline='OURS_PRICE_RIDGE_M1_M3',
            generation_metric_schema='counterfact-cake-generation-metrics-v1',generation_profile=c['generation']['profile'],
            generation_eval_seed=20261007,reference_assets_sha256=c['generation']['assets_sha256'],
            generation_source_sha=lock['source_commit'])
        tracker=init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
        # GPT-J's unchanged bind() is reuse-only because a sealed READY is required.
        if c['model_alias']=='gptj':
            parent.verify(c['native_ready'])
        if c['model_alias']=='gpt2xl':
            from .gpt2_binding import install_runtime_bindings
            install_runtime_bindings(parent,c)
        a,bench,records,H=parent.setup(c,out,cell);a.tracker=tracker
        w0mod=importlib.import_module('project.run_scripts.'+('jlz_interference_l1.cap_w0' if c['model_alias']=='llama3'
            else 'jlz_price_gptj.w0' if c['model_alias']=='gptj' else 'jlz_price_gpt2xl.w0'))
        parent.require(w0mod.choose_reuse(c,attempt) is not None,'W0_REUSE_IDENTITY_BLOCK_NO_FORWARD')
        original_commit=parent.commit_measure
        def checked_commit(a,entry,H,plan,folder):
            result=original_commit(a,entry,H,plan,folder)
            if c['model_alias']=='llama3' and entry['batch']==1:
                old=json.loads(parent.verify(c['reference_B1_commit']).read_text())
                equal=(result['weight_hashes']==old['weight_hashes'] and result['after']==old['after'])
                parent.write(folder/'reproduction-check.json',dict(exact_payload_W_H=equal,
                    reference=c['reference_B1_commit'],actual_weight_hashes=result['weight_hashes']))
                parent.require(equal,'LLAMA_B1_PAYLOAD_W_H_REPRODUCTION')
            return result
        parent.commit_measure=checked_commit
        attach_post_generation(parent,a,bench,records,H,c,out,tracker,lock)
        parent.drive(a,bench,records,H,c,out,lock,attempt,cell)
        status='W20_COMPLETE'
    except BaseException as error:
        parent.write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        if tracker is not None:
            try:tracker.finish(exit_code=0 if status=='W20_COMPLETE' else 1)
            except Exception:pass
        parent.write(out/'terminal.json',dict(status=status,cell=cell,source=lock['source_commit'],
            seconds=time.monotonic()-started,new_W0_forwards=0,checkpoint_saved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--cell',required=True)
    args=p.parse_args();execute(args.attempt.resolve(),args.cell)
