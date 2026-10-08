"""Task-private three-native baseline runner: generation only at actual W20.

Read-only native engines and RPN definitions are reused. No old module global is
patched, no W0 generation READY/reuse dependency exists, and no pre/milestone
generation is performed. Qualification is technical, fixed, <=8 prompts/cold arm.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback
from dataclasses import asdict

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .w20_common import (TASK, NONCE, ARMS, SOURCE_ENV, MILESTONES, digest,
    require, sha, verify, write, member, batches, validate_config)
from .run import (EngineAdapter, NativeTransaction, check_native, finish_native_batch,
    generation_guard, generation_records, generation_receipt, generation_phase,
    generation_state, generation_assets, EXPECTED, WRITERS)
from .metrics import ObservationView, state, observe, rows, install_W0, generation_payload
from project.run_scripts.gpt2xl_cake_blue.common import stat_seal
from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
from project.run_scripts.experiment_generation_eval.kv_qualification import run_qualification, verify_actual_receipt
from project.run_scripts.experiment_tracking import init
from project.run_scripts.jlz_price_gpt2xl.tracking import SCHEMA, log_w0, log_batch
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.base_model_eval.gpt2xl_server1_common import token_identity
from scripts.fixed_counterfact import load_prefix

SCHEDULE = 'W20_ONLY_FIRST2000'


def locked(attempt):
    c = json.loads((attempt/'config.json').read_text())
    lock = json.loads((attempt/'execution.lock.json').read_text())
    validate_config(c)
    require(c['instruction_id'] == lock['instruction_id'] == NONCE
        and c['task_id'] == lock['task_id'] == TASK, 'W20_TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV) == lock['source_commit']
        and sha(attempt/'config.json') == lock['config_sha256'], 'W20_SOURCE_CONFIG_IDENTITY')
    for field in ('source_members','runtime_sources','launchers','native_closure','source_config_members'):
        for item in lock.get(field, []): verify(item)
    require(lock.get('native_closure') and set(c['source_configs']) == set(ARMS), 'W20_NATIVE_SOURCE_CLOSURE')
    for item in c['assets']: stat_seal(item)
    for item in c['runtime'].get('source_members', [])+c.get('evaluator_sources', []): verify(item)
    for key in ('archive','tracking_env'):
        if key in lock: verify(lock[key])
    for key in ('authority','observer_identity','generation_policy'):
        if key in c: verify(c[key])
    for key in ('session_boundary_member','transition_receipt','source_reference','source_reference_lock'):
        verify(c[key])
    for item in c['policy_members']: verify(item)
    generation = c['generation']
    verify(generation['assets_manifest_member'])
    require(digest(json.loads(verify(generation['qualification_plan_member']).read_text()))
        == generation['qualification_plan_sha256'], 'W20_QUALIFICATION_PLAN_IMMUTABLE')
    for item in generation.get('source_members', []): verify(item)
    require(subprocess.check_output(['git','-C',c['model'],'rev-parse','HEAD'],text=True).strip()
        == c['model_revision'], 'W20_RUNTIME_MODEL_REVISION')
    return c, lock


def start_tracking(c, lock, out, arm):
    generation = c['generation']
    cfg = dict(server='server1',task_id=TASK,arm=arm,attempt=c['run_instance']['attempt'],
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=os.environ['SLURM_JOB_ID'],
        model='gpt2xl',model_family='gpt2',writer=WRITERS[arm],role='scientific',metric_schema=SCHEMA,
        generation_metric_schema=generation['schema'],generation_profile=generation['profile'],
        generation_eval_seed=generation['eval_seed'],reference_assets_sha256=generation['reference_assets_sha256'],
        generation_source_sha=generation['generation_source_sha'],baseline=arm,
        generation_qualification_plan_sha256=generation['qualification_plan_sha256'],
        generation_repair_instruction=NONCE,generation_schedule=SCHEDULE)
    tracker = init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,source_sha=lock['source_commit'],config_sha=lock['config_sha256'],
        startup_readback=tracker.startup,scientific_complete=False,generation_schedule=SCHEDULE))
    return tracker


def bind_runtime_generation(c, arm, model, tok, assets, view, engine, bench, out):
    generation = copy.deepcopy(c['generation'])
    require(generation['schedule'] == SCHEDULE and 'shared_W0_root' not in generation
        and 'old_w0_reuse' not in generation, 'W20_NO_W0_GENERATION_DEPENDENCY')
    plan = json.loads(verify(generation['qualification_plan_member']).read_text())
    require(digest(plan) == generation['qualification_plan_sha256'], 'W20_QUALIFICATION_PLAN_HASH')
    expected_path = str(out/'qualification'/'qualification-actual.json')
    require(generation['qualification_receipts'][arm] == expected_path, 'W20_QUALIFICATION_ARM_PATH')
    require(not Path(expected_path).exists(), 'W20_CREATE_ONCE_ACTUAL_QUALIFICATION')
    actual = run_qualification(model,tok,assets,plan,out=out/'qualification',
        state_callback=lambda:generation_guard(view,engine,bench),source_identity=generation['source_identity'])
    actual_member = actual['member']
    require(actual_member['path'] == expected_path, 'W20_ACTUAL_QUALIFICATION_PATH')
    actual = verify_actual_receipt(actual_member,expected_plan_sha256=generation['qualification_plan_sha256'],
        expected_model_identity=generation['model_identity'])
    generation.update(generation_route=actual['selected_route'],declared_route=actual['selected_route'],
        generation_microbatch=actual['fixed_microbatch'],qualification_receipt_member=actual_member)
    write(out/'generation-qualification-binding.json',dict(plan=generation['qualification_plan_member'],
        plan_sha256=generation['qualification_plan_sha256'],actual=actual_member,
        selected_route=actual['selected_route'],fixed_microbatch=actual['fixed_microbatch'],
        route_results=actual['route_results'],qualification_reused=False,
        technical_qualification_not_scientific_generation=True,generation_schedule=SCHEDULE,
        native_extra_fits=0,scientific_source_unchanged=True))
    return dict(c,generation=generation)


def log_generation_progress(tracker, payload, out=None):
    require(payload.get('phase') == 'generation_evaluation', 'W20_SOURCE_PROGRESS_PHASE')
    fields = ('completed_cases','total_cases','completed_prompts','total_prompts','generated_tokens',
        'new_cases','reused_cases','elapsed_sec','cases_per_sec','prompts_per_sec','tokens_per_sec',
        'physical_forward_calls','prefill_query_tokens','decode_query_tokens','step')
    require(set(payload) <= {'phase'}|{'generation_progress/'+name for name in fields}
        and 'generation_progress/step' in payload, 'W20_PROGRESS_ONLY')
    values = dict(payload,phase='W20_generation')
    from project.run_scripts.experiment_tracking.schema import metrics as validate_metrics
    validate_metrics(values,scientific=True)
    accepted = tracker.log(values)
    if out is not None:
        write(Path(out)/'generation-progress-transport'/('step-'+str(values['generation_progress/step'])+'.json'),
            dict(accepted=accepted,status='SDK_QUEUE_ACCEPTED_NOT_REMOTE_ACK' if accepted else 'LOGGING_DEGRADED',
                payload=values,scientific_result_not_restarted=True,generation_schedule=SCHEDULE))
    return accepted


def log_generation(tracker, observed):
    require(observed['summary']['planned_count'] == 2000
        and observed['identity']['endpoint'] == 'W20' and observed['identity']['cohort'] == 'ALL_SEEN'
        and observed['identity']['ordered_occurrences'] == list(range(1,2001))
        and observed['RNG_restored'] is True and observed['observer_no_mutation'] is True,
        'W20_GENERATION_COMPLETE_BEFORE_LOG')
    values = dict(edits=2000,pre_state_edits=1900,post_state_edits=2000,batch=20)
    values.update(generation_payload('all_seen/post',observed['summary']))
    return safe_log(tracker,lambda:values,'w20_generation_scalar_axes')


def native_loop(c, lock, out, arm, records, model, view, engine, bench, tracker, commits, generation):
    """Twenty unchanged cumulative applies/RPN endpoints; one terminal generation."""
    require(arm in ARMS and len(records) == 2000, 'W20_ARM_FIRST2000')
    previous = state(view,engine.history()); cursor=[];w0raw=rows(out/'W0',previous)
    gen_records = generation_records(records)
    terminal_generation = None
    for number,current,seen in batches(records):
        folder=out/f'batch-{number:02d}';ids=[r['case_id'] for r in current];started=time.monotonic()
        require(state(view,engine.history()) == previous, 'BATCH_ENTRY_LINK')
        pack=dict(record_ids=ids,identity=digest(dict(requests=current,contexts=engine.context_snapshot(),
            hparams=asdict(engine.hp))))
        require(ids == c['packs'][number-1]['ids'], 'NATIVE_TOKEN_ORDER_CONTEXT')
        pre=observe(view,bench,seen,current,engine.history(),f'B{number}_PRE',folder/'pre',current_ids=ids)
        with NativeTransaction(view,engine,bench,arm) as tx:
            returned,native=engine.apply(current,number)
            require(returned is model, 'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(w).all()) for w in view.weights.values())
                and all(bool(torch.isfinite(h).all()) for h in engine.history().values()), 'NATIVE_NONFINITE_COMMIT')
            check_native(arm,native,engine,number)
            native_after,transform=finish_native_batch(engine,arm,number,view)
            selected=seen if number in MILESTONES else current
            post=observe(view,bench,seen,selected,engine.history(),f'W{number}',folder/'post',current_ids=ids)
            after=state(view,engine.history())
            if number == 20:
                terminal_generation=generation.observe(gen_records,'W20',cohort='ALL_SEEN',
                    state_identity=generation_state(after))
                require(terminal_generation['summary']['planned_count'] == 2000
                    and terminal_generation['identity']['ordered_occurrences'] == list(range(1,2001)),
                    'W20_GENERATION_FULL_COHORT')
                generation_phase(out,'W20_POST',terminal_generation,20)
            receipt=dict(task=TASK,arm=arm,batch=number,case_ids=ids,source=lock['source_commit'],config=digest(c),
                before=previous,after=after,native=native,native_counts=native['delta'],pre=pre['summary'],post=post['summary'],
                native_after=native_after,terminal_transform=transform,prune_applied=transform['prune_applied'],
                post_current=post['current'],post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',
                seen_requests=len(seen),native_pack=pack['identity'],ledger=digest(cursor+ids),observer_no_mutation=True,
                generation_schedule=SCHEDULE,
                generation=generation_receipt(terminal_generation) if number == 20 else None,
                generation_measured=number == 20,seconds=time.monotonic()-started,
                checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
            tx.finish()
            try:write(folder/'commit.json',receipt)
            except BaseException:tx.done=False;raise
        commits.append(receipt);cursor.extend(ids);previous=after
        rpn_accepted=log_batch(tracker,receipt,w0raw,ids,[r['case_id'] for r in seen])
        generation_accepted=log_generation(tracker,terminal_generation) if number == 20 else None
        write(folder/'logging.json',dict(RPN_accepted=rpn_accepted,generation_accepted=generation_accepted,
            generation_measured=number == 20,generation_schedule=SCHEDULE,
            status='LOGGING_ACCEPTED' if rpn_accepted and generation_accepted is not False else 'LOGGING_DEGRADED',
            SDK_acceptance_is_not_remote_readback=True,scientific_state_unchanged=True))
        print(json.dumps(dict(event='native_batch_committed',arm=arm,batch=number,edits=len(cursor),
            generation_measured=number == 20,native_counts=native['delta'])),flush=True)
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    require(len(commits) == 20 and len(cursor) == 2000 and terminal_generation is not None,
            'FULL_20_BATCH_W20_GENERATION_COMPLETION')
    write(out/'generation-W20.json',dict(generation_schedule=SCHEDULE,completion_verified=True,
        generation_measured_at_edits=2000,observation=generation_receipt(terminal_generation),
        physical_state=generation_state(previous),qualification_receipt=generation.qualification_member,
        checkpoint_saved=False,raw_local_only=True))
    return previous


def run(attempt, arm):
    attempt=Path(attempt).resolve();require(arm in ARMS,'ARM_IDENTITY')
    out=attempt/arm;require(not out.exists(),'CREATE_ONCE_ARM');out.mkdir()
    started=time.monotonic();stage='LOCK';commits=[];engine=tracker=None;terminal={}
    try:
        c,lock=locked(attempt)
        require(str(torch.__version__) == c['runtime']['torch']
            and transformers.__version__ == c['runtime']['transformers'], 'PINNED_RUNTIME')
        torch.set_num_threads(c['resources']['cpu']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        stage='ONLINE_STARTUP';tracker=start_tracking(c,lock,out,arm)
        stage='LOAD_W0';loaded=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True,use_safetensors=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        view=ObservationView(model);require(state(view,{})['W'] == c['cold_W'],'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(),'GPT2_TIED_HEAD')
        engine=EngineAdapter(c,model,tok,arm,attempt)
        require(all(bool(torch.count_nonzero(h) == 0) for h in engine.history().values()),'COLD_NATIVE_HISTORY_ZERO')
        stage='NATIVE_COLD_CONTEXT_PREPARATION';contexts=engine.prepare_contexts()
        require(isinstance(contexts,list) and contexts,'NATIVE_CONTEXT_READY')
        write(out/'native-contexts.json',dict(contexts=contexts,identity=digest(contexts),generator=engine.context_receipt,
            local_only=True,checkpoint_saved=False,raw_Git_W_B=False))
        bench=CounterFactAdapter(tok,contexts);records=load_prefix(Path(c['stream']).parent,2000);list(batches(records))
        expected=json.loads(verify(c['observer_identity']).read_text())['rows'];actual=token_identity(records,bench)
        require(len(actual) == 26000 and actual == expected,'ACTUAL_OBSERVER_TOKEN_ROW_ORDER')
        cold=state(view,engine.history())
        write(out/'runtime.json',dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),
            transformers=transformers.__version__,model=c['model'],FP32=True,eager=True,TF32=False,autocast=False,
            CPU_threads=c['resources']['cpu'],source=lock['source_commit'],config=digest(c),arm=arm,
            job=os.environ['SLURM_JOB_ID'],checkpoint_saved=False,native_solve_dtype='unchanged actual native source',
            cold_state=cold,cold_history_zero_verified=True,native_context_identity=digest(contexts),
            observer_token_order_identity=digest(actual),load_seconds=time.monotonic()-loaded,generation_schedule=SCHEDULE))
        stage='W0_RPN_OBSERVER'
        observation_config=dict(c,W0_reuse=c['source_configs'][arm].get('W0_reuse'),
            observation_identity=c['source_configs'][arm].get('observation_identity',c['observation_identity']))
        w0=install_W0(observation_config,view,engine.history(),out,bench,records)
        write(out/'W0/logging.json',dict(RPN_accepted=log_w0(tracker,w0['summary']),generation_measured=False,
            generation_schedule=SCHEDULE,SDK_acceptance_is_not_remote_readback=True))
        stage='GENERATION_ROUTE_QUALIFICATION';assets=generation_assets(c)
        observer_c=bind_runtime_generation(c,arm,model,tok,assets,view,engine,bench,out)
        write(out/'runtime-generation-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
            job_id=os.environ['SLURM_JOB_ID'],source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],
            qualification_plan=observer_c['generation']['qualification_plan_member'],
            qualification_plan_sha256=observer_c['generation']['qualification_plan_sha256'],
            qualification_actual=observer_c['generation']['qualification_receipt_member'],
            selected_route=observer_c['generation']['generation_route'],fixed_microbatch=observer_c['generation']['generation_microbatch'],
            generation_schedule=SCHEDULE,source_native_hparams_unchanged=True,immutable=True,
            remote_startup_evidence='PLAN identity verified; actual qualification local, not remotely asserted'))
        # One observer root, one edited-state observe, never a W0 generation cache.
        generation=GenerationObserver(model,tok,assets,observer_c['generation'],out/'generation-W20-raw',
            state_callback=lambda:generation_guard(view,engine,bench),
            progress_callback=lambda payload:log_generation_progress(tracker,payload,out))
        def progress(value):
            candidate=value.get('fit_global_candidate',value.get('native_z_completed',value.get('native_z')))
            require(type(candidate) is int and candidate >= 0,'ACTUAL_NATIVE_FIT_AXIS')
            payload=dict(batch=value['batch'],candidate=candidate,**{'fit/global_candidate':candidate})
            if 'fit_updates' in value:payload['optimizer/calls']=value['fit_updates']
            safe_log(tracker,lambda:payload,'actual_native_candidate_axis')
        engine.progress=progress;stage='NATIVE_20_BATCH_LOOP'
        final=native_loop(c,lock,out,arm,records,model,view,engine,bench,tracker,commits,generation)
        terminal=dict(status='COMPLETED',completed_batches=20,commits=20,edits=2000,state=final,
            native_counts=engine.counts,source=lock['source_commit'],config=digest(c),
            terminal_transforms=1 if arm == 'PRUNE' else 0,prune_applied=arm == 'PRUNE',
            repair='PRUNE_TERMINAL_BASE_FIX' if arm == 'PRUNE' else None,
            generation_schema=c['generation']['schema'],generation_schedule=SCHEDULE,generation_W20_complete=True,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    except BaseException as error:
        terminal=dict(status='FAILED',stage=stage,error_type=type(error).__name__,error=str(error)[:1800],
            completed_batches=len(commits),commits=len(commits),source=locals().get('lock',{}).get('source_commit',os.environ.get(SOURCE_ENV)),
            config=digest(c) if 'c' in locals() else None,native_counts=engine.counts if engine else {k:0 for k in EXPECTED[arm]},
            generation_schedule=SCHEDULE,generation_W20_complete=False,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        write(out/'failure.json',terminal);traceback.print_exc(limit=8)
    finally:
        if tracker:
            try:write(out/'tracking-finish.json',tracker.finish(exit_code=0 if terminal.get('status') == 'COMPLETED' else 1,timeout=45))
            except Exception as error:write(out/'tracking-finish-error.json',dict(type=type(error).__name__,science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic()-started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_gpu_measurement_scope='SINCE_LAST_QUALIFICATION_ROUTE_PEAK_RESET'
                if (out/'qualification/qualification-actual.json').exists()
                else 'PROCESS_START_OR_LAST_RESET_IN_INCOMPLETE_QUALIFICATION',
            qualification_route_peaks_retained=(out/'qualification/qualification-actual.json').exists())
        write(out/'terminal.json',terminal)
    return terminal


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    args=p.parse_args()
    if run(args.attempt,args.arm)['status'] != 'COMPLETED':raise SystemExit(1)
