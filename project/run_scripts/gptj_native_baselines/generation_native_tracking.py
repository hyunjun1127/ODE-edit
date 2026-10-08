"""Shared scalar logger reused with native USER profile; no duplicate SDK helper."""
import os
from project.run_scripts.experiment_tracking import init, schema
from project.run_scripts.experiment_generation_eval.metrics import generation_payload
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
from .generation_native_common import TASK, REPAIR_INSTRUCTION, TRACKING_SCHEDULE, PROFILE, require, write, writer_identity

PROGRESS_KEYS = schema.GENERATION_PROGRESS_METRICS | {'phase'}

def tracking_config(config, lock, arm, job_id):
    gen = config['generation']
    return dict(server='server2', task_id=TASK, arm=arm, attempt=config['tracking_attempt'],
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],job_id=job_id,
        model='gptj',model_family='gptj',writer=writer_identity(arm),role='scientific',
        metric_schema='price-first2k-scalar-v1',baseline=arm,generation_metric_schema=gen['schema'],
        generation_profile=PROFILE,generation_eval_seed=gen['eval_seed'],
        reference_assets_sha256=gen['reference_assets_sha256'],generation_source_sha=gen['source_sha'],
        generation_repair_instruction=REPAIR_INSTRUCTION,generation_schedule=TRACKING_SCHEDULE)

def start_tracking(config, lock, out, arm):
    tracker = init(env_file=config['tracking']['env_file'],spool=out/'tracking',
        config=tracking_config(config,lock,arm,os.environ['SLURM_JOB_ID']))
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,startup_readback=tracker.startup,scientific_complete=False))
    return tracker

def generation_progress_values(payload):
    require(type(payload) is dict and set(payload) == PROGRESS_KEYS
        and payload['phase'] == 'W20_generation'
        and payload['generation_progress/total_cases'] == 2000,
        'NATIVE_W20_PROGRESS_EXACT_SCALAR_KEYS')
    for key,value in payload.items():
        if key=='phase': continue
        require(type(value) in (int,float) and value >= 0,'NATIVE_PROGRESS_BUILTIN_NONNEGATIVE')
    require(payload['generation_progress/new_cases']+payload['generation_progress/reused_cases']
        == payload['generation_progress/completed_cases'] <= 2000,
        'NATIVE_PROGRESS_COMPLETENESS_ARITHMETIC')
    return schema.metrics(payload,scientific=True)

def log_generation_progress(tracker,payload):
    return safe_log(tracker,lambda:generation_progress_values(payload),'native_W20_generation_progress')

def generation_values(prefix,summary,edits,pre_edits=None,post_edits=None):
    require(prefix=='all_seen/post' and edits==2000 and post_edits==2000
        and summary['planned_count']==2000,'NATIVE_ONE_FINAL_W20_GENERATION')
    values=dict(edits=edits,pre_state_edits=pre_edits,post_state_edits=post_edits)
    values.update(generation_payload(prefix,summary))
    return schema.metrics(values,scientific=True)

def log_generation(tracker,prefix,summary,edits,pre_edits=None,post_edits=None):
    return safe_log(tracker,lambda:generation_values(prefix,summary,edits,pre_edits,post_edits),
        'native_generation_'+prefix)
