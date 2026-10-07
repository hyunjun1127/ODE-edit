"""Own producer mapping with a source-identical private transport except scalar extension."""
import os
from .generation_common import TASK, require, writer_identity, write
from . import generation_tracking_schema as schema
from project.run_scripts.jlz_interference_l1.cap_tracking import SCHEMA, log_w0, log_batch, safe_log
def start_tracking(c,lock,out,arm):
    # Extra keys are allowed only in this task's source-bound private schema.
    from .generation_tracking_client import init
    gen=c['generation']
    cfg=dict(server='server2',task_id=TASK,arm=arm,attempt='attempt-r1',
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],
        job_id=os.environ['SLURM_JOB_ID'],model='gptj',model_family='gptj',
        writer=writer_identity(arm),role='scientific',metric_schema=SCHEMA,
        baseline=arm,generation_metric_schema=gen['schema'],generation_profile=gen['profile'],
        generation_eval_seed=gen['eval_seed'],reference_assets_sha256=gen['reference_assets_sha256'],
        generation_source_sha=gen['source_sha'])
    tracker=init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,startup_readback=tracker.startup,scientific_complete=False))
    return tracker
def generation_values(prefix,summary,edits,pre_edits=None,post_edits=None):
    require(prefix in schema.PREFIXES,'GEN_PREFIX')
    result={'edits':edits}
    if pre_edits is not None:result['pre_state_edits']=pre_edits
    if post_edits is not None:result['post_state_edits']=post_edits
    for key in schema.FIELDS:
        if key in summary:result[prefix+'/'+key]=summary[key]
    require(prefix+'/generation/planned_count' in result,'GEN_SUMMARY_REQUIRED_COUNTS')
    return schema.metrics(result,scientific=True)
def log_generation(tracker,prefix,summary,edits,pre_edits=None,post_edits=None):
    payload=generation_values(prefix,summary,edits,pre_edits,post_edits)
    accepted=tracker.log(payload)
    if accepted is False:
        write(tracker.spool/'generation-log-degraded.json',dict(status='LOGGING_DEGRADED_REJECTED_POINT',
            prefix=prefix,edits=edits,scientific_rerun=False))
    return accepted
