"""Task caller of the immutable shared generation payload and scalar logger."""
import os
from .generation_common import TASK, require, writer_identity, write
from project.run_scripts.experiment_generation_eval.metrics import generation_payload
from project.run_scripts.experiment_tracking import init, schema
from project.run_scripts.experiment_tracking.method import GENERATION_PREFIXES
from project.run_scripts.jlz_interference_l1.cap_tracking import SCHEMA, log_w0, log_batch, safe_log
def start_tracking(c,lock,out,arm):
    gen=c['generation']
    cfg=dict(server='server2',task_id=TASK,arm=arm,attempt='attempt-r1',
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],
        job_id=os.environ['SLURM_JOB_ID'],model='gptj',model_family='gptj',
        writer=writer_identity(arm),role='scientific',metric_schema=SCHEMA,
        baseline=arm,generation_metric_schema=gen['schema'],generation_profile=gen['profile'],
        generation_eval_seed=gen['eval_seed'],reference_assets_sha256=gen['reference_assets_sha256'],
        generation_source_sha=gen['source_sha'])
    logger_init=init
    if 'repair' in gen:
        from .generation_tracking_client import init as private_init
        from .generation_tracking_schema import REPAIR_TASK, REPAIR_ATTEMPT
        require(type(gen['repair']) is dict,'GEN_REPAIR_CONFIG')
        cfg.update(task_id=REPAIR_TASK,attempt=REPAIR_ATTEMPT)
        if 'qualification_plan_sha256' in gen['repair']:
            cfg['qualification_plan_sha256']=gen['repair']['qualification_plan_sha256']
        logger_init=private_init
    tracker=logger_init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    write(out/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,startup_readback=tracker.startup,scientific_complete=False))
    return tracker
def generation_values(prefix,summary,edits,pre_edits=None,post_edits=None):
    require(prefix in GENERATION_PREFIXES,'GEN_PREFIX')
    result={'edits':edits}
    if pre_edits is not None:result['pre_state_edits']=pre_edits
    if post_edits is not None:result['post_state_edits']=post_edits
    if 'planned_count' in summary:
        # Production consumes reduce_cases() unchanged: raw means/counts and
        # closed missing reasons. Shared mapping alone owns public key names.
        result.update(generation_payload(prefix,summary))
    else:
        # Prior CPU-only fixtures supplied already flattened scalar fields.
        # Keep that fixture boundary; it is never the shared observer format.
        from . import generation_tracking_schema as legacy
        result.update({prefix+'/'+key:summary[key] for key in legacy.FIELDS if key in summary})
        require(prefix+'/generation/planned_count' in result,'GEN_SUMMARY_REQUIRED_COUNTS')
        legacy.metrics(result,scientific=True)
    return schema.metrics(result,scientific=True)
def log_generation(tracker,prefix,summary,edits,pre_edits=None,post_edits=None):
    return safe_log(tracker,lambda:generation_values(prefix,summary,edits,pre_edits,post_edits),
                    'generation_'+prefix)


def generation_progress_values(values):
    from .generation_tracking_schema import metrics as private_metrics
    return private_metrics(values,scientific=True)


def log_generation_progress(tracker,values):
    """Callback telemetry only; failure never causes new generation or fitting."""
    return safe_log(tracker,lambda:generation_progress_values(values),'w0_generation_progress')
