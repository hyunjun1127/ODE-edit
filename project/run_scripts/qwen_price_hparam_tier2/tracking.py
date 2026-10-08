"""Task-scoped schema adapter; reuse SH1 transport unchanged, no logger fork.

The adapter is installed only in this task's parent/sidecar processes. Shared
files and old run sources are never edited. CPU tests exercise the real worker.
"""
import math
import re
import subprocess
from types import SimpleNamespace
from project.run_scripts.experiment_tracking import schema as base
from project.run_scripts.experiment_tracking import method
from . import TASK

SCHEMA='qwen-heldout-tuning-scalar-v1'
EXTRA={'cohort_role','tier','slice_identity','resolved_config','resolved_config_sha256','preset','M1'}
_config=base.config
_metrics=base.metrics
_axes=method.define_axes


def config(values):
    base.require(type(values) is dict and EXTRA<=values.keys(),'TUNING_METADATA_REQUIRED')
    base.require(values['task_id']==TASK and values['metric_schema']==SCHEMA
        and values['model']=='qwen' and values['model_family']=='QWEN'
        and values['writer']=='memit','TUNING_TASK_IDENTITY')
    base.require(values['tier'] in ('smoke','w0','tier1','tier2'),'TUNING_TIER')
    validation=values['tier']=='smoke'
    base.require(values['role']==('validation' if validation else 'scientific')
        and values['cohort_role']==('validation' if validation else 'heldout_tuning'), 'TUNING_COHORT_ROLE')
    base.require(values['M1'] is True and values['preset'] in ('base','native','override'),'TUNING_PRESET')
    base.require(all(type(values[k]) is str and re.fullmatch('[a-f0-9]{64}',values[k])
        for k in ('slice_identity','resolved_config_sha256')),'TUNING_HASH')
    # Only the exact resolver output is admitted, never arbitrary nested config.
    from official.ours.config import resolve, plain
    resolved=values['resolved_config']
    base.require(type(resolved) is dict,'TUNING_RESOLVED_CONFIG')
    arm=resolved.get('arm')
    conditional=values['arm'].endswith('_Q4-beta400-lamN0')
    base.require(not conditional or arm=='qwen25-Q4-beta400','CONDITIONAL_ARM_PARENT')
    expected=plain(resolve('qwen25',arm,override={'lambda_N':0.0} if conditional else None))
    base.require(resolved==expected and expected['sha256']==values['resolved_config_sha256'],
                 'TUNING_RESOLVER_IDENTITY')
    stripped={k:v for k,v in values.items() if k not in EXTRA}
    stripped.update(role='scientific',metric_schema=method.COMPARISON_SCHEMA)
    accepted=_config(stripped)
    accepted.update({k:values[k] for k in EXTRA})
    accepted.update(role=values['role'],metric_schema=SCHEMA)
    return accepted


def metrics(values,*,scientific=False):
    base.require(type(values) is dict,'TUNING_METRIC_MAPPING')
    w0={k:v for k,v in values.items() if k.startswith('W0_first500/')}
    rest={k:v for k,v in values.items() if k not in w0}
    base.require(not any(k.startswith('W0_first2000/') or 'generation' in k or
        k.startswith('eval/') for k in values),'TUNING_NO_EVAL2K_OR_GENERATION')
    if w0:
        base.require(values.get('edits')==0,'TUNING_W0_ZERO')
        allowed={f'W0_first500/{kind}/{field}' for kind in 'RPN' for field in method.FIELDS}
        allowed.add('W0_first500/success_harmonic_pct')
        base.require(set(w0)<=allowed,'TUNING_W0_ALLOWLIST')
        base.require(all(type(v) in (int,float) and math.isfinite(v) for v in w0.values()),'TUNING_W0_FINITE')
        base.require(all(w0.get(f'W0_first500/{k}/count')==n for k,n in [('R',500),('P',1000),('N',5000)]),
                     'TUNING_W0_EXACT500')
        # Validate arithmetic using the common prefix-independent current rules;
        # original payload/names/counts are returned, never relabelled on wire.
        check={k.replace('W0_first500/','current/post/',1):v for k,v in w0.items()}
        _metrics(dict(check,edits=0,post_state_edits=0),scientific=True)
    _metrics(rest,scientific=scientific)
    return dict(values)


def axes(run):
    _axes(run)
    run.define_metric('W0_first500/*',step_metric='edits',step_sync=False)


def install():
    from project.run_scripts.experiment_tracking import identity,client,worker
    base.config=config
    identity.config=config
    client.metrics=metrics
    worker.config=config;worker.metrics=metrics;worker.define_axes=axes
    # Keep shared parent queue, immutable identity, privacy env, failure isolation
    # and bounded finish/readback. Redirect only its exact worker module argv.
    def launch(args,**kwargs):
        args=list(args)
        index=args.index('project.run_scripts.experiment_tracking.worker')
        args[index]='project.run_scripts.qwen_price_hparam_tier2.tracking'
        return subprocess.Popen(args,**kwargs)
    client.subprocess=SimpleNamespace(Popen=launch,PIPE=subprocess.PIPE,
        DEVNULL=subprocess.DEVNULL,TimeoutExpired=subprocess.TimeoutExpired)
    return client.init


if __name__=='__main__':
    install()
    from project.run_scripts.experiment_tracking.worker import main
    main()
