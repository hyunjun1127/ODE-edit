"""Private SDK sidecar; stdout/stderr suppressed by parent, control FD JSON only."""
import json
import itertools
import math
import os
from pathlib import Path
import resource
import sys
from urllib.parse import urlsplit
from .generation_tracking_schema import (ENTITY, PROJECT, SDK_VERSION, config, metrics,
    identifier, endpoint, require, job_identity, run_name, PREFIXES,
    GenerationProgressAxis, PROGRESS_KEYS, PROGRESS_METADATA, GEN_KEYS, REPAIR_TASK)
from project.run_scripts.experiment_tracking.method import define_axes, AxisState
from project.run_scripts.experiment_tracking.readback import verify_last_rows


def settings(sdk, base_url):
    return sdk.Settings(mode='online', base_url=endpoint(base_url), console='off', save_code=False,
        disable_code=True, disable_git=True, disable_job_creation=True, x_disable_meta=True,
        x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
        quiet=True, silent=True, init_timeout=30, login_timeout=10, x_service_wait=15,
        ignore_globs=['*.log','**/*.log','requirements.txt','wandb-metadata.json','code/**','*.patch','*.diff'])


def verify_last_progress(sdk, base_url, run_id, cfg, name, expected):
    """One finish readback of the last accepted progress row, not completion."""
    scope='last logged generation progress row and exact phase only; not full history or scientific completion'
    if expected is None:
        return dict(status='NOT_MEASURED',scope=scope,rows=0)
    try:
        api=sdk.Api(overrides={'base_url':base_url},timeout=10)
        remote=api.run(ENTITY+'/'+PROJECT+'/'+run_id)
        require(remote.id==run_id and remote.name==name,'GEN_PROGRESS_READBACK_IDENTITY')
        require(all(remote.config.get(k)==v for k,v in cfg.items()),'GEN_PROGRESS_READBACK_CONFIG')
        values=metrics(expected['values'],scientific=True);step=expected['step']
        rows=list(itertools.islice(remote.scan_history(keys=['_step',*values],
            min_step=step,max_step=step+1,page_size=2),3))
        require(len(rows)==1 and rows[0].get('_step')==step,'GEN_PROGRESS_READBACK_ROW')
        row=rows[0]
        require(all(type(row.get(k)) is str and row[k]==values[k]
                    for k in PROGRESS_METADATA),'GEN_PROGRESS_READBACK_METADATA')
        require(all((type(v) is int and type(row.get(k)) is int and row[k]==v)
                or (type(v) is float and type(row.get(k)) in (int,float)
                    and math.isfinite(row[k]) and math.isclose(row[k],v,rel_tol=1e-9,abs_tol=1e-8))
                for k,v in values.items() if k in PROGRESS_KEYS),'GEN_PROGRESS_READBACK_VALUES')
        return dict(status='REMOTE_BOUNDED_PROGRESS_VERIFIED',scope=scope,rows=1,
            transport_step=step,generation_step=values['generation_progress/step'],
            phase=values['phase'],route=values['route'],model=values['model'],job_id=values['job_id'])
    except Exception:
        return dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE',scope=scope,rows=None)


def session(sdk, request, commands, emit):
    """Same production assembly is exercised with a fake SDK in CPU tests."""
    require(sdk.__version__==SDK_VERSION, 'SDK_VERSION_REVIEW_REQUIRED')
    cfg = config(request['config']); identifier(request['run_id'])
    identity=job_identity(cfg); name=run_name(cfg)
    scientific='metric_schema' in cfg;axis=AxisState();progress_axis=GenerationProgressAxis()
    require(type(request['smoke']) is bool, 'SMOKE_FLAG')
    options = settings(sdk, request['base_url'])
    sdk.setup(settings=options)
    if not sdk.login(host=request['base_url'], prompt=False, verify=True):
        emit(dict(status='SETUP_READY_NEEDS_USER_LOGIN',sdk_version=sdk.__version__))
        return
    run = sdk.init(entity=ENTITY, project=PROJECT, group=cfg['task_id'], id=request['run_id'],
        name=name, config=cfg, mode='online',
        resume='never', dir=request['spool'], save_code=False, settings=options)
    require(run is not None and not run.offline and run.id==request['run_id'], 'NOT_ONLINE')
    if scientific:
        define_axes(run)
        for prefix in PREFIXES:run.define_metric(prefix+'/*',step_metric='edits',step_sync=False)
        if cfg['task_id']==REPAIR_TASK:
            run.define_metric('generation_progress/step')
            run.define_metric('generation_progress/*',step_metric='generation_progress/step',step_sync=False)
    # Remote access verification before declaring startup ready / loading a GPU model.
    api = sdk.Api(overrides={'base_url':request['base_url']}, timeout=15)
    remote = api.run(ENTITY+'/'+PROJECT+'/'+run.id)
    require(remote.id == run.id, 'REMOTE_RUN_IDENTITY')
    require(remote.name == name, 'REMOTE_JOB_NAME_MISMATCH')
    require(all(remote.config.get(k)==v for k,v in identity.items()), 'REMOTE_JOB_CONFIG_MISMATCH')
    require(all(remote.config.get(k)==v for k,v in cfg.items()),'REMOTE_CONFIG_MISMATCH')
    url = run.url
    parsed = urlsplit(url)
    require(parsed.scheme=='https' and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment, 'RUN_URL')
    emit(dict(status='READY_ONLINE',run_id=run.id,url=url,sdk_version=sdk.__version__,run_name=name,job_identity=identity,config=cfg))
    failures = 0; count = 0;next_step=0;last_rows={};last_progress=None;final_generation_logged=False
    for message in commands:
        if message['op']=='log':
            payload=metrics(message['values'],scientific=scientific,
                            canonical=(cfg['task_id']==REPAIR_TASK),
                            final_only=cfg['attempt']=='final-generation-v1');step=message['step']
            final_point=cfg['attempt']=='final-generation-v1' and bool(GEN_KEYS & payload.keys())
            require(not final_point or not final_generation_logged,'GEN_FINAL_SINGLE_PAYLOAD_ONLY')
            require(step is None or type(step) is int and step>=0, 'INVALID_STEP')
            if scientific:
                if PROGRESS_KEYS & payload.keys():
                    require(cfg['task_id']==REPAIR_TASK and payload['job_id']==cfg.get('job_id'),
                            'GEN_PROGRESS_CALLER_IDENTITY')
                axis.accept(payload)
                progress_axis.accept(payload)
                require(step is None or step>=next_step,'TRANSPORT_STEP_DECREASE')
            if request['smoke']:
                require(count<3 and set(payload)=={'setup_ok','step'}, 'SMOKE_THREE_SCALARS_ONLY')
            count+=1
            if final_point:final_generation_logged=True
            try:
                run.log(payload, step=step)
                actual_step=next_step if step is None else step
                next_step=actual_step+1
                if scientific:
                    kind='evaluation' if 'edits' in payload else 'fit' if 'fit/global_candidate' in payload else None
                    if kind:last_rows[kind]=dict(step=actual_step,values=payload)
                    if PROGRESS_KEYS & payload.keys():last_progress=dict(step=actual_step,values=payload)
                emit(dict(status='LOGGING_ACCEPTED',points=count,delivery='SDK_ASYNC_NOT_REMOTE_ACK'))
            except Exception:
                failures+=1
                emit(dict(status='LOGGING_DEGRADED',points=count,failures=failures))
        elif message['op']=='finish':
            code=message['exit_code'];require(type(code) is int and code in (0,1),'EXIT_CODE')
            try:
                run.finish(exit_code=code)
            except Exception:
                emit(dict(status='LOGGING_DEGRADED_FINISH',points=count,failures=failures+1));return
            status='FINISHED_UNVERIFIED' if failures else 'FINISHED_SDK_FLUSHED'
            readback=verify_last_rows(sdk,request['base_url'],run.id,cfg,name,last_rows) if scientific else None
            progress_readback=verify_last_progress(sdk,request['base_url'],run.id,cfg,name,last_progress) if scientific else None
            if request['smoke']:
                require(count==3 and failures==0,'SMOKE_COUNT')
                api = sdk.Api(overrides={'base_url':request['base_url']}, timeout=15)
                remote=api.run(ENTITY+'/'+PROJECT+'/'+run.id)
                history=list(remote.scan_history(keys=['setup_ok','step'],page_size=3))
                require(len(history)==3 and [r['step'] for r in history]==[0,1,2]
                        and all(r['setup_ok']==1 for r in history), 'READBACK_NOT_THREE_POINTS')
                names=[f.name for f in remote.files()]
                forbidden=lambda n: n.endswith(('.py','.patch','.diff','.log','.pt','.pth','.npz')) or n in ('requirements.txt','wandb-metadata.json') or n.startswith('code/')
                require(not any(forbidden(n) for n in names),'REMOTE_UPLOAD_BOUNDARY')
                status='READY_ONLINE_VERIFIED'
            emit(dict(status=status,run_id=run.id,url=url,points=count,failures=failures,
                      remote_points=3 if request['smoke'] else None,run_name=name,job_identity=identity,
                      method_readback=readback,generation_progress_readback=progress_readback,
                      scientific_completion_claim=False));return
        else:
            raise ValueError('UNKNOWN_OPERATION')
    emit(dict(status='LOGGING_DEGRADED_PARENT_CLOSED',points=count))


def main():
    # Limits apply only to this isolated telemetry process and its SDK children.
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    control = os.fdopen(int(sys.argv[1]),'w',buffering=1)
    def emit(value):
        control.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        request=json.loads(sys.stdin.readline())
        import wandb
        commands=(json.loads(line) for line in sys.stdin)
        session(wandb,request,commands,emit)
    except Exception:
        # Never stringify SDK errors: they may embed requests/credentials/config.
        emit(dict(status='LOGGING_BLOCKED_OR_FAILED',error='REDACTED_SDK_OR_PROTOCOL_ERROR'))


if __name__=='__main__':
    main()
