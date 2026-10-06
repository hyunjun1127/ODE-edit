"""Private SDK sidecar; stdout/stderr suppressed by parent, control FD JSON only."""
import json
import os
from pathlib import Path
import resource
import sys
from urllib.parse import urlsplit
from .schema import ENTITY, PROJECT, SDK_VERSION, config, metrics, identifier, endpoint, require


def settings(sdk, base_url):
    return sdk.Settings(mode='online', base_url=endpoint(base_url), console='off', save_code=False,
        disable_code=True, disable_git=True, disable_job_creation=True, x_disable_meta=True,
        x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
        quiet=True, silent=True, init_timeout=30, login_timeout=10, x_service_wait=15,
        ignore_globs=['*.log','**/*.log','requirements.txt','wandb-metadata.json','code/**','*.patch','*.diff'])


def session(sdk, request, commands, emit):
    """Same production assembly is exercised with a fake SDK in CPU tests."""
    require(sdk.__version__==SDK_VERSION, 'SDK_VERSION_REVIEW_REQUIRED')
    cfg = config(request['config']); identifier(request['run_id'])
    require(type(request['smoke']) is bool, 'SMOKE_FLAG')
    options = settings(sdk, request['base_url'])
    sdk.setup(settings=options)
    if not sdk.login(host=request['base_url'], prompt=False, verify=True):
        emit(dict(status='SETUP_READY_NEEDS_USER_LOGIN',sdk_version=sdk.__version__))
        return
    run = sdk.init(entity=ENTITY, project=PROJECT, group=cfg['task_id'], id=request['run_id'],
        name=cfg['server']+'-'+cfg['arm']+'-'+cfg['attempt'], config=cfg, mode='online',
        resume='never', dir=request['spool'], save_code=False, settings=options)
    require(run is not None and not run.offline and run.id==request['run_id'], 'NOT_ONLINE')
    # Remote access verification before declaring startup ready / loading a GPU model.
    api = sdk.Api(overrides={'base_url':request['base_url']}, timeout=15)
    remote = api.run(ENTITY+'/'+PROJECT+'/'+run.id)
    require(remote.id == run.id, 'REMOTE_RUN_IDENTITY')
    url = run.url
    parsed = urlsplit(url)
    require(parsed.scheme=='https' and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment, 'RUN_URL')
    emit(dict(status='READY_ONLINE',run_id=run.id,url=url,sdk_version=sdk.__version__))
    failures = 0; count = 0
    for message in commands:
        if message['op']=='log':
            payload=metrics(message['values']);step=message['step']
            require(step is None or type(step) is int and step>=0, 'INVALID_STEP')
            if request['smoke']:
                require(count<3 and set(payload)=={'setup_ok','step'}, 'SMOKE_THREE_SCALARS_ONLY')
            count+=1
            try:
                run.log(payload, step=step)
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
                      remote_points=3 if request['smoke'] else None));return
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
