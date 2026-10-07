"""Isolated W0-only SDK worker, lineage shared worker SHA784f2406b11d.

Bounded API startup+finish only; no models, evaluator, raw upload or polling.
"""
import itertools
import json
import math
import os
import resource
import sys
from urllib.parse import urlsplit
from project.run_scripts.experiment_tracking.worker import settings
from .schema import (ENTITY,PROJECT,SDK_VERSION,config,metrics,identifier,require,
                     job_identity,run_name,AxisState,define_axes,performance)
from .tracking import chosen_run_id


def readback(sdk,base_url,run_id,cfg,name,expected):
    """At most 26 scalar curve rows, not a recurring/full-history monitor."""
    from .curves import curve_coverage
    scope='W0 first2000 +20 current +4 all-seen curves; scalar identity/value readback only'
    try:
        local=curve_coverage([row['values'] for row in expected],cfg)
        require(len(expected)==21,'W0_LOCAL_CURVE_ROWS')
        remote=sdk.Api(overrides={'base_url':base_url},timeout=10).run(ENTITY+'/'+PROJECT+'/'+run_id)
        require(remote.id==run_id and remote.name==name and all(remote.config.get(k)==v for k,v in cfg.items()),
                'W0_REMOTE_IDENTITY')
        first=expected[0]['step'];last=expected[-1]['step']
        require([row['step'] for row in expected]==list(range(first,first+21)),
                'W0_FINAL_CURVE_TRANSPORT_BLOCK')
        # SDK keys projects the selected columns, and W0/current/prefix rows
        # deliberately have different metric sets. Read all strict scalar
        # columns only in this final 21-step block, not prior progress/history.
        rows=list(itertools.islice(remote.scan_history(min_step=first,max_step=last+1,
                                                      page_size=26),26))
        require(len(rows)==21,'W0_REMOTE_CURVE_COVERAGE')
        require([row['edits'] for row in rows]==list(range(0,2001,100)),'W0_REMOTE_AXIS_ORDER')
        for actual,item in zip(rows,expected):
            require(actual.get('_step')==item['step'],'W0_REMOTE_TRANSPORT_STEP')
            for key,value in item['values'].items():
                got=actual.get(key)
                if type(value) in (int,float):
                    require(type(got) in (int,float) and math.isclose(got,value,rel_tol=1e-9,abs_tol=1e-8),
                            'W0_REMOTE_SCALAR_VALUE')
                else:
                    require(type(got) is type(value) and got==value,'W0_REMOTE_REFERENCE_STATE')
        # SDK-owned _step/_timestamp/_runtime are not metric payload fields.
        actual_coverage=curve_coverage([{key:actual[key] for key in item['values']}
                                      for actual,item in zip(rows,expected)],cfg)
        return dict(status='REMOTE_W0_CURVE_COVERAGE_VERIFIED',scope=scope,curve_rows=21,
            coverage=actual_coverage,local_coverage=local,scientific_completion_claim=False)
    except Exception:
        return dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE',scope=scope,curve_rows=None,
            scientific_completion_claim=False)


def session(sdk,request,commands,emit):
    require(sdk.__version__==SDK_VERSION,'SDK_VERSION_REVIEW_REQUIRED')
    cfg=config(request['config']);chosen_run_id(request['run_id'])
    identity=job_identity(cfg);name=run_name(cfg);axis=AxisState()
    options=settings(sdk,request['base_url'])
    sdk.setup(settings=options)
    if not sdk.login(host=request['base_url'],prompt=False,verify=True):
        emit(dict(status='SETUP_READY_NEEDS_USER_LOGIN',sdk_version=sdk.__version__));return
    run=sdk.init(entity=ENTITY,project=PROJECT,group=cfg['task_id'],id=request['run_id'],
        name=name,config=cfg,mode='online',resume='never',dir=request['spool'],save_code=False,settings=options)
    require(run is not None and not run.offline and run.id==request['run_id'],'NOT_ONLINE')
    define_axes(run)
    remote=sdk.Api(overrides={'base_url':request['base_url']},timeout=15).run(ENTITY+'/'+PROJECT+'/'+run.id)
    require(remote.id==run.id and remote.name==name and all(remote.config.get(k)==v for k,v in cfg.items()),
            'REMOTE_W0_CONFIG_IDENTITY')
    url=run.url;parsed=urlsplit(url)
    require(parsed.scheme=='https' and parsed.hostname and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment,'RUN_URL')
    emit(dict(status='READY_ONLINE',run_id=run.id,url=url,sdk_version=sdk.__version__,
        run_name=name,job_identity=identity,config=cfg))
    failures=0;count=0;next_step=0;expected=[];rejections=0
    for message in commands:
        if message['op']=='log':
            try:
                payload=metrics(message['values'],cfg);step=message['step']
                require(step is None or type(step) is int and step>=next_step,'TRANSPORT_STEP_DECREASE')
                axis.check(payload)
            except Exception:
                rejections+=1
                emit(dict(status='LOGGING_REJECTED',rejected_points=rejections,
                    code='W0_SCHEMA_OR_AXIS_REJECTED',delivery='NOT_ACCEPTED'));continue
            count+=1
            actual_step=next_step if step is None else step;next_step=actual_step+1
            axis.accept(payload)
            if performance(payload):expected.append(dict(step=actual_step,values=payload))
            try:
                run.log(payload,step=step)
                emit(dict(status='LOGGING_ACCEPTED',points=count,delivery='SDK_ASYNC_NOT_REMOTE_ACK'))
            except Exception:
                failures+=1
                emit(dict(status='LOGGING_DEGRADED',points=count,failures=failures,
                    delivery='REMOTE_ACK_NOT_ESTABLISHED'))
        elif message['op']=='finish':
            code=message['exit_code'];require(type(code) is int and code in (0,1),'EXIT_CODE')
            try:run.finish(exit_code=code)
            except Exception:
                emit(dict(status='LOGGING_DEGRADED_FINISH',points=count,failures=failures+1));return
            status='FINISHED_UNVERIFIED' if failures or rejections else 'FINISHED_SDK_FLUSHED'
            verification=readback(sdk,request['base_url'],run.id,cfg,name,expected)
            emit(dict(status=status,run_id=run.id,url=url,points=count,failures=failures,
                rejected_points=rejections,run_name=name,job_identity=identity,
                cohort_readback=verification,method_readback=verification,
                scientific_completion_claim=False));return
        else:
            raise ValueError('UNKNOWN_OPERATION')
    emit(dict(status='LOGGING_DEGRADED_PARENT_CLOSED',points=count))


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    control=os.fdopen(int(sys.argv[1]),'w',buffering=1)
    def emit(value):control.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        request=json.loads(sys.stdin.readline())
        import wandb
        session(wandb,request,(json.loads(line) for line in sys.stdin),emit)
    except Exception:
        emit(dict(status='LOGGING_BLOCKED_OR_FAILED',error='REDACTED_SDK_OR_PROTOCOL_ERROR'))


if __name__=='__main__':main()
