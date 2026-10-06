"""Server2 one-shot CPU setup smoke, not the shared experiment logger.

Never print credentials, provider objects, exceptions, environment or SDK logs.
The shared production logger remains owned by SH1.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/wandb-setup')
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='initial');name=p.parse_args().attempt
    if '/' in name or name in ('.','..'):raise ValueError('INVALID_ATTEMPT')
    out=ROOT/name;out.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    signal.alarm(90)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',WANDB_CONSOLE='off',
        WANDB_SAVE_CODE='false',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_SILENT='true',
        WANDB_MODE='online',WANDB_DIR=str(out/'spool'))
    (out/'spool').mkdir()
    import wandb
    from wandb.sdk.lib import wbauth
    policy=dict(entity='wkdguswns2256',project='layer allocation',mode='online',console='off',save_code=False,
        disable_code=True,disable_git=True,x_disable_stats=True,x_disable_meta=True,init_timeout=30,silent=True,capture_loggers=None)
    settings=wandb.Settings(**policy)
    result=dict(nonce='USER-GH-ALL-SH-WANDB-REALTIME-20261006-SERVER2',server='server2',
        session='01a0493a-074c-7f91-9a13-769116326fef',sdk=wandb.__version__,python=sys.executable,
        sdk_path=wandb.__file__,SDK_READY=True,privacy_settings_validated=True,entity=policy['entity'],project=policy['project'],
        mode='online',status='SETUP_READY_NEEDS_USER_LOGIN',authenticated=False,online_smoke=False,remote_points=0,
        GPU=0,model=0,new_Slurm=0,existing_jobs_changed=0,independent_reviewer=0,common_helper_implemented_here=False)
    run=None
    try:
        # SDK resolves existing auth; neither its credential object nor key is serialized.
        auth=wbauth.authenticate_session(host=wbauth.HostUrl(settings.base_url),source='server2 CPU smoke',
            no_offline=True,no_create=True,prompt=False,verify=False)
        result['credential_present']=auth is not None
        result['API_base_url_source']='SDK_DEFAULT'
        if auth is None:return
        run=wandb.init(entity=policy['entity'],project=policy['project'],mode='online',job_type='setup-smoke',
            name='server2-setup-smoke',group='wandb-realtime-setup',config={'server':'server2'},settings=settings)
        result['authenticated']=True;result['run_id']=run.id;result['run_url']=run.url
        path=f"{policy['entity']}/{policy['project']}/{run.id}"
        for step in range(3):run.log({'step':step,'setup_ok':1},step=step)
        run.finish();run=None
        remote=wandb.Api(timeout=20).run(path)
        rows=list(remote.scan_history(keys=['step','setup_ok'],page_size=10))
        if len(rows)!=3 or [r['step'] for r in rows]!=[0,1,2] or any(r['setup_ok']!=1 for r in rows):
            result['status']='ONLINE_READBACK_NOT_VERIFIED'
        else:result.update(status='READY_ONLINE_VERIFIED',online_smoke=True,remote_points=3)
    except Exception as e:
        # Exception messages/SDK objects may carry credentials; type only.
        result.update(status='LOGGING_BLOCKED',error_type=type(e).__name__)
    finally:
        if run is not None:
            try:run.finish(exit_code=1)
            except Exception:result['finish_error']=True
        result['dependencies']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
        path=out/'receipt.json'
        with path.open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='dependencies'},sort_keys=True))

if __name__=='__main__':main()
