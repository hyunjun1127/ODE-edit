"""One CPU-only run of exactly three synthetic points, then remote readback."""
import argparse
import json
from pathlib import Path
import sys
from .client import init,LoggingBlocked
from .schema import load_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--server',required=True)
    p.add_argument('--source-sha',required=True);a=p.parse_args()
    settings=load_env(a.env_file);a.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    result=dict(server=a.server,entity=settings['WANDB_ENTITY'],project=settings['WANDB_PROJECT'],
        base_url=settings['WANDB_BASE_URL'],sdk_python=settings['ODEEDIT_WANDB_PYTHON'],
        GPU=0,new_slurm=0,maximum_points=3,offline_pass=False,credentials_recorded=False)
    tracker=None
    try:
        tracker=init(env_file=a.env_file,spool=a.out/'spool',smoke=True,
            config=dict(server=a.server,task_id='wandb-realtime-setup',arm='cpu-smoke',attempt=a.out.name,source_sha=a.source_sha))
        for i in range(3):
            if not tracker.log({'setup_ok':1,'step':i},step=i):raise RuntimeError('POINT_REJECTED')
        outcome=tracker.finish(timeout=50)
        result.update(outcome)
        result['status']='READY_ONLINE_VERIFIED' if outcome.get('status')=='READY_ONLINE_VERIFIED' and not outcome['dropped_points'] else 'LOGGING_BLOCKED_SMOKE'
    except LoggingBlocked as e:
        # LoggingBlocked has only fixed internal status codes, never an SDK message.
        result['status']=str(e)
        if (a.out/'spool/receipt.json').is_file():
            result['sdk_version']=json.loads((a.out/'spool/receipt.json').read_text()).get('result',{}).get('sdk_version')
    except Exception:
        result['status']='LOGGING_BLOCKED_SMOKE'
    finally:
        if tracker is not None and not tracker.closed:
            tracker.finish(exit_code=1)
    with (a.out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    return 0 if result['status']=='READY_ONLINE_VERIFIED' else 2


if __name__=='__main__':sys.exit(main())
