"""One bounded observation and immutable monitor ledger; no background process."""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from .import_assets import ROOT
from .records import save

def observe(job,output,interval,reason):
    result=subprocess.run(['scontrol','show','job',str(job),'-o'],capture_output=True,text=True,timeout=30)
    if result.returncode:
        if 'Invalid job id' not in result.stderr:raise RuntimeError(result.stderr)
        scheduler=subprocess.run(['sacct','-j',str(job),'--format=JobID,JobName,User,State,ExitCode,Elapsed,AllocTRES,NodeList','-P'],
                  capture_output=True,text=True,check=True,timeout=30).stdout
        if len(scheduler.strip().splitlines())<2:raise RuntimeError('SCHEDULER_ACCOUNTING_NOT_YET_AVAILABLE')
    else:scheduler=result.stdout
    now=dt.datetime.now(dt.timezone.utc)
    progress=sorted((output/'progress').glob('*.json'))
    latest=json.loads(progress[-1].read_text()) if progress else None
    row=dict(stage='A',job=job,active_agent_handle='01a04939-f93a-7b50-bca0-65438eab2062',
             last_observed_utc=now.isoformat(),next_check_utc=(now+dt.timedelta(minutes=interval)).isoformat(),
             interval_minutes=interval,reason=reason,scheduler=scheduler,latest_progress=latest,
             output=str(output),last_artifact_utc=dt.datetime.fromtimestamp(progress[-1].stat().st_mtime,dt.timezone.utc).isoformat() if progress else None,
             mechanism='active assistant turn; no background monitor',
             terminal_exists=(output/'terminal.json').exists(),failure_exists=(output/'failure.json').exists())
    path=ROOT/'monitor-ledger'/f'{time.time_ns()}-{job}.json';save(path,row)
    print(json.dumps(dict(path=str(path),job=job,latest_progress=latest,next_check_utc=row['next_check_utc'],
                         terminal_exists=row['terminal_exists'],failure_exists=row['failure_exists']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=int,required=True);p.add_argument('--reason',required=True);a=p.parse_args()
    observe(a.job,a.output,a.interval,a.reason)
