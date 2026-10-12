"""USER-requested finite read-only milestone watcher; never retries science.

CPU-only, one job, at most 48h. Deliver each milestone once to the authorized
root thread; failed transport is retained locally, not called delivered.
"""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--peer',type=Path,required=True)
    p.add_argument('--target-session',required=True)
    a=p.parse_args()
    if not a.job_id.isdigit():raise ValueError('job ID')
    root=a.attempt/'watch';root.mkdir(exist_ok=False)
    out=a.attempt/'output';start=time.monotonic();seen=set()
    def emit(key,data):
        if key in seen:return
        seen.add(key)
        event=dict(event=key,job_id=a.job_id,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)
        path=root/(key+'.json');path.write_text(json.dumps(event,indent=2))
        message=root/(key+'.txt')
        message.write_text('SH4 read-only progress, USER-FE-QWEN-CF-FP32-SERVER4-20261012-R1. No new action request. '+json.dumps(event))
        receipt=root/(key+'-delivery.json')
        try:
            subprocess.run(['python3',str(a.peer),'--host','codex-server1','--target-session',a.target_session,
                            '--message',str(message),'--receipt',str(receipt),'--related-active','--delivery-only'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60,check=False)
        except subprocess.TimeoutExpired:
            (root/(key+'-transport-timeout.json')).write_text(json.dumps({'status':'COMMUNICATION_HOLD'}))
    while time.monotonic()-start<48*3600:
        receipt=read(out/'tracking/receipt.json')
        if receipt and receipt.get('startup_readback'):
            emit('online-startup',dict(tracking=receipt['startup_readback'],run_id=receipt.get('run_id'),
                                      note='SDK/startup state, not scientific completion'))
        for label in ('W0-first2000','B01-current','B05-all-seen','B10-all-seen','B15-all-seen','B20-all-seen'):
            raw=read(out/'factual'/f'{label}.json')
            audit=read(out/'raw-audits'/f'{label}.json')
            if raw and audit:
                emit(label,dict(summary=raw['summary'],independent_reducer=audit,raw_path=str(out/'factual'/f'{label}.json'),
                                tracking_status=receipt.get('status') if receipt else 'NOT_OBSERVED'))
        quality=read(out/'QUALITY_EARLY_STOP.json')
        if quality:emit('quality-stop',dict(status=quality['status'],batch=quality['batch'],violations=quality['violations']))
        final=read(a.attempt/'final-cpu-audit.json')
        if final:emit('final-cpu-audit',dict(status=final['status'],W0=final['W0'],W20=final['W20']))
        try:
            state=subprocess.check_output(['sacct','-X','-j',a.job_id,'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True,timeout=20).strip()
        except (subprocess.SubprocessError,OSError):state='ACCOUNTING_UNAVAILABLE'
        terminal=next((r.split('|') for r in state.splitlines() if r.split('|')[0]==a.job_id),None)
        if terminal and terminal[1].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'):
            emit('terminal',dict(accounting=terminal,scientific_complete=bool(final),
                 tracking_finish=read(out/'tracking-finish.json'),failure_log=str(a.attempt/'logs')))
            return
        time.sleep(60)
    emit('watch-expired',dict(status='MONITOR_TIME_LIMIT_NOT_SCIENCE_COMPLETION'))


if __name__=='__main__':main()
