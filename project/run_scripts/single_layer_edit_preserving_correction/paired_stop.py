"""Explicit one-shot cancellation of this attempt's allowlisted M children.

Invoked by the active agent after T failure; not a daemon or automatic retry.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from .common import member,write
from .control import call


BAD=('FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE')


def checked_targets(text,allowlist):
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        job,owner,state,name,*rest=line.split('|')
        if job not in allowlist or owner!='janghj' or name!='odeedit_enfc_M_s4':
            raise ValueError('CANCEL_TARGET_NOT_EXACT_LINKED_M:'+line)
        if state in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'):
            result.append(job)
        else:raise ValueError('UNEXPECTED_LIVE_M_STATE:'+line)
    return result


def stop(lockpath):
    lock=json.loads(lockpath.read_text());parent=lockpath.parent
    submission=json.loads((parent/'submission.json').read_text())
    if not lock.get('parallel_override') or lock['T_job']!=49928 or submission['T_job']!=49928:
        raise ValueError('NO_EXACT_FAILCANCEL_AUTHORITY')
    Tstate=call(['sacct','-n','-X','-j','49928','--format=JobID,User,State,ExitCode,ElapsedRaw','-P'])
    failure=Path(lock['T_failure_path'])
    if not failure.exists() and not any(s in Tstate for s in BAD):raise ValueError('T_HAS_NOT_FAILED_NO_CANCEL')
    queue=call(['squeue','-h','-r','-j',submission['job'],'-o','%i|%u|%T|%j|%b|%R'])
    ids=checked_targets(queue,submission['cancel_allowlist'])
    request=write(parent/'paired-stop-request.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        T_job=49928,T_scheduler=Tstate,T_failure=None if not failure.exists() else member(failure),
        M_array=submission['job'],M_queue=queue,exact_cancel_ids=ids,linked_submission=member(parent/'submission.json'),
        authority=lock['parallel_override'],completed_M_excluded=True,other_jobs_changed=0,
        rollback='NOT_VERIFIED; cancellation is not evidence of cleanup',automatic_M_resubmit=False))
    result=subprocess.run(['scancel',*ids],text=True,capture_output=True) if ids else None
    ref=write(parent/'paired-stop-cancel-result.json',dict(request=request,exact_ids=ids,
        returncode=0 if result is None else result.returncode,
        stdout='' if result is None else result.stdout,stderr='' if result is None else result.stderr,
        terminal_confirmation='PENDING_AGENT_EXACT_BOUNDED_CHECK',files_deleted=0))
    if result is not None and result.returncode:raise RuntimeError('EXACT_CANCEL_REQUEST_FAILED')
    print(json.dumps(ref))


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);a=p.parse_args();stop(a.lock)


if __name__=='__main__':main()
