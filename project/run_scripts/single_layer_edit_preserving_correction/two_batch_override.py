"""Explicit user reduction: preserve two running M children, cancel pending rest."""
import datetime
import json
import subprocess
from .common import ROOT,member,write
from .control import call


def main():
    out=ROOT/'receipts/two-batch-override-r1';authority=out/'user-recall.txt'
    subpath=ROOT/'M/attempt-metadata-r1/submission.json';sub=json.loads(subpath.read_text())
    if sub['job']!='50050' or not authority.exists():raise ValueError('EXACT_USER_AUTHORITY')
    if (out/'cancel-request.json').exists():raise ValueError('NO_DUPLICATE_CANCEL')
    queue=call(['squeue','-h','-r','-j','50050','-o','%i|%u|%T|%j|%b|%R'])
    rows={x.split('|')[0]:x.split('|') for x in queue.splitlines()};ids=[]
    for i in range(2,10):
        jid='50050_'+str(i);r=rows.get(jid)
        if r is None:raise ValueError('EXPECTED_PENDING_CHILD_MISSING:'+jid)
        if r[1:4]!=['janghj','PENDING','odeedit_enfc_M_s4']:raise ValueError('NOT_OWNED_PENDING:'+jid)
        ids.append(jid)
    if any(rows.get('50050_'+str(i),[])[1:4]!=['janghj','RUNNING','odeedit_enfc_M_s4'] for i in (0,1)):
        raise ValueError('TWO_RUNNING_MAPPING_CHANGED')
    request=write(out/'cancel-request.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authority=member(authority),submission=member(subpath),queue=queue,exact_pending_ids=ids,
        preserve_running=['50050_0','50050_1'],remaining_scope='2 independent cold100 episodes, 8arms each, 16finalL4',
        historical_submission=10,other_jobs_changed=0,files_deleted=0))
    # Scheduler-side state filter prevents cancelling a child that raced to RUNNING.
    result=subprocess.run(['scancel','--state=PENDING',*ids],capture_output=True,text=True)
    receipt=write(out/'cancel-result.json',dict(request=request,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,exact_requested_ids=ids,state_filter='PENDING'))
    if result.returncode:raise RuntimeError('PENDING_CANCEL_REQUEST_FAILED')
    queue_after=call(['squeue','-h','-r','-j','50050','-o','%i|%u|%T|%j|%b|%R'])
    if any(x.split('|')[0] not in ('50050_0','50050_1') for x in queue_after.splitlines()):
        raise ValueError('EXTRA_CHILD_REMAINS_AFTER_CANCEL')
    accounting=call(['sacct','-n','-P','-X','-j','50050','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,Start,End'])
    final=write(out/'confirmed-state.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cancel=receipt,queue=queue_after,accounting=accounting,preserved=['50050_0','50050_1'],
        cancelled_requested=ids,remaining_episodes=2,remaining_requested_population=200,remaining_arm_endpoints=16,
        initial_gate_not_observed=True,source_locks_immutable=True,new_submit=0,new_T=0,S_R_L=0,deleted_or_moved=0))
    print(json.dumps(dict(receipt=final,queue=queue_after,accounting=accounting)))


if __name__=='__main__':main()
