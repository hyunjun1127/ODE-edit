"""One deliberate exact-receipt reconciliation; no broad cancellation/retry."""
from pathlib import Path
import datetime
from .submit import read, verify, command, metadata, dependencies, require, RegistrationError
from official.experiments.prepare import write_new

BASE = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
ROOTS = ('cf-checkpoint-r1', 'alpha-sphere-cf-r1', 'zsre-six-r1')
OUT = BASE/'no-gpu-qualification-r1/cancellation'
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}

def main(*, roots=ROOTS, out=OUT, selected_keys=None):
    OUT = Path(out)
    require(not OUT.exists(), 'CANCELLATION_ATTEMPT_EXISTS_RECONCILE_NO_RETRY')
    targets = {}
    for root in roots:
        receipt = read(BASE/root/'registration-r1/submission.json')
        lock = read(verify(receipt['execution_lock']))
        for key, job in receipt['jobs'].items():
            if selected_keys is not None and key not in selected_keys:continue
            profile = next(p for p in lock['profiles'] if p['key']==key)
            require(profile['mode'] in ('chain','qualification','base_w0','collect'), 'UNEXPECTED_OLD_ROLE')
            verify(lock['launchers'][key])
            targets[str(job)] = dict(key=key, profile=profile, launcher=lock['launchers'][key],
                source=lock['source'], cwd=lock['frozen_source']['directory'], root=root)
    if selected_keys is not None:
        require({v['key'] for v in targets.values()} == set(selected_keys), 'EXACT_SELECTED_KEYS_REQUIRED')
    def inspect(job):
        expected=targets[job]
        try:row=metadata(command(['scontrol','show','job',job,'--oneliner']))
        except RegistrationError as error:
            require('Invalid job id specified' in str(error),'SCHEDULER_QUERY_FAILED')
            rows=command(['sacct','-n','-X','-j',job,'-o','JobID,User,State','-P']).splitlines()
            require(len(rows)==1,'TERMINAL_ACCOUNTING_REQUIRED')
            actual,user,state=rows[0].split('|')[:3]
            require(actual==job and user=='janghj' and state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'), 'PURGED_ACTIVE_JOB_NOT_ALLOWED')
            return dict(JobId=job,JobState=state,Dependency='',source='ORIGINAL_SEALED_RECEIPT_AND_TERMINAL_ACCOUNTING',mutation=False)
        require(row['UserId'].split('(')[0]=='janghj' and row['JobId']==job and
                row['ReqNodeList']=='devbox' and row['Command']==expected['launcher']['path'] and
                row['WorkDir']==expected['cwd'], 'EXACT_JOB_OWNER_SOURCE_COMMAND_MISMATCH')
        verify(expected['launcher'])
        return row
    before={j:inspect(j) for j in targets}
    cost=command(['sacct','-n','-X','-j',','.join(targets),'-o','JobID,State,ElapsedRaw,AllocTRES,Start,End','-P'])
    write_new(OUT/'before.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),targets=targets,states=before,accounting=cost))
    # Hold exact pending descendants before any afterany ancestor is cancelled.
    for job in targets:
        row=inspect(job)
        if row['JobState']=='PENDING':
            command(['scontrol','hold',job])
            write_new(OUT/('held-'+job+'.json'),inspect(job))
    order=[];pending=set(targets)
    while pending:
        parents={p for j in pending for _,p in dependencies(before[j].get('Dependency','')) if p in pending}
        leaves=sorted(pending-parents,key=int,reverse=True)
        require(leaves,'CANCELLATION_DAG_CYCLE')
        order.extend(leaves);pending.difference_update(leaves)
    cancelled=[];kept=[]
    for job in order:
        row=inspect(job)
        if row['JobState'] in ACTIVE:
            command(['scancel',job]);cancelled.append(job)
            write_new(OUT/('cancel-'+job+'.json'),dict(before=row,after=inspect(job)))
        else:kept.append(dict(job=job,state=row['JobState']))
    result=dict(cancelled=cancelled,kept_terminal=kept,after={j:inspect(j) for j in targets},
                artifacts='KEEP',automatic_retry=False)
    write_new(OUT/'result.json',result)
    print({k:result[k] for k in ('cancelled','kept_terminal')})

if __name__=='__main__':main()
