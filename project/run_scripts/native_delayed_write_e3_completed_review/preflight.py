import datetime
import os
import subprocess
from .common import *

def main():
    assert not (LOCAL/'accounting.json').exists(), 'NO_DUPLICATE_SCHEDULER_QUERY'
    assert subprocess.check_output(['hostname'],text=True).strip()=='server4'
    assert os.environ['CODEX_THREAD_ID']=='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
    assert sha(ATTEMPT/'execution.lock.json')=='ff2c8fd6b685a933b980b2255504200b1744a8827dbae128ee1aa4da38bd3b71'
    assert sha(ATTEMPT/'submission.json')=='c6569653df755513837cd0b80a57a1320c12e52c27a038bb2f31c1aa7376054b'
    sub=read(ATTEMPT/'submission.json');assert sub['jobs']==dict(collector='52824',science='52823')
    fields='JobIDRaw,JobName%64,User,State,ExitCode,ElapsedRaw,AllocTRES,ReqTRES,Start,End,NodeList,Timelimit'
    cmd=['sacct','-X','-j','52823,52824','--noheader','--parsable2','--format='+fields]
    r=subprocess.run(cmd,text=True,capture_output=True,check=True)
    q=subprocess.run(['squeue','-j','52823,52824','--noheader','--format=%i|%j|%u|%T|%R|%E'],text=True,capture_output=True)
    names={'52823':'odeedit_delayed_E3_science_s4','52824':'odeedit_delayed_E3_collector_s4'}
    parsed=[]
    for line in r.stdout.strip().splitlines():
        v=line.split('|');row=dict(zip([x.split('%')[0] for x in fields.split(',')],v))
        assert row['JobIDRaw'] in names and row['User']=='janghj' and row['JobName']==names[row['JobIDRaw']]
        parsed.append(row)
    assert {r['JobIDRaw'] for r in parsed}==set(names)
    result=dict(instruction_id=INSTRUCTION,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=cmd,parent_rows=parsed,squeue=q.stdout,squeue_stderr=q.stderr,squeue_exit=q.returncode,
        exact_scope_only=True,new_GPU=0,mutation=0)
    save(LOCAL/'accounting.json',result);save(AUDIT/'accounting.json',result)
    prior=REPO/'audits/servers/server4/native-delayed-write-e3-20260924-v1/full-read-and-inputs.json'
    old=read(prior);verified=[]
    for m in old['design_members']:
        p=Path(m['path']);assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];verified.append(m)
    lock=read(ATTEMPT/'execution.lock.json');members=[]
    for m in lock['members']:
        p=Path(m['path']);assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];members.append(m)
    cp=[]
    for family,entries in lock['checkpoints'].items():
        for batch,m in entries.items():
            st=Path(m['path']).stat()
            assert (st.st_size,st.st_mtime_ns,st.st_ino)==(m['bytes'],m['mtime_ns'],m['inode'])
            cp.append(dict(family=family,batch=batch,path=m['path'],bytes=st.st_size,prior_sha256=m['sha256'],current_stat_match=True,full_rehash=False))
    sources=[record(p) for p in sorted((ROOT/'execution-source-r1/project/run_scripts/native_delayed_write_e3').glob('*')) if p.is_file()]
    binding=dict(instruction_id=INSTRUCTION,prior_FULL_READ=record(prior),authority_design_members=verified,
        frozen_lock=record(ATTEMPT/'execution.lock.json'),frozen_submission=record(ATTEMPT/'submission.json'),
        frozen_members_verified=len(members),execution_sources=sources,checkpoints=cp,
        review_envelope=record(REPO/'messages/head/2026-09-24-delayed-write-e3-completed-review-sh4.md'),
        protocol=record(REPO/'PROTOCOL.md'),fullread_reuse='exact prior design bytes verified; new review envelope fully read',
        model_calls=0,Slurm_mutation=0,raw_modified=False,independent_agent=False)
    save(AUDIT/'input-source-binding.json',binding)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if any(r['State'] in ('PENDING','RUNNING','COMPLETING') for r in parsed):print('NOT_TERMINAL_STOP_NO_POLL')

if __name__=='__main__':main()
