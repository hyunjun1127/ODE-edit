"""Exact owner cancellation for USER mask rerun, unrelated sources immutable."""
from pathlib import Path
import json,subprocess
from datetime import datetime,timezone
from official.experiments.prepare import read,write_new,file_sha
from project.run_scripts.server2_qwen_submit import verify,fields,check,cmd

OLD=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-native-eval-r2')
EVAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server2/zsre-2k-reeval-20261009/registration-r1')
OUT=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/control-r1')
GPU={'61954','61956','61958','61960','61966','61968','61970','61972'}
TARGETS=GPU|{'61955','61957','61959','61961','61967','61969','61971','61973','61974'}
def now():return datetime.now(timezone.utc).isoformat()
def exact(j,root,eval_job=False):
    script=j['method']+'.sh' if eval_job else j['cell']+'-'+j['kind']+'.sh'
    try:raw,f=fields(j['job_id'])
    except subprocess.CalledProcessError:
        raw=cmd(['sacct','-X','-P','-n','-j',j['job_id'],'--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%500'])
        p=raw.splitlines()[0].split('|')
        check(p[0]==j['job_id'] and p[1]==j['name'] and p[2]=='janghj' and p[4]=='server2' and p[5]==str(root),'TERMINAL_ACCOUNTING_IDENTITY')
        check(p[3] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'),'NOT_TERMINAL')
        return dict(at=now(),raw=raw,fields=dict(JobState=p[3]),command_evidence='SEALED_REGISTRATION_ONLY_TERMINAL_NO_MUTATION')
    check(f.get('UserId','').startswith('janghj(') and f.get('ReqNodeList')=='server2'
        and f.get('Command')==str(root/'scripts'/script) and f.get('WorkDir')==str(root)
        and f.get('JobName')==j['name'],'OWNER_SOURCE_COMMAND')
    return dict(at=now(),raw=raw,fields=f)
def run():
    check(not OUT.exists(),'NO_DUPLICATE_CONTROL')
    verify(OLD)
    check(read(OLD/'source-lock.json')['code_commit']=='5503935821b0ececb4aef09a5bccb5308879a6b5','OLD_SOURCE')
    jobs=read(OLD/'released.json')['jobs'];ejobs=read(EVAL/'released.json')['jobs']
    before={j['job_id']:exact(j,OLD) for j in jobs}
    ebefore={j['job_id']:exact(j,EVAL,True) for j in ejobs}
    write_new(OUT/'before.json',dict(at=now(),jobs=before,eval=ebefore,authority='USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1'))
    # Four protected eval heads refer to cancelled resource frontiers. Hold only
    # pending heads until the new resource-only graph is bound; never cancel.
    for j in ejobs:
        if j['job_id'] not in {'61942','61943','61944','61945'}:continue
        b=exact(j,EVAL,True);check(b['fields']['JobState']=='PENDING','PROTECTED_EVAL_NOT_PENDING')
        cmd(['scontrol','hold',j['job_id']]);write_new(OUT/'eval-holds'/(j['job_id']+'.json'),dict(before=b,after=exact(j,EVAL,True)))
    # Keep BLUE archive jobs but remove their unrelated, soon-cancelled serial edge.
    for jid,dep in [('61963','afterany:61962:61953'),('61965','afterany:61964:61963')]:
        j=next(j for j in jobs if j['job_id']==jid);b=exact(j,OLD)
        check(b['fields']['JobState']=='PENDING','PROTECTED_ARCHIVE_NOT_PENDING')
        cmd(['scontrol','update','JobId='+jid,'Dependency='+dep])
        write_new(OUT/'protected-archive-rebind'/(jid+'.json'),dict(before=b,after=exact(j,OLD),science_changed=False))
    for j in reversed(jobs):
        if j['job_id'] not in TARGETS:continue
        b=exact(j,OLD)
        if b['fields']['JobState']=='PENDING':cmd(['scontrol','hold',j['job_id']])
    cancelled=[];kept=[]
    for j in reversed(jobs):
        if j['job_id'] not in TARGETS:continue
        b=exact(j,OLD)
        if b['fields']['JobState'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'):
            kept.append(dict(job=j,before=b,action='KEEP_TERMINAL'));continue
        check(b['fields']['JobState'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'),'UNKNOWN_STATE')
        cmd(['scancel',j['job_id']]);a=exact(j,OLD)
        cancelled.append(dict(job=j,before=b,after=a))
        write_new(OUT/'cancelled'/(j['job_id']+'.json'),cancelled[-1])
    write_new(OUT/'cancellation.json',dict(at=now(),cancelled=cancelled,kept=kept,
        protected_GPU=['61898','61900','61962','61964'],protected_eval_source_unchanged=True,raw_CP_KEEP=True))
    print(json.dumps(dict(cancelled=[x['job']['job_id'] for x in cancelled],kept=[x['job']['job_id'] for x in kept])))
if __name__=='__main__':run()
