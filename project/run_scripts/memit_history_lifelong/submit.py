"""One admission snapshot, held exact inspection, release then STOP (no polling)."""
import argparse,fnmatch,json,os,re,subprocess
from pathlib import Path
from .io import save,file_sha

def run(argv):
    p=subprocess.run(argv,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(f'{argv[0]} exit={p.returncode}: {p.stderr}')
    return p.stdout

def submit(lock_path):
    lock_path=Path(lock_path).resolve();lock=json.loads(lock_path.read_text());attempt=lock_path.parent
    assert not (attempt/'submission.json').exists(),'ALREADY_REGISTERED'
    assert lock['resource']['cap']==1 and lock['resource']['memory_mib']==121856
    owner=run(['id','-un']).strip();assert owner=='janghj'
    queue=run(['squeue','-a','-h','-u',owner,'-w','ubuntu','-o','%i|%j|%T|%b|%N'])
    deps=[]
    for line in queue.splitlines():
        jid,name,state,gres,node=line.split('|')
        assert name!='odeedit_memit_history_10k_s3','EXISTING_ACTIVE_SAME_TASK:'+jid
        if any(fnmatch.fnmatch(name,p) for p in ['odeedit_*','bfode_*','motivation_*','session01_*','project_*']):
            assert re.fullmatch(r'\d+(?:_\d+)?',jid),'UNSUPPORTED_JOB_ID'
            deps.append(jid)
    dependency='afterany:'+':'.join(deps) if deps else None
    save(attempt/'admission.json',dict(owner=owner,queue_resource_snapshot=queue,cap=1,existing_project_jobs=deps,new_gpus=1,dependency=dependency,lock_sha256=file_sha(lock_path)))
    script=Path(lock['source_root'])/'project/run_scripts/memit_history_lifelong/run.sbatch'
    (attempt/'logs').mkdir(exist_ok=True)
    argv=['sbatch','--parsable','--hold','--export=NONE','--output='+str(attempt/'logs/%j.out'),'--error='+str(attempt/'logs/%j.err')]
    if dependency:argv+=['--dependency='+dependency]
    argv += [str(script),lock['source_root'],str(lock_path)]
    answer=run(argv).strip();jid=answer.split(';')[0];assert jid.isdigit()
    save(attempt/'submission.json',dict(job_id=jid,argv=argv,source_commit=lock['source_commit'],lock_sha256=file_sha(lock_path),sbatch_response=answer))
    text=run(['scontrol','show','job',jid,'--oneliner'])
    # Values may contain spaces; Slurm field labels also include ':' and '/'.
    matches=list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=',text));fields={}
    for i,m in enumerate(matches):fields[m.group(1)]=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
    checks={
      'owner':fields.get('UserId','').split('(')[0]==owner,
      'job_id':fields.get('JobId')==jid,
      'name':fields.get('JobName')=='odeedit_memit_history_10k_s3',
      'held':fields.get('JobState')=='PENDING' and fields.get('Priority')=='0',
      'node':fields.get('ReqNodeList')=='ubuntu',
      'partition':fields.get('Partition')=='gpu',
      'cpu':fields.get('NumCPUs')=='8' and fields.get('CPUs/Task')=='8',
      'gpu':'gres/gpu=1' in fields.get('ReqTRES','') and fields.get('TresPerNode')=='gres/gpu:1',
      'memory':fields.get('MinMemoryNode') in ('119G','121856M'),
      'requeue':fields.get('Requeue')=='0',
      'wall':fields.get('TimeLimit')=='7-00:00:00',
      'command':fields.get('Command')==str(script),
      'argv':fields.get('SubmitLine')==' '.join(argv),
      'dependency':(all(x in fields.get('Dependency','') for x in deps) and fields.get('Dependency','').startswith('afterany:')) if deps else fields.get('Dependency')=='(null)',
      'source':file_sha(script)==lock['launcher_sha256'],
    }
    save(attempt/'held-inspection.json',dict(job_id=jid,checks=checks,fields=fields,all_pass=all(checks.values())))
    assert all(checks.values()),'HELD_INSPECTION_FAILED (job remains held): '+str(checks)
    release=run(['scontrol','release',jid])
    save(attempt/'release.json',dict(job_id=jid,command=['scontrol','release',jid],returncode=0,stdout=release,actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED',status=('AWAITING_INITIAL_GATE' if lock.get('monitor_until_initial_gate') else 'MONITORING_PAUSED_AWAITING_USER')))
    print(json.dumps(dict(job_id=jid,status='RELEASE_COMMAND_SUCCEEDED',actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);submit(p.parse_args().lock)
