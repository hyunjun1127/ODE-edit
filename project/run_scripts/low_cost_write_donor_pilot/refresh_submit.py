"""Cap-aware held/inspect/release, two technical or four main processes."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import time

from .runtime import file_sha,save
from .sequential_submit import command,project_rows

def inspect(text,job,attempt,mode):
 n={'technical':2,'repair-I1':1,'main':4}[mode]; throttle=min(n,2)
 name='odeedit_refresh_tech_s4' if mode!='main' else 'odeedit_refresh_seq1000_s4'
 wall='02:00:00' if mode!='main' else '12:00:00'
 array_id='0' if n==1 else f'0-{n-1}'
 for field in [f'JobId={job}',f'UserId={pwd.getpwuid(os.getuid()).pw_name}(',f'JobName={name}',
               f'ArrayTaskId={array_id}','NumCPUs=8',f'TimeLimit={wall}','ReqNodeList=server4',str(attempt/'source/project/run_scripts/low_cost_write_donor_pilot/refresh.sbatch'),
               'JobState=PENDING','Reason=JobHeldUser']:
  assert field in text,('HELD_REQUEST_MISMATCH',field)
 assert f'ArrayTaskThrottle={throttle}' in text or f'ArrayTaskId={array_id}%{throttle}' in text
 assert 'MinMemoryNode=59G' in text or 'MinMemoryNode=60416M' in text
 assert any(s in text for s in ['gres/gpu:rtx_pro_6000=1','gres:gpu:rtx_pro_6000:1','TresPerNode=gres/gpu:rtx_pro_6000:1'])

def submit(attempt,mode):
 a=Path(attempt).resolve();w=Path(__file__).resolve().parents[3]
 n={'technical':2,'repair-I1':1,'main':4}[mode];throttle=min(n,2)
 lock=json.loads((a/'execution.lock.json').read_text());control=a/f'{mode}-submission';control.mkdir(mode=0o700,exist_ok=False)
 command(['scripts/check-session-boundary.sh','01a04939-b5c7-7a03-ba2d-ef3343d62cfd'],w)
 gate=None
 if mode=='main':
  gp=a/'technical/comparison-receipt.json';gate=json.loads(gp.read_text());assert gate['status']=='PASS','TECHNICAL_GATE_REQUIRED'
  for r in gate['inputs']:assert file_sha(r['path'])==r['sha256']
  gate=dict(path=str(gp),sha256=file_sha(gp))
 cap=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
 row=[l.split('\t') for l in cap.read_text().splitlines() if l.startswith('server4\t')]
 assert len(row)==1 and row[0][1:4]==['server4','2','60416']
 patterns=row[0][4].split(',')
 sq=command(['squeue','-h','-w','server4','-o','%i|%u|%j|%T|%D|%b|%E'],w)
 pending=command(['squeue','-h','-q','lab_gpu_s4','-o','%i|%u|%j|%T|%D|%b|%E'],w)
 existing=list({r['job']:r for r in project_rows(sq,patterns)+project_rows(pending,patterns)}.values())
 deps=sorted({r['job'].split('_')[0] for r in existing})
 assert all(re.fullmatch(r'[0-9]+',d) for d in deps),'DEPENDENCY_ID_UNRESOLVED'
 helper='afterany_all_existing_admissions' if deps else command(['env',f'AGENT_GPU_CAPS_FILE={cap}','scripts/check-slurm-resource-cap.sh','server4',str(throttle),f'{60416*throttle}M'],w)
 script=a/'source/project/run_scripts/low_cost_write_donor_pilot/refresh.sbatch'
 assert any(m['path']==str(script) and m['sha256']==file_sha(script) for m in lock['members'])
 memory=command(['python3','scripts/slurm_memory_policy.py','audit',str(script)],w)
 partition=command(['scontrol','show','partition','gpu','-o'],w)
 output=a/('technical' if mode!='main' else 'output');output.mkdir(mode=0o700,exist_ok=False)
 slurm=control/'slurm';slurm.mkdir(mode=0o700)
 admission=save(control/'admission.json',dict(epoch=time.time(),hostname=command(['hostname'],w).strip(),cap_registry=dict(path=str(cap),sha256=file_sha(cap)),existing=existing,node_snapshot=sq,pending_snapshot=pending,helper=helper,memory=memory,partition=partition,cap_proof=f'array%{throttle} only after all existing admissions' if deps else f'noexisting+array%{throttle}<=2',unrelated_job_mutation=0,gate=gate))
 array_id='0' if n==1 else f'0-{n-1}'
 args=['sbatch','--parsable','--hold',f'--array={array_id}%{throttle}',f'--output={slurm}/%A_%a.out',f'--error={slurm}/%A_%a.err']
 if mode!='main':args+=['--job-name=odeedit_refresh_tech_s4','--time=02:00:00']
 if deps:args+=['--dependency=afterany:'+':'.join(deps)]
 args +=[str(script),str(a),mode]
 job=command(args,w).strip().split(';')[0];assert job.isdigit()
 save(control/'accepted.json',dict(job=job,mode=mode,command=args,admission=admission,lock_sha256=file_sha(a/'execution.lock.json')))
 held=command(['scontrol','show','job',job,'-o'],w);save(control/'held.json',dict(job=job,raw=held))
 inspect(held,job,a,mode)
 command(['scontrol','release',job],w)
 released=command(['squeue','-h','-r','-j',job,'-o','%i|%u|%j|%T|%D|%b|%E'],w)
 rows=project_rows(released,['odeedit_refresh_*']);assert len(rows)==n
 r=save(control/'released.json',dict(job=job,mode=mode,rows=rows,status='RELEASED',mapping=['native','I1'] if mode=='technical' else ['I1'] if mode=='repair-I1' else lock['array_mapping'],after_initial='CONTINUE_MIDDLE_USER_OVERRIDE',dependencies=deps))
 print(json.dumps(r))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--mode',choices=['technical','repair-I1','main'],required=True)
 args=p.parse_args();submit(args.attempt,args.mode)
