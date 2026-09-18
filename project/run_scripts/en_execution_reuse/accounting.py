"""Exact task parents only; no scheduler writes or general queue query."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import subprocess
from .preparation import ROOT,member,create_json

FIELDS='JobIDRaw%30,JobName%60,User%40,State%30,ExitCode,ElapsedRaw,AllocTRES%150,Start,End,NodeList%100'


def parse(text,expected):
    rows=list(csv.DictReader(io.StringIO(text),delimiter='|'))
    if len(rows)!=len(expected) or {r['JobIDRaw'] for r in rows}!=set(expected):
        raise ValueError('EXACT_PARENT_ACCOUNTING_CARDINALITY')
    result=[]
    for row in rows:
        job=row['JobIDRaw'];role=expected[job]
        if not job.isdecimal() or row['User']!='janghj' or row['NodeList']!='server4':
            raise ValueError('EXACT_OWNER_NODE_PARENT')
        expected_name='odeedit_en_reuse_g256_'+('prep' if role=='PREP' else 'B1')+'_s4'
        if row['JobName']!=expected_name or row['State']!='COMPLETED' or row['ExitCode']!='0:0':
            raise ValueError('EXACT_SUCCESSFUL_TASK_PARENT')
        allocation=dict(x.split('=',1) for x in row['AllocTRES'].split(',') if '=' in x)
        if allocation.get('gres/gpu')!='1' or allocation.get('cpu')!='8' or allocation.get('mem') not in ('59G','60416M'):
            raise ValueError('EXACT_ALLOCATION')
        elapsed=int(row['ElapsedRaw'])
        if elapsed<0 or row['Start']=='Unknown' or row['End']=='Unknown':raise ValueError('ALLOCATION_INTERVAL')
        start=datetime.fromisoformat(row['Start']);end=datetime.fromisoformat(row['End'])
        if (end-start).total_seconds()!=elapsed:raise ValueError('ELAPSED_INTERVAL_MISMATCH')
        result.append(dict(role=role,job=job,parent=row,allocated_GPU_seconds=elapsed,allocated_GPU_hours=elapsed/3600))
    intervals=sorted((r['parent']['Start'],r['parent']['End']) for r in result)
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('TASK_SINGLE_GPU_INTERVAL_OVERLAP')
    return dict(status='EXACT_COMPLETED_PARENTS_ONLY',jobs=result,
        allocated_GPU_seconds=sum(r['allocated_GPU_seconds'] for r in result),
        allocated_GPU_hours=sum(r['allocated_GPU_seconds'] for r in result)/3600,
        maximum_task_GPU_overlap=1,step_or_extern_double_count=0,utilization='NOT_MEASURED')


def collect(prep,main,output):
    expected={};inputs=[]
    for role,path in (('PREP',prep),('B1',main)):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT/role) or path.name!='submission.json':raise ValueError('TASK_SUBMISSION_SCOPE')
        value=json.loads(path.read_text());job=value['job']
        if not job.isdecimal() or job in expected or value['released'] is not True:raise ValueError('EXACT_RELEASED_PARENT')
        lock=json.loads(Path(value['lock']['path']).read_text())
        if member(value['lock']['path'])['sha256']!=value['lock']['sha256']:raise ValueError('SUBMISSION_LOCK_HASH')
        expected_stage='GENERATED_REFERENCE_PREPARATION' if role=='PREP' else 'MATCHED_B1'
        if lock['stage']!=expected_stage:raise ValueError('SUBMISSION_STAGE')
        expected[job]=role;inputs.append(member(path))
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('TASK_LOCAL_ACCOUNTING_ONLY')
    args=['sacct','-X','-j',','.join(expected),'--format='+FIELDS,'--parsable2']
    text=subprocess.check_output(args,text=True,env=None)
    result=parse(text,expected)
    result.update(time_utc=datetime.now(timezone.utc).isoformat(),args=args,submission_receipts=inputs,
        allocation_timestamp_timezone='Slurm server local time; retained verbatim',new_GPU=0,new_submissions=0)
    return create_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prep',required=True);p.add_argument('--main',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();print(collect(a.prep,a.main,a.output))
