"""CPU-only independent scalar reducer, registered afterany once; no polling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def reduce_rows(rows):
    output={}
    for kind in ('R','P','N'):
        part=[r for r in rows if r['kind']==kind];desired='true' if kind=='N' else 'new'
        if not part:continue
        good=sum(r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll'] for r in part)
        output[kind]=dict(n=len(part),successes=good,new_nll=sum(r['new_nll'] for r in part)/len(part),true_nll=sum(r['true_nll'] for r in part)/len(part),
            strict=sum(r[desired+'_strict'] for r in part),correct_tokens=sum(r[desired+'_token_correct'] for r in part),tokens=sum(r[desired+'_token_count'] for r in part))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
    out=a.run/'output';target=a.run/'collected';target.mkdir(exist_ok=False)
    files=list(out.glob('*.json'));manifest=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    terminal=json.loads((out/'terminal.json').read_text()) if (out/'terminal.json').exists() else {'status':'NO_PROGRAM_TERMINAL'}
    errors=[];metrics=[];fixed=short=large=0;shorts={};oracle_rows=[]
    for f in files:
        v=json.loads(f.read_text())
        if f.name.startswith('fixed-'):fixed+=len(v.get('oracle_records',[]));oracle_rows+=v.get('oracle_records',[])
        if f.name.startswith('short-'):
            n=len(v.get('oracle_records',[]));short+=n;shorts[f.name]=n;oracle_rows+=v.get('oracle_records',[])
            if n>12:errors.append(f.name+':SHORT_CAP')
        if f.name.startswith('B100-oracle-'):large+=1;oracle_rows.append(v['timing'])
        if f.name=='kernels.json':
            if v.get('schema')!=2 or len(v['cases'])!=12 or len(v['comparisons'])!=6:errors.append('KERNEL_SCHEMA_INVENTORY')
            for row in v['comparisons']:
                if not isinstance(row.get('reference_kernel'),str) or not isinstance(row.get('candidate_kernel'),str) or not isinstance(row.get('reference'),dict) or not isinstance(row.get('candidate'),dict):errors.append('KERNEL_LABEL_STAT_SCHEMA')
        if 'observer-' in f.name and 'rows' in v:
            rows=v['rows'];identities=[r['identity'] for r in rows]
            if len(set(identities))!=len(identities):errors.append(f.name+':DUPLICATE_IDENTITY')
            if not all(math.isfinite(r[k]) for r in rows for k in ('true_nll','new_nll')):errors.append(f.name+':NONFINITE')
            reductions=reduce_rows(rows)
            expected={'R':100,'P':200,'N':1000} if f.name.startswith('B100') else {'R':4,'P':8,'N':40}
            if any(reductions.get(k,{}).get('n')!=n for k,n in expected.items()):errors.append(f.name+':DENOMINATOR')
            metrics.extend(dict(file=f.name,kind=k,**r) for k,r in reductions.items())
    if fixed>32 or short>96 or fixed+short>160 or large>8:errors.append('TOTAL_BUDGET')
    recorded=terminal.get('budget',{})
    if recorded and (recorded['separate']['fixed']!=fixed or recorded['separate']['B100']!=large or sum(recorded['short'].values())!=short):errors.append('INCOMPLETE_CALL_RECORDS_COUNTED_ATTEMPTS_PRESERVED_IN_TERMINAL')
    cmd=['sacct','-j',a.job,'-X','--parsable2','--noheader','--format=JobIDRaw,JobName,User,State,ExitCode,ElapsedRaw,AllocTRES']
    accounting=subprocess.run(cmd,text=True,capture_output=True)
    receipt=dict(program=terminal,independent_reducer_errors=errors,budget=dict(fixed=fixed,short=short,B100=large,per_short=shorts),
        source_lock_sha256=sha(a.run/'execution.lock.json'),artifact_manifest=manifest,scheduler_accounting=accounting.stdout,scheduler_error=accounting.stderr,
        oracle_seconds_inclusive=sum(r['seconds'] for r in oracle_rows),native_entry_observer_IO='SEPARATE; not double added to parent allocation',
        status='COLLECTED_WITH_LIMITATIONS' if errors else 'SCALAR_REDUCED',new_science=0,checkpoint_saved=False)
    if (out/'reuse-receipt.json').exists():receipt['prior_reuse_not_new_work']=json.loads((out/'reuse-receipt.json').read_text())
    (target/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (target/'metrics.csv').open('w') as f:
        if metrics:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    report=f'''# JLZ 효율화 technical benchmark — 자동 CPU 수집

- Program 상태: {terminal['status']}
- 실제 기록 oracle: small fixed {fixed}, short {short}, B100 {large}. 실패 중단 호출은 terminal 장부와 별도 대조.
- 원 56684 변경 없음, 새 scientific chain 0, checkpoint_saved=false.
- 독립 scalar 검산 이슈: {json.dumps(errors,ensure_ascii=False)}
- Oracle inclusive 시간: {receipt['oracle_seconds_inclusive']:.3f}초. 이는 allocation이나 native/observer 포함 총비용이 아니다.
- 실제 pretrained 통과/속도 주장은 원 qualification/timing 표에 한정한다. 1000-chain 효율·성능은 검증하지 않았다.
- 상세 owner/red 검토는 사용자 recall. 이 collector는 GPU 실행/후속 제출을 하지 않는다.
'''
    (target/'factual-report-ko.md').write_text(report)
    print(json.dumps(dict(status=receipt['status'],path=str(target),errors=errors)))

if __name__=='__main__':main()
