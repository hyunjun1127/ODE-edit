"""CPU afterany collector. Independent raw reduction; missing is never PASS."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from .common import INSTRUCTION,member,write,require,digest,sha
from .observe import reduce_rows

def raw(path):
    rows=[]
    for file in sorted(path.glob('chunk-*.json')):
        rows.extend(json.loads(file.read_text())['rows'])
    return rows

def transition(before,after):
    left={r['identity']:r for r in before};right={r['identity']:r for r in after}
    require(set(left)<=set(right),'MISSING_PAIRED_ROWS')
    result={}
    for kind in sorted({r['kind'] for r in before}):
        pairs=[(r,right[k]) for k,r in left.items() if r['kind']==kind]
        success=lambda r:r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll']
        lost=sum(success(l) and not success(r) for l,r in pairs);gained=sum(not success(l) and success(r) for l,r in pairs)
        birth=sum(success(l) for l,r in pairs)
        result[kind]=dict(denominator=len(pairs),birth_success=birth,lost=lost,gained=gained,
                         retained=birth-lost,birth_conditional_retention=(birth-lost)/birth if birth else None,
                         true_nll_delta=sum(r['true_nll']-l['true_nll'] for l,r in pairs)/len(pairs),
                         new_nll_delta=sum(r['new_nll']-l['new_nll'] for l,r in pairs)/len(pairs))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--jobs',required=True)
    args=p.parse_args();root=args.attempt;config=json.loads((root/'config.json').read_text());out=root/'report';out.mkdir(exist_ok=False)
    schedule=config['evaluation_schedule'];inventory=[];table=[];armstatus={};paired={};errors=[]
    if config.get('w0_reuse'):
        bridge=config['w0_reuse']['receipt']
        require(sha(bridge['path'])==bridge['sha256'],'W0_BRIDGE_CHANGED')
        reuse=json.loads(Path(bridge['path']).read_text());observation=reuse['observations']
        require(sha(observation['path'])==observation['sha256'],'W0_RAW_CHANGED')
        w0rows=json.loads(Path(observation['path']).read_text())['rows']
        write(out/'W0-reuse.json',dict(bridge=bridge,status='REUSED_HISTORICAL_NO_NEW_FORWARD',
              numerical_bitwise_equivalence='NOT_ESTABLISHED',layout_difference=reuse['layout_difference']))
    else:w0rows=raw(root/'output/shared-SHARED/W000')
    for arm in ['JLZ_A','JLZ_B']:
        base=root/'output'/('main-'+arm);complete=True;birth=[];all_endpoints={};commit_count=0
        terminal=base/'terminal.json'
        if not terminal.exists() or json.loads(terminal.read_text())['status']!='COMPLETED':complete=False
        for e in schedule['endpoints'][1:]:
            b=e['batch'];path=base/f'W{b:03d}';receipt=path/'summary.json'
            c=base/f'B{b:03d}/commit.json'
            if c.exists():
                commit=json.loads(c.read_text());commit_count+=1
                require(commit['candidate_count']==25 and commit['Adam_updates']==24,'COMMIT_BUDGET')
                require(commit['history_appends']==len(config['profile']['eligible_layers']),'HISTORY_COUNT')
            if not receipt.exists():complete=False;continue
            rows=raw(path)
            summary=reduce_rows(rows)
            require({k:v['denominator'] for k,v in summary.items()}=={k:e[k] for k in ('R','P','N')},'RAW_DENOMINATORS')
            stored=json.loads(receipt.read_text());require(summary==stored['summary'],'INDEPENDENT_REDUCER_MISMATCH')
            ids=set(config['packing'][b+1]['ids'])  # packing pilot two, then main1..20
            birth.extend([r for r in rows if r['case_id'] in ids]);all_endpoints[b]=rows
            for kind,values in summary.items():table.append(dict(arm=arm,batch=b,scope=e['scope'],kind=kind,**values))
        complete=complete and commit_count==20
        armstatus[arm]=dict(status='COMPLETED' if complete else 'PARTIAL_OR_TECHNICAL_FAILED',commits=commit_count)
        if 20 in all_endpoints:
            final=all_endpoints[20]
            paired[arm]=dict(atwrite_to_W20=transition(birth,final),
                W0_to_W20=transition(w0rows,final),
                W10_to_W20=transition(all_endpoints[10],final) if 10 in all_endpoints else 'NOT_MEASURED',
                active=reduce_rows([r for r in final if r['active_at_endpoint']]),
                superseded=reduce_rows([r for r in final if not r['active_at_endpoint']]))
            paired[arm]['cohorts']={str(i+1):transition([r for r in birth if r['case_id'] in set(config['packing'][i+2]['ids'])],final) for i in range(20)}
    # Only this exact registered task mapping, one terminal accounting query.
    jobids=args.jobs.split(',');require(all(x.isdigit() for x in jobids),'INVALID_JOB_IDS')
    account=subprocess.run(['sacct','-n','-P','-j',','.join(jobids),'-o','JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,ExitCode,MaxRSS'],capture_output=True,text=True)
    write(out/'accounting.json',dict(job_ids=jobids,returncode=account.returncode,output=account.stdout,stderr=account.stderr,
          parent_allocation_only=True,allocation_not_utilization=True))
    for path in sorted((root/'output').rglob('*.json')):inventory.append(member(path))
    write(out/'artifact-index.json',inventory);write(out/'paired.json',paired)
    if table:
        with (out/'metrics.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# JLZ v4 A/B 2k 실행 사실보고','',f'Instruction: `{INSTRUCTION}`.',
           '', '이 보고서는 저장된 raw의 CPU 재집계이며, 우월성·기전 해석을 하지 않는다.',
           '', '| Arm | 상태 | commit |','|---|---|---:|']
    lines += [f"| {arm} | {d['status']} | {d['commits']}/20 |" for arm,d in armstatus.items()]
    lines += ['', '| Arm | Batch | Scope | 지표 | 성공/분모 | TF strict |', '|---|---:|---|---|---:|---:|']
    lines += [f"| {r['arm']} | {r['batch']} | {r['scope']} | {r['kind']} | {r['numerator']}/{r['denominator']} | {r['strict_numerator']}/{r['strict_denominator']} |" for r in table]
    lines += ['', '없는 endpoint는 NOT_MEASURED. Scheduler 상태와 scientific completeness는 별도다.',
              'W0는 사용자 지시에 따라 기존 결과를 재사용한다. W0-reuse.json의 identity/원 source/배치 차이를 참고하며 신규 W0 forward는 없다.' if config.get('w0_reuse') else 'W0는 본 attempt 공통 평가다.',
              'noCP; exact_resume=NOT_AVAILABLE. Raw는 local 보존, NO_BROADCAST_NOT_REQUIRED.',
              '비용: accounting.json의 parent allocation만 합산; component 미계측은 NOT_SEPARATED.',
              'paired.json은 all-offered/active/superseded 및 cohort at-write→W20 산술 전이다.']
    (out/'report-ko.md').write_text('\n'.join(lines)+'\n')
    status='COMPLETED' if all(x['status']=='COMPLETED' for x in armstatus.values()) else 'PARTIAL_OR_TECHNICAL_FAILED'
    write(out/'terminal.json',dict(status=status,arms=armstatus,raw_rows_independently_reduced=True,
         job_ids=jobids,report=member(out/'report-ko.md'),checkpoint_saved=False))

if __name__=='__main__':main()
