"""Independent CPU-only partial/complete reducer. Never schedules/retries."""
import argparse
import csv
import json
from pathlib import Path
from .common import INSTRUCTION,require,member,write,sha
from .observe import reduce_rows

def success(row):
    return row['true_nll']<row['new_nll'] if row['kind']=='N' else row['new_nll']<row['true_nll']

def paired(before,after):
    index={r['identity']:r for r in before};require(len(index)==len(before),'PAIRED_DUPLICATES')
    result={}
    for kind in ('R','P','N'):
        rows=[r for r in after if r['kind']==kind and r['identity'] in index]
        if not rows:continue
        n=len(rows);lost=gained=0
        for r in rows:
            q=index[r['identity']]
            for label in ('true','new'):
                require(q[label+'_token_count']==r[label+'_token_count'],'PAIRED_TOKEN_DENOMINATOR')
                if label+'_token_identity' in q:require(q[label+'_token_identity']==r[label+'_token_identity'],'PAIRED_TOKEN_IDENTITY')
            lost+=success(q) and not success(r);gained+=not success(q) and success(r)
        result[kind]=dict(denominator=n,lost=int(lost),gained=int(gained),
            before_success=sum(success(index[r['identity']]) for r in rows),after_success=sum(success(r) for r in rows),
            true_nll_change=sum(r['true_nll']-index[r['identity']]['true_nll'] for r in rows)/n,
            new_nll_change=sum(r['new_nll']-index[r['identity']]['new_nll'] for r in rows)/n)
    return result

def collect(attempt,out):
    config=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    summary={};inventory=[];tables=[];complete=True;cohorts={};cost=[]
    bridge=json.loads(Path(config['w0_reuse']['receipt']['path']).read_text())
    require(sha(bridge['observations']['path'])==bridge['observations']['sha256'],'W0_RAW_IDENTITY')
    w0=json.loads(Path(bridge['observations']['path']).read_text())['rows']
    for arm in ('A','B'):
        root=attempt/('main-'+arm);terminal=root/'terminal.json';count=0;previous=None;errors=[];birth=[];final=[];case_birth={}
        term=json.loads(terminal.read_text()) if terminal.exists() else dict(status='NOT_RUN_OR_NO_TERMINAL')
        for b in range(1,6):
            commit=root/f'batch-{b:02d}'/'commit.json'
            if not commit.exists():errors.append('MISSING_COMMIT_'+str(b));continue
            r=json.loads(commit.read_text());count+=1
            require(r['source']==lock['source_commit'],'COMMIT_SOURCE')
            require(r['batch']==b and r['actual_B']==100 and r['candidate_count']==25 and r['Adam_updates']==24,'COMMIT_BUDGET')
            require(r['history_appends']==5 and not r['terminal_gradient_measured'] and r['no_checkpoint'],'COMMIT_STATE_SCHEMA')
            require(r['replay'] is False and r['Q2']==(arm=='A' and b==1),'V9_COMMIT_SCOPE')
            fit_root=commit.parent/'fit'
            candidate_paths=sorted(fit_root.glob('candidate-*.json'))
            require(len(candidate_paths)==25,'CANDIDATE_COMPLETENESS')
            for k,path in enumerate(candidate_paths,1):
                row=json.loads(path.read_text())
                require(row['candidate']==k and row['Adam_updates_after']==min(k,24) and row['gradient_measured']==(k<25) and row['replay'] is False,'CANDIDATE_BUDGET')
            diagnostic=json.loads((fit_root/'diagnostic-manifest.json').read_text())
            require(len(diagnostic)==10 and all(set(x['tensors'])=={'M','planned_D','realized_Y'} for x in diagnostic),'DIAGNOSTIC_ALLOWLIST')
            for item in diagnostic:require(sha(item['path'])==item['sha256'],'DIAGNOSTIC_INTEGRITY')
            inventory.extend(diagnostic)
            if arm=='A' and b==1:
                q2=json.loads((commit.parent/'Q2/receipt.json').read_text())
                require(q2['restore_verified'] and q2['ridge_payload_unchanged'] and q2['additional_fits']==0 and q2['history_appends']==0,'Q2_INTEGRITY')
                for path in sorted((commit.parent/'Q2').rglob('*.json')):inventory.append(member(path))
            if previous is not None:require(r['before']==previous,'SEQUENTIAL_STATE_JOIN')
            previous=r['after'];inventory.append(member(commit))
            for cid in r['current_ids']:require(cid not in case_birth,'DUPLICATE_OFFERED_CASE');case_birth[cid]=b
            cost.append(dict(arm=arm,phase='main_batch',batch=b,seconds=r['seconds']))
            for path in sorted((root/f'batch-{b:02d}'/'fit').glob('*.json')):inventory.append(member(path))
            observer=root/f'observe-W{b:02d}';chunks=sorted(observer.glob('chunk-*.json'));rows=[]
            for path in chunks:
                raw=json.loads(path.read_text());require(raw['state']==r['after'],'OBSERVER_ENDPOINT_STATE');rows.extend(raw['rows'])
                inventory.append(member(path))
            if not (observer/'summary.json').exists():errors.append('INCOMPLETE_OBSERVER_'+str(b));continue
            reduced=reduce_rows(rows);N=500 if b==5 else 100
            require({k:v['denominator'] for k,v in reduced.items()}==dict(R=N,P=2*N,N=10*N),'OBSERVER_DENOMINATORS')
            declared=json.loads((observer/'summary.json').read_text());require(declared['summary']==reduced,'INDEPENDENT_REDUCTION_MISMATCH')
            inventory.append(member(observer/'summary.json'))
            if b==5:final=rows
            birth.extend(r for r in rows if case_birth[r['case_id']]==b)
            cost.append(dict(arm=arm,phase='observer',batch=b,seconds=declared['seconds']))
            for kind,v in reduced.items():tables.append(dict(arm=arm,batch=b,kind=kind,**v))
        good=term['status']=='COMPLETED' and count==5 and not errors
        complete &= good;summary[arm]=dict(terminal=term,main_commits=count,complete500=good,missing=errors)
        if final:
            cohorts[arm]=dict(at_write_to_W5=paired(birth,final),W0_to_W5=paired(w0,final),
                W0_comparability=bridge['comparison_scope'],active_W5=reduce_rows([r for r in final if r['active_at_endpoint']]),
                by_birth={str(b):paired([r for r in birth if case_birth[r['case_id']]==b],
                    [r for r in final if case_birth[r['case_id']]==b]) for b in range(1,6)})
        for path in (attempt/('prep-'+arm)).rglob('*.json'):inventory.append(member(path))
        if terminal.exists():inventory.append(member(terminal))
    out.mkdir(parents=True,exist_ok=True)
    write(out/'summary.json',summary);write(out/'artifact-index.json',inventory);write(out/'paired-cohorts.json',cohorts)
    write(out/'cost.json',dict(timers=cost,overlap='batch includes entry/fit/commit; observer separate; parent allocation NOT_RECORDED until explicit accounting review',
        qualification_separate=True,W0_new_forward=0,new_baseline=0))
    if tables:
        with (out/'metrics.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    report=['# JLZ v9 ridge 실행 사실 보고','',f"- instruction: `{INSTRUCTION}`",
            f"- 실행 source: `{lock['source_commit']}`",'- 신규 baseline/checkpoint: 0','',
            '| Arm | main commits | 500 완료 | terminal |','|---|---:|---|---|']
    for arm,r in summary.items():report.append(f"| {arm} | {r['main_commits']} | {r['complete500']} | {r['terminal']['status']} |")
    report+=['','품질·방법 우월성 해석 없음. 미실행·부분 자료는 negative result가 아니다.',
             '집계는 저장 scalar raw 기반 CPU 검산이며 새 모델 평가를 수행하지 않았다.']
    (out/'report-ko.md').write_text('\n'.join(report)+'\n')
    write(out/'terminal.json',dict(status='COMPLETED_500_BOTH' if complete else 'PARTIAL_OR_TECHNICAL_FAILED',
          instruction=INSTRUCTION,source=lock['source_commit'],config_sha256=sha(attempt/'config.json'),
          report=member(out/'report-ko.md'),artifacts=member(out/'artifact-index.json'),main_commits=sum(r['main_commits'] for r in summary.values()),
          no_new_evaluation=True,no_checkpoint=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();collect(a.attempt,a.report)
