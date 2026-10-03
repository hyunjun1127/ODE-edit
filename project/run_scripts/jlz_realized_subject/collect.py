"""Independent CPU-only fixed-five-batch raw audit, afterany failure collector."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from .common import require,write,member,sha,digest,INSTRUCTION


def validate_rows(rows, reference, records):
    """Bind full row sequence and token identities to the preflight W0 schema."""
    ids={r['case_id'] for r in records}
    ordinal={r['case_id']:i for i,r in enumerate(records)}
    expected=sorted((r for r in reference if r['case_id'] in ids),
                    key=lambda r:(ordinal[r['case_id']],{'R':0,'P':1,'N':2}[r['kind']],r['prompt_index']))
    require([r['identity'] for r in rows]==[r['identity'] for r in expected],'RAW_ORDER_IDENTITY')
    keys=('case_id','kind','prompt_index','new_token_identity','true_token_identity',
          'new_token_count','true_token_count')
    for actual,wanted in zip(rows,expected):
        require(all(actual[k]==wanted[k] for k in keys),'RAW_TOKEN_TARGET_IDENTITY')


def validate_budget(path):
    rows=[json.loads(p.read_text()) for p in sorted(path.glob('candidate-*.json'))]
    require([r['candidate'] for r in rows]==list(range(1,26)),'CANDIDATE_COVERAGE')
    for i,r in enumerate(rows,1):
        require(r['Adam_updates_after']==min(i,24) and r['optimizer_builder_bridges']==1,'ADAM_BRIDGE_BUDGET')
        require(r['component_builder_backwards']==(4 if i in (2,9,25) else 0),'COMPONENT_COVERAGE')
        require(r['terminal_backward']==(i==25) and r['terminal_update'] is False,'TERMINAL_BUDGET')
        require(r['coefficients']==dict(norm=.5,allocation=.1,kl=.0625),'COEFFICIENT')
        require(not any(r[k] for k in ('clamp','pulse','E_penalty','replay','R_or_P_saved','checkpoint_saved')),'SCIENCE_NOCP')
    return dict(candidates=25,updates=24,seconds=sum(r['seconds'] for r in rows),
        prediction_tokens=sum(r['prediction_tokens'] for r in rows),
        masked_backward_calls=sum(r['backward_calls'] for r in rows),
        component_builder_backwards=sum(r['component_builder_backwards'] for r in rows),
        geometry_solves=1+25*4,timers='candidate wall includes nested native/builder; never sum nested timers again')


def reduce(rows):
    require(len({r['identity'] for r in rows})==len(rows),'DUPLICATE_RAW_ID')
    result={}
    for kind in ('R','P','N'):
        selected=[r for r in rows if r['kind']==kind]
        if not selected:continue
        for r in selected:
            for label in ('new','true'):
                require(math.isfinite(r[label+'_nll']),'NONFINITE_RAW')
                c,n=r[label+'_token_correct'],r[label+'_token_count']
                require(type(c) is int and type(n) is int and 0<=c<=n and n>0,'TOKEN_DENOMINATOR')
                require(type(r[label+'_strict']) is bool and r[label+'_strict']==(c==n),'STRICT_TOKEN_MISMATCH')
        d='true' if kind=='N' else 'new';success=[r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll'] for r in selected]
        denom=len(selected);correct=sum(r[d+'_token_correct'] for r in selected);count=sum(r[d+'_token_count'] for r in selected)
        result[kind]=dict(denominator=denom,numerator=sum(success),rate=sum(success)/denom,
            desired_token_correct=correct,desired_token_count=count,token_micro=correct/count,
            prompt_macro=statistics.mean(r[d+'_token_correct']/r[d+'_token_count'] for r in selected),
            strict_numerator=sum(r[d+'_strict'] for r in selected),strict_denominator=denom,
            true_nll_mean=statistics.mean(r['true_nll'] for r in selected),new_nll_mean=statistics.mean(r['new_nll'] for r in selected),
            desired_nll_mean=statistics.mean(r[d+'_nll'] for r in selected),
            margin_true_minus_new=statistics.mean(r['true_nll']-r['new_nll'] for r in selected),
            true_nll_max=max(r['true_nll'] for r in selected),new_nll_max=max(r['new_nll'] for r in selected))
    return result


def pair(before,after):
    b={r['identity']:r for r in before};a={r['identity']:r for r in after}
    require(a.keys()==b.keys(),'PAIRED_ROW_IDENTITY')
    result={}
    for kind in ('R','P','N'):
        ids=[key for key,r in b.items() if r['kind']==kind]
        if not ids:continue
        def ok(r):return r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll']
        result[kind]=dict(denominator=len(ids),lost=[key for key in ids if ok(b[key]) and not ok(a[key])],
            gained=[key for key in ids if not ok(b[key]) and ok(a[key])],retained=[key for key in ids if ok(b[key]) and ok(a[key])])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path);args=p.parse_args()
    root=args.attempt;out=args.out or root/'collection';out.mkdir(exist_ok=False)
    config=json.loads(args.config.read_text());data=json.loads(Path(config['stream']).read_text());summary={};table=[];inventory=[]
    lock=json.loads((root/'execution.lock.json').read_text())
    require(lock['instruction']==INSTRUCTION and lock['config_sha256']==sha(args.config),'COLLECT_SOURCE_CONFIG')
    require(sha(config['W0_reuse']['path'])==config['W0_reuse']['sha256'],'W0_REUSE_SHA')
    reference=json.loads(Path(config['W0_reuse']['path']).read_text())['rows']
    for arm in ('A','B'):
        path=root/('main-'+arm);terminal=path/'terminal.json';observed=[];commits=[];previous=None;atwrite=[];finalrows=None
        terminal_row=json.loads(terminal.read_text()) if terminal.exists() else {}
        status=terminal_row.get('status','NOT_EXECUTED_OR_NO_TERMINAL');cost=[]
        if terminal_row:
            require(terminal_row['source']==lock['source_commit'] and terminal_row['config_sha256']==lock['config_sha256'],'TERMINAL_SOURCE_CONFIG')
            inventory.append(member(terminal))
        for b in range(1,6):
            cp=path/f'batch-{b:02d}/commit.json'
            if cp.exists():
                c=json.loads(cp.read_text());require(c['ids']==[r['case_id'] for r in data[(b-1)*100:b*100]],'COMMIT_ORDER')
                require(c['candidates']==25 and c['updates']==24 and c['history']['history_appends']==5,'BUDGET_OR_HISTORY')
                require(c['source']==lock['source_commit'] and c['config']==digest(config),'COMMIT_SOURCE_CONFIG')
                cost.append(dict(batch=b,**validate_budget(cp.parent/'fit')))
                if previous:require(c['before']==previous['after'] and c['before_aux']==previous['after_aux'],'STATE_CHAIN')
                previous=c;commits.append(b);inventory.append(member(cp))
            op=path/f'observe-W{b:02d}';sp=op/'summary.json'
            if not sp.exists():continue
            raw=[]
            for chunk in sorted(op.glob('chunk-*.json')):
                raw.extend(json.loads(chunk.read_text())['rows']);inventory.append(member(chunk))
            validate_rows(raw,reference,data[:500] if b==5 else data[(b-1)*100:b*100])
            metrics=reduce(raw);N=500 if b==5 else 100
            require({k:v['denominator'] for k,v in metrics.items()}==dict(R=N,P=2*N,N=10*N),'RAW_COVERAGE')
            recorded=json.loads(sp.read_text())
            require(cp.exists() and recorded['state']==c['after'] and recorded['no_mutation'] is True,'OBSERVER_COMMIT_STATE')
            for kind,values in metrics.items():
                for key in ['denominator','numerator','desired_token_correct','desired_token_count','strict_numerator']:
                    require(values[key]==recorded['summary'][kind][key],'STORED_REDUCTION_'+key)
                for key in ['rate','token_micro','prompt_macro','true_nll_mean','new_nll_mean']:
                    require(math.isclose(values[key],recorded['summary'][kind][key],rel_tol=1e-10,abs_tol=1e-10),'STORED_MEAN_'+key)
                table.append(dict(arm=arm,endpoint=b,family=kind,**values))
            inventory.append(member(sp));observed.append(b)
            ids={r['case_id'] for r in data[(b-1)*100:b*100]};atwrite.extend(r for r in raw if r['case_id'] in ids)
            if b==5:finalrows=raw
        summary[arm]=dict(status=status,commits=commits,observed=observed,candidate_cost=cost,
            program_seconds=terminal_row.get('seconds'),scheduler_allocation_seconds='NOT_MEASURED',
            peak_host_kib=terminal_row.get('peak_host_kib'),peak_gpu_bytes=terminal_row.get('peak_gpu_bytes'),
            main_complete=status=='COMPLETED' and commits==observed==[1,2,3,4,5])
        if finalrows:
            pairs=dict(atwrite_to_W5=pair(atwrite,finalrows),W0_to_W5=pair(reference,finalrows))
            write(out/(arm+'-paired-raw-local.json'),pairs)
            summary[arm]['paired']={tag:{kind:{k:len(v) if isinstance(v,list) else v for k,v in row.items()} for kind,row in families.items()} for tag,families in pairs.items()}
            summary[arm]['strata']={str(flag):reduce([r for r in finalrows if r.get('active_at_endpoint')==flag]) for flag in (True,False)}
    write(out/'summary.json',dict(instruction=INSTRUCTION,arms=summary,source_lock=member(root/'execution.lock.json'),
        analysis='independent CPU arithmetic; no new model forward',noCP=True,scientific_promotion=False))
    if table:
        with (out/'metrics.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    lines=['# JLZ v10 T′ 저장자료 CPU 집계','', '실행 source 및 평가 원본을 보존한다. 새 GPU 평가·checkpoint·자동 재제출은 없다.','']
    for arm,row in summary.items():lines.append(f"- {arm}: {row['status']}; commit {row['commits']}; 평가 {row['observed']}.")
    lines+=['','미완료는 0점으로 대입하지 않는다. TF 정확도는 teacher-forced이며 자유생성 정확도가 아니다.',
            '기존 baseline은 별도 역사 참고이며 새 baseline fit은 수행하지 않았다. 실제 배정 GPU 시간은 사용자 완료 회수 시 scheduler accounting과 결속한다.']
    report=out/'report-ko.md';report.write_text('\n'.join(lines)+'\n')
    products=[member(x) for x in sorted(out.iterdir()) if x.is_file()]
    write(out/'manifest.json',dict(inputs=inventory,products=products,config=member(args.config),source=member(Path(__file__))))
    # Publication succeeded before terminal. Scheduler success is not coverage.
    write(out/'terminal.json',dict(status='COMPLETED' if all(r['main_complete'] for r in summary.values()) else 'SCIENTIFIC_INCOMPLETE',
        report=member(report),manifest=member(out/'manifest.json'),GPU_forward=0,checkpoint_saved=False))


if __name__=='__main__':main()
