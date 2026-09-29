"""Afterany CPU coverage and independent strict NLL reducer. No retry or scheduler writes."""
import argparse
import collections
import csv
import math
from .common import *

def paired(rows):
    groups={}
    for r in rows:
        require(math.isfinite(r['nll']) and len(r['token_nll'])==r['target_count'],'RAW_FINITE_CARDINALITY')
        require(abs(sum(r['token_nll'])/r['target_count']-r['nll'])<=max(2e-5,abs(r['nll'])*1e-6),'RAW_NLL_MEAN')
        key=r['pair_id'];g=groups.setdefault(key,{})
        require(r['label'] not in g,'DUPLICATE_LABEL');g[r['label']]=r
    out=[]
    for key,g in groups.items():
        require(set(g)=={'true','new'},'MISSING_PAIR')
        a,b=g['true'],g['new'];desired=a if a['kind']=='N' or a['role'].startswith('base_') else b
        out.append(dict(pair_id=key,case_id=a['case_id'],role=a['role'],kind=a['kind'],prompt_index=a['prompt_index'],
            true_nll=a['nll'],new_nll=b['nll'],margin=a['nll']-b['nll'],preferred=(a['nll']<b['nll']) if desired is a else (b['nll']<a['nll']),
            strict=desired['strict'],correct_tokens=desired['token_correct'],tokens=desired['target_count']))
    return out

def summary(rows):
    r=paired(rows);groups=collections.defaultdict(list)
    for x in r:groups[x['role']+':'+x['kind']].append(x)
    return [dict(panel=k,prompts=len(v),cases=len({x['case_id'] for x in v}),success=sum(x['preferred'] for x in v),
        strict=sum(x['strict'] for x in v),token_correct=sum(x['correct_tokens'] for x in v),tokens=sum(x['tokens'] for x in v),
        mean_true=sum(x['true_nll'] for x in v)/len(v),mean_new=sum(x['new_nll'] for x in v)/len(v)) for k,v in groups.items()]

def transitions(before,after):
    a={x['pair_id']:x for x in paired(before)};b={x['pair_id']:x for x in paired(after)}
    require(set(a)==set(b),'PAIR_IDENTITY')
    return dict(denominator=len(a),before_success=sum(x['preferred'] for x in a.values()),
        lost=[k for k in a if a[k]['preferred'] and not b[k]['preferred']],gained=[k for k in a if not a[k]['preferred'] and b[k]['preferred']])

def collect(lockfile):
    lock=read(lockfile);cfg=read(lock['configuration']['path']);validate_execution(cfg)
    require(sha(lock['configuration']['path'])==lock['configuration']['sha256'],'COLLECTOR_CONFIG_IDENTITY')
    root=Path(lock['attempt']);out=root/'collector';out.mkdir(exist_ok=False)
    rows=[];coverage=[];trans=[];cost=[];artifacts=[]
    for spec in cfg['trajectories']:
        d=root/'output'/spec['name'];terminal=read(d/'terminal.json') if (d/'terminal.json').is_file() else dict(status='MISSING_TERMINAL')
        complete=0;errors=[];atwrite={}
        try:
            for b in range(1,STEPS+1):
                p=d/f'B{b:03d}'
                if not (p/'commit.json').is_file():continue
                c=read(p/'commit.json');complete+=1
                require(len(c['offered_ids'])==BATCH_SIZE,'OFFERED_DENOMINATOR')
                require(c['history_appends']==(5 if c['accepted'] else 0),'HISTORY_COUNT')
                if not c['accepted']:require(c['before']['weight']==c['after']['weight'] and c['before']['history']==c['after']['history'] and c['before']['anchors']==c['after']['anchors'],'REJECT_STATE')
                if b>1:require(read(p/'entry.json')['state']==read(d/f'B{b-1:03d}'/'commit.json')['after'],'STATE_CHAIN')
                raw=read(p/'current.json');atwrite.update({r['row_id']:r for r in raw})
                for r in summary(raw):rows.append(dict(trajectory=spec['name'],batch=b,scope='current',**r))
                if (p/'all-offered.json').is_file():
                    final=read(p/'all-offered.json');require(len(final)==b*BATCH_SIZE*6,'ALL_OFFERED_ROWS')
                    for r in summary(final):rows.append(dict(trajectory=spec['name'],batch=b,scope='all_offered',**r))
                    trans.append(dict(trajectory=spec['name'],batch=b,scope='atwrite_to_now',**transitions(list(atwrite.values()),final)))
                for f in ('fixed-pre.json','fixed-post.json','neighborhood.json'):
                    if (p/f).is_file():
                        for r in summary(read(p/f)):rows.append(dict(trajectory=spec['name'],batch=b,scope=f,**r))
                if (p/'fixed-post.json').is_file():
                    old=[r for r in read(d/'entry-metrics.json') if r['role'].endswith('observer')]
                    trans.append(dict(trajectory=spec['name'],batch=b,scope='entry_observer_to_now',**transitions(old,read(p/'fixed-post.json'))))
            if terminal['status']=='COMPLETED':require(complete==STEPS and len(list((d/'weights'/spec['name']).glob('*.pt')))==len(SAVE_STEPS),'TERMINAL_COMPLETENESS')
        except Exception as e:errors.append(repr(e))
        coverage.append(dict(trajectory=spec['name'],terminal=terminal['status'],completed_commits=complete,errors=errors))
        cost.append(dict(trajectory=spec['name'],program_seconds=terminal.get('program_seconds'),native_fits=terminal.get('scientific_native_fits'),
             timers=terminal.get('timers','NOT_RECORDED'),calls=terminal.get('calls',{}),parent_allocation='NOT_RECORDED_BY_PROGRAM',allocation_not_utilization=True))
        for p in sorted(d.rglob('*')) if d.exists() else []:
            if p.is_file():artifacts.append(record(p))
    save(out/'coverage.json',coverage);save(out/'paired-transitions.json',trans);save(out/'compute.json',cost);save(out/'artifact-index.json',artifacts)
    save(out/'summary.json',rows)
    with (out/'metrics.csv').open('x',newline='') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    completed=all(r['terminal']=='COMPLETED' and r['completed_commits']==STEPS and not r['errors'] for r in coverage)
    status='COMPLETED' if completed else 'INCOMPLETE_OR_TECHNICAL_FAILED'
    report='# 다층 BS1 × 100 USER 변경 프로그램 사실 보고\n\n'+status+'\n\n|경로|terminal|commit/100|검산오류|\n|---|---|---:|---|\n'
    report+='\n'.join(f"|{r['trajectory']}|{r['terminal']}|{r['completed_commits']}|{r['errors']}|" for r in coverage)
    report+='\n\n수치는 metrics.csv, paired-transitions.json 및 compute.json 참조. CPU collector의 exit0은 과학완결성 PASS가 아니다. 실제 모델 source와 원입력은 lock 결속. Exact editor resume은 NOT_AVAILABLE. NO_BROADCAST_NOT_REQUIRED.\n'
    with (out/'report-ko.md').open('x') as f:f.write(report)
    save(out/'terminal.json',dict(status=status,trajectories=coverage,source=lock['source'],config_sha=lock['configuration']['sha256'],report=record(out/'report-ko.md')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);a=p.parse_args();collect(a.lock)
