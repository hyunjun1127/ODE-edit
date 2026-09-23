"""Independent CPU row reducer and afterany collector. No model/runtime imports."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import ROOT, PANELS, INSTRUCTION, read, save, sha, digest


def read_rows(root):
    rows=[r for p in sorted(Path(root).glob('*.json')) for r in read(p)]
    if len({r['row_id'] for r in rows})!=len(rows):raise RuntimeError(('DUPLICATE_ROW',str(root)))
    for r in rows:
        if not math.isfinite(r['nll']):raise RuntimeError(('NONFINITE_ROW',r['row_id']))
        if not 0<=r['token_correct']<=r['target_count']:raise RuntimeError('TOKEN_DENOMINATOR')
    return rows


def pair_rows(rows,panel_index):
    grouped=collections.defaultdict(dict)
    for r in rows:
        p=panel_index[r['row_id']]
        assert (r['pair_id'],r['case_id'],r['kind'],r['label'],r['target_count'],r['input_sha']) == (p['pair_id'],p['case_id'],p['kind'],p['label'],len(p['target_ids']),digest([p['input_ids'],p['positions'],p['target_ids']]))
        grouped[r['pair_id']][r['label']]=r
    out=[]
    for pair,d in grouped.items():
        if 'natural' in d:continue
        a,b=d['true'],d['new'];assert a['kind']==b['kind']
        prefer_true=a['kind'] in ('N','BASE')
        desired=a if prefer_true else b
        p=panel_index[a['row_id']]
        out.append(dict(pair_id=pair,panel=a['panel'],kind=a['kind'],case_id=a['case_id'],subject=a['subject'],prompt_cluster=a['prompt_cluster'],
            true_nll=a['nll'],new_nll=b['nll'],m=a['nll']-b['nll'],g=b['nll']-a['nll'],
            success=a['nll']<b['nll'] if prefer_true else b['nll']<a['nll'],
            tie=a['nll']==b['nll'],strict=desired['strict'],desired_nll=desired['nll'],
            token_correct=desired['token_correct'],token_count=desired['target_count'],
            prompt_index=p['prompt_index'],new_target=p['new_target'],active_target_hash=p.get('active_target_hash')))
    return out


def summary(name,pairs,raw):
    out=[]
    for panel,kind in sorted({(r['panel'],r['kind']) for r in pairs}):
        rr=[r for r in pairs if (r['panel'],r['kind'])==(panel,kind)]
        out.append(dict(endpoint=name,panel=panel,kind=kind,count=len(rr),success=sum(r['success'] for r in rr),
            rate=sum(r['success'] for r in rr)/len(rr),strict=sum(r['strict'] for r in rr),
            token_correct=sum(r['token_correct'] for r in rr),token_total=sum(r['token_count'] for r in rr),
            true_nll_mean=float(np.mean([r['true_nll'] for r in rr])),new_nll_mean=float(np.mean([r['new_nll'] for r in rr])),
            desired_nll_mean=float(np.mean([r['desired_nll'] for r in rr])),desired_nll_q95=float(np.quantile([r['desired_nll'] for r in rr],.95)),
            desired_nll_q99=float(np.quantile([r['desired_nll'] for r in rr],.99)),ties=sum(r['tie'] for r in rr)))
    general=[r for r in raw if r['kind']=='GENERAL']
    if general:
        out.append(dict(endpoint=name,panel='GeneralEval128',kind='GENERAL',count=len(general),success=None,rate=None,
            strict=sum(r['strict'] for r in general),token_correct=sum(r['token_correct'] for r in general),token_total=sum(r['target_count'] for r in general),
            true_nll_mean=float(np.mean([r['nll'] for r in general])),new_nll_mean=None,desired_nll_mean=float(np.mean([r['nll'] for r in general])),
            w0_forward_kl=float(np.mean([r['w0_forward_kl'] for r in general])),
            w0_top1_agreement=sum(r['w0_top1_agree'] for r in general)/sum(r['target_count'] for r in general)))
    h=[r for r in pairs if r['panel']=='H_diag_B1_R100_P200'];bycase=collections.defaultdict(list)
    for r in h:bycase[r['case_id']].append(r)
    assert all(len(v)==3 for v in bycase.values())
    out.append(dict(endpoint=name,panel='H_diag_B1_R100_P200',kind='R+twoP_joint',count=len(bycase),
        success=sum(all(r['success'] for r in v) for v in bycase.values()),strict=sum(all(r['strict'] for r in v) for v in bycase.values())))
    return out


def cluster_ci(values,identities,iterations=2000):
    # Connected case/subject/duplicate-prompt components; never independent prompts.
    parent=list(range(len(values)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for i,r in enumerate(identities):
        for key in (('case',r['case_id']),('subject',r['subject']),('prompt',r['prompt_cluster'])):
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    groups=collections.defaultdict(list)
    for i,v in enumerate(values):groups[find(i)].append(v)
    sums=np.array([sum(x) for x in groups.values()]);counts=np.array([len(x) for x in groups.values()])
    rng=np.random.default_rng(20260924);boot=[]
    for _ in range(iterations):
        ix=rng.integers(0,len(sums),len(sums));boot.append(float(sums[ix].sum()/counts[ix].sum()))
    return dict(cluster_count=len(sums),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),bootstrap_seed=20260924,bootstrap_iterations=iterations)


def paired(name,before,after):
    left={r['pair_id']:r for r in before};right={r['pair_id']:r for r in after}
    assert left.keys()==right.keys()
    out=[]
    for panel,kind in sorted({(r['panel'],r['kind']) for r in before}):
        ids=[i for i,r in left.items() if (r['panel'],r['kind'])==(panel,kind)]
        x=[left[i] for i in ids];y=[right[i] for i in ids]
        changes=[b['desired_nll']-a['desired_nll'] for a,b in zip(x,y,strict=True)]
        row=dict(contrast=name,panel=panel,kind=kind,count=len(ids),gained=sum(not a['success'] and b['success'] for a,b in zip(x,y)),
            lost=sum(a['success'] and not b['success'] for a,b in zip(x,y)),strict_gained=sum(not a['strict'] and b['strict'] for a,b in zip(x,y)),
            strict_lost=sum(a['strict'] and not b['strict'] for a,b in zip(x,y)),
            desired_nll_delta=float(np.mean(changes)),delta_q05=float(np.quantile(changes,.05)),delta_q95=float(np.quantile(changes,.95)),
            true_nll_delta=float(np.mean([b['true_nll']-a['true_nll'] for a,b in zip(x,y)])),
            new_nll_delta=float(np.mean([b['new_nll']-a['new_nll'] for a,b in zip(x,y)])))
        row.update(cluster_ci(changes,x));out.append(row)
    return out


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def overwrite_metrics(name,pairs,ledger,panel_index):
    if name=='W0':batch=0
    else:batch=int(name.rsplit('_W',1)[1])
    selected=[r for r in ledger if r['batch']==batch]
    original={};variants={}
    for r in pairs:
        if r['panel']=='H_diag_B1_R100_P200':original[(r['case_id'],r['kind'],r['prompt_index'])]=r
        elif r['panel']=='H_active_variants':variants[(r['case_id'],r['kind'],r['prompt_index'],r['active_target_hash'])]=r
    result=[]
    for status in ('ACTIVE','SUPERSEDED','NOT_RECEIVED'):
        ids={r['case_id'] for r in selected if r['status']==status}
        for kind in ('R','P'):
            rr=[r for r in original.values() if r['case_id'] in ids and r['kind']==kind]
            active=[]
            for item in selected:
                if item['case_id'] not in ids or item['active_target'] is None:continue
                for pi in ([0] if kind=='R' else [0,1]):
                    base=original[(item['case_id'],kind,pi)]
                    if item['active_target']==item['original_target']:active.append(base)
                    else:active.append(variants[(item['case_id'],kind,pi,digest(item['active_target']))])
            result.append(dict(endpoint=name,batch=batch,status=status,kind=kind,original_count=len(rr),original_success=sum(r['success'] for r in rr),
                active_count=len(active),active_success=sum(r['success'] for r in active),active_strict=sum(r['strict'] for r in active)))
    return result


def report(output,destination,job_id=None):
    output=Path(output);dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    lock=read(output.parent/'execution.lock.json');panels=read(PANELS/'rows.json');pindex={r['row_id']:r for r in panels}
    plan=read(ROOT/'inputs/design/review/e3-dependency-plan.json');ledger=read(PANELS/'active-overwrite-ledger.json')
    stages=[]
    for stage in plan['stages']:
        p=output/stage['stage_id']/'gate-result.json'
        stages.append(dict(stage=stage['stage_id'],status='PASS' if p.is_file() else 'NOT_RUN',receipt=str(p) if p.is_file() else None,sha256=sha(p) if p.is_file() else None))
    complete=(output/'G60/gate-result.json').is_file() and not (output/'failure.json').exists()
    if complete:stages[-1]['status']='PASS_CPU_REDUCER'
    endpoint_summary=[];transitions=[];overwrite=[];factorial=[];patch_summary=[];coverage=[]
    endpoints={};raws={}
    expected=[r['row_id'] for r in panels]
    if (output/'G21/gate-result.json').is_file():
        for root in sorted((output/'G20').iterdir()):
            if not (root.name=='W0' or root.name.startswith('BASE_')):continue
            raw=read_rows(root);assert [r['row_id'] for r in raw]==expected
            rr=pair_rows(raw,pindex);endpoints[root.name]=rr;raws[root.name]=raw
            endpoint_summary+=summary(root.name,rr,raw);overwrite+=overwrite_metrics(root.name,rr,ledger,pindex)
        for name,rr in endpoints.items():
            if name!='W0':transitions+=paired(name+' minus W0',endpoints['W0'],rr)
    for p in plan['pairs']:
        key=f"{p['family']}_s{p['s']:03d}_t{p['t']:03d}"
        stage='G30' if p['phase']=='core' else 'G50';patchstage='G40' if p['phase']=='core' else 'G60'
        src=output/stage/key;states={}
        for state in ('00','10','01','11'):
            paths=list((src/state).glob('*.json'))
            if paths:
                raw=read_rows(src/state)
                if [r['row_id'] for r in raw]==expected:states[state]=pair_rows(raw,pindex)
        coverage.append(dict(pair=key,phase=p['phase'],factorial_complete=len(states)==4,
            patch_complete=(output/patchstage/'gate-result.json').exists(),write_kind=p['write_kind']))
        if len(states)==4:
            maps={s:{r['pair_id']:r for r in rr} for s,rr in states.items()}
            for panel,kind in sorted({(r['panel'],r['kind']) for r in states['00']}):
                rr=[r for r in states['00'] if (r['panel'],r['kind'])==(panel,kind)]
                values=[]
                for r in rr:
                    i=r['pair_id'];values.append(maps['11'][i]['desired_nll']-maps['10'][i]['desired_nll']-maps['01'][i]['desired_nll']+maps['00'][i]['desired_nll'])
                row=dict(pair=key,panel=panel,kind=kind,count=len(rr),final_NLL_interaction_mean=float(np.mean(values)),
                    q05=float(np.quantile(values,.05)),q95=float(np.quantile(values,.95)),not_module_additive_fraction=True)
                row.update(cluster_ci(values,rr));factorial.append(row)
            for variant in ['dose_0p5','dose_1','dose_minus1']+[f'rotation_{x}' for x in lock['rotation_seeds']]:
                rpath=output/patchstage/key/variant
                if not rpath.is_dir():continue
                raw=read_rows(rpath)
                if [r['row_id'] for r in raw]!=expected:continue
                rr=pair_rows(raw,pindex);patch_summary+=paired(key+'/'+variant+' minus actual11',states['11'],rr)
                # Supplementary lost-only is defined from fixed W0 -> unmodified actual11.
                w0={x['pair_id']:x for x in endpoints['W0']};actual={x['pair_id']:x for x in states['11']}
                lost_ids={i for i in actual if w0[i]['success'] and not actual[i]['success']}
                lost_before=[x for x in states['11'] if x['pair_id'] in lost_ids];lost_after=[x for x in rr if x['pair_id'] in lost_ids]
                if lost_before:patch_summary+=paired(key+'/'+variant+' LOST_ONLY_auxiliary',lost_before,lost_after)
    for name,rows in [('endpoint-summary.csv',endpoint_summary),('paired-transitions.csv',transitions),('active-overwrite.csv',overwrite),
        ('factorial-interaction.csv',factorial),('path-patch-paired.csv',patch_summary),('coverage.csv',coverage),('stage-status.csv',stages)]:write_csv(dest/name,rows)
    accounting='NOT_QUERIED'
    if job_id:
        assert str(job_id).isdigit()
        accounting=subprocess.check_output(['sacct','-X','-j',str(job_id),'--noheader','--parsable2','--format=JobIDRaw,JobName,User,State,ExitCode,ElapsedRaw,AllocTRES,Start,End'],text=True)
    cost=read(output/'science-terminal.json') if (output/'science-terminal.json').exists() else read(output/'failure.json') if (output/'failure.json').exists() else {'status':'NO_TERMINAL_RECEIPT'}
    save(dest/'compute.json',dict(parent_accounting=accounting,program=cost,allocated_not_utilization=True,nested_timers_not_additive=True,old_native_training_charged=0))
    save(dest/'lineage.json',dict(instruction_id=INSTRUCTION,execution_lock_sha256=sha(output.parent/'execution.lock.json'),source_sha256=lock['source_sha256'],
        panel_manifest_sha256=lock['panel_sha256'],input_fullSHA='MEMIT receiver fresh; Alpha and model prior verification + current stat',
        independent_reducer=True,independent_agent_review=False,checkpoint_saved=False,exact_new_resume='NOT_AVAILABLE',
        source_analysis_sha256=sha(__file__),NO_BROADCAST_NOT_REQUIRED=True))
    status='COMPLETED' if complete else 'TECHNICAL_FAILED' if (output/'failure.json').exists() else 'INCOMPLETE'
    report_text=['# BASE delayed-write E0(endpoint) → E1 → E3 사실 보고','',f'- 상태: `{status}`. task GPU 동시1, 신규 z/write/history append0, 신규 checkpoint0.',
        '- BASE_ALPHAEDIT 42657 / BASE_MEMIT 42658의 저장 endpoint만 사용했다. 새 baseline 또는 continuation이 아니다.',
        '- E1은 공통 W0+두 family 각12시점. E3은 core4+고정확장8이며 결과 부호로 선정하지 않았다.',
        '- true/new NLL은 별도 TF completion 평균. m=true−new, g=−m. R/P는 new<true, N/Base는 true<new; tie=failure.',
        '- TF accuracy는 target token correctness이며 자유생성 정확도가 아니다. module HδK와 final NLL interaction을 가산 기여율로 해석하지 않는다.',
        '- BaseEval subject/fact 분리는 CounterFact 제공 canonical subject와 정규화 alias 수준. 외부 entity alias resolver는 NOT_AVAILABLE.',
        '- GeneralEval128은 기존 C4 validation의 고정 natural-token window. W0 full-vocabulary teacher는 RAM-only; KL/top1 계산 후 scalar만 보존.',
        '- 원래 H100/P200와 prefix별 ACTIVE/SUPERSEDED 목표는 active-overwrite.csv에서 분리했다.',
        '- bootstrap은 case/subject/중복 prompt 연결 cluster, seed20260924, 2000회. 하나의 고정 trajectory에 대한 기술통계이며 새 순서 불확실성이나 인과 우월성 검증이 아니다.','',
        '## Endpoint 표','', '| Endpoint | Panel | Kind | 성공/분모 | TF strict | Mean desired NLL |','|---|---|---|---:|---:|---:|']
    for r in endpoint_summary:
        nll=r.get('desired_nll_mean');report_text.append(f"| {r['endpoint']} | {r['panel']} | {r['kind']} | {r.get('success','NA')}/{r['count']} | {r.get('strict','NA')} | {nll:.6f} |" if nll is not None else f"| {r['endpoint']} | {r['panel']} | {r['kind']} | {r.get('success','NA')}/{r['count']} | {r.get('strict','NA')} | NA |")
    report_text += ['', '## 단계 및 해석 경계','', '| Stage | Status |','|---|---|']+[f"| {r['stage']} | {r['status']} |" for r in stages]
    report_text += ['', '## 산출물 및 비용','', '- [Endpoint](endpoint-summary.csv), [paired transitions](paired-transitions.csv), [active overwrite](active-overwrite.csv), [factorial](factorial-interaction.csv), [patch](path-patch-paired.csv), [coverage](coverage.csv).',
        '- [Compute](compute.json): parent allocation만 집계하며 batch/extern을 중복 가산하지 않는다. model/forward 누적 timer는 중첩값이므로 합산하지 않는다. 기존 native training 비용은 read-only 입력 lineage이며 이번 실행에 재청구하지 않는다.',
        '- 고정 source/입력/receiver SHA와 실행 gate 증거는 [lineage](lineage.json)에 결속했다. actual W/M continuation은 수행하지 않았으며 MEMIT에 M을 만들지 않았다.',
        '- 같은 입력 key를 사용할 뿐 query-specific hook을 배포 가능한 repair나 새 학습 trajectory라고 부르지 않는다. E0 새 batch/E2/E4–E6는 NOT_AUTHORIZED.',
        '- 실패/미완료이면 남은 셀은 negative result가 아니다. source-backed 실패 원문은 local failure receipt에 보존한다.',
        '- 별도 독립 agent reviewer는 사용하지 않았다. owner 검토와 별도 CPU reducer를 구분한다.']
    with (dest/'report-ko.md').open('x') as f:f.write('\n'.join(report_text)+'\n')
    if complete:
        previous=read(output/'G60/gate-result.json')
        save(output/'G70/gate-result.json',dict(status='PASS',stage='G70',binding=previous['binding'],predecessor_sha256=sha(output/'G60/gate-result.json'),
            detail=dict(report=str(dest/'report-ko.md'),report_sha256=sha(dest/'report-ko.md'),independent_reducer=True)))
    inventory=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()]
    save(dest/'artifact-index.json',dict(root=str(output),members=inventory,new_fullSHA=True))
    save(dest/'terminal.json',dict(status=status,instruction_id=INSTRUCTION,covered_pairs=sum(r['factorial_complete'] for r in coverage),
        full_patch_pairs=sum(r['patch_complete'] for r in coverage),execution_lock_sha256=sha(output.parent/'execution.lock.json'),
        report_sha256=sha(dest/'report-ko.md'),monitoring_active=False,automatic_resume=False))
    print('CPU_COLLECTOR_TERMINAL',status,str(dest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True);p.add_argument('--job-id')
    a=p.parse_args();report(a.output,a.destination,a.job_id)
