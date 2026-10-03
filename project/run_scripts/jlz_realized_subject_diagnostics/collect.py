"""One pre-registered afterany CPU collector, no retry/forward/checkpoints."""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from .common import NONCE,CAPTURE,require,write,sha,member
from .reduce import metrics,paired,paired_rows,interaction,validate


def rows(path):return [r for p in sorted(path.glob('chunk-*.json')) for r in json.loads(p.read_text())['rows']]


def csv_write(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(records)


def flatten(prefix,value):
    result={}
    for k,v in value.items():
        if isinstance(v,dict):result.update(flatten(prefix+k+'_',v))
        else:result[prefix+k]=v
    return result


def collect(attempt,gpu_job=None):
    out=attempt/'collection';out.mkdir(exist_ok=False);root=attempt/'output'
    lock=json.loads((attempt/'execution.lock.json').read_text());config=json.loads((attempt/'config.json').read_text())
    require(lock['instruction']==NONCE,'COLLECT_AUTHORITY')
    terminal=json.loads((root/'terminal.json').read_text()) if (root/'terminal.json').exists() else dict(status='NO_RUNNER_TERMINAL')
    if 'source' in terminal:require(terminal['source']==lock['source'],'TERMINAL_SOURCE')
    accounting=None
    if gpu_job:
        # One exact completed dependency accounting query; never a polling loop.
        p=subprocess.run(['sacct','-X','-j',str(gpu_job),'--noheader','--parsable2',
            '--format=JobIDRaw,JobName,User,State,ExitCode,ElapsedRaw,AllocTRES'],text=True,capture_output=True)
        accounting=dict(job_id=str(gpu_job),returncode=p.returncode,rows=p.stdout.strip().splitlines(),stderr=p.stderr)
        write(out/'accounting.json',accounting)
    products=[];pairs=[];pair_details=[];summary={};raws={};missing=[]
    for c in (0,)+CAPTURE:
        path=root/'D1'/f'c{c:02d}'
        if not (path/'summary.json').exists():missing.append('D1_c'+str(c));continue
        raw=rows(path);validate(raw);recorded=json.loads((path/'summary.json').read_text());m=metrics(raw)
        require({k:v['denominator'] for k,v in m.items()}==dict(R=100,P=200,N=1000),'D1_COVERAGE')
        if raws:validate(raw,next(iter(raws.values())))
        require(recorded['no_mutation'] is True,'D1_MUTATION')
        for k,v in m.items():
            require(v['numerator']==recorded['summary'][k]['numerator'],'INDEPENDENT_NUMERATOR')
            require(abs(v['new_NLL']['mean']-recorded['summary'][k]['new_nll_mean'])<1e-10,'INDEPENDENT_NLL')
            products.append(dict(probe='D1',candidate=c,family=k,**flatten('',v)))
        summary[str(c)]=m;raws[c]=raw
    if 0 in raws:
        # Bind canonical string identity directly to the sealed source data.
        from .common import digest
        data=json.loads(Path(config['stream']).read_text())[:100];expected=[]
        for rec in data:
            rw=rec['requested_rewrite']
            panels=dict(R=[rw['prompt'].format(rw['subject'])],P=rec['paraphrase_prompts'],N=rec['neighborhood_prompts'])
            for kind,prompts in panels.items():
                for ix,prompt in enumerate(prompts):expected.append(digest([rec['case_id'],kind,ix,prompt,rw['target_new']['str'],rw['target_true']['str']]))
        require([r['identity'] for r in raws[0]]==expected,'RAW_INPUT_CASE_PROMPT_TARGET_ORDER')
        for c,raw in raws.items():
            for k,v in paired(raws[0],raw).items():pairs.append(dict(before='W0',after=c,family=k,**v))
            pair_details.extend(dict(before='W0',after=c,**r) for r in paired_rows(raws[0],raw))
    if 25 in raws:
        for c,raw in raws.items():
            if c in (0,25):continue
            for k,v in paired(raw,raws[25]).items():pairs.append(dict(before=c,after=25,family=k,**v))
            pair_details.extend(dict(before=c,after=25,**r) for r in paired_rows(raw,raws[25]))
    lookup=json.loads((attempt/'lookup-local.json').read_text());masked={};d2summary={}
    for mode in ('NONE','SUBJECT_ONLY','NONSUBJECT_ONLY','ALL'):
        path=root/'D2'/mode
        if not (path/'summary.json').exists():
            if lookup['identifiable']:missing.append('D2_'+mode)
            continue
        raw=rows(path);validate(raw)
        require([r['identity'] for r in raw]==lookup['common_four_mask_subset'],'D2_COMMON_LOOKUP_ORDER')
        m=metrics(raw);masked[mode]=raw;d2summary[mode]=m
        for k,v in m.items():products.append(dict(probe='D2',candidate=25,mask=mode,family=k,**flatten('',v)))
    if len(masked)==4:
        interactions,stats=interaction(masked);write(out/'interaction-local.json',dict(rows=interactions))
        csv_write(out/'interaction-summary.csv',[dict(denominator=len(interactions),**stats)])
        for mode,raw in masked.items():
            for k,v in paired(masked['NONE'],raw).items():pairs.append(dict(before='D2_NONE',after='D2_'+mode,family=k,**v))
            pair_details.extend(dict(before='D2_NONE',after='D2_'+mode,**r) for r in paired_rows(masked['NONE'],raw))
        for label in ('NONE','ALL'):
            p=root/'qualification'/f'mask-{label}.json';require(p.exists() and json.loads(p.read_text())['passed'],'MASK_PARITY_RECEIPT')
    candidate_rows=[];layers=[]
    for path in sorted((root/'fit').glob('candidate-*.json')):
        r=json.loads(path.read_text());c=r['candidate']
        require(r['Adam_updates_after']==min(c,24) and r['optimizer_builder_bridges']==1,'CANDIDATE_BUDGET')
        require(r['actual_B']==100 and r['coefficients']==dict(norm=.5,allocation=.1,kl=.0625),'SCIENCE_INVARIANTS')
        candidate_rows.append(dict(candidate=c,updates=r['Adam_updates_after'],total_mean=r['total_mean'],**r['losses'],
            seconds=r['seconds'],native_seconds=r['native_seconds'],builder_seconds=r['builder_seconds'],
            backward_calls=r['backward_calls'],prediction_tokens=r['prediction_tokens']))
        for l,e in r['energy'].items():
            total_c=sum(v['c'] for v in r['energy'].values())
            layers.append(dict(candidate=c,layer=l,**e,normalized_cost_share=e['c']/total_c if total_c else None,mean_share=r['layer'][l]['mean_share'],context_share=r['layer'][l]['context_share'],
                context_v_over_anchor=r['layer'][l]['native_group_weighted_context_rho'],R_gradient=r['gradient']['R_mean']['by_layer'][l],
                q_gradient=r['gradient']['q_optimizer_SUM']['by_layer'][l]))
    if [r['candidate'] for r in candidate_rows]!=list(range(1,26)):missing.append('FIT_25_CANDIDATES')
    commit=root/'ephemeral-commit.json'
    if not commit.exists():missing.append('EPHEMERAL_COMMIT')
    else:require(json.loads(commit.read_text())['history_appends']==5,'FIVE_HISTORY_APPENDS')
    if not terminal.get('restored',False):missing.append('FINAL_RESTORE')
    for name,table in [('metrics.csv',products),('paired.csv',pairs),('candidate.csv',candidate_rows),('layers.csv',layers)]:csv_write(out/name,table)
    write(out/'paired-per-row-local.json',dict(rows=pair_details))
    cost_rows=[dict(component=k,value=v,nesting='program_cost_tree_not_additive') for k,v in flatten('',terminal.get('cost',{})).items()]
    for path in sorted((root/'instrument').glob('components-c*.json')):
        detail=json.loads(path.read_text())
        cost_rows.extend(dict(component='extra_components_c'+str(detail['candidate'])+'_'+k,value=v,nesting='included_in_fit') for k,v in detail['cost'].items())
    csv_write(out/'compute.csv',cost_rows)
    result=dict(status=terminal['status'],independent_CPU_reduction=True,independent_reviewer_agent=False,
        D1=summary,D2=d2summary,lookup=dict(N=lookup['denominator'],identifiable=lookup['identifiable'],excluded=lookup['denominator']-lookup['identifiable']),
        missing=missing,terminal=terminal,scientific_promotion=False,checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        comparison='single cold B100, observer only; no baseline fit, no B2, no causal GH conclusion')
    write(out/'summary.json',result)
    lines=['# JLZ v10 A 단일 B1 D1/D2 사실 보고','',f"실행 상태: `{terminal['status']}`. 단일 cold W0/H0, 고정100요청, 25후보/24갱신 계약이다.",
        '',f"실행 source: `{lock['source']}`. 모델/runtime/input/평가 identity는 execution lock에 결속했다.",
        '독립 구현 CPU reducer로 저장 true/new NLL, strict 및 분모를 다시 집계했다. 독립 reviewer agent는 사용하지 않았다.','',
        '| 후보 | R /100 | P /200 | N /1000 |','|---|---:|---:|---:|']
    for c in (0,)+CAPTURE:
        m=summary.get(str(c));vals=[str(m[k]['numerator']) for k in ('R','P','N')] if m else ['NOT_MEASURED']*3
        lines.append(f"| {'W0' if c==0 else c} | {' | '.join(vals)} |")
    lines+=['',f"D2 공통 식별 subset: {lookup['identifiable']}/{lookup['denominator']}. 제외 문항은 D1 N1000에서 제외하지 않았다.",'',
        '| D2 mask | NS numerator | denominator |','|---|---:|---:|']
    for mode,m in d2summary.items():lines.append(f"| {mode} | {m['N']['numerator']} | {m['N']['denominator']} |")
    lines+=['','NS margin은 new NLL−true NLL이다. Interaction은 ALL−SUBJECT−NONSUBJECT+NONE의 산술 차이다.',
        'TF는 teacher-forced 정확도이며 자유생성 결과가 아니다. c25만 공식 terminal이며 중간 후보는 관측이다.',
        '',f"누락/미검증: {', '.join(missing) if missing else '요구된 저장 coverage 검사에서 누락 없음'}.",
        f"Program seconds: {terminal.get('seconds','NOT_RECORDED')}; peak GPU allocated bytes: {terminal.get('peak_CUDA_allocated_bytes','NOT_RECORDED')}.",
        f"Peak host KiB: {terminal.get('peak_host_KiB','NOT_RECORDED')}. 할당 비용은 accounting.json의 exact parent 행과 구분한다.",
        'Fit 시간은 관측 backward/capture를 포함한 상위 timer다. 중첩 timer를 추가 합산하지 않는다. 미분리 I/O는 NOT_SEPARATED.',
        '', '복원 주장은 selected W/H와 RNG/context/hook 및 nonselected pointer/version guard 범위다. 전체 모델 byte 검증으로 확대하지 않는다.',
        'Disk checkpoint 저장0, exact crash-resume 불가. RAM snapshot은 teardown 후 해제한다.',
        '이 결과로 원인·우열·장기 효과를 판정하지 않는다. 과학 판정은 GH 소유다.',
        'NO_BROADCAST_NOT_REQUIRED: 같은 host의 local 결과를 GH가 접근할 수 있다. Raw/weight/prompt는 Git에 게시하지 않는다.','']
    report=out/'report-ko.md'
    with report.open('x') as f:f.write('\n'.join(lines))
    # Code-only figure is independent of scientific success; missing data omitted.
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        xs=[c for c in CAPTURE if str(c) in summary]
        for k in ('R','P','N'):ax.plot(xs,[summary[str(c)][k]['percent'] for c in xs],marker='o',label=k)
        ax.set(xlabel='Evaluated candidate (c25 official)',ylabel='Preference (%)',ylim=(0,101));ax.legend();fig.tight_layout();fig.savefig(out/'D1.png',dpi=160);plt.close(fig)
    inputs=[member(p) for p in sorted(root.rglob('*.json'))]
    write(out/'manifest.json',dict(source=lock['source'],analysis_source=member(Path(__file__)),reducer=member(Path(__file__).with_name('reduce.py')),
        inputs=inputs,products=[member(p) for p in sorted(out.iterdir()) if p.is_file()],execution_lock=member(attempt/'execution.lock.json')))
    write(out/'terminal.json',dict(status='D1_D2_COMPLETE' if terminal['status']=='D1_D2_COMPLETE' and not missing else 'PARTIAL_OR_TECHNICAL_BLOCKED',
        report=member(report),manifest=member(out/'manifest.json'),GPU_forward=0,new_fit=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--gpu-job');a=p.parse_args();collect(a.attempt,a.gpu_job)
