"""Afterany CPU collector: no repair, submission, model loading or science gate."""
import argparse
import csv
import statistics
import math
from pathlib import Path
from .common import *


def validate_scores(rows,catalog):
    require(len({r['row_id'] for r in rows})==len(rows),'DUPLICATE_SCORE')
    for r in rows:
        ref=catalog[r['row_id']];target=ref['target_ids']
        require(r['input_sha']==digest(ref['input_ids']) and r['position_sha']==digest(ref['positions']) and r['target_ids_sha']==digest(target),'SCORE_TOKEN_IDENTITY')
        require(len(r['token_nll'])==r['target_count']==len(target)==len(r['token_predictions']),'DENOMINATOR')
        require(all(math.isfinite(v) for v in r['token_nll']) and math.isfinite(r['nll']),'SCORE_FINITE')
        require(abs(statistics.mean(r['token_nll'])-r['nll'])<=2e-5,'NLL_REDUCTION')
        correct=[a==b for a,b in zip(target,r['token_predictions'])]
        require(sum(correct)==r['token_correct'] and all(correct)==r['strict'],'TF_REDUCTION')
    return rows


def paired(rows):
    groups={}
    for r in rows:
        require(r['label'] not in groups.setdefault(r['pair_id'],{}),'DUPLICATE_PAIR')
        groups[r['pair_id']][r['label']]=r
    result=[]
    for pid,g in groups.items():
        require(set(g)=={'true','new'},'MISSING_LABEL')
        a,b=g['true'],g['new']; desired='true' if a['role'].startswith('base_') or a['kind']=='N' else 'new'
        d=g[desired];other=g['new' if desired=='true' else 'true']
        result.append(dict(pair_id=pid,case_id=a['case_id'],role=a['role'],kind=a['kind'],prompt_index=a['prompt_index'],
            true_nll=a['nll'],new_nll=b['nll'],new_preferred=b['nll']<a['nll'],desired=desired,
            desired_preferred=d['nll']<other['nll'],desired_nll=d['nll'],strict=d['strict'],
            token_correct=d['token_correct'],target_count=d['target_count'],
            prompt_accuracy=d['token_correct']/d['target_count'],w0_kl=a.get('w0_kl')))
    return result


def summarize(rows):
    if not rows:return dict(prompts=0,success=None,strict=None)
    return dict(prompts=len(rows),success=sum(r['desired_preferred'] for r in rows),strict=sum(r['strict'] for r in rows),
        mean_true_nll=statistics.mean(r['true_nll'] for r in rows),mean_new_nll=statistics.mean(r['new_nll'] for r in rows),
        desired_mean_nll=statistics.mean(r['desired_nll'] for r in rows),
        token_micro=sum(r['token_correct'] for r in rows)/sum(r['target_count'] for r in rows),
        prompt_macro=statistics.mean(r['prompt_accuracy'] for r in rows),
        w0_kl=statistics.mean(r['w0_kl'] for r in rows if r['w0_kl'] is not None) if any(r['w0_kl'] is not None for r in rows) else None)


def transitions(before,after):
    a={r['pair_id']:r for r in before};b={r['pair_id']:r for r in after}
    require(a.keys()==b.keys(),'TRANSITION_IDENTITY')
    lost=[pid for pid in a if a[pid]['desired_preferred'] and not b[pid]['desired_preferred']]
    gained=[pid for pid in a if not a[pid]['desired_preferred'] and b[pid]['desired_preferred']]
    return dict(prompts=len(a),entry_success=sum(r['desired_preferred'] for r in a.values()),gross_lost=len(lost),recovered=len(gained),
        mean_desired_nll_change=statistics.mean(b[k]['desired_nll']-a[k]['desired_nll'] for k in a) if a else None,
        lost_ids=lost,gained_ids=gained)


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def collect(lock_path):
    lock=read(lock_path);root=Path(lock['output']);out=root/'collector';out.mkdir(exist_ok=False)
    inventory=[];coverage=[];summary=[];patch=[];retention=[];cost=[]
    for index in range(15):
        branch=f'B{(10,50,90)[index//5]:03d}-L{index%5+4}';p=root/f'branch-{index:02d}'
        terminal=p/'COMPLETED.json';failure=p/'TECHNICAL_FAILED.json'
        state='COMPLETED' if terminal.exists() else 'TECHNICAL_FAILED' if failure.exists() else 'MISSING_OR_NOT_TERMINAL'
        commits=list((p/'commits').glob('*.json')) if p.exists() else []
        weights=list((p/'weights').glob('S*.pt')) if p.exists() else []
        coverage.append(dict(branch=branch,status=state,commits=len(commits),snapshots=len(weights)))
        if state=='COMPLETED':
            t=read(terminal);require(t['source']==lock['source_commit'] and t['config_sha']==lock['config']['sha256'],'TERMINAL_BINDING')
            require(len(commits)==100 and len(weights)==2,'SCIENTIFIC_COMPLETENESS')
            cost.append(dict(branch=branch,**t['cost']))
        if not (p/'observations/entry.json').exists():continue
        catalog={r['row_id']:r for r in read(p/'token-rows.json')['rows']}
        def load_scores(file):return paired(validate_scores(read(file)['rows'],catalog))
        entry=load_scores(p/'observations/entry.json')
        atwrite=[]
        for step in range(1,101):
            file=p/f'observations/S{step:03d}-at-write.json'
            if file.exists():atwrite+=load_scores(file)
        for n in MILESTONES:
            file=p/('observations/entry.json' if n==0 else f'observations/S{n:03d}-post.json')
            if not file.exists():continue
            rr=load_scores(file)
            for role,kind in sorted({(r['role'],r['kind']) for r in rr}):
                rows=[r for r in rr if r['role']==role and r['kind']==kind]
                summary.append(dict(branch=branch,step=n,role=role,kind=kind,**summarize(rows)))
            for role in ('base_observer','history_observer'):
                now=[r for r in rr if r['role']==role];prev=[r for r in entry if r['pair_id'] in {x['pair_id'] for x in now}]
                if now:retention.append(dict(branch=branch,step=n,reference='CP_ENTRY',role=role,**transitions(prev,now)))
            now=[r for r in rr if r['role']=='continuation' and r['kind'] in ('R','P')]
            ids={r['pair_id'] for r in now};at=[r for r in atwrite if r['pair_id'] in ids]
            if n and len(at)==len(now):retention.append(dict(branch=branch,step=n,reference='ACTUAL_AT_WRITE',role='continuation',
                acquisition_failure=sum(not r['desired_preferred'] for r in at),**transitions(at,now)))
        nativefile=p/'observations/S100-post.json'
        if nativefile.exists():
            native=load_scores(nativefile)
            for mode in ('zero','targeted','random'):
                file=p/f'observations/patch-{mode}.json'
                if not file.exists():continue
                rr=load_scores(file)
                for role in sorted({r['role'] for r in rr}):
                    now=[r for r in rr if r['role']==role];ids={r['pair_id'] for r in now}
                    prev=[r for r in native if r['pair_id'] in ids]
                    patch.append(dict(branch=branch,mode=mode,role=role,**transitions(prev,now)))
        for f in sorted(p.rglob('*')):
            if f.is_file():inventory.append(record(f))
    write_csv(out/'coverage.csv',coverage);write_csv(out/'milestones.csv',summary);write_csv(out/'compute.csv',cost)
    save(out/'retention.json',retention);save(out/'patch.json',patch);save(out/'artifact-index.json',inventory)
    complete=all(r['status']=='COMPLETED' for r in coverage)
    save(out/'terminal.json',dict(status='COMPLETED' if complete else 'INCOMPLETE_OR_TECHNICAL_FAILED',branches=coverage,
        scheduler_completed_is_not_scientific_complete=True,source=lock['source_commit'],initial_monitoring='USER_RECALL_ONLY'))
    lines=['# Temporal-routing 진단 — 자율 수집 결과','',f"- 상태: {'COMPLETED' if complete else 'INCOMPLETE_OR_TECHNICAL_FAILED'}",
        '- TF 지표이며 자유생성 정확도가 아니다. Step2 이후는 서로 다른 누적 trajectory다.',
        '- 모델 복원 snapshot만 저장; exact_editor_resume=NOT_AVAILABLE.',
        '- Scientific negative result는 실패로 분류하지 않는다. 상세 리뷰는 사용자 호출 이후 별도 수행.',
        '', '| branch | 상태 | commit | snapshot |','| --- | --- | ---: | ---: |']
    lines += [f"| {r['branch']} | {r['status']} | {r['commits']} | {r['snapshots']} |" for r in coverage]
    with (out/'report-ko.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
        for ax,b in zip(axes,(10,50,90)):
            for l in LAYERS:
                r=[x for x in summary if x['branch']==f'B{b:03d}-L{l}' and x['role']=='history_observer']
                if r:ax.plot([x['step'] for x in r],[x['success']/x['prompts'] for x in r],marker='.',label=f'L{l}')
            ax.set_title(f'B{b:03d}');ax.set_xlabel('Committed edit');ax.set_ylim(0,1);ax.legend()
        axes[0].set_ylabel('Fixed history NLL preference fraction');fig.tight_layout();fig.savefig(out/'history-preference.png',dpi=150);plt.close(fig)
    return dict(status='COMPLETED' if complete else 'INCOMPLETE_OR_TECHNICAL_FAILED',output=str(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);a=p.parse_args();print(collect(a.lock))
