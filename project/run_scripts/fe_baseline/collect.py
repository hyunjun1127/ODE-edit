"""CPU-only independent arithmetic, count/link audit and partial-aware report."""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from . import *
from .profile import profile

def rows(folder):return [r for p in sorted(Path(folder).glob('chunk-*.json')) for r in json.loads(p.read_text())['rows']]
def success(r):return r['true_nll']<r['new_nll'] if r['kind']=='N' else r['new_nll']<r['true_nll']
def metrics(values):
    out={}
    require(len({r['identity'] for r in values})==len(values),'DUPLICATE_ROWS')
    for kind in ('R','P','N'):
        group=[r for r in values if r['kind']==kind]
        if not group:continue
        desired='true' if kind=='N' else 'new';den=len(group)
        require(all(math.isfinite(r[x+'_nll']) for r in group for x in ('new','true')),'NONFINITE_RAW')
        token_n=sum(r[desired+'_token_correct'] for r in group);token_d=sum(r[desired+'_token_count'] for r in group)
        out[kind]=dict(numerator=sum(success(r) for r in group),denominator=den,
            strict=sum(r[desired+'_strict'] for r in group),token_n=token_n,token_d=token_d,
            TFmicro=token_n/token_d,TFpromptmacro=sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group)/den,
            new_nll=float(np.mean([r['new_nll'] for r in group])),true_nll=float(np.mean([r['true_nll'] for r in group])),
            desired_nll_p50=float(np.quantile([r[desired+'_nll'] for r in group],.5)),desired_nll_p95=float(np.quantile([r[desired+'_nll'] for r in group],.95)),
            desired_nll_p99=float(np.quantile([r[desired+'_nll'] for r in group],.99)))
    rates=[out[k]['numerator']/out[k]['denominator'] for k in ('R','P','N') if k in out]
    out['harmonic']=0. if len(rates)!=3 or min(rates)==0 else 3/sum(1/v for v in rates)
    return out

def paired(before,after,label):
    old={r['identity']:r for r in before};out=[]
    for kind in ('R','P','N'):
        group=[r for r in after if r['kind']==kind and r['identity'] in old]
        for r in group:
            require(all(r[k]==old[r['identity']][k] for k in ('case_id','kind','prompt_index','new_token_identity','true_token_identity')),'PAIR_IDENTITY')
        out.append(dict(comparison=label,kind=kind,denominator=len(group),lost=sum(success(old[r['identity']]) and not success(r) for r in group),
            gained=sum(not success(old[r['identity']]) and success(r) for r in group),retained=sum(success(old[r['identity']]) and success(r) for r in group)))
    return out

def csv_write(path,values):
    if not values:return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def collect(attempt,accounting=True):
    out=attempt/'collector';out.mkdir(exist_ok=False);main=attempt/'main';errors=[];table=[];transitions=[];birth=[];members=[]
    cfg=json.loads((attempt/'config.json').read_text());identities=json.loads(verify(cfg['observer_identity']).read_text());ref={r['identity']:r for r in identities}
    horizon=profile(cfg['settings']['requests']);batches=horizon['batches'];count=horizon['requests']
    batch_folders=sorted(main.glob('batch-*'),key=lambda p:int(p.name.split('-')[1]))
    commits=[];previous=cfg['cold_W0_H0'];W0=rows(main/'W0')
    for folder in [main/'W0',*(p/'pre' for p in batch_folders),*(p/'post' for p in batch_folders)]:
        raw=rows(folder)
        if not raw:continue
        try:
            for r in raw:require(r['identity'] in ref and all(r[k]==ref[r['identity']][k] for k in ref[r['identity']]),'TOKEN_ROW_REFERENCE')
            stat=metrics(raw);summary=folder/'summary.json'
            if summary.exists():
                expected=json.loads(summary.read_text())['summary']
                require(all(stat[k]['numerator']==expected[k]['numerator'] and stat[k]['denominator']==expected[k]['denominator'] for k in ('R','P','N')),'INDEPENDENT_REDUCER')
            for k in ('R','P','N'):table.append(dict(endpoint=str(folder.relative_to(main)),cohort='all_measured',kind=k,**stat[k],harmonic=stat['harmonic']))
            if folder.name=='post':
                batch=int(folder.parent.name.split('-')[1]);current_ids={x['case_id'] for x in json.loads(verify(cfg['specs']).read_text())[(batch-1)*100:batch*100]}
                current=[r for r in raw if r['case_id'] in current_ids];require(len(current)==1300,'CURRENT_DENOMINATOR');birth+=current
                transitions+=paired(W0,raw,f'W0_to_W{batch}')
                if batch in horizon['milestones']:
                    require(len(raw)==batch*100*13,'ALL_SEEN_DENOMINATOR')
                    transitions+=paired(birth,raw,f'atwrite_to_W{batch}')
                    specs=json.loads(verify(cfg['specs']).read_text())
                    subsets={'first100':{x['case_id'] for x in specs[:100]},'first500':{x['case_id'] for x in specs[:500]}}
                    subsets.update({f'birth{b}':{x['case_id'] for x in specs[(b-1)*100:b*100]} for b in range(1,batch+1)})
                    for tag,ids in subsets.items():
                        sm=metrics([r for r in raw if r['case_id'] in ids])
                        for k in ('R','P','N'):table.append(dict(endpoint=f'W{batch}',cohort=tag,kind=k,**sm[k],harmonic=sm['harmonic']))
                    for active in (True,False):
                        sub=[r for r in raw if r['active_at_endpoint']==active]
                        if sub:
                            sm=metrics(sub)
                            for k in ('R','P','N'):table.append(dict(endpoint=f'W{batch}',cohort='active' if active else 'superseded',kind=k,**sm[k],harmonic=sm['harmonic']))
        except Exception as e:errors.append(dict(path=str(folder),error=str(e)))
    for p in (f/'commit.json' for f in batch_folders if (f/'commit.json').exists()):
        c=json.loads(p.read_text());require(c['before']==previous and c['batch']==len(commits)+1,'COMMIT_CHAIN_LINK');previous=c['after'];commits.append(c)
    fits=[json.loads(p.read_text()) for p in sorted((main/'targets').glob('*-fit.json'))]
    replays=list((main/'targets').glob('*-replay.json'));layers=list(main.glob('batch-*/layer-*.json'))
    if fits:require([r['index'] for r in fits]==list(range(len(fits))) and all(1<=r['evaluations']<=35 and r['Adam_updates']==r['evaluations']-1 for r in fits),'FIT_BUDGET_OCCURRENCE')
    terminal=json.loads((main/'terminal.json').read_text()) if (main/'terminal.json').exists() else {'status':'MISSING_TERMINAL'}
    complete=terminal['status']==f'W{batches}_COMPLETE' and len(commits)==batches and len(layers)==5*batches and len(fits)==len(replays)==count and not errors
    if complete:require(len(W0)==len(rows(main/f'batch-{batches:02d}/post'))==13*count,'W0_FINAL_FULL_DENOMINATORS')
    for p in sorted(main.rglob('*.json')):members.append(member(p))
    account='NOT_QUERIED'
    if accounting and terminal.get('job'):
        run=subprocess.run(['sacct','-n','-P','-j',terminal['job'],'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],text=True,capture_output=True);account=run.stdout if run.returncode==0 else run.stderr
    result=dict(status=f'W{batches}_COMPLETE_CPU_VERIFIED' if complete else 'PARTIAL_OR_TECHNICAL_FAILED',profile=horizon,commits=len(commits),history_appends=len(layers),
        own_joins=max(len(commits)-1,0),target_fits=len(fits),replays=len(replays),fit_evaluations=sum(r['evaluations'] for r in fits),
        Adam_updates=sum(r['Adam_updates'] for r in fits),fit_physical_forwards=sum(r['physical_forwards'] for r in fits),
        fit_recompute_forwards=sum(r['checkpoint_recompute'] for r in fits),target_fit_seconds=sum(r['seconds'] for r in fits),
        terminal=terminal,errors=errors,accounting=account,checkpoint_saved=False,exact_resume='NOT_AVAILABLE',independent_reviewer=0,
        independent_CPU_arithmetic=True,source=json.loads((attempt/'execution.lock.json').read_text())['source_commit'])
    cost_keys=('key_seconds','prepare_seconds','solve_seconds','residual_and_hash_seconds','history_seconds','write_seconds')
    layer_data=[json.loads(p.read_text()) for p in layers]
    result['layer_exclusive_seconds']={k:sum(r.get(k,0) for r in layer_data) for k in cost_keys}
    result['observer_seconds']=sum(json.loads(p.read_text()).get('seconds',0) for p in main.glob('**/summary.json'))
    csv_write(out/'metrics.csv',table);csv_write(out/'paired.csv',transitions)
    write(out/'verification.json',result);write(out/'raw-manifest.json',members)
    (out/'report-ko.md').write_text(f'# FE-MEMIT sequential{count} — CPU 사실 보고\n\n'+
        f"상태: {result['status']}; commit {len(commits)}/{batches}, history {len(layers)}/{5*batches}, target {len(fits)}/{count}.\n\n"+
        f'공개 FE firstforward를 W0 target 고정/BS100×{batches}에 적용했다. 논문 bulk2000 직접 재현이 아니다. TF strict는 자유생성 Accuracy가 아니다.\n\n'+
        'metrics.csv는 원 NLL에서 독립 재계산한 선호·TF·꼬리·cohort이고 paired.csv는 exact identity 기반 lost/gained다. target fit/recompute/replay 비용은 포함하며 새 checkpoint/exact resume는 없다.\n\n'+
        '원본 raw/실패/비용은 local 보존. 품질 해석은 GH. 모니터링/자동 재시도 없음.\n\n'+
        f'## 최종 W{batches} (존재하는 측정만)\n\n'+
        '\n'.join(f"- {r['kind']}: {r['numerator']}/{r['denominator']}, TF strict={r['strict']}/{r['denominator']}, new/true NLL={r['new_nll']:.8g}/{r['true_nll']:.8g}" for r in table if r['endpoint']==f'batch-{batches:02d}/post' and r['cohort']=='all_measured')+'\n\n'+
        '## 비용·한계\n\n'+f"Target fit {result['target_fit_seconds']:.3f}s, logical evaluations {result['fit_evaluations']}, Adam {result['Adam_updates']}; activation recompute forwards {result['fit_recompute_forwards']}. 상세 exclusive timers는 verification.json.\n\n"+
        '공개 BF16/TF4.51.3/generated-context 대신 FP32/eager/TF32off/고정 context/실제 pinned runtime를 쓴 순차 이식이다. 과거25step MEMIT와35step FE는 budget/history 타이밍/rounding/target age가 달라 동등 강도를 주장하지 않는다. 기존 baseline pairedraw 비교는 이 collector에서 NOT_MEASURED이며 새로운 baseline fit은0.\n')
    write(out/'terminal.json',dict(status='COLLECTED',verification=member(out/'verification.json'),report=member(out/'report-ko.md'),scientific_complete=complete))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args().attempt
    try:print(json.dumps(collect(a)))
    except BaseException as e:
        write(a/'collector-error.json',dict(status='COLLECTOR_FAILED',error_type=type(e).__name__,error=str(e)));raise
if __name__=='__main__':main()
