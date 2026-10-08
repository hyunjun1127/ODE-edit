"""Finite sealed smoke -> W0 -> Tier1 -> predeclared selection -> Tier2 DAG.

One GPU warmup, two independent GPU1 arm processes in sweep, no retries/Tier3.
Tier2 cold refits are explicitly charged because no Tier1 state is persisted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from project.run_scripts.jlz_interference_l1.cap_storage import write
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_interference_l1.cap_common import rows_from,require
from .prepare import ARMS


def read(path):return json.loads(Path(path).read_text())


def select(rows,q7_loss=5.0):
    base=rows['Q0']['PS']
    eligible=[a for a,r in rows.items() if a!='Q0' and r['NS_loss_pp']<=1.5 and r['PS']>base]
    frontier=[a for a in eligible if not any(
        rows[b]['PS']>=rows[a]['PS'] and rows[b]['NS_loss_pp']<=rows[a]['NS_loss_pp']
        and (rows[b]['PS']>rows[a]['PS'] or rows[b]['NS_loss_pp']<rows[a]['NS_loss_pp'])
        for b in eligible if b!=a)]
    ranked=sorted(frontier,key=lambda a:(-rows[a]['PS'],rows[a]['NS_loss_pp'],a))
    q7='Q7-native';reference=q7 in rows and rows[q7]['NS_loss_pp']<=q7_loss
    selected=ranked[:3]
    if reference and q7 not in selected:selected=ranked[:2]+[q7]
    return dict(ordinary_eligible=eligible,pareto_frontier=frontier,selected=selected,
        selected_roles={a:'NORMAL_FRONTIER' if a in frontier else 'NATIVE_REFERENCE_ONLY' for a in selected},
        Q7_normal_admission=q7 in eligible,Q7_extreme_excluded=not reference,
        candidate_shortfall=max(0,2-len(selected)),do_not_fill_failed=True,Q0_tier2_required=True)


def summarize(root,stage,arm,w0):
    out=root/(stage+'-'+arm);result=read(out/'result.json')
    require(result['status']=='COMPLETE','RESULT_NOT_COMPLETE')
    b=result['batches'];commit=read(out/f'batch-{b:02d}/commit.json')
    folder=out/f'batch-{b:02d}/post';raw=rows_from(folder,commit['after'])
    baseline={r['identity']:r for r in w0}
    require(all(r['identity'] in baseline for r in raw),'PAIRED_IDENTITY_MISSING')
    matched=reduce_rows([baseline[r['identity']] for r in raw]);summary=reduce_rows(raw)
    require(summary==result['final']['summary'],'FINAL_INDEPENDENT_REDUCER')
    rates={k+'S':100*summary[k]['rate'] for k in 'RPN'}
    paired={}
    for k in 'RPN':
        group=[r for r in raw if r['kind']==k]
        def success(r):return r['true_nll']<r['new_nll'] if k=='N' else r['new_nll']<r['true_nll']
        paired[k]=dict(count=len(group),lost=sum(success(baseline[r['identity']]) and not success(r) for r in group),
            gained=sum(not success(baseline[r['identity']]) and success(r) for r in group),
            margin_delta_mean=sum(r['margin_true_minus_new']-baseline[r['identity']]['margin_true_minus_new'] for r in group)/len(group))
    return dict(**rates,NS_loss_pp=100*(matched['N']['rate']-summary['N']['rate']),
        matched_W0_NS=100*matched['N']['rate'],paired=paired,metrics=summary,
        P_new_nll=summary['P']['new_nll_mean'],P_margin=summary['P']['true_nll_mean']-summary['P']['new_nll_mean'],
        size=read(out/f'batch-{b:02d}/size-diagnostics.json'),
        realization_path=str(out/f'batch-{b:02d}/writer/realization.json'),
        batches=b,numeric_projection_failures=0,source=result['source'],config_sha=result['config_sha'],
        identity=read(out/'tracking/identity.json'))


def run(root,phase):
    root=Path(root);started=time.monotonic();executions=[]
    def stage(name,arm,batches,gpu):
        command=[sys.executable,'-u','-m','project.run_scripts.qwen_price_hparam_tier2.run',
            '--attempt',str(root),'--stage',name,'--arm',arm,'--batches',str(batches)]
        begin=time.monotonic()
        with (root/(name+'-'+arm+'.stdout')).open('x') as stream:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False,env=env)
        row=dict(stage=name,arm=arm,batches=batches,exit_code=process.returncode,seconds=time.monotonic()-begin)
        executions.append(row);write(root/f'execution-{name}-{arm}.json',row)
        require(process.returncode==0,'STAGE_TECHNICAL_FAILURE:'+name+':'+arm)
    status='TECHNICAL_FAILED'
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(devices)==(1 if phase=='warmup' else 2),'EXACT_GPU_ALLOCATION')
    def wave(name,arms,batches):
        for begin in range(0,len(arms),2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(stage,name,arm,batches,devices[i])
                         for i,arm in enumerate(arms[begin:begin+2])]
                for future in futures:future.result()
    try:
        if phase=='warmup':
            stage('smoke','Q0',1,devices[0])
            stage('w0','Q0',0,devices[0])
            status='WARMUP_COMPLETE';return
        require(read(root/'smoke-Q0/result.json')['smoke_WH_equal'],'SMOKE_MUST_PASS')
        w0=rows_from(root/'w0-Q0/W0',read(root/'w0-Q0/result.json')['state'])
        tier1={}
        wave('tier1',ARMS,1)
        for arm in ARMS:tier1[arm]=summarize(root,'tier1',arm,w0)
        if tier1['Q4-beta400']['size']['last_shared_active_fraction']<.8:
            extra='Q4-beta400-lamN0'
            write(root/'conditional-arm.json',dict(parent='Q4-beta400',trigger='shared_active_fraction<0.8',
                observed=tier1['Q4-beta400']['size']['last_shared_active_fraction'],arm=extra,only_change={'lambda_N':0.0}))
            stage('tier1',extra,1,devices[0]);tier1[extra]=summarize(root,'tier1',extra,w0)
        write(root/'tier1-results.json',tier1,limit=8*1024**2)
        rules=read(root/'selection-rules.json')
        selection=select(tier1,rules['Q7_extreme_N_collapse']['matched_W0_current_NS_loss_pp_exclusive'])
        write(root/'tier2-selection.json',selection)
        tier2={}
        wave('tier2',['Q0',*selection['selected']],5)
        for arm in ['Q0',*selection['selected']]:tier2[arm]=summarize(root,'tier2',arm,w0)
        write(root/'tier2-results.json',tier2,limit=8*1024**2)
        passing=[a for a,r in tier2.items() if r['RS']>=99 and r['NS_loss_pp']<=1.2
                 and r['numeric_projection_failures']==0]
        ranked=sorted(passing,key=lambda a:(-tier2[a]['PS'],tier2[a]['NS_loss_pp'],a))
        write(root/'tier3-recommendation.json',dict(status='RECOMMENDATION_ONLY' if ranked else 'NO_PASS',
            ranked=ranked,selected=ranked[0] if ranked else None,Tier3_submitted=False,
            gates=dict(RS_min=99,NS_loss_pp_max=1.2,numeric_projection_failures=0)))
        status='TIER2_COMPLETE'
    except BaseException as error:
        write(root/(phase+'-first-error.json'),dict(type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        write(root/(phase+'-terminal.json'),dict(status=status,executions=executions,
            total_wall_seconds=time.monotonic()-started,gpus=len(devices),
            actual_fit_batches=sum(r['batches'] for r in executions if r['stage']!='w0'),
            cost_includes_smoke_W0_and_cold_Tier2_refits=True,Tier3_submitted=False,
            no_checkpoints=True,archive='NOT_APPLICABLE_NO_CHECKPOINT',automatic_retry=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--phase',choices=['warmup','sweep'],required=True)
    args=p.parse_args();run(args.attempt,args.phase)
