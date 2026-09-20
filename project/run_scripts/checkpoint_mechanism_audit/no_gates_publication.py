"""CPU final collection of existing no-gates results; no scheduler/model calls."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *
from .publication import bound
from . import reporting


PRIOR=REPO/'experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1'


def collect(output,scheduler):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ctx=read(PRIOR/'context.json')
    prior_context=bound(PRIOR/'context.json')
    key=read(RUN/'results/keys-r1/terminal.json')
    op=read(RUN/'results/operator-r1/terminal.json')
    act=read(RUN/'results/activation-r1/terminal.json')
    jobs=read(scheduler)['jobs']
    assert {str(j['job']) for j in jobs}=={'51137','51138','51139'}
    assert all(j['state'] in ('COMPLETED','FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED') for j in jobs)
    history={x['history_batch']:x for x in op.get('histories',[])}
    intervals={tuple(x['interval']):x for x in act.get('intervals',[])}
    cells=[]
    original=pd.read_csv(PRIOR/'cell_status.csv').fillna('').to_dict('records')
    native={int(r['target_batch']):r for h in history.values() for r in h.get('native_demand_statuses',[])}
    for row in original:
        ident=row['cell_id'];reason='Reused immutable historical evidence; not rerun'
        status=row['status']
        if ident=='C01':reason='Historical FAILED retained; removed as execution prerequisite by explicit user override'
        elif ident=='C02':status=key['status'];reason='Native700 + geometry512 key artifact available; no parity certification'
        elif ident=='F00':status='PASS' if key.get('F00') else 'FAILED';reason='Fixed geometry keys × actual11interval E/D algebra'
        elif ident.startswith('O'):
            x=history.get(int(ident[1:]),{});status=x.get('status','BLOCKED');reason=x.get('error','General LU/RHS computation; residual is observation only')
        elif ident.startswith('N'):
            x=native.get(int(ident[1:]),{});status=x.get('status','BLOCKED');reason=x.get('reason') or 'Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED'
        elif ident=='E00':
            status='PASS' if all(history.get(i,{}).get('counterfactual_status')=='PASS' for i in (0,1,10,100)) else 'BLOCKED'
            reason='Fixed B1/B2 K/R ×4histories ×(original+20permutations); not editing arm'
        elif ident.startswith('G'):
            pair=tuple(map(int,ident[1:].split('_')));x=intervals.get(pair,{})
            status=x.get('status','BLOCKED');reason=x.get('error','Actual s0/.5/1 margins and entry-gradient paths; no duplicated parity forwards')
        elif ident=='Z00':status='PASS';reason='Terminal collection and report, not numerical certification'
        cells.append(dict(cell_id=ident,status=status,reason=reason,numerical_validation='NOT_ESTABLISHED',
            new_execution='SKIPPED_USER_DIRECTED' if ident in ('C00','C01') else 'REUSED' if ident in ('A00','A01','B00') else 'ANALYSIS'))
    write_json(out/'cell-statuses.json',dict(cells=cells,**EXECUTION_POLICY))
    tables,sources=reporting.prepare_optional(RUN/'results',{})
    for name,frame in tables.items():
        if not frame.empty:frame.to_csv(out/name,index=False)
    probe=tables['fixed_probe_history.csv'];modes=tables['native_write_modes.csv'];recon=tables['reconstruction.csv'];activation=tables['activation_margin.csv']
    stats={};paragraphs=[];hyp={}
    if not probe.empty:
        assert not probe.duplicated(['history_batch','case_id']).any()
        pivot=probe.pivot(index='case_id',columns='history_batch',values='nu')
        assert len(pivot)==512
        if 0 in pivot and 100 in pivot:
            difference=pivot[100]-pivot[0]
            stats['fixed_probe']=dict(histories=len(pivot.columns),keys=len(pivot),nu0_mean=float(pivot[0].mean()),
                nu100_mean=float(pivot[100].mean()),decreased=int((difference<0).sum()),increased=int((difference>0).sum()),
                unchanged=int((difference==0).sum()),max_solve_column_residual=max(h['solve']['residual']['max_relative'] for h in history.values() if 'solve' in h),
                negative_raw=int(probe.raw_score.lt(0).sum()),nu_outside_ideal=int((probe.nu.le(0)|probe.nu.gt(1)).sum()))
            p=stats['fixed_probe']
            paragraphs.append(f"고정 geometry512의 nu 평균은 M0 {p['nu0_mean']:.9g} → M100 {p['nu100_mean']:.9g}; 감소 {p['decreased']}/512, 증가 {p['increased']}/512. Raw 음수 {p['negative_raw']}행, ideal 범위 밖 {p['nu_outside_ideal']}행은 clipping하지 않았다. 최대 RHS 상대 잔차 {p['max_solve_column_residual']:.9g}는 계산 관측이며 통과 기준이 아니다.")
            hyp['H2']=dict(status='SUPPORTED' if p['decreased']==512 else 'MIXED',provisional=True,
                basis=f"동일512key의 M0→M100 nu 감소 {p['decreased']}/512, 평균 {p['nu0_mean']:.9g}→{p['nu100_mean']:.9g}. 고정-key operator의 제한적 관측이며 raw nonsymmetric/수치동등성 미확립, 실제 forgetting 인과와 구분.")
    if not modes.empty:
        assert not modes.duplicated(['target_batch','mode']).any()
        mode_summary=modes.groupby('target_batch').agg(modes=('mode','size'),gain_min=('gain','min'),gain_max=('gain','max'),
            target_loading_sum=('target_loading_squared','sum'),write_energy=('write_energy','sum')).reset_index()
        mode_summary.to_csv(out/'native-mode-summary.csv',index=False)
        ratios=[]
        for r in recon.to_dict('records'):
            if r.get('status')=='PASS':
                energy=float(mode_summary.loc[mode_summary.target_batch.eq(r['target_batch']),'write_energy'].iloc[0])
                ratios.append(abs(energy-float(r['factor_delta_norm'])**2))
        stats['native_modes']=dict(target_batches=mode_summary.target_batch.astype(int).tolist(),rows=len(modes),
            maximum_energy_identity_absolute_error=max(ratios,default=None))
        paragraphs.append('Native demand의 raw B 직접 thin-SVD 요약:\n\n'+reporting.markdown(mode_summary)+
            '\n\n이 에너지 분해는 factor write의 대수적 분해다. B1 actual residual/cross term을 별도로 보존하고, B2 이후를 실제 다음 checkpoint와 동일하다고 인증하지 않는다. 작은 key singular value만으로 gain을 추론하지 않았다.')
        hyp['H3']=dict(status='MIXED',provisional=True,basis=f"{len(mode_summary)}개 demand/{len(modes)}mode의 gain²×target-loading으로 factor write energy를 분해했다. 이는 정의상 대수관계이며 causal 설명의 독립 검증이 아니다. B1 actual reference 외는 RECONSTRUCTED_NATIVE_WRITE, dense-factor 검증 생략 및 원 C01 FAILED 때문에 actual write 전체에 대한 수치 인증은 미확립.")
    if not activation.empty:
        assert not activation.duplicated(['interval_start','interval_end','identity']).any()
        summary=[]
        for (a,b,role),g in activation.groupby(['interval_start','interval_end','selection_role']):
            nonzero=g[~g.derivative_zero & ~g.actual_change_zero]
            summary.append(dict(interval_start=int(a),interval_end=int(b),role=role,n=len(g),
                actual_margin_change_mean=float(g.actual_margin_change.mean()),predicted_mean=float(g.predicted_margin_change.mean()),
                absolute_remainder_mean=float(g.nonlinear_remainder.abs().mean()),
                lost=int((g.entry_margin.gt(0)&g.endpoint_margin.le(0)).sum()),
                sign_agree=int((nonzero.actual_margin_change.gt(0)==nonzero.predicted_margin_change.gt(0)).sum()),sign_denominator=len(nonzero),
                DK_squared_mean=float(g.DK_squared_norm.mean()),cross_mean=float(g.cross_term.mean())))
        frame=pd.DataFrame(summary);frame.to_csv(out/'activation-summary.csv',index=False)
        stats['activation']=dict(rows=len(activation),intervals=len(intervals),groups=summary)
        paragraphs.append('선정된 NS panel 실제 margin/entry-gradient 요약:\n\n'+reporting.markdown(frame)+
            '\n\nLost/retained는 archive outcome으로 사후 선정했다. 재측정 lost 수가16과 다를 수 있으며 원 panel을 교체하지 않았다. DK energy와 margin은 다른 값이고 부호가 있는 gradient 및 nonlinear remainder를 함께 본다. 새 backward는 실제 연구 분석이며 parity 전용 backward가 아니다.')
        hyp['H4']=dict(status='MIXED',provisional=True,basis=f"두 사후선정 구간 {len(activation)}개 NS문항에서 actual s0/.5/1와 all-valid-token gradient/EK/DK를 측정. 부호 일치/불일치와 remainder를 표에 모두 보존. Outcome선정에 의한 loss 자체를 독립 인과증거로 삼지 않고, 원 evaluator 동등성 미확립도 유지한다.")
    # Parent R/P is joined by exact request/interval in the already sealed observer table.
    parent=pd.read_parquet(ATTEMPT/'results/archival/mechanism_parent_rp.parquet')
    panel=pd.read_csv(ATTEMPT/'results/archival/mechanism_panel.csv')
    parent_rows=[]
    for (a,b,role),group in panel.groupby(['interval_start','interval_end','selection_role']):
        subset=parent[parent.case_id.isin(group.case_id.unique()) & parent.checkpoint_batch.isin([a,b])]
        for tag,g in subset.groupby('metric_tag'):
            left=g[g.checkpoint_batch.eq(a)];right=g[g.checkpoint_batch.eq(b)]
            j=left.merge(right,on=['case_id','identity'],suffixes=('_a','_b'),validate='one_to_one')
            assert len(j)==len(left)==len(right)
            parent_rows.append(dict(interval_start=int(a),interval_end=int(b),role=role,metric=tag,rows=len(j),
                unique_requests=int(j.case_id.nunique()),entry_success=int(j.success_a.sum()),endpoint_success=int(j.success_b.sum()),
                lost=int((j.success_a & ~j.success_b).sum()),recovered=int((~j.success_a & j.success_b).sum()),
                mean_margin_change=float((j.safety_margin_b-j.safety_margin_a).mean())))
    pd.DataFrame(parent_rows).to_csv(out/'parent-rp-summary.csv',index=False)
    paragraphs.append('선정 panel의 parent R/P(이미 봉인된 평가 재사용):\n\n'+reporting.markdown(pd.DataFrame(parent_rows))+
        '\n\n동일 parent가 복수 NS 또는 두 selection-role에 등장할 수 있다. 각 role 안 request/identity를 중복제거했고 role 사이 합을 독립 분모로 더하지 않는다.')
    f00=pd.read_csv(RUN/'results/keys-r1/fixed_probe_activation.csv')
    numeric=[c for c in f00.select_dtypes(include='number') if c not in ('interval_start','interval_end','case_id')]
    fsum=f00.groupby(['interval_start','interval_end'])[numeric].mean().reset_index()
    fsum['probe_count']=512
    fsum.to_csv(out/'fixed-probe-activation-summary.csv',index=False)
    controls=tables['counterfactuals.csv']
    if not controls.empty:
        csum=controls.groupby(['history_batch','target_batch','kind']).agg(rows=('write_energy','size'),
            energy_mean=('write_energy','mean'),energy_min=('write_energy','min'),energy_max=('write_energy','max'),
            target_error_mean=('target_error_relative','mean')).reset_index()
        csum.to_csv(out/'counterfactual-summary.csv',index=False)
        paragraphs.append('固定K/Rとhistory・R列置換の対照（実編集ではない）:\n\n'+reporting.markdown(csum))
    write_json(out/'analysis-summary.json',stats)
    stage_records={'keys':key,'operator':op,'activation':act}
    ledger=[]
    for j in jobs:
        mode=next(mode for mode in stage_records if read(RUN/f'execution-{mode}-r1/submission.json')['job']==str(j['job']))
        t=stage_records[mode]
        ledger.append(dict(stage=mode,job=j['job'],scheduler_state=j['state'],allocated_gpu_seconds=j['elapsed_seconds'],
            runner_seconds=t.get('elapsed_seconds',sum(x.get('wall_seconds',0) for x in t.get('intervals',[]))),
            numerical_status='NOT_ESTABLISHED',compute_status=t['status']))
    history_timing=[dict(history_batch=h,**{k:x.get(k) for k in ('seconds','factor_seconds','solve_verification_seconds','bank_prepare_seconds','downstream_algebra_and_verification_seconds','cache_io_seconds','peak_gpu_bytes','peak_rss_kib')}) for h,x in history.items()]
    write_csv(out/'history-compute.csv',history_timing)
    new_gpu=sum(j['elapsed_seconds']*j['allocated_gpus'] for j in jobs)
    timing=dict(prior_allocated_gpu_seconds=269,new_allocated_gpu_seconds=new_gpu,total_allocated_gpu_seconds=269+new_gpu,
        jobs=jobs,history_timing=history_timing,key_wall_seconds=key['elapsed_seconds'],
        key_peak_gpu_bytes=key['peak_gpu_bytes'],activation_intervals=[{k:v for k,v in x.items() if k not in ('members','endpoint_archive_parity')} for x in intervals.values()],
        original_cpu_costs_reused_not_recharged={k:v for k,v in ctx['timing'].items() if 'cpu' in k or 'hash' in k},
        nested_timers_not_added_to_allocation=True,validation_only_new_gpu_calls=0)
    oldgate=ctx['gate_summary_ko'].split('이 수치는 B1 부분 재현')[0]
    gate_text=oldgate+'\n\n**최신 정책 변경:** 위 실패를 PASS로 바꾸지 않았다. C00/C01 재실행0. 수치 gate/검증 전용 GPU 호출을 제거하고 필수 artifact 가용성·source/shape/dtype/finite/I/O/상태복원/resource만 유지해 나머지 분석을 실행했다. 새 PASS는 계산 완료이며 numerical_validation=NOT_ESTABLISHED다. Tolerance를 확대해 통과시킨 실행이 아니다.'
    ctx.update(EXECUTION_POLICY)
    ctx.update(optional_results=str(RUN/'results'),completion='NO_GATES_ANALYSIS_TERMINAL_REPORT',
        gate_summary_ko=gate_text,analysis_summary_ko='\n\n'.join(paragraphs),cost_ledger=ctx['cost_ledger']+ledger,
        timing=timing,hypotheses={**ctx['hypotheses'],**hyp},prior_context=prior_context,
        analysis_source_head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        actual_runtime=read(RUN/'results/keys-r1/runtime.json'),parent_RP_rows=len(parent),
        prior_outputs_reused=True,report_numerical_certification=False,
        extra_tables={name:bound(out/name) for name in ('activation-summary.csv','parent-rp-summary.csv','native-mode-summary.csv',
            'history-compute.csv','fixed-probe-activation-summary.csv','counterfactual-summary.csv') if (out/name).exists()})
    ctx['numerical_parity']={**EXECUTION_POLICY,'historical_evidence':ctx['numerical_parity'],
        'new_validation_only_calls':0,'historical_C01_status':'FAILED','new_C00_C01_execution':'SKIPPED_USER_DIRECTED'}
    ctx['source_bindings'] += [dict(kind='no_gates_source_lock',**bound(RUN/f'execution-{mode}-r1/execution.lock.json')) for mode in stage_records]
    ctx['source_bindings'] += [dict(kind='no_gates_override',**bound(RUN/'authority/recall.json')),dict(kind='new_scheduler',**bound(scheduler))]
    for item in ctx['artifact_index']:
        name=item['required_output']
        if name in tables:
            p=out/name
            item.update(status='MEASURED_LOCAL_RETAINED' if p.exists() else 'NOT_MEASURED',**(bound(p) if p.exists() else {}))
    write_json(out/'report-context.json',ctx)
    write_json(out/'collection-receipt.json',dict(status='CPU_TERMINAL_COLLECTION',cells=bound(out/'cell-statuses.json'),
        context=bound(out/'report-context.json'),summary=bound(out/'analysis-summary.json'),
        new_jobs=[j['job'] for j in jobs],prior_gpu_seconds=269,new_gpu_seconds=new_gpu,**EXECUTION_POLICY))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scheduler-receipt',required=True)
    a=p.parse_args();collect(a.output,a.scheduler_receipt)
