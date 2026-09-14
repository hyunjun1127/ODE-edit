"""Completed-review supplement from existing CPU aggregate CSVs; no model calls."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from statistics import fmean
from .refresh_report import read, table, sha
from .review_metrics import write_csv

ID='ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-COMPLETED-REVIEW-SH4-V1'
ORDER=['N4','REFIT4','FROZEN2','I2','FROZEN4','I4']

def build(attempt, root, control):
    rates=read(root,'batchmetrics'); pairs=read(root,'paired-transitions')
    costs=read(root,'compute-ledger'); targets=read(root,'target-counters')
    state=json.loads((root/'state-review-receipt.json').read_text())
    supplement=json.loads((root/'state-review-supplement-receipt.json').read_text())
    assert supplement['status']=='CPU_SMALL_RECEIPT_SUPPLEMENT_PASS' and supplement['restores']==4 and supplement['remaining_unreferenced_stat_only_files']==0
    metric_receipt=json.loads((root/'metric-reduction-receipt.json').read_text())
    assert state['status']=='CPU_STATE_AND_COUNTER_REDUCTION_PASS'
    assert metric_receipt['logical_batches']==60 and metric_receipt['new_forward']==0
    assert len(read(root,'metric-input-inventory'))==60 and len(read(root,'terminal-general'))==6
    assert len(read(root,'request-cluster-uncertainty'))==99
    zero_exposure=[r for r in pairs if r['contrast']=='AT_WRITE_TO_W60' and r['population']=='cohort60' and r['group']=='ALL']
    assert len(zero_exposure)==18
    assert all(int(r['lost'])==int(r['gained'])==0 and float(r['new_nll_delta_mean'])==float(r['true_nll_delta_mean'])==0 for r in zero_exposure)
    alloc=json.loads((control/'main-allocation.json').read_text())
    assert len(alloc['rows'])==4 and all(r['status']=='COMPLETED' and r['exit']=='0:0' for r in alloc['rows'])
    # Raw scheduler text stays local. Public receipt contains only parsed own-job metadata.
    public_alloc={k:v for k,v in alloc.items() if k!='raw_sacct'}
    (root/'main-allocation.json').write_text(json.dumps(public_alloc,indent=2)+'\n')
    finals=[r for r in rates if r['batch']=='60' and r['population'] in ['current','suffix','entry_old','fullseen','first_suffix500']]
    write_csv(root/'final-populations.csv',finals)
    key=lambda r:(r['policy'],r['population'],r['metric'],r['group'])
    indexed={key(r):r for r in finals}
    for p in ORDER:
        for m in ['RS','PS','NS']:
            a,b,c=[indexed[(p,pop,m,'ALL')] for pop in ['entry_old','suffix','fullseen']]
            assert int(a['numerator'])+int(b['numerator'])==int(c['numerator'])
            assert int(a['prompt_denominator'])+int(b['prompt_denominator'])==int(c['prompt_denominator'])
    contrasts=[r for r in pairs if r['contrast']=='CROSS_POLICY_SAME_ITEMS_DIFFERENT_TRAJECTORIES' and r['batch']=='60' and r['population'] in ['suffix','entry_old','fullseen']]
    write_csv(root/'terminal-contrasts.csv',contrasts)
    compact=[]
    for r in contrasts:
        desired='true' if r['metric']=='NS' else 'new'
        compact.append(dict(before=r['before'],after=r['after'],population=r['population'],group=r['group'],metric=r['metric'],d=r['prompt_denominator'],delta=r['delta_numerator'],delta_pp=r['delta_pp'],lost=r['lost'],gained=r['gained'],strict_delta=r['new_strict_delta_numerator'],desired_NLL_mean=r[desired+'_nll_delta_mean'],desired_NLL_p95=r[desired+'_nll_delta_p95'],desired_NLL_p99=r[desired+'_nll_delta_p99'],competing_NLL_mean=r[('new' if desired=='true' else 'true')+'_nll_delta_mean']))
    write_csv(root/'contrast-summary.csv',compact)
    summary=[]
    for p in ORDER:
        rows=[r for r in costs if r['policy']==p]
        counters=[r for r in targets if r['policy']==p]
        if p in ORDER[2:]:
            assert len(counters)==(2000 if p in ['FROZEN2','I2'] else 4000), 'MISSING_NEW_REQUEST_CHUNK_COUNTERS'
        else:
            assert not counters, 'REFERENCE_STRUCTURED_COUNTERS_NOT_RECORDED'
        item=dict(policy=p,reused=p in ORDER[:2],online_seconds=sum(float(r['online_seconds']) for r in rows),evaluation_seconds=sum(float(r['evaluation_seconds']) for r in rows),Adam=sum(int(r['adam']) for r in counters) if counters else 'REFERENCE_LOG_RECONSTRUCTION_24000' if p=='N4' else 'REFERENCE_LOG_RECONSTRUCTION_27525',loss=sum(int(r['loss']) for r in counters) if counters else 'REFERENCE_LOG_RECONSTRUCTION_25000' if p=='N4' else 'REFERENCE_LOG_RECONSTRUCTION_29525',recorded_request_chunks=len(counters),zero_Adam_chunks=sum(int(r['adam'])==0 for r in counters) if counters else 'NOT_RECORDED_STRUCTURED',frozen_reuse_chunks=sum(r['frozen_reuse']=='True' for r in counters) if counters else 'NOT_APPLICABLE')
        if p in ORDER[2:]:
            i=ORDER[2:].index(p); term=json.loads((attempt/f'output/cell-{i}/terminal.json').read_text()); job=alloc['rows'][i]
            item.update(allocated_GPU_seconds=job['allocated_gpu_seconds'],program_wall_seconds=term['seconds'],load_seconds=term['model_seconds'],peak_GPU_allocated_bytes=term['peak_allocated_bytes'],peak_GPU_reserved_bytes=term['peak_reserved_bytes'],host_batch_MaxRSS=job['batch_step_max_rss'])
            # Preserve all terminal scalar telemetry without guessing renamed fields.
            item['terminal_scalar_telemetry']=json.dumps({k:v for k,v in term.items() if isinstance(v,(int,float))},sort_keys=True)
        else:item.update(allocated_GPU_seconds=5237 if p=='N4' else 6027,new_allocation_charge=0)
        summary.append(item)
    native=summary[0]['online_seconds']
    for r in summary:r['online_ratio_N4']=r['online_seconds']/native
    write_csv(root/'compute-summary.csv',summary)
    frontier=[]
    for r in summary:
        frontier.append(dict(policy=r['policy'],reference='online_ratio_1.5_descriptive_not_admission_gate',measured=r['online_ratio_N4'],status='WITHIN' if r['online_ratio_N4']<=1.5 else 'EXCEEDS',automatic_pruning=False))
    write_csv(root/'quality-frontier.csv',frontier)
    geometry=read(root,'target-geometry');path_geometry=read(root,'weight-path-geometry')
    geometry_summary=[]
    for p in ORDER[2:]:
        for chunk in sorted({int(r['chunk']) for r in geometry if r['policy']==p}):
            rr=[r for r in geometry if r['policy']==p and int(r['chunk'])==chunk]
            row=dict(policy=p,chunk=chunk,requests=len(rr),zero_step=sum(r['zero_step']=='True' for r in rr),early_stop=sum(r['early_stop']=='True' for r in rr),anchor_scope=rr[0]['aj_scope'])
            for field in ['u_norm','u_chunk_displacement_norm','Z_chunk_displacement_norm','stored_aj_minus_a0_norm','adam_m_norm','adam_v_norm','current_residual_norm']:
                values=[float(r[field]) for r in rr];row[field+'_mean']=fmean(values);row[field+'_max']=max(values)
            geometry_summary.append(row)
    write_csv(root/'request-chunk-summary.csv',geometry_summary)
    coverage=[
      ('Final six policies/full6000 old+new','PASS','metric-reduction-receipt.json; final-populations.csv'),
      ('Current/Historical all60; strict/token/two-P','PASS','batchmetrics.csv'),
      ('W55/W60 same first500; at-write; B60 exposure0','PASS','paired-transitions.csv'),
      ('Active/superseded/unknown, both NLL tails','PASS','terminal-contrasts.csv; nll-distributions.csv'),
      ('Request-cluster CI','PASS_DESCRIPTIVE_SINGLE_ORDER','request-cluster-uncertainty.json'),
      ('New output fullhash/checkpoints/carry/history','SEE_STATE_RECEIPT', 'state-review-receipt.json'),
      ('Native reference internal u/m/v/teacher','NOT_RECORDED','Reference source/CP audit reused; no invented moments'),
      ('Intermediate actual tensor cross-hash bridge','PARTIAL','Stored CP/prepared bridged; other states dual ledgers only'),
      ('GPU continuation / full nonzero-offset parity','NOT_TESTED','I1/fixedW exact gate does not prove all I2/I4 trajectories'),
      ('Pure writer / target F-B separately','NOT_RECORDED_WHERE_NESTED','compute-ledger.csv; do not add nested IO twice'),
      ('Quality/cost reference lines','DESCRIPTIVE_NO_AND_GATE','quality-frontier.csv; no policy pruning'),
      ('Oldentry whole5000 temporal forgetting','NOT_MEASURED','Final cross-policy difference only'),
      ('Audit128/MMLU68/FutureN/Late/selection','DEFERRED_NOT_EVALUATED','No new evaluation or policy selection'),
    ]
    write_csv(root/'coverage.csv',[dict(item=a,status=b,evidence=c) for a,b,c in coverage])
    worktree=Path(__file__).resolve().parents[3]
    documents=['PROTOCOL.md','project/proposals/2026-09-14-refit4-write-refresh-seq1000-gh-instruction.md','plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md','plans/global/2026-09-14-refit4-write-refresh-seq1000-cells.csv','plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json','plans/global/2026-09-14-refit4-write-refresh-design-checks.json','audits/global/2026-09-14-lowcost-seq10-review-ko.md']
    documents=[dict(path=n,sha256=sha(worktree/n),bytes=(worktree/n).stat().st_size) for n in documents]
    assert sha(attempt/'execution.lock.json')=='235c11e48f18d01277c3bf5c24a32fba44ad26f79b21b70a1b1d2baee3d625e5'
    technical=attempt/'technical/comparison-receipt.json'
    assert sha(technical)=='17eb65ffaa7d4713cfb128b1e30a441b8907db539d5825d37d19823bffe59e1f'
    docs=dict(instruction_id=ID,full_read_documents=documents,pause_receipt=dict(path=str(attempt/'user-pause-override-20260914-r1.json'),sha256=sha(attempt/'user-pause-override-20260914-r1.json')),execution_lock=dict(path=str(attempt/'execution.lock.json'),sha256=sha(attempt/'execution.lock.json')),first_table_sha256=sha(root/'first-final-table.csv'),source_main_at_recall='cc348e012707f2507dc2d66cfdb5af937a44b4fc',analysis_base='a2553e2',analysis_worktree=str(worktree),host='server4',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',source_original_unchanged=True,recall_scope='CPU_COMPLETED_REVIEW_ONLY',GPU=0,other_paused_tasks_unchanged=True)
    docs['technical_exact_gate']=dict(path=str(technical),sha256=sha(technical),verification='PRIOR_GPU_GATE_REUSED_NEW_SMALL_RECEIPT_HASH',new_GPU_comparison=False)
    docs['original_attachment_sha256']='6dbd71d8a495afdbc958677540019d7a55d13fbb0d6dc5091f31891615b05b35'
    docs['posted_instruction_sha256']='55c8eb6ca5730b197b83852ed65f20d00519bf89247ec2a5afd7894a75d99f54'
    docs['attachment_relation']='Whitespace/line-ending normalized equality previously verified; not identical bytes'
    (root/'evidence-reuse-manifest.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2)+'\n')
    observed=[]
    for before,after in [('FROZEN2','I2'),('FROZEN4','I4'),('I2','I4')]:
        delta=[int(indexed[(after,'fullseen',m,'ALL')]['numerator'])-int(indexed[(before,'fullseen',m,'ALL')]['numerator']) for m in ['RS','PS','NS']]
        observed.append(f'{after}−{before}: RS{delta[0]:+d}/PS{delta[1]:+d}/NS{delta[2]:+d}')
    text=['## 완료 recall 후 추가 검산과 상세 대조',
      '이번 지시 '+ID+'는 완료 검증/CPU 분석만 재개했다. 네 신규 scheduler COMPLETED0와 scientific terminal40batch/기존reference20batch를 별도로 확인했다. 이전 partial-final-table-v2 및 pause receipt는 불변이며 아래 완료표로 덮어쓰지 않았다.',
      '### 주요 대조: after−before (동일 문항, 다른 누적 경로)',
      table([r for r in compact if (r['before'],r['after']) in [('FROZEN2','REFIT4'),('FROZEN2','I2'),('FROZEN4','I4'),('I2','I4')] and r['group']=='ALL'],['before','after','population','metric','d','delta','delta_pp','lost','gained','strict_delta','desired_NLL_mean','desired_NLL_p95','desired_NLL_p99']),
      '양의 desired NLL 차이는 정답 target 악화, 양의 성공 수 차이는 선호 개선이다. NS의 desired target은 true이며 표의 strict_delta는 new strict로 별도 정의한다. NS 총점 증가만으로 true NLL가 개선됐다고 결론내리지 않는다. competing_NLL_mean 및 양쪽 NLL는 contrast-summary/terminal-contrasts에 함께 제공한다.',
      '### 과거 유효 target과 신규 target',
      table([r for r in finals if r['population'] in ['entry_old','suffix'] and r['group']=='ACTIVE_TARGET'],['policy','population','metric','numerator','prompt_denominator','rate','new_strict_numerator']),
      '### 요청당 optimization과 실측 비용',table(summary,['policy','reused','Adam','loss','recorded_request_chunks','zero_Adam_chunks','frozen_reuse_chunks','online_seconds','online_ratio_N4','evaluation_seconds','allocated_GPU_seconds']),
      table(summary,['policy','program_wall_seconds','load_seconds','peak_GPU_allocated_bytes','peak_GPU_reserved_bytes','host_batch_MaxRSS']),
      f"신규 main {alloc['allocated_gpu_seconds']} GPU-sec = {alloc['allocated_gpu_seconds']/3600:.6f} GPUh; 기술 실패942+수정검증476=1418 GPU-sec. 이번 신규 합계 {alloc['allocated_gpu_seconds']+1418} GPU-sec = {(alloc['allocated_gpu_seconds']+1418)/3600:.6f} GPUh. 재사용 N4/REFIT4 할당11264 GPU-sec는 새 비용0으로 처리한다.",
      f"동시 allocation 최대 {alloc['concurrency']['max_concurrent_GPUs']} GPU, 2GPU 겹친 구간 {alloc['concurrency']['at_least_two_GPU_seconds']}초. 이는 GPU utilization 실측이 아니다. 새 raw rehash bytes={state.get('rehash_bytes')}이며 referenced input이 포함된 검산량과 output의 실제 disk allocation은 동일하지 않다.",
      f"Primary+supplement full SHA/size {state['rehash_files']+supplement['additional_output_full_sha_files']+supplement['additional_source_rehash_files']} files / {state['rehash_bytes']+supplement['additional_output_rehash_bytes']+supplement['source_reference']['bytes']} bytes. 신규 output {state['new_output_coverage']['files']}개 전부 검사, 미참조 미검사0. Supplemental restore4개도 확인했다. 비L4 불변은 frozen runtime의 full nonselected guard와 성공 receipt 결속이며 독립적인 전체 모델 tensor 재감사/새 GPU replay가 아니다.",
      '### Target refresh와 실제 write 관측',
      table(geometry_summary,['policy','chunk','requests','zero_step','early_stop','u_chunk_displacement_norm_mean','Z_chunk_displacement_norm_mean','stored_aj_minus_a0_norm_mean','current_residual_norm_mean','anchor_scope']),
      table([r for r in path_geometry if r['batch']=='60'],['policy','path_length_sum_actual_delta_frobenius','summed_stored_delta_net_frobenius','checkpoint_W_minus_W50_frobenius','path_to_checkpoint_net_ratio','delta_sum_minus_checkpoint_displacement_frobenius']),
      '위 norm은 실제 저장 FP32 tensor의 Frobenius norm을 FP64로 집계한 값이며 native history/C_reg metric이 아니다. Subwrite path 합과 W60−W50 net은 다르다. FP64 delta 합과 checkpoint 차이를 제시해도 FP32 순차 replay 성공을 뜻하지 않는다. subwrites.csv의 다음 chunk currentY−이전Y는 직전 write의 관측 response이며, 마지막 subwrite 이후 별도 canonicalY가 없어 그 response는 NOT_MEASURED다. Canonical Kc 미저장으로 predicted DKc realization은 NOT_TESTED이며 새 forward로 채우지 않았다. Frozen 후속 aj는 최초 snapshot이지 새 clean readout이 아니다.',
      '### 현재 비교가 구분하는 것과 구분하지 못하는 것',
      'FROZEN2와 REFIT4는 같은 두 write/계수이나 second fresh target·teacher/clamp reset·추가 Adam 사용이 함께 다르다. REFIT4−FROZEN2의 관측을 budget 하나의 독립 효과라고 해석할 수 없다. I2−FROZEN2와 I4−FROZEN4는 write 수와 최대 Adam budget을 맞춘 정책 대조지만 실제 target 경로·loss 횟수·자기 W/M trajectory가 다르다. 단일 동일-state 원인 기여율은 식별되지 않았다.',
      '전체6000 관측 '+ '; '.join(observed)+'. 전체 평균은 신규1000의 strict/retention과 과거5000의 변화를 대체하지 않으므로 위 모집단별 반대 방향과 tails를 함께 읽어야 한다. 방법의 최종 허용 claim과 후속 후보 선택은 하지 않았다.',
      'I1/fixed-W exactPASS는 zero-offset native 연결과 고정W optimizer carry에 한정된다. changed-W/nonzero-offset I2/I4는 저장 상태와 수식/호출 순서를 감사한 것이며 별도 native 동등성 정답이 있는 검증이 아니다. 모델 backend 수치, 자기경로와 teacher/reset 차이를 현재 관측만으로 모두 분리할 수 없다. 0Adam/frozen재사용도 residual solve와 actual write가 있으므로 0write라고 표기하지 않는다.',
      '### 원설계 산출물 coverage',table([dict(item=a,status=b,evidence=c) for a,b,c in coverage],['item','status','evidence']),
      '추가 집계 재현 순서: 위 primary state reducer 뒤 `refresh_state_review --supplement-only`를 같은 --attempt/--out으로 실행하고, `python -m project.run_scripts.low_cost_write_donor_pilot.refresh_completed_report --attempt <attempt-r2> --out <publication> --control <completed-review-v1 local control>`로 추가 표와 본문을 생성한다. 마지막 refresh_report.seal()로 package manifest를 봉인한다. 모두 새 output namespace에서 실행한다. 원본 partial/report/raw는 변경하지 않는다. 원자료 누락을 새 evaluator 또는 인접 state로 보간하지 않았다.'
    ]
    with (root/'diagnostic-report-ko.md').open('a') as f:f.write('\n\n'+'\n\n'.join(text)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--control',type=Path,required=True);a=p.parse_args();build(a.attempt,a.out,a.control)
