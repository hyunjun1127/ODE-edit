"""Read-only raw reduction into a raw-free short-history factual package."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import numpy as np
from project.run_scripts.ordered_response_barrier_ode.artifacts import reduce_evaluation_payload
from .provenance import save, ARM_ORDER, git
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return dict(n=0,mean=None,median=None,p90=None,max=None)
    if not np.isfinite(values).all():raise ValueError('NONFINITE_ANALYSIS')
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),max=float(values.max()))


def summarize(evaluation, records, entry=None):
    ids=[int(r['case_id']) for r in records];hashes=[r['request_sha256'] for r in records]
    public=reduce_evaluation_payload(evaluation,case_ids=ids,request_sha256=hashes,
        request_order_sha256=canonical_hash(hashes),entry_evaluation=entry)
    facts={};request_rows=[]
    for label,kind,reverse in [('RS','rewrite',False),('PS','rephrase',False),('NS','locality',True)]:
        new=evaluation[kind+'_target_new'];true=evaluation[kind+'_target_true']
        if len(new)!=len(true):raise ValueError('PAIRED_ENDPOINT_DENOMINATOR')
        success=[];advantages=[]
        for a,b in zip(new,true,strict=True):
            if (a['case_id'],a['prompt_index'],a['prompt'])!=(b['case_id'],b['prompt_index'],b['prompt']):
                raise ValueError('PAIRED_PROMPT_IDENTITY')
            margin=b['nll']-a['nll'];hit=margin<0 if reverse else margin>0
            success.append(hit);advantages.append(margin)
            request_rows.append(dict(case_id=a['case_id'],metric=label,prompt_index=a['prompt_index'],
                target_new_nll=a['nll'],target_true_nll=b['nll'],nll_advantage=margin,success=int(hit),
                strict_teacher_forced=bool(a['all_tokens_correct'])))
        facts.update({f'{label}_n':sum(success),f'{label}_d':len(success),f'{label}_rate':sum(success)/len(success)})
        for target,rows in [('new',new),('true',true)]:
            for key,value in distribution([r['nll'] for r in rows]).items():facts[f'{kind}_{target}_nll_{key}']=value
        facts[f'{kind}_strict_n']=sum(r['all_tokens_correct'] for r in new)
        facts[f'{kind}_strict_d']=len(new)
        for key,value in distribution(advantages).items():facts[f'{kind}_advantage_{key}']=value
    return facts,request_rows,public


def old_loss(before,after):
    old_new={r['case_id']:r for r in before['rewrite_target_new']}
    old_true={r['case_id']:r for r in before['rewrite_target_true']}
    new_new={r['case_id']:r for r in after['rewrite_target_new']}
    new_true={r['case_id']:r for r in after['rewrite_target_true']}
    if not (old_new.keys()==old_true.keys()==new_new.keys()==new_true.keys()):raise ValueError('OLD_JOIN_BOUNDARY')
    rows=[]
    for case in old_new:
        prior=old_true[case]['nll']-old_new[case]['nll']
        post=new_true[case]['nll']-new_new[case]['nll']
        rows.append(dict(case_id=case,entry_success=prior>0,post_success=post>0,
            before_nll_advantage=prior,after_nll_advantage=post,
            newly_forgotten=(prior>0 and post<=0),initially_failed=prior<=0))
    return dict(old_before_RS_n=sum(r['entry_success'] for r in rows),old_before_RS_d=len(rows),
        old_after_RS_n=sum(r['post_success'] for r in rows),old_after_RS_d=len(rows),
        new_failure_n=sum(r['newly_forgotten'] for r in rows),
        entry_success_d=sum(r['entry_success'] for r in rows),initially_failed_n=sum(r['initially_failed'] for r in rows)),rows


def csv_once(path,rows):
    path=Path(path)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    path.chmod(0o600)


def md_table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(str(r.get(c,'NOT_RECORDED')).replace('|','\\|') for c in columns)+' |' for r in rows])


def paired_deltas(requests):
    """Paired prompt observations only, without unmatched-row substitution."""
    lookup={}
    for row in requests:
        key=tuple(row[k] for k in ('cell','arm','case_id','metric','prompt_index'))
        if key in lookup:raise ValueError('DUPLICATE_ENDPOINT_KEY')
        lookup[key]=row
    grouped={}
    for key,row in lookup.items():
        cell,arm,case,metric,prompt=key
        if arm=='O_NATIVE':continue
        baseline=lookup.get((cell,'O_NATIVE',case,metric,prompt))
        if baseline is None:raise ValueError('MISSING_OFFICIAL_PAIRED_ENDPOINT')
        for field in ('target_new_nll','target_true_nll','nll_advantage','success','strict_teacher_forced'):
            grouped.setdefault((cell,arm,metric,field),[]).append(float(row[field])-float(baseline[field]))
    rows=[]
    for (cell,arm,metric,field),delta in grouped.items():
        # NS success uses target-true likelihood; do not call target-new NLL lower 'better' there.
        sign=-1 if field.endswith('_nll') else 1
        if metric=='NS' and field=='nll_advantage':sign=-1
        rows.append(dict(cell=cell,arm=arm,metric=metric,field=field,delta='method-minus-O_NATIVE',
            **distribution(delta),negative=sum(v<0 for v in delta),equal=sum(v==0 for v in delta),
            positive=sum(v>0 for v in delta),favorable_direction='lower' if sign<0 else 'higher',
            favorable=sum(sign*v>0 for v in delta),unfavorable=sum(sign*v<0 for v in delta)))
    return rows


def matched_progress(main,nodes):
    rows=[]
    for cell in range(4):
        for arm in ('JV_NATIVE','ORB_RAY_N'):
            endpoint=next((r for r in main if r['cell']==cell and r['arm']==arm),None)
            candidates=[r for r in nodes if r['cell']==cell and r['arm']==arm and r.get('V0',0)>0]
            for level in (.75,.5,.25):
                valid=[r for r in candidates if abs(r['V_exit']/r['V0']-level)<=.02]
                picked=valid[0] if valid else None
                row=dict(cell=cell,arm=arm,requested_V_ratio=level,tolerance=.02,interpolation=0,
                    status='NOT_REACHED' if not valid else 'MATCHED_SAVED_NODE',
                    node=picked['node'] if picked else None,V_ratio=picked['V_exit']/picked['V0'] if picked else None)
                if picked and endpoint and picked['node']==picked['N']-1:
                    row.update(endpoint_evaluation='TERMINAL_RECORDED',RS_n=endpoint['RS_n'],RS_d=endpoint['RS_d'],
                        rewrite_new_nll_mean=endpoint['rewrite_new_nll_mean'],new_failure_n=endpoint['new_failure_n'],
                        entry_success_d=endpoint['entry_success_d'])
                else:row['endpoint_evaluation']='NOT_EVALUATED_INTERMEDIATE_NODE' if picked else 'UNMATCHED'
                rows.append(row)
    return rows


def build_package(root, output, *, allow_boundary=False):
    root,output=Path(root).absolute(),Path(output).absolute()
    if output.exists():raise ValueError('CREATE_ONCE_REPORT_EXISTS')
    terminals=[root/f'cell-{i}'/'terminal.json' for i in range(4)]
    if not allow_boundary and not all(p.is_file() for p in terminals):raise ValueError('FOUR_CELL_TERMINAL_NOT_COMPLETE')
    sample=json.loads((root/'sample.lock.json').read_text());source=json.loads((root/'source.lock.json').read_text())
    records=[r for r in sample['records'] if r['fixture']=='D10B']
    inputs=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        if p.is_symlink() or not stat.S_ISREG(p.lstat().st_mode):raise ValueError('RAW_MEMBER_TYPE')
        inputs.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,mode=oct(stat.S_IMODE(p.stat().st_mode))))
    output.mkdir(parents=True,mode=0o700)
    main=[];requests=[];oldrows=[];nodes=[];shadows=[];normalization=[];compute=[];status=[];integrity=[];setup=[];conflicts=[]
    for cell in range(4):
        directory=root/f'cell-{cell}';complete=directory/'terminal.json'
        failure=directory/'failure-boundary.json'
        state=json.loads(complete.read_text()) if complete.exists() else json.loads(failure.read_text()) if failure.exists() else {'status':'NOT_TERMINAL'}
        status.append({**state,'cell':cell})
        if state.get('status')=='TERMINAL_VALID':
            if state['source_head']!=source['head'] or state['source_tree']!=source['tree'] or state['cold_w0_restore'] is not True:
                raise ValueError('SOURCE_COLD_RESTORE_BOUNDARY')
            if (state['primary_arm_count'],state['primary_request_endpoints'])!=(4,40):raise ValueError('TERMINAL_DENOMINATOR')
        entry_path=directory/'D10B-entry.json'
        if not entry_path.exists():continue
        entry=json.loads(entry_path.read_text())
        same_prompt=[(a,b) for a in entry['old_evaluation']['rewrite_target_new']
            for b in entry['evaluation']['rewrite_target_new'] if a['prompt']==b['prompt']]
        conflicts.append(dict(cell=cell,D10A_requests=10,D10B_requests=10,
            exact_prompt_overlap=len(same_prompt),exact_prompt_different_target=sum(a['target_token_ids']!=b['target_token_ids'] for a,b in same_prompt),
            semantic_conflict='NOT_EVALUATED',selection_or_controller_influence_count=0))
        runtime=json.loads((directory/'runtime.lock.json').read_text())
        setup.append(dict(cell=cell,load_seconds=runtime['load_seconds'],target_seconds=entry['target_seconds'],
            model_load_count=runtime['model_load_count'],allocated_seconds=state.get('gpu_allocated_seconds',state.get('allocated_seconds')),
            model_forward_setup='NOT_RECORDED_SCHEMA_GAP',gpu=runtime['cuda_device'],gpu_id=runtime['cuda_visible']))
        pre,_,_=summarize(entry['evaluation'],records)
        main.append(dict(cell=cell,arm='PRE_EDIT_WARM',fixture='D10B',status='VALID',**pre))
        for arm in ARM_ORDER:
            file=directory/f'D10B-{arm}.json'
            if not file.exists():continue
            raw=json.loads(file.read_text())
            endpoint=raw['endpoint'] if 'endpoint' in raw else raw['historical']['endpoint']
            if raw['w0_restore'] is not True or raw['source_entry_sha']!=entry['entry_weight_sha'] or raw['fixed_z_sha']!=entry['fixed_z_sha']:
                raise ValueError('ARM_ENTRY_TARGET_RESTORE_BOUNDARY')
            checks=dict(cell=cell,arm=arm,w0_restore=True,same_entry=True,same_fixed_z=True,endpoint_status=endpoint['status'],
                history_append_count=endpoint.get('history_append_count','NOT_RECORDED'),
                numeric_storage_cast_count=endpoint.get('numeric_storage_cast_count','NOT_RECORDED'),
                inner_history_append_count=endpoint.get('inner_history_append_count','NOT_RECORDED'))
            for field in ('numeric_storage_cast_count','inner_history_append_count'):
                if isinstance(checks[field],int) and checks[field]!=0:raise ValueError('FORBIDDEN_COUNTER_BOUNDARY')
            integrity.append(checks)
            facts,req,pub=summarize(endpoint['evaluation'],records,entry['evaluation'])
            loss,old=old_loss(raw['old_before'],raw['old_after'])
            action=raw.get('actual_physical_action',{})
            row=dict(cell=cell,arm=arm,fixture='D10B',status=raw['status'],**facts,**loss,
                V_ratio=raw.get('V_ratio'),native_net_raw=action.get('native_net_raw'),
                native_net_normalized=action.get('native_net_normalized'),frobenius_net_sq=action.get('frobenius_net_sq'),
                E_T=raw.get('E_T','NOT_RECORDED_HISTORICAL_OR_OFFICIAL_PATH'),
                L_N=raw.get('L_N','NOT_RECORDED_HISTORICAL_OR_OFFICIAL_PATH'),
                raw_integrated_work=raw.get('raw_integrated_work'),raw_path_length=raw.get('raw_path_length'),
                write_wall_seconds=raw['total_seconds']-raw['evaluation_seconds'],eval_wall_seconds=raw['evaluation_seconds'],
                model_forward_calls=raw['forward_count'],main_JVP_calls=raw['main_jvp_count'],
                diagnostic_JVP_calls=raw['diagnostic_jvp_count'],peak_gpu_bytes=raw['peak_allocated_gpu_bytes'])
            main.append(row);requests.extend(dict(cell=cell,arm=arm,**r) for r in req)
            oldrows.extend(dict(cell=cell,arm=arm,**r) for r in old)
            compute.append({k:row[k] for k in ('cell','arm','write_wall_seconds','eval_wall_seconds','model_forward_calls','main_JVP_calls','diagnostic_JVP_calls','peak_gpu_bytes')})
            for node in raw.get('nodes',[]):
                clean={k:v for k,v in node.items() if k not in ('normalization_diagnostics','normalization_shadows')}
                nodes.append(dict(cell=cell,**clean))
                for v in node.get('normalization_diagnostics',[]):
                    j=v['column'];q=node['q_layers'][j]
                    if q<=0:raise ValueError('NORMALIZATION_RECORDED_Q_BOUNDARY')
                    # This is an exact coordinate conversion of recorded raw response moments,
                    # not a new response/JVP or an imputation of unrecorded values.
                    normalization.append(dict(cell=cell,arm=arm,node=node['node'],layer=node['active_layers'][j],**v,
                        g_i_native_whitened=[x/math.sqrt(q) for x in v['gain_by_request']],
                        r_i_native_whitened=[x/q for x in v['response_energy_by_request']],
                        conversion='g_raw/sqrt(q_l), r_raw/q_l from same recorded node'))
                normalization.extend(dict(cell=cell,arm=arm,node=node['node'],**v) for v in node.get('normalization_shadows',[]))
            shadows.extend(dict(cell=cell,**r) for r in raw.get('same_state_fields',[]))
            save(output/f'endpoint-{cell}-{arm}.json',pub)
    paired=paired_deltas(requests);matched=matched_progress(main,nodes)
    files={'pilot_main_table.csv':main,'endpoint_metrics.csv':requests,'old_edit_metrics.csv':oldrows,
           'trajectory_nodes.csv':nodes,'same_state_fields.csv':shadows,'normalization_diagnostics.csv':normalization,
           'paired_endpoint_deltas.csv':paired,'matched_progress.csv':matched,'integrity.csv':integrity,'setup_accounting.csv':setup,
           'conflict_observations.csv':conflicts,
           'compute_accounting.csv':compute,'run_registry.csv':[dict(cell=s['cell'],status=s.get('status',s.get('type')),completed=s.get('completed_primary')) for s in status]}
    for name,rows in files.items():csv_once(output/name,rows)
    save(output/'external-inputs.json',inputs)
    for name in ['science.lock.json','sample.lock.json','source.lock.json','resource.lock.json','cpu_algebra_checks.json']:
        save(output/name,json.loads((root/name).read_text()))
    for name in ('cap4-control-override.json','gpu-hour-ledger-before-submit.json','technical-exclusion.json','held-inspection.json'):
        if (root/name).is_file():save(output/name,json.loads((root/name).read_text()))
    analysis_repo=Path(__file__).resolve().parents[3]
    analysis_source=dict(head=git(analysis_repo,'rev-parse','HEAD'),tree=git(analysis_repo,'rev-parse','HEAD^{tree}'),
        path=str(analysis_repo),analyzer_sha256=sha(__file__),separate_from_execution=True)
    save(output/'analysis-source.lock.json',analysis_source)
    save(output/'gpu_fidelity_checks.json',[dict(cell=i,**json.loads((root/f'cell-{i}'/'gpu_fidelity_checks.json').read_text()))
        for i in range(4) if (root/f'cell-{i}'/'gpu_fidelity_checks.json').exists()])
    primary=[r for r in main if r['arm'] in ARM_ORDER]
    verdict='PRIMARY_FOUR_CELL_TECHNICAL_PASS' if len(primary)==16 and all(s.get('status')=='TERMINAL_VALID' for s in status) else 'INCOMPLETE_OR_BOUNDARY'
    decision=dict(status=verdict,primary_arm_endpoints=len(primary),expected=16,primary_request_endpoints=sum(r['RS_d'] for r in primary),
        expected_request_endpoints=160,scientific_promotion=False,server4_rerun_mutation=0,
        refinement='PENDING_POST_PRIMARY_COMMON_GATE',historical_anomalies='HISTORICAL_STATE_UNAVAILABLE_NO_REPLAY')
    save(output/'decision_summary.json',decision)
    headline=[]
    for r in main:
        headline.append(dict(cell=r['cell'],arm=r['arm'],RS=f"{r['RS_n']}/{r['RS_d']} ({100*r['RS_rate']:.2f}%)",
            PS=f"{r['PS_n']}/{r['PS_d']} ({100*r['PS_rate']:.2f}%)",NS=f"{r['NS_n']}/{r['NS_d']} ({100*r['NS_rate']:.2f}%)",
            old_RS=f"{r.get('old_before_RS_n','—')}→{r.get('old_after_RS_n','—')}",
            old_new_failure=f"{r.get('new_failure_n','—')}/{r.get('entry_success_d','—')}",V_ratio=r.get('V_ratio','—')))
    turn_rows=[]
    for cell in range(4):
        selected=[s for s in shadows if s['cell']==cell and s['comparator']=='ORB_RAY_N' and s['lambda_value']==.1]
        vals=[s['Rturn'] for s in selected if s.get('Rturn') is not None]
        turn_rows.append(dict(cell=cell,recorded=len(selected),nonzero_angle_rows=len(vals),
            Rturn_positive=sum(v>0 for v in vals),**{f'Rturn_{k}':v for k,v in distribution(vals).items()}))
    csv_once(output/'turning_summary.csv',turn_rows)
    missing=[dict(field='Frobenius_cosine',status='NOT_RECORDED_SCHEMA_GAP',impact='QN angle is not Frobenius angle; no equivalence claim'),
        dict(field='intermediate_matched_progress_endpoint_evaluation',status='NOT_EVALUATED',impact='saved residual alone is not matched-quality/retention evidence'),
        dict(field='Official_ORBFH_continuous_native_path',status='NOT_RECORDED_SCHEMA_GAP',impact='actual endpoint native action present; do not infer integrated work'),
        dict(field='model_forward_setup_breakdown',status='NOT_RECORDED_SCHEMA_GAP',impact='scoped arm counts and setup seconds remain factual')]
    csv_once(output/'missing_fields.csv',missing)
    report=f'''# Native-response v3.1 B10 short-history mechanism pilot

상태: {verdict}. scientific_promotion=false. 본 자료는 단일 Official D10A(B10) warm history 뒤 독립 D10B(B10) 비교다. B9 재현, lifelong 결과 또는 locality 보장으로 해석하지 않는다. Server4 retention rerun과 별도 source/process/표본이며 진행 중인 다른 실험 결과는 사용하지 않았다.

## 지표와 분모

RS/PS는 각각 rewrite/rephrase에서 target-new NLL < target-true NLL인 prompt 수/전체 prompt 수다. NS는 반대로 neighborhood target-true NLL < target-new NLL이다. tie는 실패다. NLL은 token 평균이며 낮을수록 해당 continuation의 likelihood가 높다. strict teacher-forced coverage는 target token 전부 argmax와 일치하는 prompt 수다. old loss는 D10A의 warm-entry 성공 요청 중 D10B endpoint에서 실패한 요청만 센다. initially failed는 별도다.

{md_table(headline,['cell','arm','RS','PS','NS','old_RS','old_new_failure','V_ratio'])}

## 1. 수학 및 fidelity

CPU algebra/overlay 체크와 GPU raw-direction FD, RHS scaling, joint-state materialized parity는 각각 cpu_algebra_checks.json, gpu_fidelity_checks.json에 실제 결과를 결속했다. 주 controller는 FP64 nonnegative active-set reference solver이며 model/forward/overlay는 FP32다. MEMIT의 native ephemeral FP64 solve는 보존한다. Fixed qN_ref와 entry-normalization을 node 또는 comparator별로 다시 정하지 않는다.

## 2. Historical anomaly

LM-ORBFH B9, LM-JAC B6, LA-JAC B5: historical report ref=f2dcfd4ab6fcb2917ae0a29cbc384cf95a82eb3c. 그 보고서의 execution sources는 별도 source.lock.json에 있다. exact historical entry W/method-state/z가 이 pilot 입력에 없으므로 HISTORICAL_STATE_UNAVAILABLE. 과거 B1–B8 replay는 하지 않았다.

## 3. Same-state physical turning

same_state_fields.csv는 같은 raw dictionary/JVP로 native joint, ORB ray, diagonal ray, raw ORB snapshot 및 실제 Frobenius metric을 비교한다. gᵀc의 차이, native Rturn>0, endpoint usefulness는 서로 다른 판정이다. 영벡터 angle은 N/A이며 primary 선택에 shadow λ 또는 normalization을 사용하지 않았다.

{md_table(turn_rows,['cell','recorded','nonzero_angle_rows','Rturn_positive','Rturn_mean','Rturn_median','Rturn_p90','Rturn_max'])}

## 4. Actual path

trajectory_nodes.csv는 node entry/exit V, raw/native normalized action, integrated work E, path length L, Euler model error, response velocity mismatch 및 barrier defect를 분리한다. 바뀌는 dictionary의 계수 합을 physical displacement로 쓰지 않는다. 실제 materialized block ΔW의 native/Frobenius action은 pilot_main_table.csv에 별도 기록한다. finite Euler defect/near-stall/낮은 성능은 제외 사유가 아니다.

## 5. Refinement

공통 four-cell primary 표가 먼저다. D2 fixed T2 N2/4/8은 별도 후속 budget/공통 gate를 통과한 경우에만 수행한다. 이 primary package에서 미실행 refinement를 convergence PASS로 주장하지 않는다.

## 6. Matched-progress usefulness

사전 값 V/V0={{.75,.5,.25}}, saved node의 절대 오차 .02 이내만 비교 가능하다. interpolation은 금지한다. 같은 V는 같은 efficacy가 아니다. Primary terminal이 서로 다른 progress이면 단순 endpoint 성능 차이를 matched-progress 이득으로 부르지 않는다. 중간 상태의 evaluation이 없으면 NOT_EVALUATED이며 추정하지 않는다.

## 7. D10A loss

old_edit_metrics.csv는 request별 before/after NLL advantage, entry-success/post-failure 및 initially-failed를 분리한다. old prompts/margins는 평가 callback에서만 읽으며 controller·normalization·stopping에 입력하지 않는다. 파생 first-hit 평가는 주 endpoint의 old-edit 관측값을 대체하지 않는다.

## 8. Native와 Frobenius

QN은 native S에 의한 action을 entry qN_ref로 나눈 것이다. 다른 weight block의 native-whitened G=I는 Frobenius metric을 뜻하지 않는다. GF는 독립 계산한다. absolute cross-model action을 같은 척도라고 해석하지 않는다.

## 9. Compute와 미실행

compute_accounting.csv의 실제 model.forward/JVP 수와 CUDA-synchronized wall을 분리한다. FLOPs 추정으로 대체하지 않는다. target/setup/endpoint observation의 비용 범위를 구분하며, native dense endpoint action의 관측 비용은 evaluator 범위다. refinement/audit는 남은 2GPUh/cell 및 8GPUh total 예산으로만 진행한다. 예산 부족은 NOT_RUN_BUDGET이며 outcome 기반 탈락은 없다.

## 10. 독립성 및 한계

Source {source['head']} / tree {source['tree']}. 표본 38개 hash-rank seal과 reserved 1000 overlap0은 sample.lock.json에 결속한다. 4 cells×4 primary arms×10 requests가 완성 분모다. 현재 실제 분모는 {len(primary)}/16 arm endpoints, {decision['primary_request_endpoints']}/160 request endpoints다. raw 입력은 수정하지 않았고 보고서에는 prompts/targets/token IDs/weight/cache tensors를 복사하지 않는다. automatic promotion·rescue·budget expansion=0.

Analysis source {analysis_source['head']} / tree {analysis_source['tree']}는 실행 source와 분리했다. `integrity.csv`는 source/entry/fixed-z/restore를 독립 재검산하며, `paired_endpoint_deltas.csv`는 cell·request·prompt 단위 method−Official 차이다. 다른 model/family 사이 absolute metric 크기를 직접 비교하지 않는다. native scalar 기하와 Frobenius 기하의 동등성은 가정하지 않는다.

### 누락/관측 경계

{md_table(missing,['field','status','impact'])}

### 계산량과 setup

{md_table(compute,['cell','arm','write_wall_seconds','eval_wall_seconds','model_forward_calls','main_JVP_calls','diagnostic_JVP_calls','peak_gpu_bytes'])}

`write_wall_seconds`는 전체 arm 시간에서 공통 endpoint evaluator callback 시간을 제외한 관측값이며, dictionary/solve/shadow algebra/transaction 비용을 포함한다. setup/target/model load는 `setup_accounting.csv`에 별도로 기록한다. 동시 GPU contention 및 CPU metric 관측 비용이 wall에 영향을 줄 수 있으므로 FLOPs 또는 순수 GPU kernel 비용이라고 부르지 않는다.
'''
    path=output/'factual-report-ko.md'
    with path.open('x') as h:h.write(report)
    path.chmod(0o600)
    members=[dict(path=str(p.relative_to(output)),sha256=sha(p),bytes=p.stat().st_size,mode='0600') for p in sorted(output.iterdir()) if p.is_file()]
    manifest=dict(members=members,members_root=canonical_hash(members),input_root=canonical_hash(inputs),scientific_promotion=False)
    save(output/'manifest.json',manifest)
    receipt=dict(status=verdict,manifest_sha256=sha(output/'manifest.json'),members_root=manifest['members_root'],
        report_sha256=sha(path),input_root=manifest['input_root'],source_head=source['head'],input_mutation_count=0,
        imputation_count=0,scientific_promotion=False)
    receipt['identity']=canonical_hash(receipt);save(output/'rooted-receipt.json',receipt)
    for member in inputs:
        if sha(member['path'])!=member['sha256']:raise ValueError('RAW_INPUT_CHANGED_DURING_ANALYSIS')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-boundary',action='store_true');a=p.parse_args()
    print(json.dumps(build_package(a.root,a.output,allow_boundary=a.allow_boundary),sort_keys=True))
