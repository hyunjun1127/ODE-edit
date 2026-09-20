"""CPU-only raw-free reports from one bounded EN adaptive attempt.

Missing evidence stays missing. Cumulative counters are never added across
batches, and a shared B1 native/geometry/gradient is fully charged to each
standalone EN estimate rather than divided by the number of comparisons.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import metrics

ARMS=('N4','EN_EXACT','EN_NUM','EN_ADAPT')
CHAINS=('N4','EN_EXACT','EN_ADAPT')


def _read(path, receipts):
    if not path.is_file():return None
    raw=path.read_bytes()
    receipts[str(path.resolve())]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)


def _csv(path, rows, fields=()):
    rows=list(rows)
    keys=list(fields)
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool)


def _fmt(value):
    return '미측정' if value is None else f'{value:.6g}' if _number(value) else str(value)


def _endpoint_rows(observation,batch,arm,scope):
    result=[]
    for family in ('RS','PS','NS'):
        a=observation['aggregates'][family];p=observation['metrics'][family]
        result.append(dict(batch=batch,arm=arm,scope=scope,family=family,requests=len(observation['request_ids']),
            preference_success=p['numerator'],preference_prompts=p['denominator'],preference_rate=p['rate'],
            tf_token_correct=a['tf_token_micro']['numerator'],tf_valid_tokens=a['tf_token_micro']['denominator'],
            tf_token_micro=a['tf_token_micro']['rate'],tf_prompt_macro=a['tf_prompt_macro']['value'],
            tf_strict_correct=a['tf_strict']['numerator'],tf_strict_prompts=a['tf_strict']['denominator'],tf_strict_rate=a['tf_strict']['rate'],
            new_nll_prompt_macro=a['nll']['new_prompt_macro'],true_nll_prompt_macro=a['nll']['true_prompt_macro'],
            desired_nll_prompt_macro=a['nll']['desired_prompt_macro'],desired_margin_prompt_macro=a['desired_margin_prompt_macro']))
    return result


def _paired_rows(analysis,batch,arm,comparison):
    output=[]
    for family,group in analysis.get('families',{}).items():
        rows=group.get('rows',[]);ci=group.get('cluster_bootstrap') or {}
        item=dict(batch=batch,arm=arm,comparison=comparison,family=family,prompt_denominator=group['denominator'],
            lost_ids=group.get('lost_ids',[]),gained_ids=group.get('gained_ids',[]),
            strict_lost_ids=group.get('strict_lost_ids',[]),strict_gained_ids=group.get('strict_gained_ids',[]))
        for key in ('new_nll','true_nll','desired_nll','desired_margin'):
            values=[r[key+'_delta'] for r in rows if key+'_delta' in r]
            item[key+'_delta_prompt_macro']=sum(values)/len(values) if values else None
        for name,value in ci.items():
            if value:
                item[name+'_paired_delta']=value['delta'];item[name+'_paired_ci95']=value['percentile95']
                item[name+'_defined_bootstrap_resamples']=value['defined_resamples']
        item['token_lost_positions']=sum(len(r.get('token_lost_positions',[])) for r in rows)
        item['token_gained_positions']=sum(len(r.get('token_gained_positions',[])) for r in rows)
        output.append(item)
    if analysis.get('joint'):
        group=analysis['joint'];rows=group.get('rows',[])
        output.append(dict(batch=batch,arm=arm,comparison=comparison,family='R_plus_twoP_joint',request_denominator=group['denominator'],
            lost_ids=[r['case_id'] for r in rows if r.get('preference_lost')],gained_ids=[r['case_id'] for r in rows if r.get('preference_gained')],
            strict_lost_ids=[r['case_id'] for r in rows if r.get('strict_lost')],strict_gained_ids=[r['case_id'] for r in rows if r.get('strict_gained')],
            cluster_bootstrap=group.get('cluster_bootstrap')))
    return output


def _metadata(output,receipts):
    result={}
    for name in ('execution.lock.json','execution.lock','submission.json','accounting.json','slurm-accounting.json'):
        for parent in (output,output.parent):
            path=parent/name
            if path.is_file():
                value=_read(path,receipts)
                result[name]=dict(path=str(path.resolve()),sha256=receipts[str(path.resolve())]['sha256'],value=value)
                break
    return result


def _accounting(metadata):
    """Only explicit allocation values; process wall time is not allocation."""
    rows=[]
    for name,entry in metadata.items():
        if 'accounting' not in name:continue
        value=entry['value'];items=value if isinstance(value,list) else value.get('jobs',[value]) if isinstance(value,dict) else []
        for item in items:
            if not isinstance(item,dict):continue
            # Root can provide the explicit canonical GPU-hour field regardless
            # of scheduler CLI formatting; missing fields are never inferred.
            gpu_h=item.get('allocation_gpu_hours',item.get('gpu_hours'))
            gpu_seconds=item.get('allocation_gpu_seconds',item.get('gpu_seconds'))
            if gpu_h is None and _number(gpu_seconds):gpu_h=gpu_seconds/3600
            rows.append(dict(scope='ACTUAL_ALLOCATION',component='scheduler_allocation',job_id=item.get('job_id',item.get('JobID')),
                state=item.get('state',item.get('State')),exit_code=item.get('exit_code',item.get('ExitCode')),
                allocation_gpu_hours=gpu_h,seconds=item.get('elapsed_seconds',item.get('ElapsedRaw')),
                source=entry['path'],allocation_cost_status='MEASURED' if _number(gpu_h) else 'GPU_ALLOCATION_NOT_SUPPLIED'))
    return rows


def report(output,destination):
    output,destination=Path(output),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    receipts={};missing=[]
    def read(name,required=False):
        result=_read(output/name,receipts)
        if result is None and required:missing.append(name)
        return result
    complete=read('complete.json');failure=read('technical-failure.json')
    t0=read('T0-result.json') or read('T0/t0-observations.json') or read('T0/t0-failure.json')
    metadata=_metadata(output,receipts)
    endpoints=[];paired=[];retention=[];frontier=[];candidates=[];costs=[];statuses=[]
    refs=[];latest_cost=None;core_cost={};shared_native={};shared_geo={};shared_gradient={}
    for batch in range(1,4):
        arms=ARMS if batch==1 else CHAINS
        baseline=read(f'B{batch}/W0-current.json')
        current_ids=baseline['request_ids'] if baseline else None
        if baseline:endpoints.extend(_endpoint_rows(baseline,batch,'W0','current'))
        for group in (('SHARED',) if batch==1 else CHAINS):
            native=read(f'B{batch}/{group}-native.json')
            if native:
                sec=native.get('seconds')
                shared_native[batch,group]=sec
                costs.append(dict(scope='SHARED_ACTUAL_COMPONENT',batch=batch,arm=group,component='native_fit',seconds=sec,source=f'B{batch}/{group}-native.json'))
            spectrum=read(f'B{batch}/{group}-spectrum.json')
            if spectrum:
                timing=spectrum.get('timing',{})
                geo=sum(timing[k] for k in ('current_capture','weighted_TSQR_SVD','spectrum_projection') if _number(timing.get(k)))
                shared_geo[batch,group]=geo if timing else None
                shared_gradient[batch,group]=timing.get('objective_gradient')
                costs.extend([dict(scope='SHARED_ACTUAL_COMPONENT',batch=batch,arm=group,component='geometry_capture_SVD_projection',seconds=shared_geo[batch,group]),
                              dict(scope='SHARED_ACTUAL_COMPONENT',batch=batch,arm=group,component='R_plus_H_gradient',seconds=shared_gradient[batch,group])])
                for eps,rows in spectrum.get('selection',{}).get('frontiers',{}).items():
                    for row in rows:frontier.append(dict(batch=batch,group=group,epsilon=eps,**row))
            objective=read(f'B{batch}/{group}-native-objective.json')
            if objective:
                refs.append(dict(batch=batch,arm=group,phase='native',L_R=objective.get('L_R'),L_H=objective.get('L_H'),J=objective.get('J'),reference_documents=objective.get('reference_documents'),reference_positions=objective.get('reference_positions')))
        for arm in arms:
            summary=read(f'B{batch}/{arm}-summary.json',True)
            obs=read(f'B{batch}/{arm}-metrics.json',True)
            analysis=read(f'B{batch}/{arm}-paired-analysis.json',True)
            controller=read(f'B{batch}/{arm}-controller.json',arm!='N4')
            if summary:
                statuses.append(dict(batch=batch,arm=arm,selection_status=summary.get('selection_status'),observer_alias=summary.get('observer_alias'),
                                     controller_alias=(controller or {}).get('alias'),selected_trial=(controller or {}).get('selected_trial')))
                costs.append(dict(scope='SHARED_ACTUAL_COMPONENT',batch=batch,arm=arm,component='post_selection_observer_analysis_dev_teacher_bundle',seconds=summary.get('observer_seconds'),observer_alias=summary.get('observer_alias')))
            if obs:
                endpoints.extend(_endpoint_rows(obs,batch,arm,'all_seen'))
                if current_ids:endpoints.extend(_endpoint_rows(metrics.subset(obs,current_ids),batch,arm,'current'))
                first=obs['request_ids'][:min(100,len(obs['request_ids']))]
                endpoints.extend(_endpoint_rows(metrics.subset(obs,first),batch,arm,'first100'))
            if analysis:
                for name in ('atwrite_to_now','first100_from_B1'):
                    if analysis.get(name):paired.extend(_paired_rows(analysis[name],batch,arm,name))
                r=analysis.get('W0_correct_neighborhood')
                if r:
                    retention.append(dict(batch=batch,arm=arm,scope='all_seen',
                        preference=r['w0_correct_preference_retention'],strict=r['w0_correct_strict_retention'],token=r['w0_correct_token_retention'],
                        recovered_ids=r.get('recovered_ids',[]),newly_lost_ids=r.get('newly_lost_ids',[])))
                # Active-past scope is defined by the runtime ledger; do not
                # substitute all prior records when active IDs are not stored.
                if analysis.get('active_past'):
                    for family,a in analysis['active_past'].items():
                        endpoints.append(dict(batch=batch,arm=arm,scope='active_past',family=family,
                            preference_rate=None,preference_status='NOT_IN_STORED_ACTIVE_PAST_AGGREGATE',
                            tf_token_correct=a['tf_token_micro']['numerator'],tf_valid_tokens=a['tf_token_micro']['denominator'],
                            tf_token_micro=a['tf_token_micro']['rate'],tf_prompt_macro=a['tf_prompt_macro']['value'],
                            tf_strict_rate=a['tf_strict']['rate'],new_nll_prompt_macro=a['nll']['new_prompt_macro'],
                            true_nll_prompt_macro=a['nll']['true_prompt_macro'],desired_nll_prompt_macro=a['nll']['desired_prompt_macro']))
            versus=read(f'B{batch}/{arm}-versus-N4.json',arm!='N4')
            if versus:paired.extend(_paired_rows(versus,batch,arm,'versus_N4'))
            candidate_actual=0.;candidate_standalone=0.;seen_alias=set()
            if controller:
                for row in controller.get('ledger',[]):
                    obj=row.get('objective',{});geo=row.get('geometry',{})
                    candidates.append(dict(batch=batch,arm=arm,trial=row.get('trial'),scale=row.get('scale'),accepted=row.get('accepted'),alias=row.get('alias'),
                        J=obj.get('J'),L_R=obj.get('L_R'),L_H=obj.get('L_H'),actual_gradient_inner_product=row.get('actual_gradient_inner_product'),
                        ideal_gradient_inner_product=row.get('ideal_gradient_inner_product'),strict_decrease=row.get('strict_decrease'),armijo=row.get('armijo'),
                        response_budget=row.get('response_budget'),semantic_epsilon=row.get('semantic_epsilon'),geometry_checks=row.get('geometry_checks'),
                        **{k:v for k,v in geo.items() if k!='request_response'}))
                    alias=row.get('alias');sec=obj.get('seconds')
                    if _number(sec):
                        if alias is None:candidate_actual+=sec
                        if alias!='native' and not (isinstance(alias,str) and alias.startswith(arm+':')) and alias not in seen_alias:
                            candidate_standalone+=sec
                            if alias is not None:seen_alias.add(alias)
                costs.append(dict(scope='SHARED_ACTUAL_COMPONENT',batch=batch,arm=arm,component='candidate_objective',seconds=candidate_actual,actual_evaluations=controller.get('evaluations')))
            group='SHARED' if batch==1 else arm
            base=shared_native.get((batch,group))
            geo=0. if arm=='N4' else shared_geo.get((batch,group))
            grad=0. if arm=='N4' else shared_gradient.get((batch,group))
            core=None if any(x is None for x in (base,geo,grad)) else base+geo+grad+candidate_standalone
            if summary:
                core_cost[batch,arm]=core
                costs.append(dict(scope='STANDALONE_RECONSTRUCTED_CORE',batch=batch,arm=arm,component='native_plus_geometry_gradient_candidates',seconds=core,
                    native_seconds=base,geometry_seconds=geo,gradient_seconds=grad,candidate_seconds=candidate_standalone,
                    shared_B1_cost_divisor=1,excludes='T0,teacher preparation,history capture,observers,allocation overhead; streaming I/O and hashing within measured calls NOT_SEPARATED; no speedup claim'))
            dev=read(f'B{batch}/{arm}-Dev128.json')
            if dev:refs.append(dict(batch=batch,arm=arm,phase='Dev128_observer',L_R=dev.get('L_R'),L_H=dev.get('L_H'),J=dev.get('J'),reference_documents=dev.get('reference_documents'),alias=dev.get('alias')))
        batch_cost=read(f'B{batch}/cost.json')
        if batch_cost:latest_cost=batch_cost
    costs.extend(_accounting(metadata))
    if t0:costs.append(dict(scope='TECHNICAL_SEPARATE',component='T0',seconds=t0.get('seconds'),included_in_scientific_native_denominator=False))
    if complete:costs.append(dict(scope='PROCESS_WALL_OBSERVED',component='whole_process',seconds=complete.get('seconds'),allocation_gpu_hours=None,allocation_cost_status='PROCESS_SECONDS_NOT_ALLOCATION_GPUH'))
    status='TECHNICAL_FAILURE' if failure else 'B300_COMPLETE' if complete and complete.get('status')=='B300_COMPLETE' and not missing else 'INCOMPLETE'
    artifact_rows={'endpoint-metrics.csv':endpoints,'official-paired.csv':paired,'sequential-retention.csv':retention,
                   'threshold-frontier.csv':frontier,'candidate-ledger.csv':candidates,'costs.csv':costs,'reference-paired.csv':refs,'endpoint-status.csv':statuses}
    for name,rows in artifact_rows.items():_csv(destination/name,rows,('batch','arm') if name!='costs.csv' else ('scope','component'))
    counters=(complete or {}).get('counts') or (latest_cost or {}).get('objective')
    model_complete=complete if complete else latest_cost or {}
    manifest=dict(schema='EN_ADAPTIVE_REPORT_V1',status=status,source_output=str(output.resolve()),
        complete_marker_present=complete is not None,technical_failure_present=failure is not None,
        missing_required_evidence=missing,input_receipts=receipts,metadata={k:{x:y for x,y in v.items() if x!='value'} for k,v in metadata.items()},
        counts=counters,science_native_batches=model_complete.get('native_batches'),science_native_requests=model_complete.get('native_requests'),
        weighted_SVD=model_complete.get('weighted_SVD'),counter_policy='latest cumulative receipt only; never summed across B1/B2/B3',
        source_files_no_raw_prompts=True,precision_status=(t0 or {}).get('precision_status','NOT_ESTABLISHED'),
        save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',artifacts={name:dict(rows=len(rows),sha256=hashlib.sha256((destination/name).read_bytes()).hexdigest()) for name,rows in artifact_rows.items()})
    (destination/'report-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=[f'# EN adaptive nullspace B300 저장 자료 reducer 보고', '',f'실행 상태: **{status}**. 실제 존재하는 관측 파일만 집계했다. 누락 필수 산출물 {len(missing)}개.',
        f'실행 원자료: `{output.resolve()}`. 재현 원자료와 입력 SHA는 `report-manifest.json`에 결속했다.', '',
        'N4, EN_EXACT, EN_NUM, EN_ADAPT의 B1과 N4, EN_EXACT, EN_ADAPT의 B2/B3가 승인 범위다. EN_NUM sequential 및 B1000/10k는 이 보고의 실행 범위에 없다.', '',
        f'T0 precision: **{manifest["precision_status"]}**. 수치 동등성 미확립은 탐색적 한계로 남긴다. 이전 waiver나 CPU 테스트를 실제 새 모델 경로 PASS로 해석하지 않는다.', '',
        '| Batch | Arm | Scope | RS | PS | NS |', '|---|---|---|---:|---:|---:|']
    for batch in range(1,4):
        for arm in ARMS if batch==1 else CHAINS:
            for scope in ('current','all_seen','first100'):
                rows={r['family']:r for r in endpoints if r.get('batch')==batch and r.get('arm')==arm and r.get('scope')==scope}
                if rows:lines.append('| '+ ' | '.join([str(batch),arm,scope]+[_fmt(rows.get(f,{}).get('preference_rate')) for f in ('RS','PS','NS')])+' |')
    lines += ['', 'RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 동률은 실패다. `endpoint-metrics.csv`의 nl rewrite/rephrase/neighborhood accuracy는 desired target의 teacher-forced token accuracy다. Token-micro, prompt-macro, 전체 target strict를 분리했다. 자유생성 정확도가 아니다.',
        'NLL은 target 토큰 평균 후 prompt 평균으로 집계한다. Desired margin은 R/P에서 true−new, N에서 new−true NLL이며 양수가 desired 선호다. 동일 총점이어도 lost/gained ID가 다른 경우를 `official-paired.csv`에 보존했다. R+두 P joint strict와 preference도 별개다.',
        'Paired CI는 request cluster 10,000회, seed 20260920의 percentile 95% 구간이다. 한 request의 모든 neighbor와 토큰이 함께 resample된다. 성능 지표는 최종 endpoint 선택 후 observer에서만 사용했다.', '',
        '| Batch | Arm | Selection | Controller alias | Observer alias |','|---|---|---|---|---|']
    for row in statuses:lines.append('| '+' | '.join(_fmt(row.get(k)) for k in ('batch','arm','selection_status','controller_alias','observer_alias'))+' |')
    lines += ['', 'Fallback은 정상 native 선택 결과이며 기술 exception과 구별한다. Alias는 정확히 같은 FP32 endpoint를 재사용한 사실이다. Algebra-only epsilon .01/.1은 실제 모델 품질 결과가 아니다. `threshold-frontier.csv`와 `candidate-ledger.csv`에서 rank/frontier, ideal→actual rounding, L_R/L_H 및 Armijo를 함께 확인할 수 있다.', '',
        '| Cost scope | Component | Seconds | Allocation GPUh |','|---|---|---:|---:|']
    for row in costs:
        if row['scope'] in ('ACTUAL_ALLOCATION','TECHNICAL_SEPARATE','PROCESS_WALL_OBSERVED'):
            lines.append('| '+' | '.join(_fmt(row.get(k)) for k in ('scope','component','seconds','allocation_gpu_hours'))+' |')
    lines += ['', '공유 실행 비용과 standalone core 재구성은 `costs.csv`에서 분리했다. B1 native·geometry·gradient는 공유 실제 비용에서 한 번, 각 standalone EN에서 전체 비용으로 계산한다. Standalone core는 teacher 준비, history capture, observer, T0, allocation overhead를 제외한 부분비용이다. 측정 call 내부의 streaming I/O와 hashing은 포함되며 NOT_SEPARATED다. 완전한 standalone 실행 비용이나 속도향상을 주장하지 않는다.',
        '기존 teacher 생성/기존 native 자산의 과거 비용은 신규 GPU allocation에 재계상하지 않는다. 과거 비용 영수증이 없으면 미측정이다. Process wall seconds만으로 allocation GPUh를 추정하지 않는다. 누적 B1/B2/B3 카운터는 마지막 영수증 하나만 사용한다.',
        f'과학 native batches={_fmt(manifest["science_native_batches"])}, requests={_fmt(manifest["science_native_requests"])}, weighted SVD={_fmt(manifest["weighted_SVD"])}; objective counters=`{json.dumps(counters,ensure_ascii=False)}`.', '',
        'save_checkpoints=false. Edited W/M/optimizer/동등 delta·resume bundle을 저장하지 않았다. Exact crash-resume은 NOT_AVAILABLE이다. 모델·teacher·key 입력 자산은 edited checkpoint와 구별한다.',
        'Active-past 선호율이 저장 aggregate에 없으면 미측정으로 남긴다. All-seen 관측을 active-past라고 대체하지 않는다. 미승인 B1000/10k, 새 arm, sweep 또는 과학 task 자동 재개는 없다.']
    if missing:lines += ['', '누락 필수 증거:', *[f'- `{name}`' for name in missing]]
    if failure:lines += ['', '기술 실패가 존재한다. 상세 exception과 비용 원자료는 실행 디렉터리의 `technical-failure.json`을 확인한다. 실패 이후 endpoint를 완성 결과로 채우지 않았다.']
    if metadata:lines += ['', '실행·제출·회계 원본 영수증:', *[f'- `{entry["path"]}` SHA256 `{entry["sha256"]}`' for entry in metadata.values()]]
    text='\n'.join(lines)+'\n'
    (destination/'report.md').write_text(text)
    (destination/'report-ko.md').write_text(text)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();print(json.dumps(report(args.output,args.destination),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
