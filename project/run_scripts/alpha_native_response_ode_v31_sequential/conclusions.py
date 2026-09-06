"""Completed-receipt interpretation tables; no outcome gates or model access."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from .reporting import csvwrite


def rows(out, name):
    with (Path(out)/(name+'.csv')).open() as f:
        return list(csv.DictReader(f))


def number(row, key):
    value=row.get(key)
    return float(value) if value not in ('', None) else None


def fmt(value):
    return 'NOT_RECORDED' if value is None else f'{value:.6g}'


def pair(row, prefix):
    return f"{row[prefix+'_num']}/{row[prefix+'_den']}"


def layer_trend(data):
    groups=defaultdict(list)
    for row in data:groups[(row['alias'],row['arm'],int(row['batch']))].append(row)
    result=[]
    for (alias,arm,batch),rr in sorted(groups.items()):
        total=sum(float(x['endpoint_DeltaW_squared']) for x in rr)
        l8=sum(float(x['endpoint_DeltaW_squared']) for x in rr if int(x['layer'])==8)
        early=total-l8
        work=sum(number(x,'integral_normalized_native_velocity_action') or 0 for x in rr)
        history=sum(number(x,'integral_history_velocity_action') or 0 for x in rr)
        raw=sum(number(x,'integral_raw_native_velocity_action') or 0 for x in rr)
        measured=any(x['velocity_trajectory_status']=='RECORDED' for x in rr)
        l8work=sum(number(x,'integral_normalized_native_velocity_action') or 0 for x in rr if int(x['layer'])==8)
        result.append(dict(alias=alias,arm=arm,batch=batch,endpoint_energy=total,
            L8_endpoint_energy=l8,L4_7_endpoint_energy=early,
            L8_endpoint_energy_share=l8/total if total else None,
            normalized_native_work=work if measured else None,
            L8_native_work_share=l8work/work if work else None,
            raw_native_work=raw if measured else None,history_work=history if measured else None,
            history_over_raw_work=history/raw if raw else None,
            L4_7_signed_predicted_progress=sum(number(x,'signed_predicted_target_progress') or 0 for x in rr if int(x['layer'])!=8) if measured else None,
            sum_of_batch_energy_is_not_cumulative_W10_W0_energy=True))
    return result


def make(out, completed):
    out=Path(out)
    final=rows(out,'final_metrics');current=rows(out,'current_batch_metrics')
    layers=layer_trend(rows(out,'physical_layer_batch_metrics'))
    csvwrite(out/'layer_redistribution_summary.csv',layers)
    transition=rows(out,'prompt_transition_metrics');groups=defaultdict(list)
    for x in transition:groups[(x['alias'],x['arm'],int(x['batch']))].append(x)
    transitions=[]
    for (alias,arm,batch),rr in sorted(groups.items()):
        yes=lambda x,k:x[k]=='True'
        transitions.append(dict(alias=alias,arm=arm,batch=batch,canonical_prompt_denominator=len(rr),
            W0_success_denominator=sum(yes(x,'W0_success') for x in rr),
            W0_failure_denominator=sum(not yes(x,'W0_success') for x in rr),
            W0_success_to_failure=sum(yes(x,'new_failure') for x in rr),
            W0_failure_to_recovery=sum(yes(x,'recovery') for x in rr)))
    csvwrite(out/'neighborhood_transition_summary.csv',transitions)
    text=['\n## 최종 해석을 위한 질문별 근거 (미완료 arm은 대입하지 않음)',
        '\n### 1. 실제 sequential 누적과 복원성',
        '\n각 완료 batch의 W/M commit→next-entry hash, history append1 및 commit 중 target/writer/evaluator 재실행0은 state_continuity_summary.csv와 sequential_commit_checks.csv에 있다. Semantic outcome과 무관하게 분모와 history를 유지했다. 실제 W/M checkpoint1/5/10 저장·tensor 재해시는 checkpoint_inventory.csv에 있으며, 전체 trajectory replay를 검증했다는 뜻은 아니다.',
        '\n### 2–3. L8 집중·물리적 재배분·감속을 구분',
        '\n| Model | Arm | B | actual batch ΔW energy | L8 energy share | L4–7 energy | L4–7 signed progress | L8 native work share | history/raw work |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in layers:
        if x['batch'] not in (1,5,10):continue
        text.append(f"| {x['alias']} | {x['arm']} | {x['batch']} | {fmt(x['endpoint_energy'])} | {fmt(x['L8_endpoint_energy_share'])} | {fmt(x['L4_7_endpoint_energy'])} | {fmt(x['L4_7_signed_predicted_progress'])} | {fmt(x['L8_native_work_share'])} | {fmt(x['history_over_raw_work'])} |")
    text+=['\nShare 감소는 절대 L4–7 write/기여 증가와 같지 않다. 위 energy는 해당 batch의 actual endpoint ΔW 제곱합이며, native work는 h를 한 번 곱한 velocity-action 적분이다. Official one-pass의 Euler velocity work는 NOT_RECORDED이며 0으로 대입하지 않았다. Batch별 metric/qN_ref 및 target이 달라지므로 coefficient만으로 physical migration을 주장하지 않는다.',
        '\n### 4. 같은 state에서 history 비용이 선택에 미친 직접 영향',
        '\n| Model | Arm | Batch/node | native cosine(actual vs initial cost) | Frobenius cosine | L8 c actual | L8 c initial-cost shadow | progress actual | progress shadow |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for x in rows(out,'history_cost_shadows'):
        a=json.loads(x['c_actual']);b=json.loads(x['c_initial']);angle=json.loads(x['frobenius_angle'])
        text.append(f"| {x['alias']} | {x['arm']} | {x['batch']}/{x['node']} | {fmt(number(x,'native_cosine'))} | {fmt(angle.get('native_cosine'))} | {fmt(a[-1])} | {fmt(b[-1])} | {fmt(number(x,'actual_predicted_progress'))} | {fmt(number(x,'shadow_predicted_progress'))} |")
    text+=['\n실제 G_initial, c와 physical coefficients는 history_cost_shadows.csv에 보존했다. 이는 actual basis/response/N0/qN_ref를 고정한 비용-only observer이며, history-free method나 retention에 대한 전체 인과 효과가 아니다. Zero-field는 receipt status를 유지하며 cosine을 생성하지 않는다.',
        '\n### 5. 신규 editability와 누적 retention은 별개의 비교',
        '\n첫 B100 current 비교는 초기 operating point 차이다. Final W10 전체 1,000 요청은 누적 결과다. 두 값을 섞어 초기 차이를 forgetting으로 설명하지 않는다.',
        '\n| Model | Arm | scope | RS | PS | NS | rewrite new/true NLL mean | rephrase new/true NLL mean |',
        '|---|---|---|---:|---:|---:|---|---|']
    for x in [x for x in current if int(x['batch'])==1]+final:
        scope=x.get('scope') or 'first current B100'
        text.append(f"| {x['alias']} | {x['arm']} | {scope} | {pair(x,'RS')} | {pair(x,'PS')} | {pair(x,'NS')} | {fmt(number(x,'rewrite_target_new_nll_mean'))} / {fmt(number(x,'rewrite_target_true_nll_mean'))} | {fmt(number(x,'rephrase_target_new_nll_mean'))} / {fmt(number(x,'rephrase_target_true_nll_mean'))} |")
    text+=['\nNLL median/p90/max, strict preference 및 teacher-forced token correctness는 current_batch_metrics.csv / final_metrics.csv의 별도 열에 보존한다. 자유 생성 accuracy는 NOT_RECORDED다.',
        '\n### 6. Official 대비 retention/locality 및 문항 전이',
        '\npaired_seen_endpoint_metrics.csv는 exact request/prompt identity를 확인한 O/JV/L8 end-to-end 차이와 양방향 성공 전이를 제공한다. B1 이후 arm-local W/M/z가 다르므로 same-state causal contrast라고 부르지 않는다.',
        '\n| Model | Arm | Wk | W0-success→failure / W0-success | W0-failure→recovery / W0-failure | all NS prompts |',
        '|---|---|---:|---|---|---:|']
    for x in transitions:
        text.append(f"| {x['alias']} | {x['arm']} | {x['batch']} | {x['W0_success_to_failure']}/{x['W0_success_denominator']} | {x['W0_failure_to_recovery']}/{x['W0_failure_denominator']} | {x['canonical_prompt_denominator']} |")
    text+=['\nRetained/failed/recovered/overwrite-candidate rewrite cohort의 unconditional/conditional 분모는 retention_cohort_metrics.csv에 별도로 있다. 동일 NS 총점이 같은 문항 보존이라는 뜻이 아니며, NS를 general pretrained capability의 보장으로 확대하지 않는다.',
        '\n### 7. L8-only로 설명되지 않는 JV 이득',
        '\nSame-state L8/single-layer shadow는 추가 forward/JVP0인 algebraic comparison이며 actual L8 chain과 다르다. joint objective improvement와 best single-layer는 node_mechanism_summary.csv에 있다. '+('두 actual L8 chain이 완료되어 final/paired table에서 실제 trajectory 비교가 가능하다.' if {4,5}.issubset(completed) else 'Actual L8 chain은 아직 둘 다 완료되지 않았으므로 Case G/H 또는 L8-only 대비 최종 이득 판정은 보류한다.'),
        '\n### 8. 추가 계산 비용',
        '\n실제 process wall/GPU-hours는 run_registry.csv, model F/B/input-token-slot/solve/JVP와 구간별 비용은 compute_accounting.csv에 있다. write_including_endpoint에서 endpoint_observation bracket을 빼도 pure solver latency와 같지 않다. Endpoint bracket에는 evaluator와 physical metric observer/capture overhead가 포함된다. History bracket도 append-only 시간이 아니다. L8는 actual JVP4/batch, JV20/batch이나 기존 all-layer field build 재사용 비용은 그대로 계상한다.',
        '\n### 9. 검증 범위 밖과 경쟁 설명',
        '\n1,000 edits 이후 안정성, 추가 seed/order, all-capability preservation, history-free whole-method counterfactual, learned preservation, unconditional causal superiority는 검증하지 않았다. L8 집중의 간섭 증가와 early-layer write 감소의 downstream key-drift 완화는 경쟁 가설이며, 이번 관측만으로 원인을 확정하지 않는다. 새로운 threshold/수식/selector/trajectory replay/imputation0.']
    return text
