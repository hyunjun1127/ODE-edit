"""Compact repair submission/initial publication from sealed local evidence."""
import argparse
import json
from pathlib import Path
from .common import ROOT,member,write,sha


def publish(observation_path):
    repo=Path.cwd();parent=ROOT/'M/attempt-metadata-r1';rca=ROOT/'receipts/metadata-repair-r1'
    obs=json.loads(observation_path.read_text());lock=json.loads((parent/'execution.lock.json').read_text())
    sub=json.loads((parent/'submission.json').read_text());resource=json.loads((parent/'resource-admission.json').read_text())
    if obs['job']!=sub['job'] or obs['lock']!=member(parent/'execution.lock.json'):raise ValueError('HANDOFF_BINDING')
    initials=obs['initials'];paused=bool(initials)
    state='M_INITIAL_VALID_WITH_T_SKIPPED_MONITORING_PAUSED' if paused else 'M10_REPAIR_RELEASED_AWAITING_ACTUAL_INITIAL'
    stage='initial' if paused else 'submission'
    package=repo/'experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/metadata-repair-M-r1'/stage
    package.mkdir(parents=True,exist_ok=False)
    evidence=dict(status=state,job=sub['job'],registered_M=10,full_M_complete=False,
        observed_time=obs['time'],queue=obs['queue'],initials=initials,observation=member(observation_path),
        execution={k:v for k,v in lock['execution'].items() if k!='members'},lock=member(parent/'execution.lock.json'),
        submission=member(parent/'submission.json'),resource=member(parent/'resource-admission.json'),
        held=member(parent/'held-inspection.json'),retained_native_plan=lock['retained_native_plan'],
        original_reuse_matrix=lock['reuse_matrix'],original_reuse_plan=lock['reuse_plan'],
        repair_authority=lock['repair_authority'],prior_attempt=lock['repair_prior_terminal'],
        old_50021_allocation_GPU_seconds=2301,prior_49928_49973_allocation_GPU_seconds=3231,
        current_allocation='RUNNING_NOT_FINAL; no task-end cost report',new_native_fits_max=7,
        new_native_targets_max=700,native_reuse=['b001','b002','b003'],final_L4_required=80,
        new_T=0,S_R_L=0,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        CPU_new_tests=7,independent_red=False,self_audit=True,
        storage_status=resource['storage_admission']['status'],free_bytes=resource['available_disk'],
        reserved_bytes_original=77309411328,cleanup_completion='NOT_VERIFIED',deleted_or_moved=0,
        monitoring_active=not paused,automatic_resume=False,scope='actual initial only, not efficacy/model numerical PASS')
    summary=write(package/'receipt.json',evidence)
    local_inputs=[member(parent/name) for name in ('execution.lock.json','submission.json','held-inspection.json',
        'resource-admission.json','validation-waiver-binding.json','skip-t-all-m-override.json','storage-waiver-submit-override.json')]
    local_inputs += [member(rca/name) for name in ('user-recall.txt','cancel-result.json','old-attempt-terminal.json',
        'retained-native-plan.json','cpu-regression.json')]
    local_inputs += [member(ROOT/'receipts/storage-waiver-r1/serialization-rca.json')]
    inputs=write(package/'input-manifest.json',dict(members=local_inputs,execution_source=lock['execution']['members'],
        unchanged_numerical_source=lock['unchanged_method_source'],raw_payload_in_git=False))
    initial_text='실제 M 초기 gate는 아직 미관측이다. 제출 완료와 과학 완료를 구분하며 이 task만 bounded 관찰한다.'
    if paused:
        i=initials[0]
        initial_text=f'''{i['episode']}의 실제 EN-F route와 선택봉인 이후 observer를 확인했다.
stop={i['stop_reason']}, accepted_rounds={i['accepted_rounds']}.
Final L4의 weights_only CPU reload/FP32/finite/fileSHA/tensorSHA, 8개 optimizer 선택봉인→공식 P/N 관측,
observer nonmutation, exact W0 reset/M0 경계를 확인했다. 대표1episode 범위이며 다른 episode의 완료/후속
GPU continuation/미분 correctness/효능 PASS로 확대하지 않는다. 정상 native fallback도 유지한다.
초기 관측 뒤 agent scheduler/log/result polling과 후속 submit을 중지한다. 등록 M은 자연 진행한다.'''
    text=f'''# ENFC M 저장 메타데이터 기술 수리·재제출 {stage}

상태: **{state}**. 사용자 “기술적 오류는 해당 오류 보고 이후 SH가 직접 수정후 재제출해”를 적용했다.

## 첫 오류와 최소 수리

50021_0의 첫 N4 final-L4 저장 후 weights_only reload가 TorchVersion unsupported global로 실패했다.
`torch.__version__`는 str처럼 출력되지만 실제 subclass이며 pickle metadata에서 안전 로더가 거절했다.
Tiny CPU 재현과 원 파일의 scoped safe-global CPU 검사로 원인을 확인했다. 무제한 pickle load로 우회하지 않았다.
수리는 endpoint metadata를 builtin str로 기록하는 것으로 제한했다. 원 runtime/native/alltoken/geometry/
optimizer/observer/guard/threshold bytes는 동일하다. 수치 실패나 효능 결론, ENOSPC 실패가 아니다.

50021_0/1은 FAILED, 실행·대기 중인 exact _2–9만 취소했다. 다른 job 변경0, 원파일 삭제0,
cancelled process rollback NOT_VERIFIED. 첫 실패·부분 산출물·원 source와 비용은 보존했다.
같은 source 재시도에서 완료된 B2/B3 native fileSHA/weights_only/finite/weightSHA/요청순서/W0/M0/P4/
history0를 CPU 검사했다. B1+B2+B3 native REUSE, B4–10 최대7fit/700target; 중단된 B4 incomplete 작업을
완료 capsule로 만들지 않았다. 재사용은 full process 또는 GPU crash-resume과 다르다.
새 좁은 CPU7검사는 serialization 및 재사용 valid/order/warm/P/weight/history 오류경계다.
기존 CPU82와 skip4/storage3의 변경 없는 근거를 재사용하며 새로운 T/FD/noop/teacher 검증0이다.

## 전체 M 제출

배열 **{sub['job']}_[0–9]%{sub['array_throttle']}**, 10/10 held owner/source/args/resource/dependency 검사 후
{sub['release_time']} release. 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0, cap2, hour cap=null.

| Index | 독립 cold episode | Native | 요청 ordinal |
| --- | --- | --- | --- |
| 0 | b001 | 원 cold7 REUSE | [0,100) |
| 1–2 | b002–b003 | 완료된 50021 cold native REUSE | [100,300), 각100 |
| 3–9 | b004–b010 | RUN_MISSING, 최대7fit | [300,1000), 각100 |

각 episode의8arm은 동일 native capsule을 공유하고 history0, W0/M0 independent reset을 유지한다.
80 final L4 endpoint/RAND±/CA-EXACT/공식평가 저장 의무는 불변이다. T dependency/failcancel0, S/R/L0.
T=**SKIPPED_USER_DIRECTED**, full_numerical_validation=**NOT_ESTABLISHED**.

## 실제 관측 범위

관측시각 {obs['time']}.
{initial_text}

## Source·저장·비용

Frozen execution `{lock['execution']['head']}` / tree `{lock['execution']['tree']}`.
Archive SHA `{lock['execution']['archive']['sha256']}`.
Execution lock SHA `{member(parent/'execution.lock.json')['sha256']}`.
원자료 root `{parent}`. 실행 source와 본 publication source는 구분한다.

제출 free={resource['available_disk']}B, 원72GiB reserve=77,309,411,328B.
USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP은 유지한다. 사용자 cleanup 완료/여유공간 원인은
미검증이며 exclusive reserve나 미래80endpoint 저장 보장이 아니다. 실제 IO/atomic·저장완결성 오류는
실패로 남긴다. SH 삭제/이동0.

기존 T49928/M49973 3231GPU초, 50021 실패·중단 시도2301GPU초를 별도 보존한다.
새 {sub['job']} allocation은 진행 중이므로 최종 비용을 확정하지 않는다. 재사용 native/teacher와
batch/extern 또는 중첩 component timer를 다시 합산하지 않는다.

## 증거와 한계

[Receipt](receipt.json), [input manifest](input-manifest.json).
별도 독립 red agent는 사용하지 않았으며 SH 자체 source/state/reuse/resource/raw-free 검사를 수행했다.
파일/CPU hash를 T numerical PASS 또는 GPU continuation으로 승격하지 않는다.
M 상세 통계와 S/R/L 확대는 이번 initial 인계 범위 밖이다.
monitoring_active={str(not paused).lower()}, automatic_resume=false.
'''
    report=package/'report-ko.md'
    with report.open('x') as f:f.write(text)
    manifest=write(package/'manifest.json',dict(members=[member(report),summary,inputs],raw_free=True))
    rooted=write(package/'rooted-receipt.json',dict(manifest=manifest,observation=member(observation_path),
        actual_initial_scope=initials,full_model_validation=False))
    if paused:
        write(ROOT/'resume-manifest-metadata-repair-r1.json',dict(evidence=evidence,report=member(report),
            manifest=manifest,rooted_receipt=rooted,monitoring_active=False,automatic_resume=False,
            old_attempts_preserved=True,resume_trigger='explicit_user_call'))
    status_path=repo/'tasks/status/odeedit_single_layer_edit_preserving_correction_s4_20260918/server4.json'
    status=json.loads(status_path.read_text())
    status.update(status=state,monitoring_active=not paused,automatic_resume=False,
        metadata_repair_r1=dict(receipt=summary,report=member(report),manifest=manifest,rooted_receipt=rooted))
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(report=member(report),manifest=manifest,rooted_receipt=rooted,status=state)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observation',required=True,type=Path);a=p.parse_args();publish(a.observation)
