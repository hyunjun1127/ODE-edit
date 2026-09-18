"""Compact M initial handoff from one already-captured bounded observation."""
import argparse
import datetime
import json
from pathlib import Path
from .common import ROOT,member,write
from .control import call


def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);a=p.parse_args()
    observation=json.loads(a.observation.read_text())
    if not observation['initials']:raise ValueError('NO_ACTUAL_M_INITIAL')
    initial=observation['initials'][0]
    parent=ROOT/'M/attempt-skip-t-v1';lockpath=parent/'execution.lock.json'
    lock=json.loads(lockpath.read_text());submission=json.loads((parent/'submission.json').read_text())
    resource=json.loads((parent/'resource-admission.json').read_text())
    # No further scheduler/result polling after this evidence is accepted.
    evidence=dict(status='M_INITIAL_VALID_WITH_T_SKIPPED_MONITORING_PAUSED',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation=member(a.observation),
        initial=initial,last_queue=observation['queue'],M_job=submission['job'],M_registered=10,
        actual_initial_scope=initial['episode'],all_M_complete=False,execution=lock['execution'],
        lock=member(lockpath),submission=member(parent/'submission.json'),
        held=member(parent/'held-inspection.json'),resource=member(parent/'resource-admission.json'),
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        storage_status=resource['storage_admission']['status'],original_reserve_bytes=77309411328,
        user_cleanup='PLANNED_NOT_VERIFIED',deletion_or_move=0,storage_sufficiency_PASS=False,
        old_T_dependency=False,old_T_failcancel=False,new_T=0,S_R_L=0,
        numerical_method_source_unchanged=lock['unchanged_method_source'],
        prior_GPU_seconds=3231,new_allocation='RUNNING_NOT_FINAL; no full-M cost report before recall',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        next_action='user completion recall only; submitted M continues naturally; no follow-up submit')
    repo=Path.cwd();package=repo/'experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/storage-waiver-M-r1'
    package.mkdir(parents=True,exist_ok=False)
    compact=dict(evidence);compact['execution']={k:v for k,v in lock['execution'].items() if k!='members'}
    summary=write(package/'submission-initial-receipt.json',compact)
    refs=[member(parent/n) for n in ('execution.lock.json','submission.json','held-inspection.json',
        'resource-admission.json','storage-waiver-full-read.json','validation-waiver-binding.json',
        'skip-t-all-m-override.json','storage-waiver-submit-override.json')]
    inputs=write(package/'input-lineage-manifest.json',dict(local_only_full_members=refs,
        old_M_source=lock['prior_attempt_lock'],prior_CPU82=lock['prior_CPU82'],
        source_members=lock['execution']['members'],reuse_matrix=lock['reuse_matrix'],reuse_plan=lock['reuse_plan'],
        independent_model_validation=False,source_only_exact_comparison=lock['unchanged_method_source']))
    report=package/'submission-and-initial-ko.md'
    text=f'''# ENFC: T 생략·저장 reserve waiver 후 M 전체 제출 및 초기 인계

상태: **M_INITIAL_VALID_WITH_T_SKIPPED / MONITORING_PAUSED_AWAITING_USER**.
본 문서는 실제 최초 M 실행 경계만 보고한다. M 전체완료, 효능 PASS, T 수치검증 PASS가 아니다.

## 제출·원문 override

사용자 “그냥 T는 건너 뛰고 M전부 올려”, “일단 올리라고 해. 내가 공간 확보 해놓을게”를 적용했다.
M array **{submission['job']}_[0–9]%{submission['array_throttle']}**, 10/10 held 검사와 release 완료.
release {submission['release_time']}. 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0,
GPU cap2, hour cap=null. T 신규/READY/afterok/old failcancel0, S/R/L0.

| Array index | Cold episode / order | Native 계산 |
| --- | --- | --- |
| 0 | b001 / [0,100) | retained B1 REUSE |
| 1–9 | b002–b010 / [100,1000), 각100 | 매번 W0/M0, 최대9shared fits |

각 episode의 N4/SCALE/CA/KL-P/EN-S/EN-F/EN-COV/EN-F4 8arm은 같은 native capsule을 공유한다.
80 final L4 endpoint 저장 의무, RAND±/CA-EXACT 진단, 원 method guard·geometry·threshold·평가 분리는 유지한다.
기존330행 reuse/10episode 계획과 적합한 canonical observer를 재사용한다. Sequential warm 결과를 cold로 바꾸지 않는다.

## 실제 초기 확인

관측 {observation['time']}, 대표 **{initial['episode']}**.
EN-F stop={initial['stop_reason']}, accepted_rounds={initial['accepted_rounds']}.
Actual selected endpoint fileSHA/CPU weights_only shape[4096,14336]/FP32/finite/tensorSHA,
8개 controller 선택봉인→공식 observer 순서, observer nonmutation, W0 exact reset 및 M0=0 경계를 확인했다.
M history0이며 다른 독립 episode의 전체완료 또는 GPU continuation을 주장하지 않는다.
정상 fallback/zero correction도 실제 method 결과로 보존하며 강제 nonzero를 만들지 않았다.
이때 아직 남은 observer 및 다른 episode 프로그램은 변경 없이 자연 진행한다.

T=**SKIPPED_USER_DIRECTED**, full_numerical_validation=**NOT_ESTABLISHED**.
기존 T 부분 PASS는 원 범위만 재사용하며 미실행 FD/미보존 projector 판정은 승격하지 않는다.
CPU82 기존 core source 재사용 + skip4/storage3 좁은 검사만 수행했다. 새 actual T/FD/noop/teacher pilot0.

## 저장·source·비용

제출 전 free {resource['available_disk']}B. 원 reserve77,309,411,328B(72GiB)는 보존하고
**USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP**으로 admission 차단만 해제했다.
사용자 공간 확보는 예정/완료미확인, exclusive reservation/공간충분 PASS가 아니다.
기존 자료 삭제·이동·checkpoint정리0. 실제 ENOSPC/쓰기오류/endpoint완결성 guard는 유지한다.
실제 source/archive/lock/실행 출력의 쓰기는 이루어졌으며 미래80endpoint 저장공간 보장을 뜻하지 않는다.

실행 source `{lock['execution']['head']}` / tree `{lock['execution']['tree']}`.
ArchiveSHA `{lock['execution']['archive']['sha256']}`.
execution.lock SHA `{member(lockpath)['sha256']}`.
Local root `{parent}`. 이번 publication source와 frozen 실행 source를 구분한다.

과거 T49928/M49973 paired-stop/source/raw와3231GPU초는 불변이다. 새 M allocation은 진행 중이므로
전체비용을 확정하지 않으며 과거 native/teacher·batch/extern을 중복계상하지 않는다.

## 검증과 종료

상세 source/lock/waiver/관측 identity: [초기 receipt](submission-initial-receipt.json),
[입력·source manifest](input-lineage-manifest.json). CPU reload는 GPU continuation/미분 검증이 아니다.
최종 M 비교/통계/효능 해석은 이번 초기 인계 범위 밖이다.

monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call.
초기 확인 뒤 scheduler/log/result polling·자동 callback·추가 제출·상세분석을 하지 않는다.
이미 제출한 M은 hold/cancel 없이 계속하고, S/R/L은 자동 확대하지 않는다.
'''
    with report.open('x') as f:f.write(text)
    manifest=write(package/'artifact-manifest.json',dict(members=[member(report),summary,inputs],
        scientific_raw_committed=False,execution_head=lock['execution']['head']))
    rooted=write(package/'rooted-receipt.json',dict(manifest=manifest,initial=initial['marker'],
        validation='INITIAL_SCOPE_ONLY_T_SKIPPED',source_prepublication=call(['git','rev-parse','HEAD'],repo)))
    resume=write(ROOT/'resume-manifest-storage-waiver-M-r1.json',dict(evidence=evidence,report=member(report),
        manifest=manifest,rooted_receipt=rooted,old_paired_stop=member(ROOT/'resume-manifest-paired-stop-r1.json'),
        old_storage_block=member(ROOT/'resume-manifest-skip-t-storage-blocked-r1.json')))
    print(json.dumps(dict(report=member(report),manifest=manifest,rooted_receipt=rooted,resume=resume)))


if __name__=='__main__':main()
