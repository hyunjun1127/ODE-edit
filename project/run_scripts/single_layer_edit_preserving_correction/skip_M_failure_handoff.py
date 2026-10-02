"""One-shot compact actual M failure handoff; no cancel/retry/runtime changes."""
import datetime
import json
from pathlib import Path
from .common import ROOT,member,write
from .control import call


def main():
    repo=Path.cwd();parent=ROOT/'M/attempt-skip-t-v1';out=parent/'episodes/b001/attempt-v1'
    lock=json.loads((parent/'execution.lock.json').read_text())
    submission=json.loads((parent/'submission.json').read_text())
    rca_path=ROOT/'receipts/storage-waiver-r1/serialization-rca.json';rca=json.loads(rca_path.read_text())
    if rca['status']!='METADATA_SERIALIZATION_RCA_CONFIRMED_NOT_METHOD_FAILURE':raise ValueError('RCA_SCOPE')
    scheduler=call(['sacct','-n','-X','-j',submission['job'],
        '--format=JobID,JobName,User,State,ExitCode,ElapsedRaw,Start,End,AllocTRES','-P'])
    queue=call(['squeue','-h','-j',submission['job'],'-o','%i|%j|%T|%R|%b|%N'])
    evidence=dict(status='M_RUNTIME_FAILURE_INITIAL_NOT_REACHED_WAITING_USER',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=submission['job'],
        scheduler_parent_only=scheduler,queue=queue,registered_released=10,
        root_cause='TorchVersion metadata in N4 final-L4 weights_only reload, runner.py:181 from runtime.py:81',
        source=lock['execution']['head'],source_tree=lock['execution']['tree'],archive=lock['execution']['archive'],
        lock=member(parent/'execution.lock.json'),submission=member(parent/'submission.json'),
        resource=member(parent/'resource-admission.json'),failure=member(out/'failure.json'),RCA=member(rca_path),
        native_b001_new_fit=0,N4_payload=rca['actual_file'],
        completed_M_endpoints=0,actual_M_initial='NOT_REACHED',EN_F_actual_controller='NOT_RUN',
        new_cancel=0,new_resubmit=0,frozen_source_changed=False,original_raw_changed=False,
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        reserve='USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP',failure_is_ENOSPC=False,
        b001_final_parent_GPU_seconds=718,other_new_M_cost='running allocation snapshots, not final',
        prior_GPU_seconds=3231,S_R_L=0,monitoring_active=False,automatic_resume=False,
        resume_trigger='explicit_user_call',
        required_direction='exact live linked M cancellation and metadata-only new immutable source/retry; not performed',
        direct_transport='first message COMMUNICATION_HOLD multiple active GH turns; retry separately recorded')
    local=write(ROOT/'receipts/storage-waiver-r1/failure-handoff.json',evidence)
    package=repo/'experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/storage-waiver-M-failure-r1'
    package.mkdir(parents=True,exist_ok=False)
    result=write(package/'failure-handoff.json',evidence)
    cpu=write(package/'cpu-serialization-rca.json',rca)
    report=package/'submission-and-failure-ko.md'
    text=f'''# ENFC: 저장 waiver 후 M10 제출 / 첫 endpoint metadata 오류

상태: **M_RUNTIME_FAILURE_INITIAL_NOT_REACHED**. M10/10 정상등록·held검사·release는 완료했으나,
대표 b001은 처음 N4 endpoint의 필수 CPU 재로드에서 실패했다. **M_INITIAL_VALID/전체 M 완료/수치 PASS가 아니다.**

## 실행 및 현재 경계

Array `{submission['job']}_[0–9]%2`, release {submission['release_time']}.
0=b001[0,100), …,9=b010[900,1000), 매번 independent W0/M0, episode당8arm.
기존 점유0 확인 후 cap2, 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0로 제출했다.
B1 nativefit REUSE, 나머지 최대9sharedfit/900target 계획. 80final L4 의무와 method guard는 유지했다.

| 구분 | 실제 관측 |
| --- | --- |
| 50021_0 / b001 | FAILED1:0, 718 allocated GPU-sec |
| b001 native | 이전 capsule 재사용, 신규fit0 |
| b001 첫 N4 payload | 파일 존재; plain weights_only reload 실패; selection seal 전 |
| b001 EN-F / M initial | NOT_RUN / NOT_REACHED |
| 나머지 children | exact시각 queue는 아래 receipt; 취소/재제출하지 않음 |

마지막 한정 scheduler 관측 {evidence['time']}. [실행·실패 receipt](failure-handoff.json)에 exact 상태를 보존했다.
모든 나머지 child가 완료 또는 실패했다고 추정하지 않는다. 기존 T 오류를 이유로 새 M을 취소하지 않았다.

## 최초 원인과 실제 CPU 검산

Frozen `runner.py:176–179`의 `tokenizer_identity=rt.identity`에 `runtime.py:81`의
`torch.__version__`가 포함된다. 해당 값은 출력상 문자열처럼 보이지만 실제 타입은
`torch.torch_version.TorchVersion`이다. `torch.save`는 그 클래스 정보를 보존하여,
바로 다음 `runner.py:181`의 `torch.load(weights_only=True,mmap=True)`가 지원하지 않는 global로 거부했다.
TypeError였던 과거 T receipt 오류와 다른 연결 오류이며, FD 불일치·효능 실패·ENOSPC로 분류하지 않는다.

CPU tiny BytesIO fixture에서 TorchVersion metadata는 같은 UnpicklingError,
동일 값의 builtin str metadata는 weights_only load 성공으로 재현했다.
실제 첫 N4 파일은 unknown-global 목록이 정확히 TorchVersion 하나임을 검사한 뒤,
설치된 torch의 그 클래스만 scoped safe_globals로 허용하여 CPU weights_only/mmap 검산했다.
FP32 [4096,14336], finite, bound native weight SHA 일치를 확인했다. 원 파일은 불변이다.
`weights_only=False`를 사용하지 않았고 새 model/forward/GPU 실행도 없었다.
이 별도 CPU 복구 읽기를 원 실행의 재로드·선택 봉인 성공이나 GPU continuation으로 승격하지 않는다.
[CPU RCA](cpu-serialization-rca.json)에 exact 파일 SHA·검사 범위를 남겼다.

모든 episode가 동일 frozen 저장 경로를 사용하므로 동일 오류에 노출된다. 기존 CPU82+skip4/storage3은
method/라우팅/기본 IO 전파 검사이며 실제 Runtime identity를 넣은 final endpoint 직렬화 경계는 누락됐다.
최소수리안은 version metadata를 plain builtin string으로 저장하고 weights_only 검사를 유지하는 것이다.
**이번 보고 시점에는 frozen runtime 수정/새 source 재실행/재제출/나머지 child 취소를 하지 않았다.**

## T·저장 waiver와 보존

T=SKIPPED_USER_DIRECTED/full_numerical_validation=NOT_ESTABLISHED. 새 T/READY/afterok/oldfailcancel0.
72GiB reserve는 원 추정치로 보존하고 USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP으로 제출차단만 해제했다.
사용자cleanup은 계획/미확인, SH4삭제·이동0. 실제 source/archive/lock 및 N4 payload 쓰기는 성공했으며,
저장완결성 검사에서 metadata 오류를 감지했다. Endpoint 의무나 actual IO 오류를 생략하지 않았다.
기존 실패·partial·teacher·native와 앞선 storage-block 보고는 불변이다.

실행 source `{lock['execution']['head']}` / tree `{lock['execution']['tree']}`.
Archive `{lock['execution']['archive']['sha256']}`, lock `{member(parent/'execution.lock.json')['sha256']}`.
Root `{parent}`. 분석/publication source는 별개다.
과거 allocation3231GPU초와 새 b001718GPU초를 분리하며 다른 live M allocation은 최종비용이 아니다.
batch/extern 및 B1native·teacher를 중복계상하지 않는다.

## 다음 권한 경계

정확한 남은 live children 중단 및 metadata-only 새 immutable attempt 재제출 방향이 필요하다.
무단 광범위 취소/원파일수정/새 science 자동추가0. S/R/L0. 정상 초기 gate는 미완료이며,
technical failure로 인계한다. 상세효능 분석/실험 확대를 수행하지 않는다.
monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call.
현재 제출된 나머지 프로그램을 자동 취소했다고 주장하지 않는다.
'''
    with report.open('x') as f:f.write(text)
    manifest=write(package/'artifact-manifest.json',dict(members=[member(report),result,cpu],raw_tensors_in_git=False))
    rooted=write(package/'rooted-receipt.json',dict(manifest=manifest,local_evidence=local,
        source=lock['execution']['head'],validation='INITIAL_NOT_REACHED_T_SKIPPED'))
    resume=write(ROOT/'resume-manifest-M50021-failure-r1.json',dict(evidence=evidence,report=member(report),
        manifest=manifest,rooted_receipt=rooted,old_resume=member(ROOT/'resume-manifest-skip-t-storage-blocked-r1.json')))
    print(json.dumps(dict(report=member(report),manifest=manifest,rooted_receipt=rooted,resume=resume)))


if __name__=='__main__':main()
