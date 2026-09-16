"""CPU-only compact pending handoff. Reads control receipts, never job outputs."""
import argparse
import json
from pathlib import Path
from .common import ROOT,TASK,ARMS,identity,save,digest

def run(worktree):
    w=Path(worktree)
    lock=json.loads((ROOT/'execution.lock.json').read_text())
    registered=json.loads((ROOT/'submission-v1/registered.json').read_text())
    released=json.loads((ROOT/'submission-v1/released.json').read_text())
    assert all('PENDING' in row for row in released['last_observation'].splitlines())
    refs=dict(full_read=identity(ROOT/'full-read-receipt.json'),execution_lock=identity(ROOT/'execution.lock.json'),
        source_archive=lock['source_archive'],CPU_checks=lock['CPU_checks'],registered=identity(ROOT/'submission-v1/registered.json'),
        release=identity(ROOT/'submission-v1/released.json'),admission=identity(ROOT/'submission-v1/admission.json'))
    status=dict(instruction_id=TASK,nonce='ODEEDIT-GH-SH4-LOCAL-Z-ADAPTIVE-ALLOCATION-20260916-R1',
        task_state='MONITORING_PAUSED_AWAITING_USER',gate='PENDING_GATE_NOT_RUN',
        scientific_completion='NOT_OBSERVED',technical_status='REGISTERED_NOT_OBSERVED_READY',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        host='server4',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
        source_head=lock['source_head'],source_tree=lock['source_tree'],source_branch='codex/server4-local-z-adaptive-allocation-seq1000-v1',
        publication_source='Includes CPU-only handoff after frozen execution commit; runtime unchanged',
        technical_job=registered['technical'],scientific_array=registered['science_array'],array_mapping=registered['mapping'],
        jobs_last_observed=released['last_observation'],last_observation_time=released['time'],
        source_root=lock['source_root'],references=refs,gpu_cap=1,concurrent_reserved_capacity=1,
        conditional_dependency='afterok:'+registered['technical'],technical_READY_required=True,
        new_scientific_chains_registered=7,new_scientific_chains_observed_running=0,
        CPU_tests=12,CPU_frozen_imports_assets=2136,model_level_validation='NOT_RUN_AT_LAST_OBSERVATION',
        teacher='47592 existing192 reused by identity; current TF32off reproduction pending; conditional same-token repair only',
        capsule='PREPARATION_NOT_YET_OBSERVED; source declares common W0/context/RNG/teacher lock',
        expected_READY=lock['common_ready'],expected_outputs={arm:str(ROOT/'arms'/arm/'attempt-v1/output') for arm in ARMS},
        output_existence='NOT_REPOLLED_AFTER_PENDING',last_batch=None,next_ordinal=None,checkpoint=None,history=None,RNG=None,
        deferred='All numerical/GPU checks, dynamic B1 commit/B2 entry and all scientific results not yet observed',
        other_tasks_changed=False,broadcast='NO_BROADCAST_NOT_REQUIRED',new_GPU_hours_measured='NOT_AVAILABLE_PENDING',
        resource=lock['resource'],storage=lock['storage'])
    resume=save(ROOT/'resume-manifest.json',status)
    audit=w/'audits/servers/server4/2026-09-16-local-z-adaptive-allocation'
    save(audit/'preflight-and-pending.json',dict(status,resume_manifest=resume,
        audit_level='PARENT_SELF_REVIEW_CPU_FIXTURES_SOURCE_INPUT_RESOURCE; NO_INDEPENDENT_RED_AGENT',
        source_access_helper='PASS for implementation and ACK staged scope',
        session_helper='BLOCK_MISSING_WORKTREE_LOCAL_CONFIG; separately checked registered host/CWD/session; shared helper unchanged',
        pending_boundary='No subsequent scheduler/log/result query or automatic callback',
        shader_diagnostics='FD_ULP_KKT_NOT_REQUIRED_NOT_RUN',
        missing_measurement='Per-iteration native clamp hit counts NOT_RECORDED; final anchor/delta/radius retained',
        storage_claim='Reserve only, not allocated guarantee; no previous data deleted'))
    save(w/'tasks/status/odeedit_local_z_adaptive_allocation_s4_20260916_v1/server4.json',dict(status,resume_manifest=resume))
    save(w/'runs/odeedit_local_z_adaptive_allocation_s4_20260916_v1/pending-receipt.json',dict(status,resume_manifest=resume))
    # All prose is compact publication; no raw prompts/teacher/tensors/logs.
    report=w/'experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/preparation-submission-ko.md'
    report.parent.mkdir(parents=True,exist_ok=True)
    text=f'''# Local-z adaptive allocation — 준비·등록·PENDING 인계

상태: **MONITORING_PAUSED_AWAITING_USER / PENDING_GATE_NOT_RUN**.
기술 준비와 7개 과학 프로그램을 등록했지만, 마지막 관측에서 모두 PENDING이다.
기술 READY·teacher 재현·실제 dynamic gate·70batch 완료를 관측한 것이 아니다.

## 범위와 정확한 source

- Instruction: `{TASK}`. main 시작 `cee9447330e0eabe097775ca6185ea4c45cf267d`.
- 실제 실행 source `{lock['source_head']}`, tree `{lock['source_tree']}`.
- Archive SHA `{lock['source_archive']['sha256']}`.
- Execution lock SHA `{refs['execution_lock']['sha256']}`.
- FULL_READ SHA `{refs['full_read']['sha256']}`; 정본과 최신 PROTOCOL, 원 native/fitter 연결 전체를 결속했다.
- cold W0/M4=M8=0, seed20260916, FP32/eager, matmul/cudnn TF32off.
- 동일 first1000 / B100×10, 신규7chain·70batch·7000 arm-request observations / unique1000.
  기존 warm N4/REFIT4/EP/CAKE 결과는 이번 paired 실행을 대체하지 않는다.

## 제출 및 cap1

| 프로그램 | Job | 범위 | 마지막 상태 |
|---|---|---|---|
| 공통 준비/기술검사 | {registered['technical']} | W0/context/teacher 재현/필수 native 연결 | PENDING |
| 과학 array | {registered['science_array']}_[0-6] %1 | 0 LD, 1 N4, 2 REFIT4, 3 L75, 4 T75, 5 L4D, 6 TD | PENDING |

과학 array는 준비 job afterok 및 exact lock에 결속된 READY를 모두 요구한다.
각1GPU/8CPU/60416MiB/exportNONE/Requeue0/12h, 최대 동시실행가능 용량은 **1GPU**다.
등록 직전 기존 project reservation은 없었다. held owner/source/node/resource/args/dependency를 검사한 뒤 release했다.
7개가 등록된 사실과 기술검증을 통과해 과학 실행이 시작된 사실은 구분한다.
마지막 관측: `{released['time']}`. 그 뒤 scheduler·결과·로그를 재조회하지 않는다.

## 구현 및 검증 수준

원 NativeSingletonFitter와 compute_z는 수정하지 않았다. LOCAL은 local residual,
TERMINAL은 own-entry Z8을 고정하고 L4에서도 actual h8를 읽는 별도 adapter다.
원 K/repeat/direct-solve/add AST를 재사용하고 CPU FP32 endpoint gate 0/1/.5/.75를 적용한다.
기존 warm sequential runner나 EP quality/alpha policy를 실행하지 않는다.

Past64는 received-event 최신 fact ledger와 고정 SHA priority이며 현재 overwrite fact를 제외한다.
selector는 E/H·canonical rewrite strict ID·S64 D만 받는다. P/N/Dev는 observer다.
NaN/손상은 기술 실패이며 품질 부적격으로 숨기지 않는다. 최종 history는 eligible layer마다1회이고,
LD/TD의 L8 zero-write/commonN4에도 M8를1회 갱신한다.

12개 CPU 테스트(7arm 실제 proposal routing mock 포함), syntax/import, frozen 2136-member
identity/stat 연결, native adapter AST, memory-policy audit, staged access/diff 검사를 통과했다.
현재 실제 Llama/model-level 검증은 **미실행 관측 상태**다. 준비 프로그램의 동일상태 E/H 5e-5,
D 5e-7, strict 동일 기준을 완화하지 않았다. generation order, candidate score order, native 연결,
history/save-restore 검사를 준비에 포함한다. 기술 비교는 saved targets를 원 native writer에
명시 재생하는 별도 기술비용이며 baseline sequential chain이 아니다. FD/ULP/KKT는 추가하지 않았다.
별도 red agent는 사용하지 않았으며 parent 자체 검토 수준이다.

## 자산·저장·비용

기존 model/P/fixed10k/reference768/teacher192를 read-only identity로 연결한다.
teacher의 실제 W0 TF32off 재현을 공통 준비에서 확인하고, 5e-7 규약 불충족 때만 동일192 tokens로
한 번 새 teacher를 생성한다. 이전 teacher 완료를 PENDING으로 되돌리지 않는다.
모든 arm은 준비가 봉인한 동일 context text/token/RNG를 받으며 arm 간 mutable state는 공유하지 않는다.

source freeze 때 free `{lock['storage']['free_bytes']}` bytes, reserve `{lock['storage']['reserve_bytes']}` bytes.
B1/B5/B10 ×7 =21 selected W4/W8/M4/M8/context/RNG/received-ledger CP,
매batch native-fit/target·후보 점수·selected delta·commit/link·평가를 보존한다.
CAKE no-checkpoint 지시는 적용하지 않는다. FP32 delta만으로 exact replay를 주장하지 않는다.
GPU off/on continuation은 별도 미검증이다. 과거 파일 삭제/재전송은 없다.

과학 계획 12000 target calls / 최대288000 Adam /300000 loss /150 solve /190 candidate는 **산술 예상**이다.
기술 계획700 fresh target calls +400 explicit target replay calls/13 solves는 과학 비용과 별도다.
12h wall은 reserve이며 실측/과학 gate/GPU-hour hardcap이 아니다. 신규 실측 allocation은 PENDING 시점 미확인이다.
Native target/loss/Adam 반환 counters 및 final anchor/delta/radius를 보존한다.
native iteration별 clamp-hit counter는 NOT_RECORDED; nested fit/total timer는 합산하지 않는다.

## 대기 인계

Local resume: `{resume['path']}`\n
SHA `{resume['sha256']}`.
`monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
등록된 프로그램은 자연 진행하며 추가 submit/callback/heartbeat/완료대기/최종분석을 예약하지 않는다.
기술 실패 시 READY가 만들어지지 않아 과학 경로는 fail-closed로 남는다. 사용자 recall 이후 사실을 확인한다.
다른 paused task는 변경하지 않았다. NO_BROADCAST_NOT_REQUIRED.
'''
    with report.open('x') as f:f.write(text)
    report_ref=identity(report)
    head=w/'messages/server-heads/server4/2026-09-16-local-z-adaptive-allocation.md'
    head.parent.mkdir(parents=True,exist_ok=True)
    with head.open('x') as f:f.write(f'# SH4 → GH Local-z PENDING 인계\n\n공통 준비 {registered["technical"]}, 7arm array {registered["science_array"]}_[0-6]%1을 afterok/READY 단일 cap1 lane으로 등록·release했다. 마지막 상태 PENDING이며 기술 및 실제 dynamic gate는 NOT_RUN_AT_LAST_OBSERVATION이다.\n\n실행 `{lock["source_head"]}` / lock `{refs["execution_lock"]["sha256"]}`. 준비·제출 보고: `{report.relative_to(w)}` SHA `{report_ref["sha256"]}`. Resume `{resume["path"]}` SHA `{resume["sha256"]}`.\n\nMONITORING_PAUSED_AWAITING_USER. polling/후속submit/자동분석0; explicit_user_call만 재개.\n')
    save(audit/'publication-receipt.json',dict(report=report_ref,resume=resume,execution_source=lock['source_head'],
        audit=identity(audit/'preflight-and-pending.json'),raw_free=True,main_publication='OWN_SCOPE_INTEGRATION_PENDING'))
    print(json.dumps(dict(report=report_ref,resume=resume),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();run(a.worktree)
