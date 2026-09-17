"""Create-once compact PENDING handoff. Reads submission receipts, not live jobs.

This publication-only module is intentionally separate from the frozen runtime
archive. It cannot submit, monitor, inspect live results, or claim model PASS.
"""
import argparse
import json
from pathlib import Path
from .common import ROOT,save,identity

REPORT='experiment-reports/servers/server4/l4-preserving-repair-seq1000-2026-09-17-v1'
AUDIT='audits/servers/server4/2026-09-17-l4-preserving-repair'
RUN='runs/odeedit_l4_preserving_repair_s4_20260917_v1'
STATUS='tasks/status/odeedit_l4_preserving_repair_s4_20260917_v1/server4.json'

def read(name):return json.loads((ROOT/name).read_text())

def text_once(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(body)
    return identity(path)

def run(w):
    lock=read('execution.lock.json');cpu=read('cpu-checks-v1.json')
    released=read('submission-technical-v1/released.json')
    held=read('submission-technical-v1/held.json')
    admission=read('submission-technical-v1/admission.json')
    resources=read('submission-technical-v1/resource.lock.json')
    assert '|PENDING|' in released['last_observation']
    assert not released['scientific_jobs_registered'] and not released['technical_PASS']
    assert lock['storage']['disk_checkpoint'] is False
    references=[identity(ROOT/p) for p in ('full-read-receipt.json','cpu-checks-v1.json',
        'execution.lock.json','source-v1.tar','source-lineage.json',
        'submission-technical-v1/admission.json','submission-technical-v1/resource.lock.json',
        'submission-technical-v1/held.json','submission-technical-v1/released.json')]
    resume=dict(instruction_id=lock['instruction_id'],host='server4',
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',owner='SH4',
        agent_state='MONITORING_PAUSED_AWAITING_USER',monitoring_active=False,automatic_resume=False,
        resume_trigger='explicit_user_call',last_observation=released,
        execution_source=lock['source_head'],execution_tree=lock['source_tree'],
        execution_lock=identity(ROOT/'execution.lock.json'),source_archive=lock['source_archive'],
        GPU_cap=2,registered_technical_job=released['job_id'],registered_science_jobs=[],
        approved_science_arms=['R-GD','R-QP'],unregistered_scope='Both fresh W0 B100x10 mains; actual technical READY required',
        current_validation=dict(CPU_tests=cpu['tests'],CPU_status=cpu['status'],
            actual_model='NOT_RUN_AT_LAST_PENDING_OBSERVATION',technical_READY='NOT_OBSERVED',science_initial_gate='NOT_RUN'),
        checkpoint='SKIPPED_USER_DIRECTED',checkpoint_payload_W_M_RNG=False,
        exact_restart='NOT_AVAILABLE',in_memory_continuation_rollback=True,
        inputs=dict(cold_capsule=lock['cold_capsule'],teacher_manifest=read('execution.lock.json')['teacher_manifest'],
            first1000_root=lock['prefix1000_ordered_root'],seed=lock['seed']),
        expected_paths=dict(technical=lock['technical_output'],READY=lock['common_ready'],
            science=[str(ROOT/'twoarms'/arm/'attempt-v1/output') for arm in lock['arms']],
            presence='NOT_QUERIED_AFTER_PENDING_HANDOFF'),
        next_authorized_action_on_recall='Inspect exact pilot once; if actual READY passes, lock measured resource plan and submit both fresh mains within cap2',
        callbacks=False,other_task_actions=0,broadcast='NO_BROADCAST_NOT_REQUIRED',evidence=references)
    local_resume=save(ROOT/'resume-manifest.json',resume)
    summary=dict(resume,local_resume=local_resume)
    save(w/STATUS,summary);save(w/RUN/'submission-receipt.json',summary)
    save(w/AUDIT/'full-read-receipt.json',read('full-read-receipt.json'))
    compact_cpu={k:v for k,v in cpu.items() if k!='output'}
    save(w/AUDIT/'cpu-checks.json',compact_cpu)
    save(w/AUDIT/'preflight.json',dict(resource_admission=admission,resource_lock=resources,
        held_source_owner_resource='PASS',held_receipt=identity(ROOT/'submission-technical-v1/held.json'),
        code_review='Two bounded workers: FP64 QP/geometry, all-token response; parent integration',
        corrected_findings=['mandatory checks missing cannot write READY','QP b persisted before use',
            'M4 check compares entry hash','nonfinite Base/gain cannot become normal off'],
        independent_model_validation=False,science_arms=['R-GD','R-QP'],native_L8_target_P8_M8=False,
        official_P_N_access='POST_SELECTION_SEAL_ONLY',no_checkpoint_override=True,
        helper_status='BLOCK_STALE_ROOT_SESSION_AND_MISSING_WORKTREE_CONFIG; direct environment/registry match, no helper PASS',
        prior_source_diff='Reused fitting/evaluator/teacher/Past helper files unchanged from cold7 execution32a92ad6',
        scheduler_polling_after_handoff=0))
    job=released['job_id'];head=lock['source_head'];tree=lock['source_tree']
    report=f'''# L4-preserving repair 두 arm — 준비·제출 사실 보고

상태: **공통 기술 pilot {job} PENDING / 본실험 두 arm 미등록 / MONITORING_PAUSED_AWAITING_USER**.
마지막 scheduler 관측은 {released['time']}이며 이후 결과·로그·scheduler를 조회하지 않았다. 현재 상태나 완료를 추정하지 않는다.

## 최신 사용자 checkpoint 미저장 적용

“checkpoint는 저장하지 말고 진행하라”가 기존 저장 요구보다 우선한다. W4/W8/M4/RNG disk checkpoint는 생성하지 않는다. RAM의 batch간 상태 전달·rollback, 평가 NLL, commit/hash/received ledger, Q/response/기술 probe 증거는 유지한다. 이들은 complete continuation checkpoint가 아니며 exact restart는 NOT_AVAILABLE이다. 기존 파일 삭제·덮어쓰기0.

## 승인 범위와 실제 실행 수준

| 대상 | 승인 범위 | 이번 등록 | 실제 검증 수준 |
| --- | --- | --- | --- |
| 공통 pilot | 첫100 native L4 fit/solve1 공유, R-GD/R-QP 기술 연결 | {job}, held 점검 후 release | 마지막 PENDING, 실제 모델 NOT_RUN |
| R-GD | fresh W0/zeroM4, B100×10 | 미등록 | 기술 READY 전, 과학 결과 없음 |
| R-QP | fresh W0/zeroM4, B100×10 | 미등록 | 기술 READY 전, 과학 결과 없음 |

두 main 계획은 unique1000/arm-request2000/20batch, native target2000·solve20·M4append20·repair endpoint 최대120이다. 실측이 아니다. 신규 N4/REFIT4/LD, L8 native target/P8/M8, 별도 paraphrase training/guard, 다른 task 실행0. Pilot native100은 science 분모에 합산하지 않고 main에 warm carry하지 않는다.

## 구현과 기술 판정 경계

자기 entry의 native full L4 write WN 후 W4를 고정하고 physical L8 full-weight repair만 적용한다. R-GD는 Base 1방향 scalar proposal, R-QP는 Base/Current/존재시Past gradient span의 최대3방향이다. FP64 Gram/QR·pN Fisher JVP·최소 condition shift·whitening·많은 guard를 처리하는 작은 active-set QP를 구현했다. Toy2D solver를 production으로 복사하지 않았다.

Actual Current/Past mean≤각 WN mean+1e-4, strict/preference exact ID subset, Base predicted/actual gain>1e-6, agreement≥.1을 유지한다. 최대6 endpoint 중 첫 acceptable만 선택하고 r/2 중 Q/b/A/H는 고정한다. 정상 off는 이번 repairΔ8=0이며 이전 누적 L8은 유지한다. M4 finalizer1/inner0/M8append0, received ledger 전량 갱신이다. Official P/N은 selection.json 봉인 뒤 observer에서만 사용한다.

CPU {cpu['tests']} tests PASS(geometry/QP28, tiny-model response7, engine mock16), import/syntax/shell 검사 PASS. Independent source 검토의 누락을 수정했다. 이는 실제 Llama FD/JVP/GN/materialization/teacher/history 검증 PASS가 아니다. 해당 필수 pilot checks가 없거나 실패하면 READY를 생성하지 않는다. 기술 규칙은 GPU 관측 전에 numerical.py/geometry.py/qp.py와 execution lock으로 고정했다.

## 자산·source·환경

- 실행 HEAD `{head}`, tree `{tree}`. 출판 commit과 구분한다.
- Archive SHA `{lock['source_archive']['sha256']}`.
- Execution lock SHA `{identity(ROOT/'execution.lock.json')['sha256']}`.
- Cold capsule SHA `{lock['cold_capsule']['sha256']}`. W0 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, method seed20260916, FP32/eager, 양 TF32off, P4 physical4→asset0→local0, M4zero.
- Fixed10k first1000 root `{lock['prefix1000_ordered_root']}`, 같은 순서/100씩10batch. 재추출0.
- 기존 C4 reference/S64/Dev128 teacher 재사용 우선; 새 teacher 생성0. 실제 W0 연결 검사는 pilot에 예정되어 있고 아직 PASS가 아니다. C4 sampling seed20260915와 method seed를 분리한다.
- 실제 host/session/repo/환경변수·tracked registry는 일치. 공용 session helper는 root의 오래된 session 설정 및 새 worktree 설정 부재로 BLOCK이다. helper/shared config를 수정하지 않았으며 helper PASS로 기록하지 않는다.

## 자원·비용·저장

제출 직전 bounded resource-only admission에서 기존 Server4 project 예약용량0, cap2를 확인했다. 공통 pilot1GPU/8CPU/60416M/exportNONE/Requeue0/wall12h를 held 상태에서 점검하고 release했다. 기존 job 변경0. 두 main은 READY 이후 실제 pilot 측정으로 wall/storage를 lock하고 가능한 두 slot에 독립 등록해야 한다.

제출 시 available disk {resources['free_bytes']:,} bytes, 최소 reserve {lock['storage']['reserve_bytes']:,} bytes(48GiB). Q 최대14.1GB 등 진단/평가와 여유를 고려한 계획이며 실제 사용량이 아니다. No-checkpoint가 반영되었다. Allocation/compute/peak/science 비용은 아직 미측정이다. 이전 cold7/teacher 비용은 새 비용에 재합산하지 않는다. Hour hardcap=null, 신규 GPU 실측을 CPU PASS에서 추정하지 않는다.

## 재현·인계

실행 lock: `{ROOT/'execution.lock.json'}`. Frozen source: `{lock['source_root']}`. 예상 pilot: `{lock['technical_output']}`. READY 파일 존재는 인계 이후 재조회하지 않았다. Local resume: `{local_resume['path']}`, SHA `{local_resume['sha256']}`.

CPU 재현은 package README/checks.py의 51개 unittest를 CUDA_VISIBLE_DEVICES=''로 실행한다. Published checks source SHA와 frozen lineage를 비교해야 하며 GPU 작업을 자동 실행하는 재현 명령은 제공하지 않는다.

monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call. PENDING에서 시작을 기다리지 않는다. Callback/자동 추가submit/완료 대기/다른 task 재개0. 다음 명시 recall에서 exact pilot 증거를 확인한 뒤 READY 성립 시 승인된 두 main을 제출할 수 있다. 본 보고는 제출 인계이지 본실험 완료 보고가 아니다.
'''
    report_ref=text_once(w/REPORT/'preparation-submission-ko.md',report)
    save(w/REPORT/'resume-manifest.json',summary)
    save(w/REPORT/'evidence-manifest.json',dict(runtime_head=head,runtime_tree=tree,
        local_evidence=references,local_resume=local_resume,report=report_ref,
        raw_payload_git=False,no_broadcast='NO_BROADCAST_NOT_REQUIRED',numerical_actual='NOT_RUN_AT_PENDING'))
    package_members=[identity(p) for parent in (w/REPORT,w/AUDIT,w/RUN) for p in sorted(parent.rglob('*')) if p.is_file()]
    rooted=save(w/REPORT/'rooted-receipt.json',dict(status='SUBMISSION_PENDING_HANDOFF',members=package_members,
        execution_lock=identity(ROOT/'execution.lock.json'),no_checkpoint=True,full_science_complete=False))
    message=f'''# SH4 L4-preserving repair — PENDING 인계

공통 pilot **{job}** 제출·held inspection·release 완료, 마지막 관측 {released['time']} **PENDING**. 실제 모델/기술 READY/과학 initial gate는 NOT_RUN 또는 NOT_OBSERVED. R-GD/R-QP main 두 개는 **미등록**이며 실제 READY와 실측 자원계획 확인 이후 등록 대상이다.

최신 checkpoint 미저장 적용: W/M/RNG disk checkpoint0, 메모리 상태 전달·rollback/평가·commit hash·Q/response 진단 유지, exact restart NOT_AVAILABLE. 기존 artifact 변경0.

CPU51 PASS≠실제 Llama PASS. Source `{head}` / tree `{tree}`. cap2, pilot1GPU/8CPU/60416M/Requeue0/12h. 기술·teacher·타project aggregate admission 준수. FD 등 필수 기술 검사 생략0, 과학식·두 arm 불변.

정본 보고: `{REPORT}/preparation-submission-ko.md`, SHA `{report_ref['sha256']}`.
Local resume `{local_resume['path']}`, SHA `{local_resume['sha256']}`.
보고 rooted receipt SHA `{rooted['sha256']}`.

MONITORING_PAUSED_AWAITING_USER; monitoring_active=false; automatic_resume=false; resume_trigger=explicit_user_call. 추가 scheduler/log/result 조회·callback·자동후속제출0. 이미 제출한 pilot은 변경하지 않고 자연 진행한다.
'''
    text_once(w/'messages/server-heads/server4/2026-09-17-l4-preserving-repair.md',message)
    print(json.dumps(dict(report=report_ref,rooted_receipt=rooted,resume=local_resume),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();run(Path(a.worktree).resolve())
