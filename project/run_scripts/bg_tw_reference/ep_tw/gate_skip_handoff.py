"""Publish only the sealed submission receipt; never inspect live job/output."""
import argparse
import json
from pathlib import Path
import re
from .control import SESSION,identity,save,sha
from .gate_skip import TASK,MODE,SKIPPED,skip_enabled

def text_once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)
    return identity(path)

def publish(worktree,attempt):
    w,a=Path(worktree),Path(attempt)
    release=json.loads((a/'submission-v1/release-receipt.json').read_text())
    lock=json.loads((a/'execution.lock.json').read_text());assert skip_enabled(lock)
    resources=json.loads((a/'submission-v1/resource-preflight.json').read_text())
    state=re.search(r'\bJobState=(\S+)',release['last_record']).group(1)
    reason=re.search(r'\bReason=(\S+)',release['last_record']).group(1)
    refs={n:identity(a/n) for n in ['full-read-receipt.json','minimal-checks.json','execution.lock.json',
        'input-verification.json','submission-v1/resource-preflight.json','submission-v1/held-inspection.json',
        'submission-v1/release-receipt.json']}
    out=Path(lock['output'])
    resume=dict(instruction_id=TASK,nonce='ODEEDIT-GH-SH4-EP-TW1-GATES-SKIP-RUN-20260915-R1',
        host='server4',owner='janghj',session=SESSION,status='WAITING_USER_RESUME',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        job_id=release['job_id'],job_name='odeedit_ep_tw1_nogate_s4',
        last_job_state=state,last_reason=reason,last_observation=release['last_observation'],
        actual_initial_batch='NOT_OBSERVED',actual_next_ordinal='NOT_OBSERVED',
        validation_mode=MODE,numerical_validation='NOT_ESTABLISHED',skipped=SKIPPED,
        numerical_G0_PASS=False,new_diagnostic_jobs=0,new_scientific_chains=1,
        fresh_start='PRETRAINED_W0_COLD_M0_FIRST1000_NOT_B2_RESUME',
        policy='EP-TW-1',batches=10,batchsize=100,unique_requests=1000,
        source_head=lock['source_head'],source_tree=lock['source_tree'],
        source_branch='codex/server4-ep-tw1-gate-skip-run-v1',source_archive=lock['source_archive'],
        source_root=lock['source_root'],execution_lock=refs['execution.lock.json'],
        resource=lock['resource'],cap_existing=len(resources['reservations']),
        cap_total_admitted_with_new=resources['total_admitted_with_new'],
        sample_lock=lock['sample_lock'],model_revision=lock['model_revision'],
        C4_reference_identity='f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0',
        C4_member_root='0c6aa4e2ddb350c61999580fe3efabaae880ec8e17aa208a8f7d85c026a6f1aa',
        teacher_manifest=lock['teacher_manifest'],teacher_reuse=lock['asset_reuse'],
        scientific_output=lock['output'],references=refs,
        expected_paths=dict(initial=str(out/'INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED.json'),
            checkpoint=str(out/'B001/checkpoint.pt'),history_ledger_RNG='per-batch checkpoint.pt',
            next_entry=str(out/'B002/entry.json'),terminal=str(out/'terminal.json'),
            terminal_evaluation=str(out/'B010/selected-evaluation.json'),failure=str(out/'failure.json')),
        expected_path_existence='NOT_INSPECTED_AFTER_RELEASE; no live output access',
        previous_failed_cost_gpu_seconds={'47884':473,'47942':69},teacher_old_gpu_seconds=98,
        failed_native_fit_286_5445_seconds_is_nested_in_473=True,new_allocated_cost='NOT_OBSERVED',
        whole1000_complete='NOT_OBSERVED',baseline_editing=0,N4_calibration=False,new_teacher_generation=0,
        deferred=['Report256','Audit','MMLU','other policies','full10k','automatic result analysis'],
        scientific_promotion=False,no_broadcast='NO_BROADCAST_NOT_REQUIRED')
    resumeref=save(a/'resume-manifest.json',resume)
    root=w/'experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1'
    root.mkdir(parents=True,exist_ok=False)
    save(root/'resume-receipt.json',dict(resume_manifest=resumeref,**resume))
    save(root/'source-input-manifest.json',dict(source=lock['source_head'],tree=lock['source_tree'],
        archive=lock['source_archive'],references=refs,method_numerical_policy_unchanged=lock['numerical_policy'],
        skip_override=lock['validation_override'],member_count=len(lock['members']),
        reused_asset_verification=json.loads((a/'input-verification.json').read_text()),
        readonly_policy_ledger=True,readonly_native_helper=True,execution_source_not_publication_source=True))
    report=f'''# EP-TW-1 — 사용자 지시 진단 생략·본실험 제출

Instruction `{TASK}`. **WAITING_USER_RESUME**, validation **SKIPPED_USER_DIRECTED**,
numerical_validation **NOT_ESTABLISHED**. G0_PASS 또는 1000요청 완료 보고가 아니다.

## 제출 사실

단일 **{release['job_id']} / odeedit_ep_tw1_nogate_s4** held→owner/source/args/resource 검사→release.
마지막 관측 `{release['last_observation']}`: **{state}, Reason={reason}, RunTime0**.
그 뒤 scheduler/log/result/첫batch/terminal 조회를 하지 않았다. Pending 이유를 자원부족으로 추정하지 않는다.
프로그램은 B1–B10을 자연 실행하고 agent는 사용자 명시 호출을 기다린다.

Fresh pretrained W0/coldM0, Llama revision `{lock['model_revision']}`, FP32/eager,
L4 down_proj / P physical4→asset0→local0 / native BLUE singleton L2=1.
Fixed10k 공식 loader 전체 asset 검증 후 first1000/B100×10 동일 order를 사용한다.
전체 root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
prefix40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
실패 commit0에서 B2 resume하지 않는다. 기존 teacher192/C4 reference768/model/P/context 재사용.
새 teacher, N4 calibration, baseline 편집, saved-episode 진단, 다른 policy, 추가 scientific chain은 모두0.

## 생략과 유지

Saved-episode/repair_pass/E→D 조건부진입, FD/grid/convergence/resolution/jitter,
direct-gW/독립방향/ULP, W0 self-KL gate, functional/materialized/synthetic parity,
validate_episode/scientific_checks 및 redundant native-map RHS 비교는 **실제로 호출하지 않는다**.
Exception 무시·tolerance 확대·CUDA assert 이후 계속실행이 아니다.
Canonical↔method NLL/strict 비교는 이미 계산한 rows의 warning이며 추가 forward0.
원 .15 FD 규칙을 PASS로 바꾸지 않고 FD 수렴과 전체 model 수치 검증은 미확립으로 남긴다.

Method E≤Ep(양의 허용량0), 정확 strict ID-set 보존, minD64 선택/RAW fallback,
별도 gE/gD, native anchor/ball/trust, target quota, P/M/history1/accepted-ledger는 그대로다.
policy.py/ledger.py 및 non-EP helper/native bytes 불변을 source identity로 확인했다.
Shape/device/dtype/finite/source/data/order, 비선택 weight/history·observer mutation,
실제 저장/selected restore/chronological commit→next-entry hard stop은 유지한다.
CP/evaluation/rollback 및 B1–B10 내부 순서는 변경하지 않는다.

CPU routing mock5/AST/shell syntax PASS는 **수치검증이 아니다**. 기존86 fixture/GPU진단 재실행0.
새 source/input 2085 소형 fullSHA와 기존34 heavyasset fullSHA+현재 stable stat을 결속했다.
새 teacher/model 전체 재해시·재생성0. 과거47942 E direct PASS는 그 단일 범위의 관측으로만 보존한다.
본실험이 첫B100/B2를 완료하면 marker는 INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED이며
기존 numerical G0_PASS가 아니다. 현재 marker는 미관측이다.

## 자원·비용·보존

기존 project active/admitted GPU0 + 신규1 =1≤cap2. 1GPU/8CPU/**60416M**/exportNONE,
Requeue0/12h, GPU-hour hardcap=null. 다른 job/source/cap/throttle 변경0.
제출 직전 free **{resources['free_bytes']} B**, free inodes {resources['free_inodes']}.
예상30GiB + 안전여유20GiB 확인; 독점 예약은 아니다. 10개 W4/M4 tensor 예상10,569,646,080B.
과거2–8 GPUh 계획값은 신규 실측이 아니며, 진단 생략 후 속도 향상 근거로 사용하지 않는다.
원47884 **473 GPU-sec**, repair47942 **69 GPU-sec**, teacher47592 **98 GPU-sec**는 별도 보존한다.
nativefit286.5445초는 원473초의 중첩 component라 더하지 않는다. 신규 할당시간은 아직 미관측이다.
기존 attempt-v1/repair-r1/실패·부분표·locks 불변, 원본 삭제/overwrite/타task재개0.
Raw prompt/teacher/gradient/tensor/checkpoint/fullstdout local-only. PNG 생성0.

## 재현·인계

실행 source `{lock['source_head']}`, tree `{lock['source_tree']}`.
Archive `{lock['source_archive']['sha256']}`, lock `{refs['execution.lock.json']['sha256']}`.
Source `{lock['source_root']}`; output `{lock['output']}`.
Resume `{resumeref['path']}` SHA `{resumeref['sha256']}`.
허용된 구현은 ep_tw 내부 skip routing만이며 actual frozen source와 이후 publication commit은 구분한다.
CLI/작은 routing 검사/launcher는 ep_tw/README.md. 새 실행명령은 sealed held-submission receipt에 있다.
`monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
실행승인과 수치검증완료/효능 주장을 구분한다. scientific_promotion=false.
NO_BROADCAST_NOT_REQUIRED, 사용자 recall 전 결과해석·후속제출0.
'''
    reportref=text_once(root/'submission-factual-report-ko.md',report)
    summary=f'''# SH4 진단 gate 생략 — 단일 본실험 제출 인계

{TASK}. FULL_READ / source frozen / held inspection→release 완료.
Job **{release['job_id']} odeedit_ep_tw1_nogate_s4**, 마지막 {release['last_observation']} **{state}, Reason={reason}**.
Fresh W0/coldM0 L4 first1000 B100×10 단일chain. cap existing0+new1≤2,1GPU/8CPU/60416M/12h/Requeue0/exportNONE.
FD/directgW/selfKL/ULP/repair prerequisite/수치parity는 실제호출0, SKIPPED_USER_DIRECTED / NOT_ESTABLISHED.
E/strict screen·minD/RAW·history·ledger·평가·저장·상태무결성은 유지. 신규diagnostic/teacher/baseline0.
실행 source {lock['source_head']} / tree {lock['source_tree']}.
Archive {lock['source_archive']['sha256']}; lock {refs['execution.lock.json']['sha256']}.
Output {lock['output']}.
보고서 {reportref['path']} SHA {reportref['sha256']}.
Resume {resumeref['path']} SHA {resumeref['sha256']}.
WAITING_USER_RESUME; numerical G0/최초commit/전체완료 미관측. 이후poll/자동resume0.
기존실패473+69초 및teacher98초 별도보존, 새 source/report만 main통합. 다른job/source변경0.
'''
    message=text_once(w/'messages/server-heads/server4/2026-09-15-ep-tw1-gate-skip.md',summary)
    save(w/'tasks/status/odeedit_ep_tw1_gate_skip_s4_v1/server4.json',dict(instruction_id=TASK,
        status='WAITING_USER_RESUME',job_id=release['job_id'],last_job_state=state,
        numerical_validation='NOT_ESTABLISHED',validation_mode=MODE,report=reportref,resume=resumeref,
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call'))
    save(w/'runs/odeedit_ep_tw1_gate_skip_s4_v1/submission.json',dict(release=release,source=lock['source_head'],resume=resumeref))
    save(w/'audits/servers/server4/2026-09-15-ep-tw1-gate-skip/minimal-checks.json',
         json.loads((a/'minimal-checks.json').read_text()))
    manifest=save(root/'analysis-manifest.json',dict(scope='COMPACT_SUBMISSION_NOT_TERMINAL_ANALYSIS',
        members=[identity(p) for p in sorted(root.iterdir()) if p.is_file()]))
    save(root/'rooted-receipt.json',dict(report=reportref,manifest=manifest,resume=resumeref,
        source=lock['source_head'],rawfree=True,numerical_validation='NOT_ESTABLISHED',scientific_promotion=False))
    return dict(report=reportref,resume=resumeref,message=message)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps(publish(x.worktree,x.attempt),ensure_ascii=False))
