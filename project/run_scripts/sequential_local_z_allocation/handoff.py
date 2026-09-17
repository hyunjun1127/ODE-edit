"""One-shot compact publication from already observed submission evidence."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .common import ROOT,TASK,ARMS,save,identity,sha
from .control import git,copy_once

REPORT='experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2'
AUDIT='audits/servers/server4/2026-09-17-sequential-local-z-allocation-v2'

def write_text(path,text):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text)
    return identity(p)

def publish(w,boundary):
    lockpath=ROOT/'execution.lock.json';lock=json.loads(lockpath.read_text())
    techpath=ROOT/'submission-technical-v1/released.json'
    tech=json.loads(techpath.read_text()) if techpath.exists() else None
    sciencepath=ROOT/'submission-science-v1/released.json'
    science=json.loads(sciencepath.read_text()) if sciencepath.exists() else None
    if boundary=='PENDING_HANDOFF':
        observed=(science or tech)['last_observation']
        assert '|PENDING|' in observed,'PENDING_HANDOFF_REQUIRES_ACTUAL_PENDING'
        actual='NOT_OBSERVED';technical='NOT_OBSERVED' if science is None else 'TECHNICAL_READY_PREVIOUSLY_VERIFIED'
    else:
        actual=boundary;technical='SEE_SCOPED_TERMINAL_RECEIPT'
    resume=dict(instruction_id=TASK,host='server4',owner='janghj',time=datetime.now(timezone.utc).isoformat(),
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',task_state='MONITORING_PAUSED_AWAITING_USER',boundary=boundary,
        source_head=lock['source_head'],source_tree=lock['source_tree'],source_archive=lock['source_archive'],
        execution_lock=identity(lockpath),worktree=str(w),branch=git(w,'branch','--show-current'),
        FULL_READ=lock['FULL_READ'],CPU_checks=lock['CPU_checks'],cold_capsule=lock['cold_capsule'],
        teacher_candidate=lock['teacher_manifest'],teacher_reproduction=technical,
        technical_submission=None if tech is None else identity(techpath),technical_job=None if tech is None else tech['job_id'],
        scientific_submission=None if science is None else identity(sciencepath),scientific_array=None if science is None else science['job_id'],
        scientific_arms=list(ARMS),registered_scientific_arms=[] if science is None else list(ARMS),
        actual_technical_validation=technical,actual_scientific_gate=actual,actual_scientific_completion='NOT_OBSERVED',
        missing_scope=['actual native/teacher/state/coverage validation','six scientific registration','first scientific B1/B2 link'] if science is None else ['scientific initial gate and completion'],
        expected_READY=lock['common_ready'],expected_outputs={a:str(ROOT/'arms'/a/'attempt-v1/output') for a in ARMS},
        disk_W_M_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',old_cold7_CP21='DELETED_BY_PRIOR_USER_REQUEST_NOT_RECREATED',
        old_repair='FUTURE_SCOPE_STOPPED;49238_ADMISSION_ONLY_COMPLETED_LAST_OBSERVATION;NO_MUTATION',
        gpu_cap=2,monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        callbacks_created=False,after_pause_additional_submit=False)
    resumeref=save(ROOT/'resume-manifest.json',resume)
    m0=json.loads((ROOT/'full-read-m0.json').read_text());cpuref=lock['CPU_checks']
    cpuresult=json.loads(Path(cpuref['path']).read_text())
    refs=[]
    for src,name in ((ROOT/'full-read-m0.json','full-read-m0.json'),(Path(cpuref['path']),'cpu-checks.json'),
                     (ROOT/'old49238-admission-initial.json','old49238-admission.json')):
        refs.append(copy_once(src,w/AUDIT/name))
    compact=dict(instruction_id=TASK,source_head=lock['source_head'],source_tree=lock['source_tree'],
        archive=lock['source_archive'],execution_lock=identity(lockpath),capsule=lock['cold_capsule'],
        teacher=lock['teacher_manifest'],reference_root=lock['reference_root'],
        order=lock['prefix1000_ordered_root'],records_digest=lock['records_digest'],source_lineage=identity(ROOT/'source-lineage.json'),
        resource=lock['resource'],storage=lock['storage'],numerical=lock['numerical'],budgets=lock['budgets'],
        planned_not_measured=lock['planned_science_upper_bounds'],runtime_invariant_source_not_actual_PASS=True)
    refs.append(save(w/REPORT/'execution-summary.json',compact))
    refs.append(copy_once(ROOT/'resume-manifest.json',w/REPORT/'resume-manifest.json'))
    if tech:
        refs.append(copy_once(techpath,w/REPORT/'technical-submission.json'))
        refs.append(copy_once(ROOT/'submission-technical-v1/admission.json',w/AUDIT/'technical-admission.json'))
    if science:refs.append(copy_once(sciencepath,w/REPORT/'scientific-submission.json'))
    tests_summary='\n'.join(cpuresult['output'].splitlines()[-5:])
    text=f'''# Sequential Local-z Allocation v2 — 준비·제출 사실 인계

상태: **{boundary} / MONITORING_PAUSED_AWAITING_USER**. 실행 완료 또는 실제 기술 PASS 보고가 아니다.

## 정본과 범위

Instruction `{TASK}`. GH publication d1d923425d69779c50675ef301f3e2567d99c529를 기준으로 정본10/설계/계약/CPU참조/지시를 전체 결속했다. FULL_READ SHA `{lock['FULL_READ']['sha256']}`.
새 N4/F48/G48/C4/C48/C45678만 W0/M0→first1000 B100×10: 6chains/60batches, unique1000/6000arm-request. 이전 repair 후속 scope 및 다른 paused task는 재개하지 않았다.

실행 source `{lock['source_head']}`, tree `{lock['source_tree']}`. Archive SHA `{lock['source_archive']['sha256']}`. Lock SHA `{identity(lockpath)['sha256']}`. 이 publication source와 actual runtime pin은 구분한다.

## 구현 및 현재 검증

새 controller는 all-token sequential prefix의 fresh local-z, L4–L8 adapter, exact0 skip/1copy, interior FP32 gate, frozen original native fit/solve와 history를 연결한다. F48은 지정 raw(.75,.5)이며 quality fallback을 넣지 않는다. 나머지 selector는 ownN4 mean/strict/pair 보호와 global B tie, search/prune reserve를 그대로 사용한다. 모든 arm history5/Current전체100/선택후P-N·Dev 관측을 프로그램에 포함했다.

Python3.12.3/SciPy1.15.3 실제 import를 확인했다. Shared 환경·원 native·기존 runner·원 자산 수정0. COBYLA maxiter는 function evaluation 상한, tol은 trust-region 하한이다. [공식1.15.3 문서](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-cobyla.html).

CPU 검증은 controller/실제 SciPy synthetic/native metric fixture/Runtime toy state/technical evidence 경계 검사이며 실제 Llama PASS가 아니다. 독립 bounded controller 및 Runtime CPU review를 수행했고 부모가 통합했다. receipt 중복키와 snapshot교체 guard 누락은 GPU실행 전 고쳐 회귀검사했다. unittest discovery 경로 오류는 모듈별 실행으로 고쳤다.

```
{tests_summary}
```

Actual 기술 상태 `{technical}`, scientific gate `{actual}`. 필수 N4/BLUE48/BLUE45678 실제 연결, teacher/repeat, cache/rollback/history5 및 C45678≥7vectors·a4≥2는 technical READY의 조건이며 CPU 결과로 대체하지 않는다. 실제 비용·peak·품질은 아직 이 인계에서 미측정이다.

## 제출·자원·모니터링

Technical job `{None if tech is None else tech['job_id']}`. 마지막 저장 scheduler 관측:

```
{'' if tech is None else tech['last_observation']}
```

Scientific array `{None if science is None else science['job_id']}`. 등록 arm `{[] if science is None else list(ARMS)}`. 기술 READY 전에 six-main을 등록/실행/PASS했다고 쓰지 않는다. 배열 순서는 C45678,N4,F48,G48,C4,C48; READY 후 가용 cap2 내 upfront 등록 코드가 준비돼 있다. 이후 제출은 agent 자동예약이 아니라 명시 recall 시 현재 gate/자원을 확인해야 한다.

49238은 이 task의 exact resource-only 조회에서 COMPLETED였고 기존 raw/에러/성능은 열지 않았다. 다른 job mutation0. 신규 기본1GPU/8CPU/60416MiB/exportNONE/Requeue0, 기술12h reserve, GPUh hardcap null. Science wall/cost는 실제 pilot 이후 산정하며 최대 산술 target89000/Adam408000/loss497000/solve890/score960/history300을 실측이라고 쓰지 않는다.

초기 실제 gate 또는 PENDING/HOLD 인계 후 polling/logtail/sleep/heartbeat/callback/자동추가제출0. 등록된 정상 프로그램은 변경하지 않는다.

## 자산·저장 및 한계

W0 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, method seed20260916/C4 seed20260915, FP32/eager/TF32off. fixed first1000 root `{lock['prefix1000_ordered_root']}`. cold7 contexts/token/RNG와 동일 reference/teacher192를 재사용 후보로 봉인하고 실제 재현 검사는 technical에 둔다. 모델/teacher fullSHA는 prior lock+현재 stable stat 재사용이며 불필요 전량 재해시0.

64GiB disk reserve, freeze 당시 free `{lock['storage']['free_bytes']}` bytes. 후보 W는 구조 공유 CPU RAM, history는 최종 append 직전 copy-on-write한다. v2 disk W/M checkpoint0: 정확 crash-resume, 후속 selected-weight 독립 재구성은 NOT_AVAILABLE. 필수native target/key/loss/counters, 후보 점수/ID·budget/cache·commit/history hash 및 평가 row는 local-only로 저장한다. 삭제된 cold7 CP21 재생성0. 이번 준비가 이전 삭제/repair task를 재개한 것은 아니다.

원 모델 로드/forward 비용과 새 teacher 생성 여부는 실제 technical 산출물 이후만 확정한다. 신규 science 미시작/미확인 범위를 NA로 남긴다. 성능 우열·수렴·support최적성·미분 또는GPU continuation PASS를 주장하지 않는다.

## 재현·다음 사용자 호출

코드 `project/run_scripts/sequential_local_z_allocation/README.md`, local root `{ROOT}`. 실행 lock과 source archive를 먼저 확인한다. CPU 재현: `python -m project.run_scripts.sequential_local_z_allocation.control check --worktree WORKTREE` (새 source-hash receipt가 없을 때만 create-once).
Resume manifest `{resumeref['path']}`, SHA `{resumeref['sha256']}`. 다음 명시 recall에서는 저장된 technical READY/failure와 실제 job 상태에 한정 접근해 등록/미등록 scope부터 구분한다. GH 중복 raw/GPU 감사는 요구하지 않는다.

NO_BROADCAST_NOT_REQUIRED. Raw/tensor/prompt/teacher/fullstdout Git0. monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call.
'''
    refs.append(write_text(w/REPORT/'preparation-submission-ko.md',text))
    ack=f'''# SH4 v2 인계

{TASK}; FULL_READ/M0 completed. {boundary}; technical job {None if tech is None else tech['job_id']}; science registered {science is not None}.
Source `{lock['source_head']}` / tree `{lock['source_tree']}`; lock SHA `{identity(lockpath)['sha256']}`.
Actual validation {technical}; science {actual}. cap2, no previous-task mutation. [준비·제출 보고](../../../{REPORT}/preparation-submission-ko.md).
Resume `{resumeref['path']}` SHA `{resumeref['sha256']}`. monitoring_active=false / automatic_resume=false / explicit_user_call only.
'''
    for rel in ('messages/acks/server4/2026-09-17-sequential-local-z-allocation-v2.md',
                'messages/server-heads/server4/2026-09-17-sequential-local-z-allocation-v2.md'):
        refs.append(write_text(w/rel,ack))
    refs.append(save(w/'tasks/status/odeedit_sequential_local_z_v2_s4_20260917/server4.json',resume))
    refs.append(save(w/'runs/odeedit_sequential_local_z_v2_s4_20260917/receipt.json',dict(instruction_id=TASK,
        source=lock['source_head'],source_tree=lock['source_tree'],lock=identity(lockpath),resume=resumeref,
        jobs=dict(technical=None if tech is None else tech['job_id'],science=None if science is None else science['job_id']),
        boundary=boundary,monitoring_active=False,automatic_resume=False)))
    manifest=save(w/REPORT/'publication-manifest.json',dict(instruction_id=TASK,members=[dict(
        path=str(Path(r['path']).relative_to(w)),bytes=r['bytes'],sha256=r['sha256']) for r in refs],
        raw_free=True,scientific_results_not_claimed=True))
    rooted=save(w/REPORT/'rooted-receipt.json',dict(manifest=dict(path=str(Path(manifest['path']).relative_to(w)),
        bytes=manifest['bytes'],sha256=manifest['sha256']),source=lock['source_head'],resume=resumeref))
    print(json.dumps(dict(report=identity(w/REPORT/'preparation-submission-ko.md'),manifest=manifest,receipt=rooted,resume=resumeref)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True)
    p.add_argument('--boundary',choices=['PENDING_HANDOFF','INITIAL_VALID','TECHNICAL_HOLD'],required=True)
    a=p.parse_args();publish(Path(a.worktree),a.boundary)
