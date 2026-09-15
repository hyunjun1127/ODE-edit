"""Publish sealed submission facts only; no scheduler/output observation."""
import argparse
from pathlib import Path
import json
from .control import identity,save,sha,SESSION
from .repair_control import TASK

def publish(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve()
    release=json.loads((a/'submission-v1/release-receipt.json').read_text())
    lock=json.loads((a/'repair.lock.json').read_text())
    science=json.loads((a/'scientific.lock.json').read_text())
    resource=json.loads((a/'submission-v1/resource-preflight.json').read_text())
    assert release['gate']=='PENDING_GATE_NOT_RUN'
    assert 'JobState=PENDING' in release['last_record']
    refs={n:identity(a/n) for n in ('full-read-receipt.json','cpu-checks.json','cpu-checks-v2.json',
        'repair.lock.json','scientific.lock.json','submission-v1/resource-preflight.json',
        'submission-v1/held-submission.json','submission-v1/held-inspection.json','submission-v1/release-receipt.json')}
    expected=dict(technical_output=lock['output'],technical_PASS=str(Path(lock['output'])/'technical-PASS.json'),
        technical_failure=str(Path(lock['output'])/'failure.json'),science_output=science['output'],
        G0=str(Path(science['output'])/'G0_PASS.json'),first_checkpoint=str(Path(science['output'])/'B001/checkpoint.pt'),
        B2_entry=str(Path(science['output'])/'B002/entry.json'),terminal=str(Path(science['output'])/'terminal.json'),
        scientific_failure=str(Path(science['output'])/'failure.json'),
        existence='NOT_INSPECTED_AFTER_PENDING_RELEASE; not asserted absent or present')
    resume=dict(instruction_id=TASK,host='server4',owner_session=SESSION,status='WAITING_USER_RESUME',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        last_observation=release['last_observation'],last_job_state='PENDING',last_reason='None',
        job_id=release['job_id'],source_head=lock['source_head'],source_tree=lock['source_tree'],
        source_archive=lock['source_archive'],source_branch='codex/server4-ep-tw1-fd-repair-v1',
        locks=refs,technical_gate='NOT_RUN_NOT_OBSERVED',scientific_G0='NOT_RUN_NOT_OBSERVED',
        last_batch=None,next_ordinal=None,expected=expected,
        conditional_program='one allocation; saved episode technical PASS -> exact-lock verified fresh W0 B100x10',
        saved_episode=lock['saved_episode'],teacher_manifest=lock['teacher_manifest'],
        diagnostic_native_fit=0,diagnostic_native_solve=0,new_teacher=0,
        old_failed_job='47884',old_failed_gpu_seconds=473,old_failed_commits=0,
        teacher_reused_job='47592',teacher_prior_gpu_seconds=98,
        old_fit_component_seconds=286.54453050531447,old_fit_is_nested_in_473_not_additional=True,
        current_allocated_gpu_seconds='NOT_OBSERVED',scientific_completed_requests='NOT_OBSERVED',
        resource=lock['resource'],admitted_including_new=resource['total_admitted_with_new'],
        new_scientific_chains_max=1,model_G0_not_inferred_from_CPU86=True,
        preserved_old_coarse_FAIL=True,scientific_promotion=False,
        forbidden_or_deferred=['baseline rerun','N4 calibration','teacher rebuild','quality restoration','ODE','additional policies','Audit','MMLU','Report256'],
        next_user_recall='bounded exact47942 state + technical/G0 receipts; no automatic retry',
        no_broadcast='NO_BROADCAST_NOT_REQUIRED')
    resumeref=save(a/'resume-manifest.json',resume)
    package=w/'experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/fd-repair-r1'
    package.mkdir(parents=True,exist_ok=False)
    save(package/'resume-receipt.json',dict(resume_manifest=resumeref,**resume))
    save(package/'source-input-manifest.json',dict(execution_source=lock['source_head'],execution_tree=lock['source_tree'],
        archive=lock['source_archive'],saved_episode=lock['saved_episode'],refs=refs,
        source_member_count=len(lock['members']),source_root=lock['source_root'],
        old_failed_source='3fb0bfb27773ec9d016e76fcdc978dc996f010a7',
        current_GPU_measurements='NOT_OBSERVED',numerical=lock['repair_numerics'],budget=lock['diagnostic_budget']))
    save(package/'evidence-reuse-manifest.json',dict(old_failure=lock['preserved_old_coarse'],
        old_cpu=lock['old_cpu_receipt'],native_episode=lock['saved_episode'],
        teacher=lock['teacher_manifest'],prior_teacher_fullSHA='REUSED_WITH_CURRENT_STAT_LOCK',
        current256MBepisode_fullSHA='VERIFIED_BEFORE_SUBMIT',
        new_model_teacher_whole_rehash=False,old_gradient_and_postfit_RNG='NOT_SAVED',
        diagnostic_not_continuation=True,CPU_tests=86,red_focused_tests=35,
        red_tests_subset_of86_not_added=True,actual_direct_Llama_or_FD='NOT_OBSERVED'))
    report=f'''# EP-TW-1 저장 episode FD repair — 준비·제출 인계

Instruction `{TASK}`. **PENDING / WAITING_USER_RESUME**. 기술 GPU PASS, scientific G0 또는 1000요청 완료 보고가 아니다.

## 실제 제출 및 중지

단일 job **{release['job_id']} / odeedit_ep_tw1_repair_s4**를 held→owner/source/args/memory 검증→release했다.
마지막 관측은 **{release['last_observation']} UTC / PENDING / Reason=None / RunTime0**이다.
그 뒤 scheduler/result/log 및 결과파일 존재 여부를 추가 조회하지 않았다.
프로그램은 그대로 두며 agent만 명시 사용자 호출을 기다린다. 별도 후속 submit/callback/heartbeat/자동분석0.

같은 GPU allocation에서 먼저 저장 episode 기술 진단을 수행한다. E/D 각각 direct-weight route와 두 방향 FD가 모두 통과하고
source/model/teacher/scientific-lock identity가 일치해야 다음 독립 process가 fresh W0/coldM0 EP-TW-1 first1000 B100×10을 실행한다.
기술 실패가 나면 shell fail-closed이며 본실험0. 과거 실패 B1을 B2부터 resume하지 않는다.

## 보존한 실패·재사용 범위

기존47884 FAILED1:0/473 allocated GPU초, native100/solve1/commit0을 그대로 보존한다.
옛 E AD0.026770689893859417 대 FD0.25586175077340434/0.22180749523904691의 coarse FAIL은 폐기하지 않는다.
국소성 미확인이 우선 진단 대상이며 AD가 맞다고 확정하거나 EP 편집 효과 실패로 해석하지 않는다.
Saved Vp/A/Z/anchor/radius/K/H 파일 {lock['saved_episode']['bytes']}B,
SHA `{lock['saved_episode']['sha256']}`를 새로 fullSHA 검증했다. 기술 과정 native target/solve0.
원 gE/gD·postfit RNG·optimizer/local teacher 실물이 없어 complete continuation checkpoint가 아니다.
새 diagnostic seed2026091503을 봉인하며 old gradient byte equality를 주장하지 않는다.
기존 teacher47592/192문서/98GPU초는 prior fullSHA+현재 stat 및 manifest로 재사용한다. 재생성/전량 중복재해시0.
Native286.5445305초는 기존473초 내부 component라 따로 더하지 않는다.

## 수리 및 검증 수준

|항목|확인 범위|
|---|---|
|FULL_READ|사용자 점검·GH review36 scalar checks·repair envelope/dispatch·v3계약·PROTOCOL·old receipt/source|
|CPU|전체86개 PASS; 독립 red35개는 그 부분집합. 실제 Llama parity 아님|
|Direct 경로|L4 selected weight leaf로 functional_call, custom residual node 우회. gC와 gW Aᵀ 및 bilinear 검산 예정|
|조기 저장|각 완료 E/D gradient·E0/D0·RNG·rows 저장; 매 signed probe C/hash/action/ULP64/rows/receipt를 판정 전에 저장|
|FD|E self+고정독립 방향 뒤 D self+고정독립 방향. 각 h/2^k k0..9 전부, 최대80 signed objective 관측|
|판정|relative .15/absolute1e-7/convergence .15 유지. 분모 max(absAD,1e-7), k>=1 연속2 resolved scale, 첫 valid window|
|해상도|실제 FP32 action·중복/zero·baseline3회 jitter·Taylor를 기록. 8eps loss-scale은 model 오차상한이 아님|
|과학 계약|policy/ledger/native fitter/map/target/teacher 불변. positive quality allowance0, RAW fallback 원규약 유지|
|Actual GPU|Direct/FD E&D/materialization/새 G0 모두 **NOT_OBSERVED**|

한 gradient sweep 내부에서 예외가 나면 미완성 g/rows 전체는 없을 수 있다. 직전 완료 단계와 오류는 보존하며
미완성 값을 완전 gradient로 부르지 않는다. CPU 검사는 모든 neural backward의 독립 증명도 아니다.
새 scientific B1의 Vp/A/gE/gD가 동일해야 저장 기술 증거를 정확 재사용한다. 다르면 새 endpoint에서 같은 bounded 검사만
진행하고 same-state라고 부르지 않는다. 검사 실패/불확정이면 수치 허용오차를 늘리거나 다른 과학 정책을 추가하지 않는다.

## 자원·비용·출력

제출 직전 project active/admitted0 + 신규1 =1≤cap2. 다른 사용자/job변경0.
1GPU/8CPU/60416M/exportNONE/no-requeue, wall12h, GPU-hour hardcap=null.
가용 {resource['free_bytes']:,}B/inodes{resource['free_inodes']:,}. 신규40GiB reserve+별도20GiB 여유 검사(독점 예약 아님).
기술300–1800초/본실험2–8GPUh는 사전 추정이며 신규 실제 allocated/peak/결과는 아직 미관측이다.
E grid 최대280 microbatch groups, D grid2560 document forwards; 각각 direct backward 및 baseline 반복·I/O/restore는 별도 ledger.
기술과 scientific의 분모·비용은 구분한다. 진단 old native100을 신규1000 성공분모에 더하지 않는다.

Technical `{lock['output']}`; conditional science `{science['output']}`.
Source `{lock['source_head']}` / tree `{lock['source_tree']}`.
Archive `{lock['source_archive']['sha256']}`.
Repair lock `{refs['repair.lock.json']['sha256']}`; science lock `{refs['scientific.lock.json']['sha256']}`.
Resume `{resumeref['path']}` / SHA `{resumeref['sha256']}`.

CPU 재현: `PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/bg_tw_reference/ep_tw -t . -p 'test_*.py'`.
제출/conditional 명령 및 경계는 `project/run_scripts/bg_tw_reference/ep_tw/REPAIR.md`, `repair.sbatch`와 sealed submission receipts에 있다.
현재 원실행을 다시 제출하라는 명령이 아니다. 이후 explicit recall에서 exact47942/technical/G0만 한정 확인한다.
Raw/tensor/prompt/teacher/probe/fullstdout local-only, Git에는 코드와 compact provenance만. PNG생성0.
NO_BROADCAST_NOT_REQUIRED. 기존 source/실패/teacher/raw 및 다른 paused task 불변, scientific_promotion=false.
'''
    rp=package/'diagnostic-report-ko.md'
    with rp.open('x') as f:f.write(report)
    summary=f'''# EP-TW-1 FD repair 제출/PENDING

{TASK}. CPU86 PASS, 실제GPU direct/FD/G0는 NOT_OBSERVED.
Job{release['job_id']} held검사/release; 마지막 {release['last_observation']} PENDING/Reason=None.
cap2 중 신규1GPU/8CPU/60416M/12h, technical PASS 후에만 freshW0 단일1000 science 조건부 실행.
진단 saved episode 재사용/nativefit0; old47884 473초/teacher98초 불변. source {lock['source_head']}.
보고서 {rp.relative_to(w)} SHA {sha(rp)}.
Resume {resumeref['path']} SHA {resumeref['sha256']}.
WAITING_USER_RESUME; 추가query/자동재개/후속submit/타task변경0. G0 또는 전체1000 완료 주장0.
'''
    message=w/'messages/server-heads/server4/2026-09-15-ep-tw1-fd-repair.md'
    message.parent.mkdir(parents=True,exist_ok=True)
    with message.open('x') as f:f.write(summary)
    save(w/'tasks/status/odeedit_ep_tw1_fd_repair_s4_v1/server4.json',dict(status='WAITING_USER_RESUME',
        job_id=release['job_id'],gate='G0_PENDING_NOT_RUN',monitoring_active=False,automatic_resume=False,
        resume_manifest=resumeref,report=identity(rp),scientific_promotion=False))
    save(w/'runs/odeedit_ep_tw1_fd_repair_s4_v1/submission-receipt.json',dict(job_id=release['job_id'],
        last_observation=release['last_observation'],last_state='PENDING',resource=lock['resource'],
        source_head=lock['source_head'],source_archive=lock['source_archive'],repair_lock=refs['repair.lock.json'],
        science_lock=refs['scientific.lock.json'],scientific_conditional_only=True,resume=resumeref))
    members=[identity(p) for p in sorted(package.iterdir()) if p.is_file()]
    manifest=save(package/'analysis-manifest.json',dict(kind='COMPACT_PREPARATION_SUBMISSION_NOT_RESULT_ANALYSIS',
        files=members,raw_payload=False,PNG_count=0,execution_source=lock['source_head']))
    save(package/'rooted-receipt.json',dict(manifest=manifest,report=identity(rp),resume=resumeref,
        status='WAITING_USER_RESUME',raw_free=True,scientific_results_observed=False))
    return dict(report=identity(rp),resume=resumeref,message=identity(message))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps(publish(x.worktree,x.attempt)))
