"""CPU-only exact storage blocker receipt. No scientific raw or job mutation."""
import datetime
import json
import os
from pathlib import Path
import shutil
from .common import ROOT,member,write
from .control import call,PACKAGE,project_queue


def main():
    repo=Path.cwd();oldpath=ROOT/'M/attempt-v1/execution.lock.json'
    old=json.loads(oldpath.read_text());disk=shutil.disk_usage(ROOT);reserve=old['disk']['reserve_bytes']
    if disk.free>=reserve:raise ValueError('STORAGE_NO_LONGER_BLOCKED_RECHECK_ADMISSION')
    root=ROOT/'receipts/skip-t-r1';attempt=ROOT/'M/attempt-skip-t-v1'
    if attempt.exists():raise ValueError('UNEXPECTED_NEW_ATTEMPT_REQUIRES_INSPECTION')
    evidence=dict(status='RESOURCE_BLOCKED_STORAGE_PRE_SUBMISSION',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        nonce='ODEEDIT-GH-SH4-ENFC-SKIP-T-ALL-M-20260918-R1',
        source_head=call(['git','rev-parse','HEAD'],repo),source_tree=call(['git','rev-parse','HEAD^{tree}'],repo),
        free_bytes=disk.free,free_GiB=disk.free/(1<<30),required_reserve_bytes=reserve,
        required_reserve_GiB=reserve/(1<<30),shortfall_bytes=reserve-disk.free,
        shortfall_GiB=(reserve-disk.free)/(1<<30),free_inodes=os.statvfs(ROOT).f_favail,
        reserve_is='retained prior conservative storage plan, not measured irreducible output size',
        components=old['disk'],old_M_lock=member(oldpath),
        original_freeze_failure='M_STORAGE_RESERVE_UNAVAILABLE before source/attempt creation or sbatch',
        owner_resource_only_queue=project_queue(),new_M_jobs=[],new_T_jobs=[],new_GPU_seconds=0,
        intended_M_episodes=10,intended_endpoints=80,intended_throttle_if_unoccupied=2,
        new_execution_lock='NOT_CREATED_RESOURCE_PREFLIGHT_FAILED',
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        full_read_CPU=member(root/'preflight-fullread.json'),
        skip_control_current=member(repo/PACKAGE/'skip_t_control.py'),
        post_preflight_delta='scheduler48h formatting2-00:00:00 only, AST parsed before freeze',
        prior_terminal_reused=member(ROOT/'receipts/paired-stop-r1/scheduler-terminal.json'),
        prior_GPU_seconds=3231,no_old_scheduler_requery=True,
        deletion=0,checkpoint_waiver=False,endpoint_policy_unchanged=True,
        science_threshold_change=False,S_R_L_submitted=0,
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        unblock='provide capacity to meet72GiB at next admission or authorize exact alternative storage/deletion; no automatic action')
    receipt=write(root/'resource-blocked.json',evidence)
    package=repo/'experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/skip-t-all-m-r1'
    report=package/'resource-blocked-ko.md';report.parent.mkdir(parents=True,exist_ok=True)
    text=f'''# ENFC T 생략 / M 전체 재제출: 저장공간 부족으로 미제출

상태: RESOURCE_BLOCKED_STORAGE_PRE_SUBMISSION. 관측시각 {evidence['time']}.

최신 사용자 “그냥 T는 건너 뛰고 M전부 올려”를 적용했다. T 신규/복구/continuation 없이 새 M10 independent cold episode만 허용한다. T=SKIPPED_USER_DIRECTED, full_numerical_validation=NOT_ESTABLISHED. S/R/L 제출0.

## 실제 제출 경계

M job mapping은 **없음(0/10 제출)**이다. T 신규0, 신규 GPU0초. source freeze의 storage preflight에서 중단되어 새 attempt/source archive/execution lock도 생성되지 않았다. 이는 M GPU 자원부족 PENDING 인계나 M_INITIAL_VALID가 아니다.

| 항목 | 실제값 |
| --- | ---: |
| 가용 저장공간 | {disk.free} bytes ({disk.free/(1<<30):.4f} GiB) |
| 기존 보존계획 reserve | {reserve} bytes (72 GiB) |
| reserve 대비 부족 | {reserve-disk.free} bytes ({(reserve-disk.free)/(1<<30):.4f} GiB) |
| 가용 inode | {evidence['free_inodes']} |
| 새 M 제출 / endpoint 생성 | 0 / 0 |

72GiB는 기존 보존계획의 보수적 reserve이며 실제 최소 산출물 크기의 실측값은 아니다. 구성은 final L4 raw {old['disk']['final_L4_raw_bytes']}B, 최대 고유 gradient {old['disk']['maximum_unique_gradient_bytes']}B, EN-F blocked factor 상한 {old['disk']['upper_ENF_blocked_bytes']}B 및 native/factors/평가/임시 I/O다. 공유 disk 감소 원인은 조사·추정하지 않았다. 원자료 삭제, 기존 checkpoint 정리, 압축률/좋은 결과 가정, endpoint 저장 축소로 우회하지 않았다.

## 준비된 변경과 재사용

- 소스 {evidence['source_head']} / tree {evidence['source_tree']}. 최신 GH2a450e6를 포함한다. 실행 source가 아니라 **제출 준비 source**다.
- 별도 waiver routing은 old T_READY/afterok/failcancel을 새 M에서 요구하지 않는다. 4개 좁은 CPU routing test PASS, import/syntax 확인. 기존 CPU82는 exact 동일 method/model/helper 범위에 재사용했으며 광범위 T 대체검사를 실행하지 않았다.
- geometry/optimizer/current per-sequence guard/ID/Armijo/수치/공유 native는 변경0. 과거 T 부분 관측은 해당 범위만 보존하며 전체 numerical PASS로 승격0.
- B1 nativefit·적합 observer REUSE, B2–B10 최대9 independent W0/M0 fits/900targets. 기존330행 reuse/10episode계획 보존. M80 final L4 저장 의무 유지.
- 취소된 b001의 부분 geometry는 완전 factor tensor closure가 없어 재사용하지 않으며, 안전한 W0/검증 native capsule에서 시작하도록 준비했다. 이전 SIGTERM rollback 성공을 주장하지 않는다.
- 새 제출 직전 자원이 허용되면 기존 점유를 합산하여 최대 array%2, 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0. hour cap=null. 기존 resource-only queue 관측은 [근거](resource-blocked.json)에 있다. job 자체는 등록되지 않았다.

## 보존·비용·종료

과거 T49928 FAILED/M49973 취소와 frozen source/raw/paired-stop은 불변이며 terminal은 기존 exact receipt를 재사용했다. 과거3231 GPU초와 이번0초를 분리한다. 이전 cancelled M을 성공 endpoint로 사용하지 않았다.

재개에는 다음 admission 시72GiB 이상 여유 또는 구체적 대체 저장경로/정리대상에 대한 권한이 필요하다. 기존 자료 삭제·타경로 무단 이동·reserve 임의 하향은 하지 않는다. 공간을 기다리는 자동 polling/daemon/추가 submit0. 사용자 호출 대기이며 M 초기 확인은 NOT_RUN이다.

재현: 아래 명령은 read-only 자원 확인이다. scientific 실행이나 T 검증이 아니다.

```bash
df -B1 /data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1
```
'''
    with report.open('x') as f:f.write(text)
    published=write(package/'resource-blocked.json',evidence)
    manifest=write(package/'artifact-manifest.json',dict(status=evidence['status'],source=evidence['source_head'],
        members=[member(report),published],raw_payload_committed=False))
    rooted=write(package/'rooted-receipt.json',dict(manifest=manifest,local_evidence=receipt,
        new_GPU=0,new_jobs=[],T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED'))
    resume=write(ROOT/'resume-manifest-skip-t-storage-blocked-r1.json',dict(evidence=evidence,report=member(report),
        manifest=manifest,rooted_receipt=rooted,old_resume=member(ROOT/'resume-manifest-paired-stop-r1.json')))
    result=dict(report=member(report),manifest=manifest,rooted_receipt=rooted,resume=resume,evidence=receipt)
    write(root/'handoff-index.json',result);print(json.dumps(result))


if __name__=='__main__':main()
