"""Create-once factual main-gate handoff; never queries or changes jobs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .common import ROOT,TASK,ARMS,identity,save,digest
from .control import copy_once,git,source_list
from .handoff import write_text,REPORT,AUDIT
from .resume_control import RESUME,TECH_ROOT


def publish(w,evidence_path):
    evidence=json.loads(evidence_path.read_text());boundary=evidence['boundary']
    assert boundary in ('MAIN_INITIAL_VALID','MAIN_GPU_RESOURCE_PENDING_HANDOFF','UNRESOLVABLE_TYPED_BLOCKER')
    lockpath=ROOT/'execution.lock.json';lock=json.loads(lockpath.read_text())
    ready_path=TECH_ROOT/'READY.json';ready=json.loads(ready_path.read_text()) if ready_path.is_file() else None
    release_path=RESUME/'submission-science-r1/released.json'
    release=json.loads(release_path.read_text()) if release_path.is_file() else None
    if boundary!='UNRESOLVABLE_TYPED_BLOCKER':
        assert ready and release and release['all_six_registered']
        assert evidence['registered']==6 and evidence['released']
    else:assert evidence['scientific_block_reason'] and evidence['repair_authority_boundary']
    now=datetime.now(timezone.utc).isoformat()
    resume=dict(instruction_id=TASK,nonce='ODEEDIT-GH-SH4-SLZV2-MAIN-GATE-RESUME-20260917-R1',
        time=now,host='server4',owner='janghj',session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
        task_state='MONITORING_PAUSED_AWAITING_USER',boundary=boundary,
        execution_head=lock['source_head'],execution_tree=lock['source_tree'],archive=lock['source_archive'],
        execution_lock=identity(lockpath),orchestration_head=git(w,'rev-parse','HEAD'),
        orchestration_tree=git(w,'rev-parse','HEAD^{tree}'),override=identity(RESUME/'override-receipt.json'),
        preserved_historical_pause=identity(ROOT/'resume-manifest.json'),last_observation=identity(evidence_path),
        technical_job='49421',technical_ready=identity(ready_path) if ready else None,
        scientific_array=None if release is None else release['job_id'],
        registered_scientific_arms=[] if release is None else list(ARMS),
        submission=None if release is None else identity(release_path),
        actual_scientific_gates=evidence.get('gate_evidence',[]),
        whole1000_completion='NOT_REVIEWED_NOT_CLAIMED',all_six_actual_validation=False,
        capsule=ready['capsule'] if ready else None,teacher=ready['teacher'] if ready else None,
        prefix1000=lock['prefix1000_ordered_root'],gpu_cap=2,
        expected_outputs={a:str(ROOT/'arms'/a/'attempt-v1/output') for a in ARMS},
        disk_W_M_checkpoint=False,exact_crash_resume='NOT_AVAILABLE',GPU_off_on_continuation='NOT_TESTED',
        source_runtime_changed=False,monitoring_active=False,automatic_resume=False,
        resume_trigger='explicit_user_call',callbacks=False,after_handoff_additional_submit=False)
    resumeref=save(RESUME/'resume-manifest.json',resume)
    dest=w/REPORT/'resume-main-r1';audit=w/AUDIT/'resume-main-r1';members=[]
    cpupath=ROOT/'checks'/(digest(source_list(w))+'.json')
    cpu=json.loads(cpupath.read_text());assert cpu['status']=='CPU_ONLY_PASS'
    members.append(copy_once(cpupath,audit/'cpu-checks-final.json'))
    if (audit/'preflight.json').is_file():members.append(identity(audit/'preflight.json'))
    for src,target in ((RESUME/'override-receipt.json',audit/'override-receipt.json'),
                       (evidence_path,dest/'boundary-evidence.json'),
                       (RESUME/'resume-manifest.json',dest/'resume-manifest.json')):
        members.append(copy_once(src,target))
    if release:
        for name in ('released.json','resource.lock.json','admission.json','science-held.json'):
            members.append(copy_once(release_path.parent/name,dest/name))
    stages=[]
    for p in sorted(TECH_ROOT.glob('*/result.json')):
        v=json.loads(p.read_text());stages.append(dict(stage=v['stage'],status=v['status'],identity=identity(p)))
    cost=json.loads(Path(ready['cost']['path']).read_text()) if ready else None
    resource=None if release is None else json.loads((release_path.parent/'resource.lock.json').read_text())
    summary=dict(stages=stages,ready=None if not ready else identity(ready_path),
        technical_cost=cost,scientific_initial_only=True,old_pending_not_reclassified=True)
    members.append(save(dest/'technical-summary.json',summary))
    if ready:
        members.append(copy_once(ready_path,dest/'technical-READY.json'))
    stage_table='\n'.join('| '+r['stage']+' | '+r['status']+' | `'+r['identity']['sha256']+'` |' for r in stages)
    gates=evidence.get('gate_evidence',[])
    gate_table='\n'.join(f"| {g['arm']} | {g['actual_B1_requests']} | {g['history_appends']} | {g['next_ordinal']} |" for g in gates)
    mapping='\n'.join(f'| {i} | {arm} | {release["job_id"]}_{i} |' for i,arm in enumerate(ARMS)) if release else '| NA | 미등록 | NA |'
    details='NOT_MEASURED' if cost is None else json.dumps(dict(seconds=cost['seconds'],native=cost['native'],
        peak_GPU_allocated=cost['peak_GPU_allocated'],peak_GPU_reserved=cost['peak_GPU_reserved'],
        peak_host_RSS_bytes=cost['peak_host_RSS_bytes']),ensure_ascii=False,indent=2)
    scheduler='NOT_RECORDED' if resource is None else resource['technical_scheduler']
    storage='NOT_RECORDED' if resource is None else json.dumps(resource['storage_plan'],ensure_ascii=False,indent=2)
    cpu_result='\n'.join(cpu['output'].splitlines()[-5:])
    text=f'''# Sequential Local-z Allocation v2 — 본실험 초기 gate 재개 인계

상태: **{boundary} / MONITORING_PAUSED_AWAITING_USER**. 여섯 arm 전체 완료·성능 검증 보고가 아니다.

## 권한·원본 보존

사용자 정정에 따라 기술 PENDING/PASS에서 멈추지 않고 본실험 기준으로 진행했다. Instruction `{TASK}`.
이전 technical49421 PENDING 인계와 원 resume manifest는 수정하지 않았다. 원 source `{lock['source_head']}` / tree `{lock['source_tree']}`, archive `{lock['source_archive']['sha256']}`, execution lock `{identity(lockpath)['sha256']}`를 유지했다.
이번 재개·제출·인계용 source는 `{resume['orchestration_head']}`이며 실제 모델 runtime source와 구분한다. 과학식/허용치/arm/sample/teacher 변경0, 새 W/M disk checkpoint0.

## CPU와 실제 기술 검증

재개·제출·held·초기 gate 결속·자원 산술 CPU 검사를 기존 테스트와 함께 수행했다. 별도 독립 red agent는 이번 재개에서 사용하지 않았고 SH4 자체 검산이다. 이 CPU 결과가 Llama 검증을 대체하지 않는다.

```
{cpu_result}
```

| 단계 | 저장 판정 | 근거 SHA256 |
| --- | --- | --- |
{stage_table}

기술검사의 native 연결·반복 점수·coverage·history는 기술 first100에서 확인한 범위다. CPU 검사 또는 파일 존재만으로 실제 PASS를 만들지 않았다. C45678 coverage는 운영상 gate-vector 수이며 최적성·solver simplex 인증이 아니다. Teacher 재사용/재생성은 technical-READY와 capsule의 실제 identity를 기준으로 한다.

Native N4/48/45678 연결은 해당 모든 layer의 실제 weight byte가 일치했다(maxabs0). 기존 W0 teacher의 original/effective D=0으로 재생성 없이 재사용했다. C45678은 N4 제외 완료 search vector11/unique endpoint11/distinct a4=6, search suffix-fit32/추가Adam888, pruning4완료, 예산 미완료 후보1이었다. 미완료 후보는 score pool에서 제외했고 기술 오류0, 예산 확대0이다. 종료 이유 SUFFIX_FIT_CAP이며 optimizer 수렴 PASS를 주장하지 않는다.

실측 기술 비용(중첩 component 중복 합산 금지):

```json
{details}
```

프로그램 wall/native timer와 Slurm allocated GPU-sec는 다른 단위다. 기술 stage/reference 재계산은 본실험6000 arm-request 분모에 넣지 않는다. Pure writer가 분리되지 않은 값은 NOT_SEPARATED다. 이 문서에서는 본실험 전체 비용·최종 결과를 분석하지 않는다.

완료 technical49421의 scheduler 기록(본 job 3379초×1GPU, batch/extern 행을 더해 중복 청구하지 않음):

```
{scheduler}
```

## 본실험 등록·자원

| Array index | Arm | Job ID |
| --- | --- | --- |
{mapping}

새 scientific scope는 동일 cold W0/M0 first1000의6chains/60batches뿐이다. Held owner/source/args/resource/dependency inspection과 release 근거는 submission receipt를 따른다. 각1GPU/8CPU/60416MiB/exportNONE/Requeue0, 프로젝트 cap2. 공통 기술은 afterok로 직렬화하며 다른 job은 변경하지 않는다. 정확한 마지막 queue/resource 상태와 판정은 boundary-evidence.json에 있다.

최종 관측에서 GPU8/8 할당·가용0과 모든 main Resources 대기를 확인했다. Priority는 양수, dependency는 해제된 상태이며 수동 hold가 아니다. 동시에 host memory의 scheduler 예약 여유40960MiB도 요청60416MiB보다 작았다. GPU 부족은 실제 필요조건 위반이지만 **GPU만의 독점 원인이라고 주장하지 않는다**. 본실험 allocated GPU-seconds는 당시0이고 actual initial gate는 미관측이다.

제출16h는 실제 common technical wall×10×1.5에 반올림 여유를 둔 자원 예상이지 실측 과학시간/성능 gate가 아니다. 기술 완료 후 사용 가능 disk가 약53.1GB로 바뀌어, 처음의 미측정64GiB reserve와 별개로 아래 실측기반44GiB no-checkpoint resource 계획을 잠갔다. 원 execution lock bytes는 보존했다. 평가/후보/target 예산 축소·데이터 삭제·teacher 생성·checkpoint 추가0이며 공유 volume의 향후 증가까지 보장하는 수학적 상한은 아니다.

```json
{storage}
```

## 실제 본실험 초기 연결

| 실제 확인 arm | B1 요청 수 | B1 history append | B2 next ordinal |
| --- | --- | --- | --- |
{gate_table or '| NOT_OBSERVED | NA | NA | NA |'}

본실험 초기 gate의 확인 규칙은 B1 selected/selection seal/commit/5층 history receipt와 B2 entry의 W/M/context/RNG 대조다. 이번 본실험은 해당 단계 미실행으로 이 실제 대조를 수행하지 않았고 NOT_OBSERVED로 남겼다. 확인하지 않은 arm의 실제 PASS나 W10 완주를 주장하지 않는다. GPU-resource PENDING으로 인계한 경우에만 GPU allocation 부족 근거와 모든6개 정상 release를 함께 요구했다. Dependency/수동 hold/Reason=None/단순 기술 PASS를 해당 예외로 쓰지 않았다.

{('기술/권한 block: '+evidence.get('scientific_block_reason','')) if boundary=='UNRESOLVABLE_TYPED_BLOCKER' else ''}

## 저장·한계·정지

W/M은 runtime RAM snapshot만 유지한다. 필수 native target/key/scalar·candidate score/ID/budget/cache·commit/history hash·evaluation rows는 local-only다. 디스크 W/M checkpoint0이므로 exact crash-resume과 사후 selected-weight 독립 재구성은 NOT_AVAILABLE, GPU off/on continuation은 NOT_TESTED다. 원 source/raw/teacher/실패/삭제 이력과 다른 paused task는 보존했다.

인계 후 agent polling/logtail/sleep loop/heartbeat/callback/자동 추가제출/상세 결과분석0. 이미 제출된 프로그램은 hold/cancel하지 않고 정해진 B1–B10 평가·저장을 자연 진행한다. 후속 상세 리뷰는 명시 USER recall이 필요하다.

Resume `{resumeref['path']}`, SHA `{resumeref['sha256']}`. `monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
NO_BROADCAST_NOT_REQUIRED. Raw/tensor/prompt/teacher/fullstdout Git0. 과학적 우열·효과 인과·최적성 판정은 이 보고 범위가 아니다.
'''
    members.append(write_text(dest/'main-initial-handoff-ko.md',text))
    run=w/'runs/odeedit_sequential_local_z_v2_s4_20260917/resume-main-r1/receipt.json'
    members.append(save(run,dict(instruction_id=TASK,boundary=boundary,resume=resumeref,
        old_execution_source=lock['source_head'],new_orchestration_source=resume['orchestration_head'])))
    manifest=save(dest/'publication-manifest.json',dict(members=[dict(path=str(Path(r['path']).relative_to(w)),
        bytes=r['bytes'],sha256=r['sha256']) for r in members],raw_free=True,
        old_sealed_publication_unchanged=True,scientific_completion_claim=False))
    receipt=save(dest/'rooted-receipt.json',dict(manifest=dict(path=str(Path(manifest['path']).relative_to(w)),
        bytes=manifest['bytes'],sha256=manifest['sha256']),resume=resumeref))
    print(json.dumps(dict(report=identity(dest/'main-initial-handoff-ko.md'),manifest=manifest,receipt=receipt,resume=resumeref)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();publish(a.worktree,a.evidence)
