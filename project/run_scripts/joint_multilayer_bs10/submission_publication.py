"""Post-submit compact publication from immutable submission evidence ONLY; no job queries."""
import argparse
from .common import *
from .publication import markdown,REPORT,AUDIT

def publish_submission(repo,attempt):
    repo=Path(repo);p=Path(attempt);lock=read(p/'execution.lock.json');cfg=read(lock['configuration']['path'])
    sub=read(p/'submission.json');release=read(p/'release.json');obs=read(p/'post-release-snapshot.json')
    require(release['status']=='RELEASED' and sub['all_9_registered'],'RELEASE_REQUIRED')
    require('PENDING' in obs['queue'] and 'ReqNodeNotAvail' in obs['queue'] and 'gres/gpu=8' in obs['node'].split('AllocTRES=')[1],'RECORDED_RESOURCE_PENDING')
    audit=read(ROOT/'cpu-audit-bs1-locked/owner-cpu-audit.json')
    receipts={name:record(p/name) for name in ('execution.lock.json','configuration.json','source.tar','admission.json',
        'submission.json','release.json','post-release-snapshot.json','branch.sbatch','collector.sbatch')}
    state=dict(instruction_id=NONCE,override_nonce=OVERRIDE_NONCE,status='MAIN_GPU_RESOURCE_PENDING_HANDOFF',
        initial='INITIAL_NOT_OBSERVED',snapshot_utc=obs['utc'],execution_source=lock['source'],execution_tree=lock['source_tree'],
        lock=receipts['execution.lock.json'],array_job=sub['array_job'],array='0-8%2',collector_job=sub['collector'],
        collector_dependency='afterany:'+sub['array_job'],mapping=sub['mapping'],all_9_registered=True,held_inspection='PASS',released=True,
        queue_reason_at_snapshot='ReqNodeNotAvail, May be reserved for other job',node_gpus_allocated_at_snapshot=8,node_gpus_configured=8,
        project_cap=2,task_cap=2,batch_size=1,steps_per_trajectory=100,layers=list(LAYERS),save_steps=list(SAVE_STEPS),snapshots_planned=36,
        actual_model_validation='NOT_OBSERVED',scientific_completion='NOT_OBSERVED',cpu_tests=audit['tests'],
        monitoring_active=False,automatic_resume=False,exact_editor_resume='NOT_AVAILABLE',
        previous_single_layer_task='STOPPED_NOT_RESUMED',new_large_transfer_bytes=0,NO_BROADCAST_NOT_REQUIRED=True)
    save(repo/'tasks/status/joint-multilayer-bs10-20260929-v1/server4.json',state)
    save(repo/AUDIT/'submission-bs1-v1.json',dict(**state,receipts=receipts,
        owner_audit=True,independent_red_agent=False,raw_free='PASS',markdown_links='PASS',actual_markdown_render='NOT_TESTED_RENDERER_NOT_INSTALLED',
        preserved='original design / 3 parent CP / stopped singleton task / existing other tasks unchanged'))
    lines='\n'.join(f"|{x['job_id']}|{x['checkpoint']}|{x['arm']}|100|" for x in sub['mapping'])
    report=f'''# 다층 공동 편집 USER BS1 제출 인계

상태: MAIN_GPU_RESOURCE_PENDING_HANDOFF / INITIAL_NOT_OBSERVED.

사용자 변경을 실행 설정에 반영했고 GH에 USER 명령임을 직접 전달하여 ACK를 받았다. 3CP × 3방법의 9경로, 모든 L4–L8 편집은 유지한다. 각 경로는 **BS1 × 100 edit**, **25·50·75·100 step마다 다섯 actual full weight 저장**이다. 총36snapshot/180tensor/39.375GiB payload 계획이다. 기존 단층 task는 STOP 유지.

## 실제 등록

모든9경로는 array **{sub['array_job']}_[0–8]%2**, CPU collector **{sub['collector']} / afterany:{sub['array_job']}**로 held 검사 후 release했다. 추가 chat 호출이 후속 경로의 제출조건이 아니다.

|Job|부모 CP|방법|offered edit|
|---|---|---|---:|
{lines}

각 GPU job1GPU/8CPU/60416MiB/server4/gpu/exportNONE/Requeue0, CPU collector8CPU/24576MiB/GPU0. Project/task cap2, 이 task는 최대2동시. Wall7일은 요청값이며 실측 ETA가 아니다. Admission 시 자기 owner S4 queue0 및 프로젝트 active0을 확인했다. 다른 task/job 변경0.

제출 직후 한정 snapshot **{obs['utc']}**: 모든array PENDING(ReqNodeNotAvail, May be reserved for other job), collector PENDING(Dependency). 노드8GPU/할당8GPU. 이것은 이후 현재상태나 scientific completion의 보증이 아니다. 수동 hold/PENDING(None)를 자원부족으로 해석한 것이 아니다. 이 snapshot 이후 job/scheduler/log/result polling·heartbeat·자동recall을 중단했다. 등록 프로그램만 자연 진행하며 결과 회수는 사용자 호출 시 수행한다.

## 검증·미관측

정본13member/fullSHA 및 봉인CSV CRLF 보존, 별도 USER BS1 order/저장 override 검산, CPU{audit['tests']} PASS. 원500 중앞100,원272 panel 유지. 부모3CP는 prior fullSHA/W/M payload receipt + 현재stat 재사용이며 새 payload검사라고 쓰지 않는다. 재전송0. Raw/model/tensor/prompt/stdout Git0.

CPU toy/import/회귀는 actual pinned Llama PASS가 아니다. 대표 joint B1 accept/정상reject→B2연결, native history5, actual hook/materialization/snapshot reconstruction, scientific900attempt/36저장은 모두 **NOT_OBSERVED**다. 새 task의 정상거절도 offered분모에 남긴다. source/config/guard는 원 과학식 유지, 타task numerical waiver 상속0. exact_editor_resume=NOT_AVAILABLE.

Owner audit 및 별도 CPU reducer/fixtures 검산이며 독립 red agent 사용0. Markdown 상대링크/표열/source/memberSHA/raw-free는 검사했으며 실제 Markdown 렌더는 renderer미설치로 NOT_TESTED다.

## 고정 source / local evidence

- execution source: `{lock['source']}` / tree `{lock['source_tree']}`
- lock SHA256: `{receipts['execution.lock.json']['sha256']}`
- config SHA256: `{receipts['configuration.json']['sha256']}`
- archive SHA256: `{receipts['source.tar']['sha256']}`
- submission SHA256: `{receipts['submission.json']['sha256']}`
- release SHA256: `{receipts['release.json']['sha256']}`
- local attempt: `{p}`

[제출 감사/receipt index](../../../../{AUDIT}/submission-bs1-v1.json), [CPU·자원 preflight](user-bs1-freeze-ko.md). 실행source와 이번 publication source를 구분한다. 원input/source/raw/다른task는 보존했고 NO_BROADCAST_NOT_REQUIRED: same-host CP/model/P 재사용, 원격 대형전송0.

종료: monitoring_active=false, automatic_resume=false. 다음 사용자 recall 전 결과회수/자동수리/추가실험0.
'''
    markdown(repo/REPORT/'submission-bs1-v1-ko.md',report)
    markdown(repo/'messages/server-heads/server4/2026-09-29-joint-multilayer-bs10.md',
        '# GH 인계 — USER BS1 변경·전량 제출\n\n'+OVERRIDE_NONCE+'\n\n'+
        f"3CP×3방법9경로, L4–L8 유지, BS1×100,25/50/75/100저장(36snapshot). CPU{audit['tests']} PASS. 실제 array{sub['array_job']}_[0–8]%2/collector{sub['collector']} afterany 전량held검사/release. {obs['utc']} 자원대기 ReqNodeNotAvail/노드8중8할당; INITIAL_NOT_OBSERVED. monitoring_active=false/automatic_resume=false. 원단층STOP/타job변경0/새대형전송0.\n\n"+
        f"Source {lock['source']}, lock {receipts['execution.lock.json']['sha256']}. 상세 보고: `{REPORT}/submission-bs1-v1-ko.md`. 기존 ACK/미제출 history는 그대로 보존.\n")
    package=[repo/REPORT/'submission-bs1-v1-ko.md',repo/AUDIT/'submission-bs1-v1.json',repo/'tasks/status/joint-multilayer-bs10-20260929-v1/server4.json',repo/'messages/server-heads/server4/2026-09-29-joint-multilayer-bs10.md']
    save(repo/AUDIT/'submission-package-manifest.json',dict(execution_source=lock['source'],members=[dict(path=str(x.relative_to(repo)),bytes=x.stat().st_size,sha256=sha(x)) for x in package],
        actual_forward_or_snapshot='NOT_OBSERVED',NO_BROADCAST_NOT_REQUIRED=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();publish_submission(a.repo,a.attempt)
