"""One explicitly invoked bounded initial check/publication; no loop or callback."""
import argparse
import datetime
import subprocess
from pathlib import Path
from .server2_entry import ROOT,NONCE
from .common import read,record,save,require,sha

def observe(attempt):
    p=ROOT/attempt;sub=read(p/'submission.json');lock=read(p/'execution.lock.json')
    require(read(p/'release.json')['all_9_registered'],'ALL_RELEASED')
    query=subprocess.check_output(['squeue','-h','-j',sub['array_job']+','+sub['collector'],'-o','%i|%j|%T|%b|%R'],text=True)
    evidence=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=query,array_job=sub['array_job'],collector=sub['collector'],
        source=lock['source'],lock=record(p/'execution.lock.json'),all_9_registered=True)
    if (p/'user-cap3.json').exists():
        evidence.update(effective_project_cap=3,effective_task_cap=3,cap_override=record(p/'user-cap3.json'))
    d=p/'output/B010-JOINT_STEP';initial=d/'initial-link.json'
    if initial.exists():
        link=read(initial);commit=read(link['first_commit']['path']);entry=read(link['next_entry']['path']);selection=read(d/'B001/selection.json')
        require(sha(link['first_commit']['path'])==link['first_commit']['sha256'] and sha(link['next_entry']['path'])==link['next_entry']['sha256'],'LINK_HASH')
        require(commit['after']==entry['state'],'LINK_STATE')
        require(selection['outcome'] in ('ACCEPTED','ZERO_WRITE_ALREADY_FEASIBLE','ATOMIC_REJECT_NO_FINITE_FEASIBLE_CANDIDATE'),'NORMAL_JOINT_OUTCOME')
        require(commit['history_appends']==(5 if selection['accepted'] else 0),'HISTORY_COUNT')
        if not selection['accepted']:
            require(all(commit['before'][k]==commit['after'][k] for k in ('weight','history','anchors','rng','context','accepted_ids')),'REJECT_ATOMICITY')
        evidence.update(status='MONITORING_PAUSED_AWAITING_USER',initial='JOINT_B1_TO_B2_OBSERVED',
            initial_link=record(initial),commit=record(d/'B001/commit.json'),next_entry=record(d/'B002/entry.json'),
            selection=record(d/'B001/selection.json'),outcome=selection['outcome'],accepted=selection['accepted'],
            proposals=selection['proposals'],trials=selection['trials'],full_guard_passes=selection['full_guard_passes'],
            method_seconds=selection['seconds'],geometry_seconds=selection['geometry_seconds'])
    elif (d/'first-error.json').exists():
        evidence.update(status='TECHNICAL_HOLD',initial='FAILED',failure=record(d/'first-error.json'))
    else:
        pending_only=query and all('|PENDING|' in x for x in query.splitlines())
        actual_resource=any(x in query for x in ('(Resources)','(ReqNodeNotAvail','(Priority)'))
        if pending_only and actual_resource:
            evidence.update(status='MONITORING_PAUSED_AWAITING_USER',initial='INITIAL_NOT_OBSERVED',
                reason='ALL_RELEASED_PENDING',node=subprocess.check_output(['scontrol','show','node','server2'],text=True))
        else:
            print(dict(status='INITIAL_NOT_YET_OBSERVED',queue=query));return
    evidence.update(monitoring_active=False,automatic_resume=False,scientific_completion='NOT_CLAIMED')
    save(p/'monitoring-pause.json',evidence)
    print(evidence)

def publish(repo,attempt):
    repo=Path(repo);p=ROOT/attempt;e=read(p/'monitoring-pause.json');lock=read(p/'execution.lock.json');sub=read(p/'submission.json')
    cfg=read(lock['configuration']['path']);cpu=read(ROOT/'cpu-audit-r1/owner-cpu-audit.json');port=read(ROOT/'preflight/port-receipt.json');transfer=read(ROOT/'inputs/transfer-receipt.json')
    audit=repo/'audits/servers/server2/joint-multilayer-bs1-20260929-v1'
    report=repo/'experiment-reports/servers/server2/joint-multilayer-bs1-20260929-v1/submission-ko.md'
    receipts={k:record(p/k) for k in ('execution.lock.json','configuration.json','source.tar','submission.json','release.json','admission.json','monitoring-pause.json')}
    save(audit/'submission.json',dict(**e,receipts=receipts,mapping=sub['mapping'],cpu_tests=cpu['tests'],cpu_actual_model=False,
        platform_tests=5,source_tree=lock['source_tree'],original_scientific_files_changed=False,
        checkpoints=port['checkpoint_files'],transfer_count=transfer['files'],transfer_bytes=transfer['bytes'],
        independent_red_agent=False,owner_audit=True,NO_BROADCAST_NOT_REQUIRED=True))
    save(repo/'transfers/verifications/2026-09-29-joint-multilayer-bs1-s4-to-s2/receipt.json',transfer)
    cap=e.get('effective_task_cap',2)
    save(repo/'tasks/status/joint-multilayer-bs1-migration-20260929-v1/server2.json',dict(**e,steps_per_trajectory=100,trajectories=9,
        gpu_cap=cap,batch_size=1,snapshot_steps=[25,50,75,100],scientific_attempts_planned=900))
    lines='\n'.join(f"|{r['job_id']}|{r['checkpoint']}|{r['arm']}|100|" for r in sub['mapping'])
    text=f'''# SH2 다층 joint BS1×100 이관 — 제출·초기 인계

Nonce `{NONCE}`. 상태 **{e['status']}**, 초기 관측 **{e['initial']}**. 전체 실험 완료 보고가 아니다.

## 전량 제출 및 범위

GPU array `{sub['array_job']}_[0–8]%{cap}`, CPU collector `{sub['collector']}` / `afterany:{sub['array_job']}`. 원 cap2에서 전량 held 검사/release 후 최신 사용자 "CAP 3으로 해"에 따라 해당 array throttle만3으로 변경했다. 원 source/config/lock은 역사cap2 그대로이며 별도 user-cap3 receipt로 실제 운영cap을 결속한다.

|Job|Parent|Arm|offered edits|
|---|---|---|---:|
{lines}

원 metadata 선택500의 앞100을 사용한다. fixed10k first100이 아니다. 원272 control/observer·다섯 L4–L8·FP32/eager·TF32 matmul=false/cuDNN=true·원 native L2=10/blue=false·JOINT 수식/예산/수치 guard 불변. 900과학attempts 및 native300targetfits/joint600solve는 계획이고 actual 완료수치가 아니다. 기술replay S4=0, S2 별도replay=0; 초기검사는 실제 과학 경로의 통합 기록이다.

## 초기 상태와 비용

관측시각 `{e['utc']}`. Outcome `{e.get('outcome','NOT_OBSERVED')}`, proposals `{e.get('proposals','NOT_OBSERVED')}`, trials `{e.get('trials','NOT_OBSERVED')}`, full guards `{e.get('full_guard_passes','NOT_OBSERVED')}`. Method elapsed `{e.get('method_seconds','NOT_MEASURED')}` sec, geometry `{e.get('geometry_seconds','NOT_MEASURED')}` sec; 중첩 timer이며 합산하지 않는다. 정상 reject는 과학 실패/기술오류와 구분한다. W/M/anchor/dual/RNG/context 연결 receipt는 submission audit에 결속했다. 미래 step/최종성능/저장완료는 미관측.

각1GPU/8CPU/60416MiB/exportNONE/Requeue0, array/task/project cap{cap}. Collector CPU8/24576MiB/GPU0. RTX A6000 48GiB; 사전peak GPU44GiB/host48GiB는 추정이며 실측보장이 아니다. CPU 토큰 검산 최대길이36, 실제 peak는 runner terminal에 기록하도록 되어 있다. Wall7일은 요청 상한이며 ETA/GPUh cap이 아니다. 다른 allocation 포함 admission했고 타job변경0.

## 저장 및 입력 보존

USER 명시 예외: offered25/50/75/100마다 actual full FP32 다섯 weights. 총36snapshot/180tensor/42,278,584,320B payload(39.375GiB)+metadata 계획, atomic save/reload와 모델출력 검증 유지. Exact editor resume NOT_AVAILABLE. 원 CP/raw는 삭제·이동하지 않았다.

S4 정확10job 취소 receipt SHA `2f934836622526bc5555ede5ea54e185323968a8f6f3cba507dd07f5c54f683d`, elapsed0/AllocTRES없음과 원 submission mapping 결속. 소형49member/2,096,027B fullSHA 수신, 대형전송0. S2 parent3CP 현재 file fullSHA 및 CPU W/M/context/RNG 검산, P file fullSHA 일치. Model10member는 S2 prior fullSHA+현재stat 재사용. S41801 broad closure와 actual native21closure를 구분했다.

## 검증 범위·재현

CPU29회귀+5S2routing/cancellation tests PASS. CPU는 actual Llama PASS가 아니다. S4 frozen runtime/solver/evaluator/원 native 파일 불변; S2 별도 entry/prepare/launch를 추가했다. Owner audit 및 독립 fixture/reducer 검사이며 별도 red agent PASS 주장은 없다. 타 task pause 유지. Raw/weights/prompts/teacher/fullstdout Git0. NO_BROADCAST_NOT_REQUIRED: 승인된 소형수신만, 새결과 원격방송0.

- 실행source `{lock['source']}` / tree `{lock['source_tree']}`
- lock SHA `{receipts['execution.lock.json']['sha256']}`
- config SHA `{receipts['configuration.json']['sha256']}`
- 출력 `{p}/output/`
- CPU: `{ROOT}/cpu-audit-r1/owner-cpu-audit.json`
- 재현 명령은 `{p}/branch.sbatch`, `{p}/collector.sbatch`에 봉인. 승인 없이 같은출력으로 재실행하지 않는다.
- 검사: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m unittest project.run_scripts.joint_multilayer_bs10.test_server2 -v`; 원29검사는 `server2_entry audit` 경로.

이 초기 인계 후 agent polling/terminal 대기/heartbeat/자동recall은 중지한다. 이미등록 runner/collector는 예정대로 진행하며, 상세 결과 검토는 사용자 recall 이후다.
'''
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x') as f:f.write(text)
    m=repo/'messages/server-heads/server2/2026-09-29-joint-multilayer-bs1-migration.md';m.parent.mkdir(parents=True,exist_ok=True)
    with m.open('x') as f:f.write(f"# SH2 초기 인계\n\n{NONCE}\n\n{e['status']} / {e['initial']}. Array{sub['array_job']}_[0–8]%{cap} + collector{sub['collector']} 전량release. source{lock['source']}. 보고 `{report.relative_to(repo)}` SHA{sha(report)}. 전체완료 아님; polling/자동recall0, 등록프로그램 자연진행, 타task변경0.\n")
    save(audit/'package-manifest.json',dict(execution_source=lock['source'],report=record(report),audit=record(audit/'submission.json'),
        metadata_only=True,publication_source='commit containing this manifest; distinct from execution_source'))
    print(report,sha(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--publish-repo');a=p.parse_args()
    if a.publish_repo:publish(a.publish_repo,a.attempt)
    else:observe(a.attempt)
