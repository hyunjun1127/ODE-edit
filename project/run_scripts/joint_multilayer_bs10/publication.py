"""Compact raw-free owner report generation; never queries scheduler or scientific outputs."""
import argparse
from .common import *

REPORT='experiment-reports/servers/server4/joint-multilayer-bs10-20260929-v1'
AUDIT='audits/servers/server4/joint-multilayer-bs10-20260929-v1'

def markdown(path,text):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(text)

def preflight(repo,configuration,audit_path,stage='preflight'):
    repo=Path(repo);cfg=read(configuration);audit=read(audit_path);validate_execution(cfg)
    gh=record(ROOT/'gh-user-bs1-override-receipt.json')
    tests=dict(tests=audit['tests'],errors=audit['errors'],failures=audit['failures'],status=audit['status'],
        source=audit['source'],token_checks=audit['token_checks'],independent_red_agent=False,actual_gpu='NOT_RUN')
    save(repo/AUDIT/f'user-bs1-{stage}.json',dict(instruction_id=NONCE,override_nonce=OVERRIDE_NONCE,
        authority=cfg['authority'],envelope=cfg['envelope'],configuration=record(configuration),cpu_audit=record(audit_path),
        gh_override_delivery=gh,tests=tests,scope=dict(trajectories=9,requests=100,steps=100,layers=list(LAYERS),snapshots=36),
        checkpoints={k:{x:r[x] for x in ('path','bytes','sha256','inode','mtime_ns','verification')} for k,r in cfg['checkpoints'].items()},
        resources=cfg['resources'],actual_gpu='NOT_RUN',job_ids=[],monitoring='NOT_SUBMITTED',
        audit_type='owner CPU audit; no separate red agent',NO_BROADCAST_NOT_REQUIRED=True))
    text='''# 다층 BS1 × 100 — USER 변경 및 제출 전 기록

상태: IMPLEMENTING_NOT_SUBMITTED. 실제 GPU/model/fit/신규 weight snapshot/제출 job은 0이다.

사용자: “BATCH SIZE 1로 하고 각 CHECKPOINT에서 100개 EDIT하는 걸로 하자. 그리고 저장은 매 25STEP마다 하는 것으로 변경해. gh에게도 USER 명령으로 바꿧다고 해”.

기존 다층 task의 3CP × NATIVE/JOINT_STEP/JOINT_CUM = 9경로를 유지한다. 각 경로는 L4–L8 전체를 편집하며 BS1 × 100 step으로 변경한다. 원 봉인500요청의 앞100을 같은 순서로 쓰고 원272 control/observer panel은 보존한다. 철회된 단층 temporal-routing task는 STOP 상태이며 재개하지 않는다. 원 BS10 설계/CSV bytes는 변경하지 않고 실행 override만 별도로 결속했다. GH direct ACK 수신 완료; 추가 승인 대기 없음.

|항목|실행 설정|
|---|---|
|scientific attempts|900 = 9 × 100|
|NATIVE target fit|300 = 3 × 100|
|Joint solve / 최대proposal / backtrack / full guard|600 / 24,000 / 144,000 / 3,000|
|저장 시점|25 / 50 / 75 / 100 offered step|
|저장 예외|36 snapshot × 5 full FP32 [4096,14336] = 180 tensor|
|snapshot payload|42,278,584,320B = 39.375GiB, header 제외|
|exact editor resume|NOT_AVAILABLE; M/RNG 전체 bundle 미저장|
|기술 replay|경로당 최대1 허용, 현재 구현 integrated 검사만/별도 replay0|

## 결속 및 구현

정본13 member SHA/size 일치, CSV CRLF 보존. 3 parent CP의 기존 fullSHA/W/M payload 검산 + 현재 size/inode/mtime을 명시 재사용했다. 신규 CP fullrehash/전송0. 원 native 본체/compute_z/compute_ks dependency read-only, L2=10/blue=false/5층 divisor5..1/history각1 유지. Joint raw-P solve→thin basis→live downstream all-token hook, AL/STEP-CUM bound/dual/물리적 full guard/정상 reject rollback을 별도 namespace로 구현했다. BS1 basis rank≤1은 요청수 변경의 결과이지 단층 edit가 아니다.

CPU 검사는 소형 random model 및 fixtures이며 실제 pinned Llama PASS가 아니다. 수치·입력·restore 검사는 실제 branch의 통합 경로에서 수행한다. hook/physical의 사전 수치 기준은 row NLL abs2.5e-4 / margin abs5e-4, 원 feasibility1e-5는 별개다. 실제 raw 비교를 판정 전에 저장한다. 과거 다른 task의 record-only waiver는 적용하지 않았다.

## 자원·저장·진행

각 GPU job 1GPU/8CPU/60416MiB, array0–8%2, CPU afterany collector8CPU/24576MiB, exportNONE/Requeue0. 프로젝트/이task cap2. 예상 host48GiB(상한59GiB), GPU75GiB(실장 RTX PRO6000 약96GiB), output 여유192GiB를 요구한다. 이는 실측 peak가 아니다. FP32 모델 약30GiB의 GPU 상주, host CP/P mmap·history clone·bounded five-weight copies, FP64 solve는 층별 순차 해제한다. snapshot39.375GiB 외 teacher/capture/raw/atomic temp/I-O 여유를 포함한다.

Wall request7일은 실측 ETA가 아니다. 과학 native/joint 첫step 시간은 아직 NOT_MEASURED다. 성능 결과로 budget/순서를 바꾸지 않는다. 고정 launch 순서는 CP별 JOINT_STEP/NATIVE/JOINT_CUM으로 대표 joint 초기 연결을 먼저 관측할 수 있게 했다. 모든9경로가 독립적이며 각 부모를 복원한다. 준비/실행/observer/후보/geometry/backward 시간은 구분하되 중첩 timer 합산 금지.

9경로 전량 held inspection/release 후, 대표 joint B1 accept 또는 정상 reject→B2 상태연결을 확인하거나 실제 resource pending을 확인하면 agent monitoring/automatic resume을 중단한다. 나머지는 사전등록 프로그램이 자연 진행한다. 전체완료 리뷰는 다음 사용자 recall에서 수행한다. NO_BROADCAST_NOT_REQUIRED: S4 원자산 재사용, 원격 raw 전송0.

## 검토 수준

owner audit + 독립 CPU fixture/reducer 검산이며 별도 red agent 사용0. 실제 GPU/initial gate/terminal은 아직 NOT_OBSERVED. 공유 helper가 exact runs 경로를 지원하지 않으면 명시 envelope의 좁은 예외로 기록하며 helper PASS로 위장하지 않는다.
'''
    text+=f"\nCPU regression: {audit['tests']} tests / errors{audit['errors']} / failures{audit['failures']}. 근거는 [{stage} audit](../../../../audits/servers/server4/joint-multilayer-bs10-20260929-v1/user-bs1-{stage}.json).\n"
    markdown(repo/REPORT/f'user-bs1-{stage}-ko.md',text)
    if stage!='preflight':return
    markdown(repo/'messages/acks/server4/2026-09-29-joint-multilayer-bs10.md',
        f'# 수신 및 USER 변경 ACK\n\n{NONCE}\n\nUSER override: {OVERRIDE_NONCE}. 다층9경로 유지, 각 BS1×100,25/50/75/100 저장. GH 전달·ACK 완료. 원 단층 task STOP 유지. CPU{audit["tests"]} PASS, GPU검증 미수행, job_ids=[] / IMPLEMENTING_NOT_SUBMITTED.\n')
    markdown(repo/'plans/updates/server4/joint-multilayer-bs10-20260929-v1/user-bs1.md',
        '# USER 변경 실행 순서\n\n정본 불변→BS1/100/save25 override→CPU 검산→immutable source/config/lock→cap2 held array9경로+CPU afterany→release→대표 joint 초기연결 또는 실제resource pending 인계→monitoring STOP. 새 단층 실행/타job 변경/추가 chain0.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--configuration',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--stage',default='preflight',choices=('preflight','freeze'))
    a=p.parse_args();preflight(a.repo,a.configuration,a.audit,a.stage)
