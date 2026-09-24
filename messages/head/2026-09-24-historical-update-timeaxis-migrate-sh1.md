# GH → SH1: historical-update-timeaxis server1 이관·실행

Instruction / nonce: ODEEDIT-GH-SH1-HISTORICAL-TIMEAXIS-MIGRATE-20260924-R1.
Parent: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1.
사용자: “server4에서 pending 중인 job들 server1에서 하는 것으로 하자. SH1에게 task 진행시켜.”

## 소유권과 범위

SH1(session01a04939-f93a-7b50-bca0-65438eab2062, devbox,
/mnt/raid5/janghj/ODE-edit, hyunjun1127/ODE-edit)가
이관·입력확보·port·취소·제출의 단독 실행 owner다. 전용 clean child branch
codex/server1-historical-update-timeaxis-migration-20260924-v1에서 수행한다.
SH4는 원 source/자산/receipt 보존과 필요한 exact 경로 회신만 담당하고 새 submit0.
SH4 task를 무단 재개하거나 무관 job을 취소하지 않는다.
2026-09-24 app metadata/direct resume에서 현재 SH1 CWD는 위 root로 확인했다.
오래된 registry의 29e4 worktree 경로는 이번 task boundary로 사용하지 않는다.

이번 대상은 현재 대화의 repaired historical task 53182(BASE_ALPHAEDIT),
53183(BASE_MEMIT), 53184(CPU afterany:53182:53183) 세 개다.
GH의 이관 전 단발 확인에서 owner janghj, server4, 모두 PENDING/elapsed0/
AllocTRES(null), GPU reason ReqNodeNotAvail, collector Dependency였다.
이는 새 SH1 취소 시점의 상태 보장이 아니다.

SH1은 exact owner/name/argv/source/lock/dependency를 다시 결속한 뒤 **PENDING인
collector53184를 먼저, GPU53182/53183을 다음에** state-filtered cancel한다.
취소 accounting 및 중복 실행 없음이 확인되어야 대체 scientific job을 release한다.
취소 전에 RUNNING으로 바뀌면 그 job은 이번 pending-only 취소 승인 밖이다.
그 상태를 보고하고 해당 family 중복 제출을 차단하라. 실행중 job을 임의 취소0.
이미 terminal이면 상태/비용/재사용 가능 범위를 분리하고 불필요 중복 실행0.
원 source/raw/입력/24CP/과거실패·취소 기록은 삭제/덮어쓰기0.

## 정본·수리 코드 확보

기존 messages/head/2026-09-24-historical-update-timeaxis-sh4.md와 정본 설계/
contract/DAG/ledger/cells/원 evaluator를 정독하고 같은 과학범위를 상속한다.
GH local 원문 root:
/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-24-historical-update-timeaxis-dispatch-v1/.
gh-sh4-instruction-ko.md SHA3c26436f1b6add8611a430ff368eb89c457ca7b2776e3c859c626d04142c828c.
dispatch-input-manifest.json SHA9ca079f1e8417d1f3ebdcb72ee324d642be297c15a997ecd75f029f10ac4388c.
design-package.tar.gz SHAb0528de91deae69c3a213df288fa3bdb57a183e2f5bd157fb3c42bea3380be3c.
실제 local 원본을 read-only 재사용하고 안전하게 수신29member SHA/size 결속.

origin/codex/server4-historical-update-timeaxis-20260924-v1의 수리 execution
730a4a9768e5650e01fd9afdc4e0f7895c86ea92/tree48b61f229bedb75ec025a8371c160b6a1aad3dab를 확보한다.
보고 tip2adbab7eaff98652229c93cc677aadc1015bbb7a와 실행source를 구분한다.
S4 원 attempt:
/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/attempt-r2-routing/.
execution.lock.json SHA54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699.
archive SHAc14df932c221c503302035efc7a4825fb2b4ab74614cbf25462aa1ce60a07948.
ACTUAL+force_removal 선분기 수리와 collector report/inventory 후 atomic COMPLETED
수리를 모두 유지한다. 과거 original6ef71ed2/실패·미관측 기록을 수정0.
CPU22 PASS는 actual GPU PASS가 아니다. 경로/launcher/env binding만 port하고
새 실행 source/archive/lock/입력 경로 map을 별도 봉인한다.

## 입력과 전송 승인

사용자 이관 요청에 따른 입력 확보를 승인한다. SH1 sole destination writer.
새 localroot /mnt/raid5/janghj/ODE-edit/local/historical-update-timeaxis/20260924-s1-v1/.
먼저 server1의 기존 exact 동일 model/tokenizer/fixed10k/context/24CP를 찾고
hash/size/조건이 맞으면 read-only REUSE. 경로만 다르다는 이유로 다시 복사0.
없는 입력만 기존 transfers/approvals/2026-09-24-historical-update-timeaxis-sh4.json의
24CP exact path/SHA/size allowlist 또는 S4 execution.lock에 실제 연결된 동일
asset 파일에서 새 root/inputs/로 선택 pull할 수 있다. S2 archive 또는 S4 기존
실물 중 동일 bytes를 사용, 두 source를 동시에 중복 수신하지 않는다.
S4 attempt-r2-routing의 소형 lock/archive/launchers/submission/release/T0 receipt와
입력 path-map도 exact 목록을 먼저 기록한 뒤 필요한 것만 받는다.
전송 전 source/destination/SHA/size/필요량·여유를 manifest로 잠그고 수신 후 검산.
sourceKEEP, --delete/덮어쓰기/범용recursive tree/model cache 전체 복사 금지.
새 입력 또는 identity 불일치가 승인 manifest로 해소되지 않으면 구체적 보고.
S4/SH2 무관 science/monitoring 재개0. 접속은 검증된 SSH 경로만, hostkey 우회0.

## 실행·수치·자원

기존 BASE_ALPHAEDIT42657/BASE_MEMIT42658의 whole-five-weight interval U,
T0→T1→T2P(고정300/32cells)→T2F156cells→T3A/T3B16cells→T4 그대로.
실제 FP64 제거/FP32 한 번 materialization, full forward, MB16, 평가 토큰화,
FP32/eager/autocastoff/TF32matmuloff/cudnnTF32true, tolerance/패널/분모/캐시 identity
유지. A6000 등 hardware 차이를 provenance에 기록하고 actual T1을 수행한다.
새 baseline/z fitting/write/history/CP0, save_checkpoints=false.
기존 input CP read/copy는 신규 checkpoint 저장이 아니다.

Server1 project/task cap2. 각 family1GPU/8CPU, 기본60416MiB portable memory 유지.
정확 필요량과 server1 local cap을 결속하면 명시 host-memory를 조정할 수 있으나
server1 repo hardceiling183296MiB/GPU 절대 초과0. CPU collector8CPU/24576MiB/GPU0.
다른 active+admitted project capacity 포함2이하, 기존 job 자원·throttle 변경0.
exportNONE/Requeue0/devbox, 실제 partition/QOS/wall 상한을 검증한다.
자원 부족이면 cap-safe dependency/hold/queue로 정상 등록하고 진행 관찰하지 않는다.
두 persistent worker의 matching atomic PASS join과 afterany CPU collector를
새 job IDs로 함께 등록한다. 과학 음성 결과를 gate FAIL로 만들지 않는다.
OOM을 이유로 precision/패널/수학 변경0. 새 mutable 경로가 S4를 가리키지 않게 검사.

## 종료 경계: 제출 후 monitoring 금지

최신 사용자의 job monitoring 중지 방침을 그대로 유지한다.
허용: 구현/이미 확인된 기술오류 최소수리/CPU regression/자산/admission 확인,
exact pending 취소 확인, held source/owner/argv/resource/dependency 검사와 release.
전체 scope 정상 등록/release 결과를 받은 뒤 scheduler/log/result/initial/terminal
조회·주기 polling·자동재개·heartbeat·후속 scientific submit을 하지 않는다.
INITIAL_NOT_OBSERVED로 제출 인계하고 STOP. 프로그램 내부 T1→T4/collector는 자연 진행.
T4 완료 회수·상세 CPU review는 사용자 recall에서 수행한다. 앞선 end-to-end agent
관찰 지시를 이번 이관의 종료 조건으로 상속하지 않는다.

## 쓰기·게시 승인

source project/run_scripts/historical_update_timeaxis/의 task-local port/tests만.
localroot 위 경로. tracked 아래 exact 범위:
- experiment-reports/servers/server1/historical-update-timeaxis-20260924-v1/
- audits/servers/server1/historical-update-timeaxis-20260924-v1/
- plans/updates/server1/historical-update-timeaxis-20260924-v1/
- messages/acks/server1/2026-09-24-historical-update-timeaxis-migration.md
- messages/server-heads/server1/2026-09-24-historical-update-timeaxis-migration.md
- tasks/status/historical-update-timeaxis-20260924-v1/server1.json
- runs/odeedit_historical_update_timeaxis_s1_20260924/
- transfers/verifications/2026-09-24-historical-update-timeaxis-sh1/
ownscope 검산 후 nonforce branch/main 게시 허용, shareddirty/native/env/useridentity 보존.
bounded preflight source/수학불변/취소exact/자원/소유/DAG/경로 검토와 CPU회귀 필요.
raw/tensor/fullstdout/prompt Git0, NO_BROADCAST_NOT_REQUIRED 사유 기록.
수신 ACK와 구현/취소/등록/release/실제 PASS를 구분해 보고하며 단계별 GH 재승인 대기0.
첫 ACK에는 nonce/이관owner/본실험 미제출 상태를 명시하라.

## SH4 통지

이 envelope의 SH4 적용은 인계 보존·필요 exact path 회신에 한한다. cancel sole
owner는 SH1이다. SH4는53182/53183/53184를 별도취소/수리/release/resubmit하지 말고,
원 input/source/raw를 보존한 채 monitoring 중지를 유지한다.
