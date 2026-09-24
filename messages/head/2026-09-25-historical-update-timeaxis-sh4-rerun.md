# GH → SH4: historical-update-timeaxis 확인·최소수리·rerun

Instruction / nonce: ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RERUN-20260925-R1.
Parent: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1.
사용자 원문: “server4 실험들 rerun 올리라고 해”.
대상 SH4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd,
server4 / /data/janghj/ODE-edit / hyunjun1127/ODE-edit.

## 승인과 대상

현재 대화의 historical-update-timeaxis 두 family(BASE_ALPHAEDIT/BASE_MEMIT)와
CPU collector를 server4에서 재개·재제출하라. SH1 이관은 철회 상태로 유지한다.
이번 지시는 계획만이 아니라 scope 내 실제 상태 확인·RCA·최소수리·rerun 제출 승인이다.
과거 모든 server4 실험을 일괄 재실행하는 승인이 아니다.
SH4 sole execution owner, 추가 GH ACK/단계별 승인을 선행조건으로 두지 않는다.

기존 정확 job53182/53183/53184, source730a4a9768e5650e01fd9afdc4e0f7895c86ea92,
tree48b61f229bedb75ec025a8371c160b6a1aad3dab,
attempt /data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/attempt-r2-routing/,
lock SHA54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699.
과거 관측은 PENDING이며 현재 실패/완료/실행 상태는 아직 확인하지 않았다.

정확 owner/name/source/argv/dependency와 accounting 및 필요한 bounded 실패자료를
먼저 확인한다. FAILED/CANCELLED/TIMEOUT이면 최초 오류와 실행된 단계/원비용을
보존하여 기술오류 최소수리 후 새 immutable attempt로 제출한다.
아직 PENDING이면 사용자 rerun 지시에 따라 교체 가능하되, collector→GPU 순서로
exact pending만 취소하고 취소 확인 뒤 새 job release. 무관 job cancel/hold0.
RUNNING이면 중복 제출/임의 강제취소하지 말고 상태를 보고한다.
이미 유효하게 완료된 family/row는 identity 검산 후 재사용하며, 전체 완료라면
불필요 중복 rerun 대신 완료 사실과 재실행 불필요 근거를 보고한다.

## 과학·입력·수리 경계

기존 messages/head/2026-09-24-historical-update-timeaxis-sh4.md의 전체 과학범위와
write 범위를 상속한다. main e21d0c0e에 게시된 설계29파일은 기존 봉인bytes 동일.
PUBLICATION.md와 motivation §1.1은 설명 추가일 뿐 panel/threshold/DAG 변경이 아니다.
원24CP/모델/dataset/토큰/evaluator/완료 T0를 identity에 맞게 재사용하고
대형 원본 재전송/삭제/덮어쓰기0. source6ef71ed2와 수리730a4a97의 이력 구분.
ACTUAL+force_removal 선분기 및 report/inventory 후 atomic COMPLETED 수리를 유지한다.

T0→T1→T2P→T2F→T3A/T3B→T4, whole-five-weight U, FP64 제거 후 FP32
한 번 materialization/full forward/MB16/고정 문항·분모·수치 계약 그대로다.
CPU 회귀검사를 actual GPU PASS로 바꾸지 않는다. 원인 확정된 구현/직렬화/경로/
자원 연결 오류는 bounded regression과 함께 최소수리 허용. 수치 기준·precision/
과학식·표본·gate를 결과에 맞춰 변경하지 않는다. numerical contract 변경이 필요하면
typed blockage를 보고하며 임의 반복-to-PASS0. 과학 음성 결과는 정상 완료다.
완료 row 재사용은 source/input/state/token/layout의 유효성이 확인된 범위만.
새 source와 입력 lock이 다른데 원 PASS 파일 존재만으로 downstream 통과0.

## 제출·자원·종료

server4 project/taskcap2(다른 active/admitted capacity 포함), family별1GPU/8CPU/
60416MiB 이하, exportNONE/Requeue0/server4. CPU collector8CPU/24576MiB/GPU0.
정확 resource/storage preflight와 wall/partition 상한 검사, held owner/source/fullargv/
resource/dependency 검사 후 release. 두 worker와 afterany collector를 upfront 등록하고
matching atomic PASS 내부 join 및 실패 수집을 유지한다. 자원 부족이면 cap-safe
pending/dependency로 걸어두고 제출 인계하라. 중복 job으로 GPU slot을 채우지 않는다.
save_checkpoints=false; 새 native fitting/edit/history/full endpoint CP0.

이번 recall은 제출에 필요한 상태·오류 확인을 허용하지만 **제출 후 job 진행 모니터링을
허용하지 않는다**. 정상 release 확인 뒤 scheduler/log/result/initial/terminal조회0,
heartbeat/자동 recall/daemon/후속 agent 재제출0. INITIAL_NOT_OBSERVED로
compact source/lock/new job mapping·원실패비용/재사용을 보고하고 STOP.
등록된 runner/reducer만 자연 진행한다. 이전 초기 gate 대기 또는 T4까지 agent
관찰 지시를 다시 적용하지 않는다. 상세 결과 회수는 사용자 recall 때 수행한다.

## 파일·게시

전용 clean branch codex/server4-historical-update-timeaxis-rerun-20260925-v1 권장.
project/run_scripts/historical_update_timeaxis/ 안의 최소수리/tests만 허용.
새 local attempt는 기존 local/historical-update-timeaxis/20260924-v1/ 아래 create-once.
기존 server4 report/audit/plans-update/status/runs/ACK 범위에 rerun-20260925-r1
기록을 추가하고 원문/과거 raw/실패/receipt는 보존한다.
ownscope source+raw-free 인계 보고를 검산 후 nonforce branch/main 게시 승인.
공유 dirty/native/env/identity 변경0, raw/tensor/prompt/fullstdout Git0,
NO_BROADCAST_NOT_REQUIRED 근거 명시. 본 task 밖 실험·SH1/SH2/SH3 재개0.
첫 ACK에 nonce/범위/제출 전 상태를, 제출 후 실제 job IDs/원 job 처리/재사용/
monitoring_active=false/automatic_resume=false를 분리해 회신하라.
