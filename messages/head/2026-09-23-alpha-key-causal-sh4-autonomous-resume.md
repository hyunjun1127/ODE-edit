# GH → SH4: E0–E4 자율 수리·재제출 전권 위임

원 task: ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
새 nonce: ODEEDIT-GH-SH4-ALPHA-KEY-AUTONOMOUS-RESUME-20260923-R1.
대상 SH4 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / /data/janghj/ODE-edit.
사용자 최신 지시: “task 재개시켜. 전권 맡길테니 알아서 하라고 해”.

## 위임

직전 diagnosis-only 및 수리/재제출 금지는 이번 사용자 지시로 해제한다.
SH4가 승인된 원본 BASE_ALPHAEDIT E0–E4 task의 구현·원인 확인·기술 수리·검증·
입력 재사용·자원계획·Slurm 재제출·한정 초기 점검을 책임지고 자율 진행한다.
매 단계 GH/사용자 승인, 중복 raw 감사, 개별 기술수리 허가를 기다리지 않는다.
이미 작성한 진단/최소재현/유효 산출물을 재사용하고 동일 진단을 반복하지 않는다.

실패 exact job/source/argv/first-error를 결속해 원 실행을 보존한 뒤
새 immutable source/attempt/lock으로 필요한 부분만 수리한다.
Parser/import/serialization/shape/device/메모리/자원·경로·runner 연결 오류 등
실행 구현의 수정과 동등 계산의 microbatch/streaming 조정은 scope 내 자율 수행한다.
해당 변경을 확인하는 좁은 CPU/실제 모델 검사 및 필요한 GPU 재실행도 승인한다.
타당하게 완료된 native z/observer/transfer evidence는 exact identity 확인 후 재사용한다.
과거 실패비용과 신규 할당비용을 분리하고 중복 합산하지 않는다.

같은 task의 실패 gate에 묶인 후속 pending/잘못된 attempt/중복 작업은
submission receipt와 owner/job/source/dependency를 먼저 확인해 정확한 allowlist로
cancel/hold/대체 dependency 재등록할 수 있다. 다른 task/job은 변경하지 않는다.
취소는 필요할 때만 수행하고 원 artifact/로그/실패 receipt는 삭제하지 않는다.
이미 유효하게 실행 중인 동일 작업을 무조건 재시작하지 않는다.

## 유지되는 범위

- Project/task cap2, 각1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0.
  다른 project allocation 포함 admission을 지키고 독립 작업만 병행한다.
- E0–E4만 구현·제출한다. 원 baseline/입력/observer 분리/고정 대조/400 native
  target 계획/branch 독립 복원/원 precision·수치계약은 유지한다.
  실패를 PASS로 덮거나 threshold·dataset·arm을 결과에 맞춰 바꾸지 않는다.
- 기존12CP/source/raw 보존, transfer 승인 경계 및 기본 신규 full-state CP 미저장.
  진단 K/R/Δ factor 등 이미 명시한 저장 예외만 유지한다.
- SEQ/ORDER/FUTURE·method 선택·full10k는 FOLLOWUP_NOT_SUBMITTED.
  전권은 현재 task를 수행할 권한이며 다른 연구범위 확장이나 대규모 삭제 권한이 아니다.
- 기술 문제는 자율 해결하되 과학 계약 변경이 필수이거나 원본 손상/새권한 문제가
  남으면 증거를 남겨 인계한다. 같은 실패를 무한 자동 재제출하지 않는다.

## 종료와 보고

승인 범위 전체를 자율 runner/사전등록 dependency로 정상등록·held검사·release한다.
GPU 부족으로 pending이면 exact jobmapping/상태근거/의존성/cap을 남기고
MAIN_GPU_RESOURCE_PENDING_HANDOFF로 종료한다. 초기 gate는 미관측 그대로 기록.
자원 확보되어 진행하면 실제 G0–G3/W50→B51 native100+SHAM 및 자율 queue/ID가
확인되는 초기 gate까지 한정 점검한 뒤 INITIAL_GATE_PASS_MONITORING_STOPPED로 종료.
두 인계 이후 scheduler/log/result polling·heartbeat·자동 agent 재개는 하지 않는다.
이미 제출한 runner/reducer는 자연 진행한다. 후기 실패는 파일에 남기며
인계 후 agent 자동수리/추가 scientific submission은 사용자 recall 전 하지 않는다.

기존 envelope 허용 source/report/audit/run/status 경로에서 수행한다.
전용 branch에 검토한 source/소형 사실 보고 nonforce push 허용, raw Git0.
GH main 통합과 이후 연구 해석은 별도이며 SH4 실행의 선행 승인으로 삼지 않는다.
첫 ACK에 현재 진단 재사용 여부와 재개 범위를 남기고 작업을 이어간다.
