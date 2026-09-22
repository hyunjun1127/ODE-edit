# GH → SH4: cap2 / GPU 부족 시 전량 pending 제출 후 종료

Instruction: ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
Override nonce: ODEEDIT-GH-SH4-ALPHA-KEY-CAP2-PENDING-20260923-R1.
대상 session01a04939-b5c7-7a03-ba2d-ef3343d62cfd / /data/janghj/ODE-edit.

사용자 최신 원문: “gpu남은게 없으면 pending으로 다 걸어놓고 task마치라고 해 gpu cap은 2다”.
이 지시는 기존 task1GPU 상한 및 GPU 부족 중 초기 gate까지 기다리는 규칙을 대체한다.

1. Server4 project GPU cap2, 본task 동시 GPU 상한2. 기존 다른 project allocation을
   포함해 총2 이내로 유지한다. 각 job 1GPU/8CPU/host≤60416MiB, exportNONE/Requeue0.
   독립 실행 가능 범위만 두 lane으로 구성하고 source/입력·output single-writer를 봉인한다.
   Slot을 채우기 위한 중복 native fit/중복 기술검증/추가 arm은 만들지 않는다.
2. CPU 구현·입력복사/SHA·저장·source/resource 봉인과 held inspection을 마친 뒤,
   승인 E0–E4 전체 작업을 upfront 등록·release한다. 101개 설계행 전체를 제출한다는
   뜻이 아니며 SEQ/ORDER/FUTURE는 여전히 FOLLOWUP_NOT_SUBMITTED다.
3. GPU 선행검사는 자율 runner 내부 또는 사전등록된 afterok dependency로 결속한다.
   실제 PASS 없이 dependent science를 실행하지 않는다. Gate 미실행을 이유로
   후속 등록을 agent의 미래 재호출에 남기지 말고, 전체 승인 실행이 미리 등록된
   프로그램/queue에서 자연 진행할 수 있게 구성한다. 별도 gate job이면 실패 시
   dependent 작업이 실행되지 않고 상태를 보존하도록 한다.
4. 제출 직전/직후 한정 자원·queue 확인으로 GPU 가용부족 및 pending을 확인하면,
   승인 범위의 전량 정상등록·held검사·release와 job ID/의존성/throttle를 보고하고
   MAIN_GPU_RESOURCE_PENDING_HANDOFF / MONITORING_PAUSED_AWAITING_USER로 종료한다.
   Dependency/array-limit으로 대기하는 나머지도 자율 queue에 포함됐음을 명시한다.
   초기 gate는 NOT_OBSERVED/NOT_RUN 그대로 두며 PASS나 전체 완료로 쓰지 않는다.
   단순 PENDING(None)/수동 hold를 GPU 부족으로 단정하지 않는다.
5. 이 pending 인계 후 자원대기·scheduler/log/result polling·heartbeat·callback·
   자동 scientific 추가제출을 하지 않는다. 이미 등록된 runner/reducer만 자연 진행.
   GPU가 확보되어 진행 중이면 기존 실제 초기 G0–G3 PASS 경계에서 모니터링 종료한다.
   Pending 인계 이후에는 초기 gate 관측을 위해 자동으로 agent를 재개하지 않는다.
6. 저장공간 부족/전송 불일치/구현 오류는 GPU 부족과 다른 blocker다. 이번 지시가
   checksum/공간/수치계약 면제나 기존자료 삭제/타job취소 권한은 아니다.

기존 원본/transfer 승인/BASE_ALPHAEDIT 조건/observer 분리/diagnostic 저장 예외/
신규 full-state checkpoint 미저장/후속 method 미승인은 모두 유지한다.
같은 task에서 적용하고 ACK·기존 등록 여부를 기존 receipt 기준으로 보고한다.
현재 지시는 승인 범위의 제출·인계까지 수행하라는 것이며 실험 완료 보고가 아니다.
