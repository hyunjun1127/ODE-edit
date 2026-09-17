# L4-preserving repair: R-GD / R-QP

새로운 두 cold W0/M4 chain만 구현한다. 모델/seed/context/teacher는 cold7 봉인 입력을 재사용하며 native L4 fitting/finalizer는 read-only다. L8의 full-weight gradient/JVP와 작은 FP64 QP를 사용하고 L8 target/P8/M8은 사용하지 않는다. Official P/N은 selection.json 봉인 이후 observer에서만 접근한다.

최신 사용자 `checkpoint는 저장하지 말고 진행하라`를 우선한다. W4/W8/M4/RNG tensor checkpoint를 저장하지 않는다. RAM snapshot/rollback, commit/hash/received ledger, 평가 NLL, Q/response/기술 probe 기록은 유지한다. Q/target 진단 tensor는 continuation checkpoint가 아니며 **exact restart는 NOT_AVAILABLE**이다. 기존 산출물을 삭제하지 않는다.

## 실행과 검증 경계

1. `control initialize`: 정본 FULL_READ identity와 no-checkpoint override를 create-once 보존.
2. `CUDA_VISIBLE_DEVICES='' python -B -m project.run_scripts.l4_preserving_repair.checks`: 작은 CPU fixtures/import/syntax. 실제 Llama PASS가 아니다.
3. 검증 source commit 후 `control freeze`: source/archive/input/수치 규칙 봉인.
4. `operations technical --lock …`: cap2 admission → held owner/source/memory 점검 → 한 pilot release.
5. 실제 READY가 있어야 `operations science --lock …`: 두 fresh W0 R-GD/R-QP를 등록. 기술 pilot native100은 science 분모가 아니며 state를 main에 carry하지 않는다.

공통 pilot은 native100/solve1을 한 번 실행하고 WN/Q0/H00을 재사용하여 R-QP와 R-GD의 actual acceptance 경로를 검사한다. GPU 관측 전 numerical.py, geometry.py, qp.py의 기술 기준을 고정하며 실패를 성능 off 또는 PASS로 바꾸지 않는다. FD 두 scale·JVP/GN·materialization·repeat·history/rollback 검사가 미해결이면 READY가 없다. 과학 arm은 매batch 자체 native fit과 response를 새로 계산한다.

R-GD의 Current/Past gradient는 0이며 actual quality 검사만 유지한다. R-QP는 Base/Current/(존재시)Past gradient span을 사용한다. 선택은 최대6 endpoint 중 첫 acceptable, 실패마다 같은 WN 복원과 r/2이다. 정상 off도 이전 누적 L8을 유지하고 M4 finalizer1·received ledger100을 처리한다.

## 자원·보존·관측

각 job 1GPU/8CPU/60416M/exportNONE/Requeue0, aggregate cap2. Pilot wall12h는 실측 전 reserve이며 science wall은 pilot 실측 후 별도 lock한다. Disk checkpoint가 없으며 Q/response/target/평가와 기술 증거에 48GiB free reserve를 검사한다. 실제 사용량·시간과 계획값은 구분한다.

초기 정책은 INITIAL_GATE_ONLY_OR_PENDING_HANDOFF이다. PENDING이면 actual NOT_RUN과 등록/미등록 범위를 인계하고 polling하지 않는다. READY 확인 전 science 등록 완료라고 쓰지 않는다. Callback/자동 추가제출/다른 task 모니터링은 없다. Main 프로그램은 등록 시 자연히 B1–B10을 진행하며 초기 gate 이후 agent는 explicit user recall을 기다린다.

Raw prompt/teacher/Q/gradient/log/tensor는 local-only이며 Git에는 source와 compact receipt만 게시한다. NO_BROADCAST_NOT_REQUIRED.
