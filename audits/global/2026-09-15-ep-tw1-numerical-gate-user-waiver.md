# GH waiver: EP-TW-1 진단 gate skip

Instruction: ODEEDIT-S06-EP-TW1-DIAGNOSTIC-GATES-SKIP-RUN-SH4-V1.
사용자가 2026-09-15 반복적인 점검 gate를 제거하고 run 제출을 명시했다.
대상은 이번 EP-TW-1 fresh W0 first1000 재제출의 수치 진단 선행조건뿐이다.
이전 saved-episode FD/direct/ULP/self-KL/parity 필수 검증 및 해당 red numerical block은 SKIPPED_USER_DIRECTED로 대체한다. 이것은 PASS/derivative correctness/trajectory parity 승인이 아니다.
동일 실행 경로에서 disabled 검사 자체를 호출하지 않고 numerical_validation=NOT_ESTABLISHED를 기록한다.
method quality screen/normalizer/native/history/sample/evaluation은 유지한다.
실제 nonfinite, 잘못된 weight 적용, 무결성/자원/저장 실패는 blanket waiver 대상이 아니다.
별도 진단·smoke·추가 수리 job 없이 단일 scientific run 제출 권한을 부여했다.
기존47884/47942 실패 및 FD 미해결 사실은 그대로 보존한다.
GH duplicate GPU/raw audit 또는 새 하위 agent gate는 요구하지 않는다.
정확 실행·write·보고 경계는 messages/head/2026-09-15-sh4-ep-tw1-gate-skip-run.md를 따른다.
