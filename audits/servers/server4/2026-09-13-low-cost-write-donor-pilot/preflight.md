# 사전 코드·입력 감사

사용자 승인 bounded blue2/red1 협업. Native AST에서 마지막 history loop만 분리하고 동일 loop로 endpoint append. 물리 L4→P0, L8→P4→local0. 원본 z/solve 식 변경0. b51 adapter 기준을 Git에서 materialize했고 58f50a performance_schema는 byte-exact 재사용했다. terminal_performance만 row-identity adapter 및 5500 지원 guard로 변경했다.

Red 최초 BLOCK은 core 예외에서 We/M/RNG 대신 W0만 정리하던 부분이었다. 공통 entry rollback 검증과 별도 실패 receipt 후 process W0 정리로 수정했다. 초기화 전 실패는 CORE_ENTRY_NOT_YET_AVAILABLE이며 준비된 상태만 finally에서 처리한다. Requires-grad/grad-none/model mode 복원을 추가했다.

비용 WARN: instrumented phase time에는 hash/capture가 포함된다. Materialization 시간도 포함하며 pure writer NOT_SEPARATED로 기록한다. 독립 CPU source/synthetic12 tests PASS, syntax PASS, memory audit160/0. GPU actual validity 미실행. 코드상 확인을 model-level parity로 부르지 않는다. Hooks/buffers 전체 exact 복원은 주장하지 않는다.

Source/setup/history/eval 분리, outcome-independent panel과 audit 미접근 guard, immutable output·source freeze를 적용한다. Audit와 suffix는 GH 후보 policy lock 전 미제출이다. Resource collision 검사는 제출 receipt에 별도 결속한다. 과학적 성능은 참고선 초과를 기록하며 자동 gate로 쓰지 않는다.
