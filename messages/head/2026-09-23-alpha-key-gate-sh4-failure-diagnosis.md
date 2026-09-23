# GH → SH4: odeedit_alpha_key_gate_s4 실패 점검

원 task: ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
진단 nonce: ODEEDIT-GH-SH4-ALPHA-KEY-GATE-FAILURE-CHECK-20260923-R1.
대상 SH4 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / /data/janghj/ODE-edit.
사용자: “odeedit_alpha_key_gate_s4 fail되었으니 확인하라고 해”.

이번 명시 recall은 지정 gate의 실패 원인 확인과 사실 보고다.
수리·재제출을 새로 승인하는 요청이 아니며, 이 진단 turn에서는
runtime/config/threshold 변경·새GPU/model forward·재제출·자동retry를 하지 않는다.
기존 scope의 실행 계획과 원 실패 증거는 보존한다.

1. 기존 submission/lock/receipt에서 exact job ID, owner, source/tree/archive,
   full argv, output/log path를 먼저 결속한다. 이름만 같은 타 job은 포함하지 않는다.
2. 한정 accounting 및 해당 log/실패 receipt를 읽어 exit/signal, 첫 exception과
   source line, OOM/시간제한/IO/환경/serialization/수치계약 실패를 구분한다.
   후속 cleanup exception을 최초 원인으로 오인하지 않는다. 미확정은 명시한다.
3. G0/G1/G2/G3 중 실제 완료/실패/미실행, W50→B51 native100/SHAM 여부,
   state restore 증거, 유효 재사용 가능한 artifact와 비용을 분리한다.
   성능 실패·numerical mismatch·단순 기술 오류를 섞지 않는다.
4. 동일 submission manifest의 exact 후속 job/array/reducer만 한정 확인하여
   afterok/dependency 차단 여부와 gate 실패 이후 실행 유무를 보고한다.
   다른 job은 조회/변경하지 않는다. 이번 진단에 cancel/hold/release 권한은 없다.
   의존성 우회 또는 예상 밖 실행이 발견되면 즉시 사실을 GH에 알린다.
5. 필요한 최소 CPU/source 재현은 허용하되 별도 진단 scratch에 기록한다.
   원 실행 source/raw/transfer receipt/12CP를 덮어쓰거나 삭제하지 않는다.
   원본12CP 전체 재해시/재전송/모델 재평가를 진단의 관성적 선행조건으로 삼지 않는다.
6. 최초 RCA·영향 범위·최소 수리 제안·수리 시 재실행 필요/재사용 가능 부분을
   사실 근거와 함께 보고한다. 제안만 남기며 이번 turn에서 구현하지 않는다.
   allocated GPU-sec는 parent 기준, component/step 중복 가산하지 않는다.

허용 새 문서/진단 경로:
- audits/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/
- experiment-reports/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/
- local/alpha-key-concentration-causal/20260923-r1/failure-diagnosis-r1/
- messages/server-heads/server4/alpha-key-causal-20260923-r1-failure.md
- 해당 task의 소형 status/receipt.

전용 branch에서 소형 진단 보고/재현 자료 push 허용, 원 execution branch 불변.
현재 project/taskcap2는 유지되지만 이번 진단 GPU 신규 할당0.
보고서를 GH에 direct 인계한 뒤 STOP_AWAITING_USER, monitoring_active=false.
주기 polling·완료대기·자동재개·다른 paused task 재개는 금지한다.
간단 nonce ACK 후 점검을 진행하고, gate 실패를 PASS로 바꾸지 않는다.
