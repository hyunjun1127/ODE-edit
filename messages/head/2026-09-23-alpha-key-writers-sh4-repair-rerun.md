# GH → SH4: alpha-key writers 실패 점검·수리·재실행

원 task ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
Recall nonce ODEEDIT-GH-SH4-ALPHA-KEY-WRITERS-REPAIR-20260923-R1.
대상 SH4 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / /data/janghj/ODE-edit.
사용자: “odeedit_alpha_key_writers_s4 이거 fail 되엇으니 확인하고 rerun 올리라고 해”.

이번 명시 recall로 동일 E0–E4 task의 writers 실패 원인 확인과 필요한 수리·검증·
재제출을 승인한다. 기존 자율 위임을 유지하며 단계별 재승인을 기다리지 않는다.

1. 실제 submission/lock/receipt에서 exact writers job ID·owner·source·argv를 먼저
   결속한다. 동일 이름의 다른 attempt를 추측해 변경하지 않는다.
   한정 accounting/log/source로 최초 exception·signal·OOM/시간/IO/수치 오류를
   구분하고 후속 cleanup 오류와 분리한다. 원인 미확정은 그대로 기록한다.
2. 실패 전 완료된 cell/entry/native z/SHAM/평가·기술검증과 미실행 범위를 분리한다.
   Source/input/state/context/P/RNG identity와 저장자료 완결성을 확인해 재사용한다.
   Gate나 geometry 등 성공한 sibling은 이름만 보고 재실행하지 않는다.
   저장 없는 state를 resume했다고 주장하지 않는다. 재계산이 꼭 필요한 범위만
   이유와 비용을 기록하고 수행한다.
3. 원 실패source/raw/log/cost를 보존하고 새 immutable attempt/source/lock에서
   최소 기술수리를 구현한다. 좁은 CPU/실제검사로 영향받는 경로를 검증한다.
   원 BASE_ALPHAEDIT 과학식/precision/threshold/관측집합/고정대조는 바꾸지 않는다.
   실패를 PASS로 덮거나 성능 좋은 cell만 재선택하지 않는다.
4. 같은 submission의 정확한 실패 writer와 연결된 막힌 pending/reducer에 한해
   필요시 cancel/새 dependency로 재등록한다. 불필요 중복등록은 막고 다른 task나
   유효하게 실행 중인 sibling은 유지한다. 기존 gate의 유효 evidence를 새 실행에
   결속하되 변경으로 무효화된 부분만 다시 검사한다.
5. Server4 project/taskcap2, 각1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0.
   신규 admission에 다른 allocation을 포함한다. GPU 부족이면 필요한 rerun과
   후속 자율작업을 모두 정상등록·검사·release 후 pending 상태로 인계하고 종료한다.
   Pending(None)을 GPU부족으로 단정하지 않는다.
6. GPU가 확보되면 수리 경로의 실제 초기 정상 동작과 자율 queue를 한정 확인한 뒤
   모니터링을 중단한다. 완료까지 기다리지 않는다. 이후 주기조회/heartbeat/자동
   agent 재개·무한 retry 금지, 프로그램은 자연 진행한다.

기본 새 full-state checkpoint 미저장과 명시 진단factor 보존 예외,
SEQ/ORDER/FUTURE 미승인, 원본12CP/source KEEP를 유지한다.
보고에는 exact failed/replacement job mapping, RCA, source diff,
reuse/rerun 표, 신규/이전 비용, actual initial 또는 pending 상태를 포함한다.
기존 허용 namespace 아래 writers-repair-r1 등 새 attempt/report/audit 사용.
전용 branch 소형 source/사실보고 nonforce push 허용, raw Git0.
첫 ACK 후 재실행까지 진행하고 최종 job ID/인계 receipt를 GH에 전달한다.
