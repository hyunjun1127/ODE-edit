# Alpha-key gate 실패 진단 인계

ACK nonce `ODEEDIT-GH-SH4-ALPHA-KEY-GATE-FAILURE-CHECK-20260923-R1`.
Exact52527/ownerjanghj/sourcea95876f의 최초 오류는 `technical.py:200 WRITER_UNEXPECTED_BOS: 0`이다. 원 native와 current tokenizer 설정 절차를 CPU로 재현한 actual12 sequence는 토큰/마스크 exact, 모두 BOS128000이 있었다. Generic fast tokenizer에서 add_bos_token=False attribute가 backend BOS template를 바꾸지 않는데 새 gate가 BOS 부재를 가정한 것이 직접 확인된 원인이다.

Parent44GPU초/exit1:0, G1 첫 forward 전 실패. Native100/SHAM/새history0, E1–E4 0/94. 후속52528/52529 Start=None/CANCELLED, gate 우회 없음. 52530은 afterany CPU reducer만 완료하여 TECHNICAL_FAILED_OR_PARTIAL로 정확히 기록했다. Secondary cleanup failure는 기록되지 않았으며 독립 restore 성공 receipt는 없다.

수리안은 원 tokenization을 유지한 native token/backend/mask/lookup exact binding으로 gate 가정을 교정하는 것이다. 이번에는 진단만 수행했고 runtime/threshold 수리·GPU·재제출0. CP12는 prior SHA+현재 stat 재사용, 삭제/전체재해시0.

[상세 RCA](../../../experiment-reports/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/report-ko.md).
전용 진단 branch 게시 후 STOP_AWAITING_USER, monitoring_active=false, automatic_resume=false. 원 실행/실패/입력 및 다른 task 보존.
