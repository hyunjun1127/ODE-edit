# SH4 완료 baseline 및 zsRE 지표 점검

- nonce: USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1.
- accepted turn: 01a122d0-e21c-74f2-8c25-5ee01998b4de.
- 정본7dacd3fb 전체 읽음. 관측2026-10-10 07:38:45 KST.
- **본표 적격 완료 baseline0, own 완료 zsRE0, 변경할 수치0.**

정확 18개 원소유 baseline의 accounting을 단발 확인했다. Qwen CF61783은 FAILED, 나머지 CF61785/87/89/91/93 및 zsRE61755/57/59/61/63/65는 CANCELLED다. 기존 config 파일SHA는 이관 영수증과 동일하고 local run output은 모두 없다. server2 이관된 후속 실행은 SH2가 보고하며 SH4 완료로 중복계상하지 않는다.

옛 Llama CF60917–60922는 CANCELLED/elapsed0/startNone다. 이 18개에서 W20/2000 완료를 주장할 근거가 없으므로 점수나 token 분모를 생성하지 않았다. 현재 알려진 task inventory에 한정한 검산이며 전체 파일시스템 전수조사는 아니다.

zsRE 검산 상태는 `NOT_APPLICABLE_NO_COMPLETED_LOCAL_ZSRE_BASELINE`이다. 실제 완료 frozen evaluator/query/profile, 전체stream CPU query parity 및 저장 correctness/ID 독립 재집계의 대상 자체가 없다. 따라서 public-query/GPU/native parity PASS라고 표시하지 않는다. Eff/Gen/Loc는 target token correctness를 요청별 평균 후 요청 간 평균×100하며 Loc는 loc_ans다. token micro·strict prompt·W0 agreement로 대체하지 않는 계약을 유지한다.

OURS/PRICE/heldout500/tuning을 본표 baseline으로 승격하지 않았다. 기존 historical 예외/CF값/미측정FluCon도 변경하지 않았다. 실제 적격 Flu/Con이 생길 때만 미반올림 bits/cosine raw×100 후 half-up2 표시하며 이번에는 적용 대상이 없다.

[JSON](../../../../audits/servers/server4/baseline-completed-zsre-audit-20261010/table-rows.json), [CSV](../../../../audits/servers/server4/baseline-completed-zsre-audit-20261010/table-rows.csv), [CPU audit](../../../../audits/servers/server4/baseline-completed-zsre-audit-20261010/audit.py).

GPU0/신규job0/jobmutation0/CP load·삭제·전송0/raw·W&B변경0/README직접변경0. GH sole README 통합. NO_BROADCAST_NOT_REQUIRED: compact metadata만 Git, 기존 raw/CP KEEP.
