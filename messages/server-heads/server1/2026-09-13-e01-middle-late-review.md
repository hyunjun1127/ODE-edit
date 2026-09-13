# E01 Middle/Late full-seen 상세 리뷰 완료

instruction_id: ODEEDIT-S06-E01-MIDDLE-LATE-FULLSEEN-DETAILED-REVIEW-SH1-V1

상태: `REVIEW_COMPLETE`, 이번 완료 리뷰만 main publication 대상. 전체 E01 완료가 아니다.

- 보고서: `experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/completed-middle-late-review-v1/diagnostic-report-ko.md`
- 보고서 SHA256: `6ca75653378015b4f9e019a73f6ba31cc51d251e1d124827758de13cb035c8c1`.
- analysis-manifest SHA256: `bf975b9d8bce02002bea2d55843016c3d9fdd0e1f83cd7714037146e781861b7`.
- rooted-receipt SHA256: `b29b719eae31e2a47ad89cb7e3ad23dbdf56b856a1dbeaf8aab694d6d298d9b8`.
- receipt identity: `0f0f9d0630eb33c2acefdb8019c28efea3afe2e1685cd392fdcac3449308bc1e`.
- 21-member root: `3ac18e8057b142a886c8d747509fa8e7b8d2668bf2dd474cd7db3495e24576a8`.
- CPU analysis source: `089dfe7eb807a904752272cb3d0697d06b94eca8`, tree `df5ae9f68cc2b160a54c3e709934c20c44ec563b`. Runtime native b51dcf5와 observation58f50a는 source-manifest에 별도 결속한다.

단발 scheduler 확인: 46439/46440 COMPLETED0:0, 2806/4663 GPU-sec. 기존 native45914/45915의 완전 W/M·Current100을 재사용했고 추가 native/z/history0이다. 125 shards·6000/10000 requests 및 78000/130000 prompt pairs가 raw identity·순서·분모·finite·merge 검산을 통과했다.

| Endpoint fullseen | RS 원본→replay | PS 원본→replay | NS 원본→replay |
| --- | --- | --- | --- |
| Middle B060 | 5984→5984/6000 | 11596→11607/12000 | 41621→41633/60000 |
| Late B100 | 9939→9939/10000 | 19136→19146/20000 | 65348→65315/100000 |

Late RS 총수 같음에도 lost4/gained4. W 상대 Frobenius 차이0.08213631549/0.07433561280, 원인 NONEXACT_CAUSE_UNRESOLVED. 성능 유사성을 trajectory parity로 승격하지 않는다. 세부 Current/old/new/cohort/active/overwrite/NLL-tail/strict/전이/계측/비용은 보고서와 CSV에 있다.

17 focused tests, CPU compile, 7개 reducer 산출물 실제 재실행 byte 일치, PNG3개 코드 재실행 byte 일치, package rehash/access PASS. 신규 GPU/model/evaluator/Slurm mutation/원격 raw 수신0. Raw-free own-scope만 non-force main 통합하며 scientific_promotion=false.

한계: E01 canonical20cell 중 L4의4cells만 관측. Replay 자체 at-write는 각200requests만 저장. B051/B091 signed/General을 B060/B100 계측으로 대체하지 않음. Target 최종 training trajectory와 exact projected spectrum은 미기록. 원인 종합은 GH 소유. `NO_BROADCAST_NOT_REQUIRED`; 본 리뷰 publication 후 `TASK_COMPLETE_STOP`.
