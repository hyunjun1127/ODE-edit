# SH3 baseline 완료 및 zsRE 산출 감사

Nonce USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1. accepted turn `01a122d0-2dca-7152-b5ba-b513af14972d`. main7dacd3fb의 전체 지시를 읽고 전용 non-main WT에서 수행했다.

**완료 본표 적격 baseline CF W20 및 zsRE W20은 각각 0개다.** 이는 점수 0이 아니라 해당 own 결과가 없다는 뜻이다. table-rows.json의 두 완료 배열은 빈 배열이며 README numeric 갱신 요청은 없다. Qwen baseline 이관본은 SH2 소유이므로 중복 보고하지 않는다.

현재 등록된 own baseline 관련 job 62344 `s3-qwen-blue-sweep-1k-lane1`은 RUNNING, 62345 `s3-qwen-blue-sweep-1k-lane2`는 PENDING으로 단발 관측했다. 두 job은 heldout 1k sweep이므로 W20/2000 본표 대상이 아니다. PRICE 500 sweep 62477/62478도 별도 작업으로 제외했다. job 상태는 inventory의 관측시각에만 유효하며 변경/장기감시하지 않았다.

알려진 baseline local 다섯 루트의 깊이7 이하 소형 terminal/result/completion/submission/status/summary/progress/receipt metadata를 한 차례 확인했다. source/worktree/cache/입력/raw chunk/symlink는 제외했다. 새 metadata 4개는 모두 1k sweep 소속이고 zsRE 완료 기록은 없다. 이전 no-own-zsRE inventory의 48개 metadata SHA/bytes가 현재도 모두 동일하다. 오류/oversized 파일은 0개다. 전체 파일시스템의 부재 증명으로 확대하지 않는다.

zsRE 실제 frozen evaluator/query/profile SHA, 전체 stream tokenizer/prefix/공백/BOS/decode-retokenize parity, predicted/target IDs 또는 correctness bits CPU 재집계, 2,000 request 및 token 분모는 **NOT_APPLICABLE_NO_OWN_COMPLETED_RAW**이다. 실제 실행이 없으므로 공통 API source를 frozen 실행 증거로 대신하지 않았다. 공통 ZSRE_PAPER.md를 읽고 현 evaluator/parity 모듈 SHA를 참고자료로만 기록했다. Eff/Gen/Loc은 각 요청 내부 target token 정답률 평균 후 요청 간 평균×100이며 Loc target은 loc_ans다. W0agreement/token micro/strict ACC로 대체하지 않는다. CPU query parity가 pretrained forward parity를 뜻하지 않는다는 제한도 유지한다.

CF factual·과거 수치·W0·FLUCON을 변경하지 않았다. 생성 결측은 DEFERRED이고 향후 완료 raw의 표시만 unrounded×100/half-up2다. OURS/Q3/tuning/sweep/historical wrong-context를 본표로 승격하지 않았다. 최근 SH3 W0 감사는 별도 reference이며 이번 baseline 완료행으로 재집계하지 않았다.

증거: audits/servers/server3/baseline-completed-zsre-audit-20261010/inventory.json 및 table-rows.json. inventory.py는 read-only metadata/hash와 단발 scheduler 조회만 수행한다. owner CPU audit이며 별도 reviewer/새 GPU/forward/CP load/재평가/Slurm mutation/삭제/온라인 history 변경은 없다. 원 dirty/frozen/raw/CP는 보존했다. NO_BROADCAST_NOT_REQUIRED. README는 GH sole 통합이다.
