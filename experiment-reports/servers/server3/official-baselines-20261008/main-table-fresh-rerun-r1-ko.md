# server3 official fresh 재실험 범위 복원

최신 nonce `GH-SH3-MAIN-TABLE-FRESH-RERUN-20261009-R1`과 authority `1bedeaf067cfc82af40e769eebace7703f0df558`에 따라 Qwen MEMIT·AlphaEdit CF 신규 chain 제외와 실행 gate를 제거했다. 이전 `f990d10d`의 제외 방침은 SUPERSEDED이며 당시 checkpoint 존재/SHA 확인 기록과 원본은 보존한다.

구현 source는 `c4a793bcd7ee7ba7507cbdd0617e7d0fe08834c2`다. CF 편집 10개, zsRE 편집 6개(합계 16개), CF qualification은 W0+여섯 방법 7 job으로 복원했다. BLUE CF main은 grid alias이며 격자/대조와 zsRE 범위는 부모 계약 그대로다. 새 실행은 고정 CF10k 앞2000의 ordered case SHA `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`에서 fresh W0/native 초기 cache로 시작한다. 다른 cohort의 과거 CP를 새 main으로 복원하지 않는다.

상태는 **IMPLEMENTED_NOT_SUBMITTED**다. 한정 `squeue -u janghj -w ubuntu` 조회(exit0)는 빈 목록이었다. 새 main job ID/name과 실행 로그는 없다. 표의 해당 칸은 빈칸이며 qualification/W0를 ING로 표시하지 않는다. GH가 root README를 단독 갱신하며 이번에는 직접 수정하지 않았다. 행별 계획 attempt/model/method/dataset/order/cold/source/config/job/state/time/report는 [compact receipt](../../../../audits/servers/server3/official-baselines-20261008/main-table-fresh-rerun-r1.json)에 있다. planned cold identity는 실제 GPU 시작 증거가 아니다.

최신 metadata preflight에서 Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28` snapshot 부재, FLU/CON reference manifest 부재, native NLTK tokenizer resource 부재, 성공한 W&B project/auth precheck 부재를 확인했다. 정확 경로는 audit에 기록했다. C0/P는 이번 가벼운 점검에서 다시 대형 해시를 읽지 않아 CONTENT_UNVERIFIED이며 과거 fullSHA receipt와 구분한다. 전체 계획의 저장공간 gate는 `FULL_PROGRAM_DISK_LOW:free=99218456576:reserve=138905911296`다. 임의 다운로드·대형 전송·삭제·환경 교체 또는 입력 gate 완화는 수행하지 않았다.

CPU runner/assets/submit 테스트 **40 PASS**, source integrity **PASS**(157 source files, external task imports0). actual Qwen/GPU qualification과 새 W20은 NOT_RUN이다. 서버3에 FLU/CON 연기 권한이 새로 부여된 것은 아니므로 generation을 DEFERRED로 임의 표시하지 않았다. 이후 승인 실행의 제출/상태/완료에는 실제 job ID/name 및 관측시각을 결속하고, CF6칸과 zsRE3칸은 W20 실측 필드만 독립 보고한다. 기존 job/source/raw/CP 변경0, 신규 제출0, 반복 monitoring0.
