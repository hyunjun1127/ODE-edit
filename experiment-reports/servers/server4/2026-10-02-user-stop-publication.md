# Server4 사용자 실험 중단·기존 산출물 게시

`ODEEDIT-GH-ALL-SH-STOP-EXPERIMENTS-S4-PUBLISH-20261002-R1-SERVER4`.
현재 상태 STOPPED_USER, monitoring_active=false, automatic_resume=false.

- GH가 56962→56960/56961 순서로 취소했고 SH4가 CANCELLED/빈 queue를 확인했다.
- [JLZ 부분 결과](jlz-twoarm-bs100x20-20261002-v1/user-stop-20261002-r1/report-ko.md): A/B 각각11batch/1,100요청 완료 후 B12 중단. 20batch 전체 완료 아님.
- 기존 source/partial code/compact 표·보고·그림110개(2,084,851B)를 원 bytes로 추가 게시했다.
  원 worktree의 삭제/dirty·기존 main 파일을 덮지 않았다. 실제 목록은
  [게시 inventory](../../../audits/servers/server4/2026-10-02-user-stop-all/existing-output-publication-inventory.json)에 있다.
- 큰 per-request 압축 CSV2개 및 geometry CSV2개는 raw 규모 산출물로 local KEEP/Git0이다.
  각각 source path/size/SHA를 inventory에 기록했다. 기존 main과 다른 로컬 variant4개는
  main/로컬 양쪽을 그대로 보존했고, 같은 파일3개는 중복 게시하지 않았다.
- 이번 게시로 새 실험·재평가·과학 수리를 수행하지 않았다. 저장된 과거 보고의 판정은 작성 당시 값이다.

## 기존 출력 묶음과 상태 구분

| 묶음 | 이번 게시의 의미 | 상태/기존 상세 근거 |
|---|---|---|
| JLZ twoarm | 이미 게시된 runtime 유지, 최신 저장 부분 표·중단 evidence 추가 | PARTIAL_CANCELLED_USER; W20 NOT_MEASURED |
| Alpha-key generated-r1/r3/r4 | 기존 자동 reducer의 보고·소형표·그림 보존 | 실패/부분/완료 원 terminal 구분; 아래 상세 리뷰 우선 |
| 과거 lifelong finalW full10k v4/v4-r1 | 기존 생성 보고·aggregate CSV·PNG·manifest 보존 | 원 보고 상태 유지, 이번 신규 evaluation0 |
| Temporal-routing 단층 | 철회 당시 미게시 partial source/CPU preflight/STOP 상태 보존 | USER_REVOKED, job0/GPU0; 실행권한 재부여 아님 |
| 과거 EN/SLMF 분석 source | 누락된 기존 CPU handoff 코드만 보존 | 실제 원 실행/현재 main과 다른 variant는 덮지 않음 |
| Historical timeaxis / delayed E3 / joint migration | 이미 main 게시된 보고·source 재사용 | 재실행·과거 raw 재분석0 |

기존 완료 리뷰:

- [2026-09-24 전체 jobs/Alpha-key 상세](completed-jobs-review-2026-09-24-v1/report-ko.md)
- [Historical timeaxis](historical-update-timeaxis-20260924-v1/completed-review-20260926-v1/report-ko.md)
- [Delayed-write E3](native-delayed-write-e3-20260924-v1/completed-review-r1/report-ko.md)
- [Joint BS1 server2 이관](joint-multilayer-bs10-20260929-v1/migration-to-s2-r1/report-ko.md)
- [단층 철회 상태](../../../tasks/status/temporal-routing-diagnostic-20260929-v1/server4.json)

단층 preflight의 향후 실행계획·snapshot 설명은 **철회 전 역사**이며 이번 게시에서 실행하지 않는다.
과거 generated report의 로컬 대용량 파일 링크는 서버에서만 열릴 수 있고 Git에서는 의도적으로 제외된다.
새로운 render/reviewer PASS로 원 보고를 승격하지 않는다. 신규 GPU0/submit0/CP0/원자료 삭제0.
NO_BROADCAST_NOT_REQUIRED: raw/tensor/prompt/fullstdout local 보존, 소형 Git 게시만 수행.
