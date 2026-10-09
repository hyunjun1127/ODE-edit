# 완료 결과 본표 통합

사용자 지시: 각 서버에서 끝난 실험들의 main table 최신화.
nonce USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1.

SH1/2/3 게시 보고서 및 지정 SHA를 GH clean worktree에서 대조했다.
각 상태는 2026-10-09 23:05 KST 부근 owner bounded snapshot이며 이후 상태를 추측하지 않는다.
GH의 추가 GPU/forward/Slurm 조회·변경·checkpoint 이동삭제는 없다.

- SH1: Llama CF MEMIT-FE61773 신규4지표, 공개-query zsRE FT61932/MEMIT61933/BLUE61935/SPHERE61937 신규12칸.
  GPTJ MEMIT_FE_HISTORY61927은 별도 variant 표의 CF4지표에 반영했다.
  native MEMIT-FE와 합치지 않는다. Llama Alpha61934/FE61936 평가 PENDING,
  history61928/61975 RUNNING 유지. CF FT/SPHERE 및 historical 예외는 불변.
- SH2: Qwen zsRE MEMIT61956/Alpha61960 신규6칸. 구 source FT61900은
  편집 완료/공개-query 재평가 필요·미등록으로 표시. GPTJ CF6은 delta0로 유지,
  GPTJ zsRE6은 아직 평가 PENDING. Qwen 나머지 상태는 SH2 snapshot으로 갱신.
- SH3: 신규 본표 적격 완료0. Qwen61813 Q3-selected 별도 완료 기록 및
  Llama61821 선택L1 실행 중 상태를 본표 수치로 승격하지 않았다.
- SH4: 요청 전달·진행 중이며 이번 통합 시점 완료 검산 receipt 미회수.
  결과 부재로 점수를 만들거나 다른 서버 replica를 중복 집계하지 않는다.

총 새 본표 수치7개 model/method/dataset 행의22칸,
별도 history variant1행의4칸을 반영했다. 표시만 소수 둘째 자리 반올림하며 원값은 owner JSON에 보존.
CF Flu/Con DEFERRED, 기존 W0 관측/역사 source 예외, PRICE 자격 제한은 유지.
재평가 미완료를 과거 token-prefix 결과로 대체하지 않는다. 실제 수치와 온라인 전송 성공은 별개다.

[SH1 원수치·분모·SHA](../../audits/servers/server1/main-table-refresh-20261009/table-rows.json) ·
[SH1 보고](../servers/server1/main-table-refresh-20261009/report-ko.md) ·
[SH2 보고](../servers/server2/main-table-refresh-20261009/report-ko.md) ·
[SH2 통합 상세](main-table-refresh-server2-20261009.md) ·
[SH3 보고](../servers/server3/main-table-refresh-20261009/report-ko.md).
