# GH → SH4: CAKE·baseline 보고와 EP alpha-cap sweep 보고 분리

instruction_id: ODEEDIT-S06-CAKE-BASELINE-CAP-REPORT-SEPARATION-SH4-V1
nonce: ODEEDIT-GH-SH4-CAKE-BASELINE-CAP-REPORT-SEPARATION-20260916-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 최신 사용자 원문과 변경 목적

“sh4가 만든 report에서 cake리뷰는 baseline들(alp, memit,blue,l4only등) 만 포함시키는 report로 분리하고 다른 실험들 alpha cap sweep은 report 따로 만들라고 해”

이 요청은 완료된 결과의 **보고서 분리·재구성**이다. 기존에도 개별 CAKE/cap 보고는 있으나 상위 종합 보고에 서로 다른 실험이 섞여 있다. CAKE+baseline 비교와 EP alpha-cap sweep을 각각 독립적인 정본으로 명확히 제공하라. 기존 분리 파일이 있다는 ACK만으로 종료하지 말고 제목·본문·표·그림·목차·기본 진입 링크를 모두 점검해 이번 구성을 실제로 완성하라. 신규 실험이나 재평가 요청이 아니다.

## 1. CAKE + baseline 전용 보고

정본 새 경로:
`experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md`

- CAKE와 동일 fixed10k/order의 기존 baseline만 포함: W0, BASE_ALPHAEDIT_NATIVE, BASE_MEMIT_NATIVE, AlphaEdit_BLUE(L4+L8), MEMIT_BLUE(L4+L8), 각 family의 BLUE-style L4-only 및 이미 측정된 L5/L6/L7/L8-only.
- 사용자 “alp”는 여기서 AlphaEdit를 의미한다. 단독 original/불명확한 AlphaEdit_L4_ONLY 라벨 대신 실제 native/BLUE-style과 물리 layer를 분명히 표시하고 raw arm ID는 provenance mapping으로 남긴다.
- AlphaEdit family+CAKE, MEMIT family+CAKE를 별도 표/소절로 유지한다. CAKE를 두 표에 참고로 반복 표시해도 같은 실행이며 분모·비용을 두 번 합산하지 않는다.
- 주 비교는 actual W100/full10000, 분모10000/20000/100000. NLL/strict/retention/cohort/paired loss-gain/비용·환경·source/hparam 차이와 CAKE 실제 원본 호출 검토를 충분히 유지한다. 기존 실제 intermediate curve도 CAKE/baseline 범위에서 재사용한다.
- CAKE B10 first1000과 W100 first1000은 다른 state임을 유지한다. 중간1k 표가 필요하면 CAKE/baseline의 해당 시점끼리만 별도 소절로 둔다.
- EP-TW-1/CAP1/CAP10/CAP100/NORM_ONLY, lowcost/REFIT4/BG/SL-ZFlow 등 다른 실험의 결과·방법·그림·비용을 본 보고서에 넣지 않는다. 다른 실험들과 합산한 총비용/순위를 CAKE 보고에 두지 않는다.
- CAKE W/M checkpoint 미저장은 사용자 지시이며 exact restart/continuation 미검증이라는 기존 한계를 보존한다.

## 2. EP-TW-1 alpha-cap sweep 독립 보고

정본 새 경로:
`experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v2/diagnostic-report-ko.md`

- CAP1(reuse), CAP10, CAP100, NORM_ONLY W0 first1000/B100×10를 중심으로 독립적으로 읽을 수 있게 목적·실제 설정·동작·결과·한계·재현을 정리한다.
- 기존 최종1k/W5→W10 동일 first500/current/at-write/후보 선택/RAW비율/maximum versus selected action/quality screen/geometry/S64 versus Dev128/paired 전이·NLL·비용 분석을 누락하지 않는다.
- 필요한 동일 first1000 N4 및 baseline 참고 비교는 sweep 내부 별도 절에 유지할 수 있다. CAKE 중심 분석·CAKE 수치/그림의 중복 수록은 CAKE 전용 보고로 정리한다. W100/10k를 W10/1k와 섞지 않는다.
- 자기 trajectory ownRAW와 독립 native sequential baseline을 구분한다. SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED 및 derivative/off-on/GPU continuation 미검증을 그대로 유지한다.
- CAP1 재사용 및 과거 실패/teacher 비용을 신규 sweep allocation과 구분한다. 이번 구성 변경으로 선택 arm·수치·해석을 새로 선별하지 않는다.

## 3. 과거 보고 보존과 진입점 정리

읽기 전용 근거는 main307ba7ae의 다음 sealed packages 및 해당 baseline 정본이다.
- `experiment-reports/servers/server4/completed-experiments-review-2026-09-16-v1/`
- `experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/`
- `experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1/`
- `experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/`
- `experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/`
- 해당 1k baseline 정본은 필요 범위만 재사용.

기존 sealed v1 전체 bytes/manifest/receipt/원시 결과는 보존한다. 새 v2에 필요한 CSV/그림/부속자료만 정확 복사 또는 코드 생성하고 원출처/SHA와 구판→신판 mapping을 명시하라. 기존 shared aggregate CSV에 혼재한 arm은 새 report용 view에서 선택하며 원행/원CSV는 수정하지 않는다. 기존 report 및 raw에 대한 중복 full rehash·모델 재평가를 분리의 선행조건으로 추가하지 말라.

`experiment-reports/servers/server4/completed-experiments-review-2026-09-16-v2/README.md`는 두 독립 정본의 제목·범위·링크를 제공하는 **보고 목록**으로 만든다. 여기에는 mixed scientific table/그림/합산비용/우열 분석을 다시 넣지 않는다. 기존 종합 v1은 역사적 합본이며 현재 읽을 정본은 두 v2라는 안내를 목록에 남긴다. server4 보고 README에서도 기본 링크를 두 독립 보고로 연결한다. unrelated report entries는 유지한다.

## 4. 실행 envelope와 제한

- 새 dedicated branch `codex/server4-cake-baseline-cap-report-split-v1`와 clean worktree에서 진행한다. 최신 main(SH2 SL-ZFlow publication 포함)을 보존하며 shared root/기존 dirty/원실행 source 변경0.
- 승인 write:
  - 위 세 새 v2 report/index 디렉터리.
  - `experiment-reports/servers/server4/README.md` (현재 보고목록 링크만).
  - `project/run_scripts/server4_completed_review/report_separation/` 신규 CPU publication/plot/test 모듈. 기존 reviewer/runtime 변경 없이 필요한 helper는 read-only 재사용.
  - `audits/servers/server4/2026-09-16-cake-baseline-cap-report-separation/`.
  - `messages/acks/server4/2026-09-16-cake-baseline-cap-report-separation.md`.
  - `messages/server-heads/server4/2026-09-16-cake-baseline-cap-report-separation.md`.
  - `runs/odeedit_cake_baseline_cap_report_separation_s4_20260916_v1/`.
  - `tasks/status/odeedit_cake_baseline_cap_report_separation_s4_20260916_v1/server4.json`.
  - local work/staging/receipt: `/data/janghj/ODE-edit/local/report-separation/20260916-v1/`.
- Slurm submission: **not allowed**. Scheduler 재조회/모니터링/새 GPU/model/forward/evaluator/teacher/FD/ULP/continuation/replay/rsync/삭제0. GPU cap2는 유지되지만 본 task 신규 GPU 사용0.
- 본 scope는 간단한 보고 재구성으로 SH4가 직접 수행한다. GH 재승인 또는 중복 raw/GPU 감사 없이 끝낸다. 기존 작업/SH2/paused task 재개0.
- pre/post 확인: 두 보고 범위 독립, 원 count/분모/order/metric 불변, 숫자와 label mapping, baseline family 분리, immutable v1 unchanged, GFM 표 pipe/행열/실제 렌더, 링크·그림·manifest/receipt 결속, 원자료 Git 혼입0. 오류는 publication 코드/새 출력만 수리하며 다른 범위 충돌은 보고.
- PNG는 코드 생성만. 기존 그림이 새 범위와 일치하면 byte 그대로 재사용 가능; 범위/legend가 바뀌면 해당 그림만 코드로 재생성하고 재현 근거를 남긴다.
- NO_BROADCAST_NOT_REQUIRED: 기존 서버 로컬 분석의 compact publication 재구성이므로 새 대형 전송 불필요.
- 이 범위 code/tests/raw-free reports/index만 검증 후 최신 main에 non-force 통합·push를 승인한다. 다른 서버 변경 보존, force/reset/임의 ours-theirs0, 새 numerical PASS/scientific promotion0.
- 최종 두 정본 **절대경로 + repo 경로 + SHA**, 새 목록 경로, 기존값 불변/표렌더·링크 검사, mainHEAD/tree 및 source/report-only 변경임을 간단히 보고하고 TASK_COMPLETE_STOP. 별도 후속 실험·자동 모니터링0.
