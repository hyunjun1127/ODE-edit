# GH → SH4: 완료 Single-layer mechanism-first B1 상세 리뷰

Instruction ID: `ODEEDIT-S06-SLMF-B1-COMPLETED-DETAILED-REVIEW-SH4-V1`
Nonce: `ODEEDIT-GH-SH4-SLMF-B1-COMPLETED-REVIEW-20260920-R1`

사용자 “server4 실험 끝난거 리뷰시켜”에 따른 explicit recall이다.
최신 게시 기준 대상은 **51058 / odeedit_slmf_B1r4_s4**이다. 먼저 exact job/lock/terminal을
한정 확인하여 이 완료 실행을 CPU에서 자세히 리뷰하고 한국어 최종 보고와 main 게시까지 진행한다.
단순 ACK/첫 표/계획만으로 종료하지 않는다. 새 과학 실행을 재개하는 지시가 아니다.

## 1. 경계·읽기·입력

- Server4/session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`.
- 최신 main 및 실제 registry/PROTOCOL을 확인하고 새 clean branch/worktree
  `codex/server4-slmf-b1-completed-review-20260920-v1`를 사용한다. Shared root/dirty/Git identity 보존.
- 새 envelope 전체 및 원 사용자 지시/설계/contract/cells, 원 dispatch,
  default-no-checkpoints와 B1-only/FD-only 최신 사용자 override, 현재 task status를 결속한다.
  이전 FULL_READ는 exact SHA일 때 재사용하고 변경된 문서·소스는 직접 읽는다.
- 실행 source `5f79085629b10b2bb8bdee88d017e18a46bb4c74`, tree
  `a5b089229ab3e4edd2e3595c5c4f5226325b433e`, lock
  `6a14ebaf32549cc9479f2d112ba1954b06ef00380fffc1091cd70090f9098f61`.
- 실제 root `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/`.
- 이 exact 51058 scheduler accounting을 한 번 확인하고 terminal/성공·실패·부분완료를 raw와 별도 판정한다.
  다른 job 전수조회/다른 task monitoring0. 이전 실패51057 및 이전 T0는 이 실행이 참조한 저장 receipts/source만 재사용한다.
  만약 대상이 다르거나 아직 실행 중이면 확인 사실과 경계를 보고하고 새 실험/무한대기하지 않는다.

## 2. 금지 및 권한

**CPU-only 완료 리뷰**: Slurm submission/write/cancel/hold/release 금지, 신규 GPU/model load/
forward/backward/evaluator/FD/native fitting/teacher 생성0. 프로젝트 cap2는 유지하되 이번 리뷰 GPU0.
기존 runtime/source/raw/threshold를 수정하지 않는다. CPU reducer/parser/plot/report의 오류만 고칠 수 있다.
Missing observation은 NOT_MEASURED, 미저장 상태는 NOT_RECORDED로 남기고 새 모델 계산으로 채우지 않는다.

`max_batches=1`, `sequential_authorized=false`, `save_checkpoints=false`, `exact_resume=NOT_AVAILABLE` 유지.
Gate를 CPU에서 재집계해도 S3/S10 권한은 생기지 않는다. 다른 paused task, 특히 SH2 hold51071은 그대로 둔다.
FD waiver는 `SKIPPED/WAIVED_USER_DIRECTED`와 원 판정 미확립을 보존하고
`full_numerical_validation=NOT_ESTABLISHED`를 actual 전체 PASS로 바꾸지 않는다.
옛 EN/CAKE/cap-sweep 보고서를 다시 확장하거나 raw를 옮기거나 삭제하지 않는다.

## 3. 첫 실제 표 → 상세 리뷰

가능한 첫 중간보고는 원 raw true/new NLL 독립 reducer로 만든 **N4/EN-KL-Q/DEC-LINE/DEC-MODES-CUM** 표다.
B1 STEP=CUM alias는 중복 endpoint/시행으로 세지 않는다. 실제 미완료 arm은 NA와 이유를 기록한다.
분모 R100/P200/N1000 및 request100/order/target/prompt identity, strict inequality/tie-failure를 확인한다.
저장 aggregate나 동일 endpoint hash만으로 성능을 추정하지 않는다. W0 기준 raw가 있으면 함께 제시한다.

최종 보고 필수:

1. **실행 완결성·provenance**: scheduler와 scientific terminal/arm seal/observer 완료를 별도 판정.
   Runtime/analysis source, actual import/config/model/tokenizer/data/order/R512/Dev128/teacher 및 native identity,
   원 snapshot·원 T0 실패·waiver를 연결한다. 새 hash와 기존 receipt+stat 재사용은 구분한다.
2. **성능과 문항 전이**: R/P/N, true/new NLL와 원하는 방향의 margin, TF-strict/P-strict/request joint,
   N4 대비 paired lost/gained와 request cluster 불확실성, W0-correctN gross lost/recovery·tail.
   Official P/N/Dev는 postseal observer인지 source/ledger로 확인한다. 한 batch로 lifelong retention을 주장하지 않는다.
3. **Reference 효과와 독립 효과 분리**: 실제512문서/valid generated positions/EOS/censor/tie,
   native→각 후보→selected Phi/Psi, token mismatch·safe 신규flip·기존flip 회복,
   worst deficit/문서 보존/Dev128/KL. MARGIN_ONLY와 actual choice recovery를 구분한다.
   Rejected 최선 후보 값을 최종 selected 성능으로 표기하지 않는다.
4. **설계대로 작동했는가**: 요구사항→frozen file/function/line→stored evidence 대응표.
   L4-only/동일 shared native100/추가 z0/Q_E/전체token response 보호,
   Current 성공 ID·strict·요청별NLL/allowed leakage·FP32 materialization,
   B1 Past NOT_APPLICABLE, history exactly-once의 관측 수준, transaction/reset을 검사한다.
   Endpoint W/M 미저장 상태를 독립 tensor 재구성이나 GPU resume PASS로 부풀리지 않는다.
5. **목적·후보·solver**: EN-KL-Q와 DEC 목적/수용 정책 차이, LINE 대비 MODES의 실현 차이,
   동일 center factors/J 재사용, rank≤5/gradient span/extra-direction angle,
   coefficient/QCQP 두 단계 objective·residual·gap·certificate, scale별 predicted/actual,
   accept/reject의 정확 조건/순서/finite budget/fallback/zero-risk를 공개한다.
   유한 후보 실패는 전체 공간 불가능성/최적수렴과 구분한다. B1 STEP=CUM으로 누적 이득을 주장하지 않는다.
6. **z 효율화와 전송 제거 실제 적용**: prefix cache/필요위치 full-vocab head/요청 batching의 실제 설정,
   per-request Adam·clamp·stop·FP32/TF32/BOS/padding/KL 경로와 cache invalidation을 frozen source로 대조.
   원 native 대비 어떤 T0 항목 PASS/warning/waived/NOT_MEASURED인지 구분하고 batch16 성공을 가정하지 않는다.
   문서별 dense-gradient GPU→CPU 제거 및 GPU 누적 경로/실제 final transfer·메모리를 확인한다.
   미계측 전송량·속도는 NOT_MEASURED; 단순 layer호출 계산이나 unmatched 이전 walltime을 speedup으로 쓰지 않는다.
7. **기전 evidence coverage**: 저장된 writer spectrum/target-loading/realization,
   reference action·cross-term·signed response 및 postselection component panel을 CPU 재집계한다.
   H1–H5마다 존재/누락·사실·제약을 적되 우월성/인과/후속방법 선택은 GH global review에 남긴다.
   미수행 suffix/component는 모델을 재실행하지 않고 coverage gap으로 보고한다.
8. **비용**: 51058 parent allocatedGPU-sec/programwall/peak와 prefix/teacher/native/geometry/factors/
   derivative/solver/candidate/full512/Current/observer/I-O를 분리한다. Accepted·rejected 모두 포함.
   원 실패/T0 parent1113sec는 prior receipt 재검산 후 별도, 신규와 중복 합산하지 않는다.
   총 연구 shared비용1회와 각 method standalone 필수 shared비용전액을 나눠 표시한다.
   Nested timers/미분리 NOT_SEPARATED, allocation≠utilization, 별도 paired timing 없으면 인과 speedup 미확립.
9. **B1 gate의 기계적 판정**: frozen 조건에 따른 각 항목 PASS/FAIL/UNRESOLVED와 이유,
   numerical waiver와 quality/cost를 분리한다. Gate 통과하더라도 다음 실행은 미승인이다.

Raw 전체 scope에 대한 경로/size/hash inventory와 축약 source/input/analysis manifest를 남긴다.
서로 다른 비교 조건의 옛 baseline은 REFERENCE_ONLY; fresh matched N4를 우선한다.
네 arm 표 + paired transitions + reference/candidate ledger 요약 + 구현 일치표 + 비용/coverage를 CSV로 내고,
핵심 그림은 코드 생성·재생성 검산한다. 한국어 Markdown 표/링크/렌더를 점검하고 미실행 검사는 그대로 밝힌다.

## 4. write 범위·검토·게시

허용 source는 `project/run_scripts/single_layer_mechanism_first/` 안 **새 CPU review/plot/test 모듈만**.
기존 production module은 변경하지 않는다. 다음 경로를 본 instruction의 소유로 추가 승인한다:

- `local/single-layer-mechanism-first/20260919-v1/completed-b1-review-20260920-v1/**`
- `experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/**`
- `audits/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/**`
- `plans/updates/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/**`
- `messages/acks/server4/2026-09-20-slmf-b1-completed-review.md`
- `messages/server-heads/server4/2026-09-20-slmf-b1-completed-review*.md`
- `tasks/status/slmf-b1-completed-review-sh4-20260920-v1/server4.json`
- `runs/odeedit_slmf_b1_completed_review_s4_20260920/**` (compact provenance only)

새 독립 리뷰는 원 submitted lock/pending 역사/실패 보고를 수정하지 않는다.
Preflight는 신원·원 입력 read-only·모델/GPU 호출0·경로, postrun은 독립 NLL reducer/
selector arithmetic/비용/분모/manifest/raw-free/그림 재현을 검사한다.
Warn은 명시하고 block은 영향받는 claim만 차단하며 다른 사실은 보존한다. 복잡한 독립 작업만
bounded subagent에 분담한다. 별도 red 미사용이면 owner audit+독립 reducer로 기록한다.
공유 helper가 명시 경로를 지원하지 않으면 좁은 예외를 기록하고 전역 권한을 확대하지 않는다.

최종 `report-ko.md`를 위 completed-b1-review package에 작성한다. Code·compact report만
own branch/main non-force 통합 승인, concurrent GH/SH 변경 보존. Raw/weights/teacher/prompt/fullstdout Git0.
`NO_BROADCAST_NOT_REQUIRED`, 새 원격 raw 전송0/삭제0. FULL_READ/M0→첫표→주요 검산/문제→최종 compact 인계를 남긴다.
최종 source/report/manifest/rooted-receipt/main SHA와 한계를 반환하고 TASK_COMPLETE_STOP.
별도 재승인/중복 GH GPU 감사를 선행조건으로 요구하지 않는다.
