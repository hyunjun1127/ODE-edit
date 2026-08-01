# Motivation adaptive-gate pair 실행 전 최소 audit

대상: `agate_memit_*_g0_v1`, `agate_alpha_*_g0_v1`

범위: 빠른 제출에 필요한 scientific lock, technical contract, resource
boundary만 점검한다.

## 네 범주의 근거

- proposal에서 온 내용:
  state별 direction/ranking non-stationarity와 dynamic relinearization,
  AlphaEdit null-space proposal transfer를 Motivation에서 검증한다.
- repo/protocol에서 확인한 사실:
  직전 `qstep4`에서 Llama direction refresh는 작고 불확실했으나 Qwen은
  컸다. coefficient refresh는 두 model 모두 양의 방향이었다. protocol은
  SH 부재와 사용자 명시적 time-critical 요청에서 GH 최소 직접 제출을
  허용한다.
- GH 추정:
  outcome-free 후보 gate가 Llama의 불필요한 refresh를 줄이면서 Qwen
  refresh 이득을 보존할 수 있고, projector 합성 뒤에도 같은 state
  dependence가 일부 남을 수 있다.
- 사용자 확인 필요:
  없음. 사용자가 두 fixed model 동시 실행, AlphaEdit ODE-style hook,
  cache 재사용, lenient 분석과 빠른 제출을 명시했다.

## Scientific/technical lock

- fresh salted rank `[124:132]`, model별 8건, prior `[0:124]`와 disjoint
- K=4, 각 hop `D/4`, probe `D/64`, first hop exact common
- G는 동일 current G state에서 refreshed/fixed candidate를 함께 평가하고
  outcome을 보기 전에 2% relative margin으로 선택
- A/B/C 및 no-op/full/split4 final controls 고정
- MEMIT 뒤 Alpha pair를 순차 제출; 각 pair 안의 Llama/Qwen은 동시 시작
- Alpha track은 pinned EasyEdit projector를 `B@P`로 합성한
  `alphaedit_projected`이며 native AlphaEdit 전체 재현으로 부르지 않음
- direct-z 1회/사례, Wikipedia covariance와 projector는 existing pinned
  artifact만 사용; download/recompute/write fallback 없음
- raw outcome 전에 feature/action/path/lineage/branch receipt commit

상세 사전 계약은
`plans/global/2026-08-01-session01-motivation-closure-spec.md`다.

## 최소 검증 결과

- targeted regression: `13 tests`, `OK`
- 전체 Motivation regression: `190 tests`, `OK`, `10.767s`
- Python compile: 새 runner/gate/projector 통과
- shell syntax: child/pair/submit helper 통과
- `git diff --check`: 통과
- fixed selection manifest:
  `2d0d98b48692e82fa04efaa8df97863c48202bc23818e38cf2bc1a2548c81cf7`
- 새 output/one-shot marker 6개: 모두 absent
- session boundary:
  `PASS ... session=019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2 model=Sol Ultra`
- project active GPU: `0`; pair 요청 `2`, server1 cap `3`
- host memory: pair `130000M`, cap check 허용
- EasyEdit working tree는 이 작업 전부터 다른 변경을 포함한 dirty 상태다.
  이번 작업은 EasyEdit에 write하지 않았고 ODE-edit hook만 추가했다. job은
  Git cleanliness에 의존하지 않고 repo의 fixed SHA-256/size manifest로 실제
  읽는 EasyEdit source/runtime/cache/projector byte를 fail-close한다.

## Agent boundary

세 개의 제한된 reviewer를 병렬 요청했으나 각 runtime에서
`gpt-5.6-terra`, `ultra` metadata를 확인할 수 없어 지시대로 어떤
file/web도 읽지 않고 즉시 종료했다. 미확인 agent 결과는 audit 근거로
사용하지 않았다. 이는 실험 실행을 막지는 않지만, 각 실험의 최종 분석
report 승격에는 사용자가 요구한 verified Terra Ultra agent gate가 별도로
남는다.

## 자원 및 중단

- parent: `2 GPU / 16 CPU / 130000M / 12:00:00`
- child: `1 GPU / 8 CPU / 65000M`
- Alpha pair helper는 MEMIT job exact `COMPLETED`와 양 model
  `all_pass=true`를 확인하기 전 제출을 거부한다.
- output/marker/job collision, dirty/unpushed ODE-edit main, session/cap
  mismatch, child nonzero, cache/projector drift, lineage/C-budget/rollback/
  firewall 위반이면 즉시 중단한다.

## GH 직접 제출 예외

- 사유:
  server1 SH가 없고 사용자가 빠른 Llama/Qwen 동시 제출·분석을 명시했다.
- exact command:
  `project/run_scripts/submit_session01_adaptive_gate_pair_server1.sh memit`,
  이후 technical complete일 때 같은 helper의 `alphaedit_projected`
- 영향 범위:
  server1의 pair allocation 한 건씩과 명시된 ignored `local/` raw/log/state
- 후속 보고:
  model별 agent report, pair post-run audit, server1 completion message,
  no-peer artifact broadcast 예외

- 최종 판정: `PASS` — locked adaptive-gate MEMIT/Alpha pair 각각 1회에만 유효
