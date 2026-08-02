# Session 01 capacity/history closure — 최소 execution preflight

- 날짜: 2026-08-02
- 작성: GH 최소 필수 red preflight
- 대상: `odeedit_capacity_history_pair_v3` (`15813`, `15817` technical recovery)
- claim boundary: same-policy 4-edit Motivation signal only

## 확인 결과

- `PASS`: 누적 capacity 식은 과거 displacement와 current actuator의 C-cross term을
  포함하며, overloaded/positive-load layer의 coefficient cap은 0이 된다.
- `PASS`: MEMIT/Alpha-history가 같은 QP, D/4→D/8 retry, 최대 3 accepted round,
  first-hit와 trust-ratio policy를 사용한다.
- `PASS`: historical Alpha solve가 upstream
  `P @ (K K.T + cache_c) + L2 I`와 `cache_c += H H.T`에 대한 low-rank 등가식을
  사용한다. History는 edit 안에서 고정되고 post-edit key가 한 번 append된다.
- `PASS`: projector와 covariance는 existing pinned artifact만 read-only load한다.
  두 모델의 preflight는 각각 25 files를 검증했고 recompute/download path를
  controller가 호출하지 않는다.
- `PASS`: evaluator source order는 model별 네 terminal controller와 receipt/hash
  검증 뒤 evaluation loader를 호출한다.
- `PASS`: Llama/Qwen policy/config hash가 공통이고 model별 rescue branch가 없다.
- `PASS`: 새 compact schema와 gate에 NFE field가 없으며 wall time과 proposal/
  accept/reject count만 보조 compute metadata로 둔다.
- `PASS`: recovery targeted regression `24 passed`; Python import, exact
  five-layer probe contract와 sanitized traceback regression 통과.
- `PASS`: 전체 Motivation regression `323 tests` exit 0; 세 launcher의
  `bash -n`과 `git diff --check` 통과.
- `PASS`: session boundary는
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2 / Sol Ultra / server1 / ODE-edit`로
  확인됐다.
- `PASS`: resource check는 4 GPU/260000M, active project GPU 0에서 허용됐다.

## 제한과 방지책

- EasyEdit worktree에는 이 연구 이전의 수정이 존재한다. 이를 clean upstream으로
  주장하지 않는다. 실험은 현재 fixed 25-file provenance를 controller/evaluator
  manifest에 저장하고, ODE-edit hook만 실행하며 EasyEdit에는 write하지 않는다.
- Terra Ultra code/science reviewer 세 개를 병렬 시도했으나 runtime metadata가
  Sol이거나 Terra를 증명하지 못했다. 모두 repo를 읽기 전에 종료했다. 독립 Terra
  review를 수행했다고 표기하지 않으며 GH가 위 필수 gate만 직접 확인했다.
- 4 edits는 lifelong 검증이 아니다. 양성 결과도 short-chain directional signal로만
  해석한다.
- active peer SH/clone이 없으므로 artifact broadcast는 no-peer exception으로
  기록한다. 임의 SSH/rsync는 하지 않는다.
- Original job `15813`은 Qwen MEMIT step `15813.2`가 `2:0`으로 먼저 실패했고,
  `.0/.1/.3`은 parent fail-fast에 의해 signal 9로 취소됐다. Evaluator는 시작되지
  않았고 partial controller artifact는 evidence에서 제외했다.
- v2는 scientific contract를 그대로 유지하고 failed `c0_v1`을 보존한 채 새
  `c0_v2` output/marker만 쓴다. Worker별 stdout/stderr와 sanitized source/function/
  line traceback을 추가해 같은 failure가 반복돼도 raw prompt 없이 원인을 식별한다.
- Job `15817`의 isolated traceback은 Llama MEMIT-QP 첫 edit에서
  `build_central_probe_panel:274` action-order mismatch를 확정했다. Capacity caller는
  five layer actions만 의도적으로 사용했지만 helper의 six-action quarter-step
  default를 override하지 않았다.
- v3 patch는 helper 기본값을 보존하고 capacity call에 exact five-layer expected IDs를
  명시한다. Uniform probe 추가, QP objective/threshold, case/model/history/metric 변경은
  없다. v1/v2 artifact는 모두 read-only 보존하고 v3 namespace만 새로 쓴다.

- 최종 판정: `PASS` — clean pushed main과 submission wrapper 재검증 후 locked 4-GPU pair 1회 제출에만 유효
