# server1 inbox — Session 01 Motivation MV-1 score-mix D0

## 명령 메타데이터

- 명령 ID: `session01-mv1mix-d0-pair-v1`
- 작성 시각: 2026-07-30
- 작성 agent: `head-server1-gh`
- 대상 서버: `server1`
- 대상 Codex session ID: 미지정 — server-head 등록 전 SH 실행 금지
- 대상 repository CWD: `/mnt/raid5/janghj/ODE-edit`
- 대상 Git repository identity: `hyunjun1127/ODE-edit`
- 우선순위: high
- ack 필요 여부: server-head session 등록 후 yes
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-30.md`,
  `runs/mv1mix_*_d0_v1/`,
  `experiment-reports/servers/server1/`

## 목적과 배경

v1 pilot은 technical fidelity를 통과했지만 finite six-arm set이 continuous
routing을 식별하지 못했다. D0는 pilot case를 제외한 calibration `[3:8]`
다섯 case에서 outcome-free `score_mix`와 same-`C` `uniform`의 realized
progress를 비교해 Motivation을 최소 비용으로 early-kill하는 diagnostic이다.

현재 server-head가 없고 사용자가 time-critical 동시 제출을 명시했으므로
GH가 `PROTOCOL.md:506-509` 예외로 exact pair 한 건만 직접 제출한다. 향후
server-head는 자신의 session ID가 `servers/active/server1.md`에 등록된 뒤에만
ack, monitoring, post-run 보고와 가능한 artifact broadcast를 인수한다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_d0_v1/`
  - log: `local/logs/slurm/session01_motivation/`
  - state: `local/state/slurm-submissions/session01_motivation/`
  - small report: `runs/mv1mix_*_d0_v1/`,
    `experiment-reports/servers/server1/`,
    `audits/servers/server1/`,
    `messages/server-heads/server1/`
- Slurm 제출 허용 여부: GH one-shot helper에만 allowed; 미등록 SH의 추가
  제출·재제출은 not allowed
- GPU cap: server1 project 동시 최대 3; 이번 pair 총 2, child별 1
- host-memory cap / 요청 memory: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-30-session01-mv1mix-d0-execution-preflight.md`의
  exact `PASS`
- artifact broadcast 의무: active peer가 없으므로 즉시 실행 예외. peer
  활성화 후 protocol helper로 검증·전송하거나 no-peer 예외를 완료 보고에 기록
- 완료 보고 경로: 위 명령 metadata와 동일
- session boundary 확인 command:
  `scripts/check-session-boundary.sh <등록된-server1-SH-session-id>`

## 예상 산출물

- model별 exact 5 event, event별 feature 1, commitment 1, outcome 5, receipt 1
- sanitized manifest/features/actions/outcomes/events/summary
- model별 한국어 독립 분석, pair post-run red audit, small run metadata

## 중단 조건

- session/CWD/repository/job/run/slice/resource envelope 불일치
- Llama/Qwen이 같은 allocation에서 동시에 시작하지 않음
- moments/projector recompute·download·write·deserialize 또는 EasyEdit write
- outcome leakage, receipt ordering, equal-`C`, rollback/replay/hash/firewall 위반
- output/marker 중복, dirty/unpushed Git, 어느 child든 nonzero

## 금지 사항

- 미등록 SH의 submit/retry/cancel, 다른 repo 또는 다른 Codex session 조작
- raw prompt/target/logit/generation, weights, checkpoint, dataset, full log의 Git 유입
- credential, token, private SSH 값의 Git 기록
- `rsync --delete`, destructive mirror, EasyEdit 수정, online dependency/model download
- D0 positive를 method gain, GO, confirmatory 또는 MV-2 승인으로 해석

---

# D1 추가 instruction — `session01-mv1mix-d1-pair-v1`

## 목적과 배경

D0의 두 model 모두 5/5 `G_mix`가 replay envelope를 넘어 early-kill
조건이 깨졌다. Canonical continue rule에 따라 pilot/D0와 겹치지 않는
calibration `[8:20]` 12 case/model을 동시에 실행해 calibration 17 case를
완성하고 slope-only frozen static comparator를 고정한다.

## 실행 권한 envelope

- 대상 Codex session ID: 미지정 — server-head 등록 전 SH 실행 금지
- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_d1_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: `runs/mv1mix_*_d1_v1/`,
    `experiment-reports/servers/server1/`, `audits/servers/server1/`
- Slurm 제출 허용 여부: GH one-shot helper에만 allowed; 미등록 SH의 추가
  제출·재제출은 not allowed
- GPU cap: server1 project 최대 3; 이번 pair 총 2, child별 1
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`, child별
  `65000M`
- red-team gate:
  `audits/global/2026-07-31-session01-mv1mix-d1-execution-preflight.md`의
  exact `PASS`
- artifact broadcast 의무: active peer 없음 예외를 완료 보고에 기록하고,
  peer 활성화 후 protocol helper로 검증·전송
- 완료 보고 경로: `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_d1_v1/`, `experiment-reports/servers/server1/`
- session boundary: 등록된 server1 SH session ID로
  `scripts/check-session-boundary.sh` 통과

## 예상 산출물

- model별 12 event, event별 feature/action/receipt 1, outcome 5
- sanitized manifest/streams/summary
- model별 독립 D1 분석과 D0+D1 slope-only frozen static policy
- pair post-run red audit

## 중단 조건과 금지 사항

- D0의 instruction과 같은 session/repository/resource/read-only/firewall
  중단 조건을 적용한다.
- D1 slice가 `[8:20]`이 아니거나 pilot/D0 case와 겹치면 중단한다.
- Static policy에 outcome을 사용하거나 D1 positive를 gain/GO/MV-2로
  해석하는 것을 금지한다.

---

# C1 추가 instruction — `session01-mv1mix-c1-pair-v1`

## 명령 메타데이터

- 대상 서버: `server1`
- 대상 Codex SH session ID: 미지정 — 다른 repo/session 및 미등록 SH 실행 금지
- GH session ID: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- repository CWD / identity:
  `/mnt/raid5/janghj/ODE-edit` / `hyunjun1127/ODE-edit`
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_c1_v1/`,
  `experiment-reports/servers/server1/`,
  `audits/servers/server1/`

## 목적과 배경

- proposal에서 온 내용: same-snapshot rewrite-side signal이 실제
  allocation opportunity를 식별하는지 먼저 검증하고, 실패하면 큰 ODE
  실험 전에 controller 방향을 kill 또는 pivot한다.
- repo/protocol에서 확인한 사실: D0+D1은 calibration 전용이며, C1은
  outcome-blind hash split의 고정 fold `0`, model별 12 case에서 adaptive
  `score_mix`와 frozen `static_mix`를 같은 `C`로 비교하는 첫 held-out
  diagnostic이다.
- GH 추정: calibration forecast는 기대 gap의 사전 추정일 뿐 method gain,
  benchmark 개선 또는 MV-2 승인값이 아니다.
- 사용자 확인 필요: 없음. 사용자는 필수 감사만 거쳐 Llama/Qwen을 순차가
  아니라 함께 빠르게 제출하라고 명시했다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_c1_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: 위 완료 보고 경로
- Slurm 제출 허용 여부: preflight와 D1 post-run gate를 통과한 GH
  `submit_session01_mv1mix_c1_pair_server1.sh` one-shot에만 `allowed`;
  미등록 SH의 submit/retry/cancel은 `not allowed`
- GPU cap: server1 project 동시 최대 3; 이번 parent 총 2, child별 1.
  overflow면 `pending_resource_cap`으로 중단
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-31-session01-mv1mix-c1-execution-preflight.md`의
  exact `PASS`와
  `audits/global/2026-07-31-mv1mix-d1-pair-v1.postrun.md`의 exact
  `C1 PREPARE`; `warn`은 GH가 명시적으로 core signal과 무관함을 기록한
  경우만 허용하고 `block`은 제출 금지
- artifact broadcast 의무: active peer clone이 없어 즉시 broadcast
  예외를 완료 보고에 기록한다. peer 활성화 후 검증된 protocol helper만
  사용하며 `rsync --delete`는 금지한다.
- Codex session boundary: SH가 생기면 이 section과
  `servers/active/server1.md`에 그 SH 전용 ID를 먼저 등록하고
  `scripts/check-session-boundary.sh <server1-SH-session-id>`를 통과해야
  한다. 현재는 GH one-shot helper의 GH session exact match만 허용한다.

## 예상 산출물

- model별 exact 12 event, event별 feature/action/receipt/direct-z 1,
  exact 6-arm outcome으로 총 72 outcome
- sanitized manifest/streams/summary와 local-only raw direct-z
- model별 별도 agent의 한국어 분석 및 small JSON
- pair-level red post-run 판정과 calibration forecast 대비 held-out gap
- 예상효과 표기는 `adaptive - frozen_static` raw progress unit로만 하며,
  retention·benchmark accuracy·완성된 ODE-Edit gain으로 환산하지 않는다.

## 중단 조건

- session/CWD/repository/job/run/fold/model/resource envelope 불일치
- 두 child가 같은 allocation에서 동시에 시작하지 않음
- policy 또는 forecast가 tracked·clean·pre-outcome frozen 상태가 아님
- exact 12-case fold나 six-arm order 불일치
- outcome leakage, equal-`C`, receipt ordering, rollback/replay/hash/firewall 위반
- moments/projector 재계산·수정·download·deserialize 또는 EasyEdit write
- output/marker 중복, dirty/unpushed Git, active duplicate job, child nonzero

## 금지 사항

- 다른 repository 또는 다른 Codex session의 inbox/job/artifact 조작
- 미등록 SH의 추가 제출·재제출·취소
- raw prompt/target/logit/generation, `.pt`, weights, checkpoint, dataset,
  full log 및 credential의 Git 유입
- online dependency/model download, EasyEdit 수정, cache 재계산
- C1 단일 model/calibration forecast를 cross-model gain, ODE dynamics,
  paper GO 또는 즉시 MV-2 승인으로 과대해석

---

# Untouched 추가 instruction — `session01-mv1mix-untouched-pair-v1`

## 명령 메타데이터

- 대상 서버 / repository:
  `server1` / `/mnt/raid5/janghj/ODE-edit` / `hyunjun1127/ODE-edit`
- 대상 Codex SH session ID: 미지정 — 다른 repo/session 및 미등록 SH 실행 금지
- GH session ID: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv1mix_*_untouched_v1/`,
  `experiment-reports/servers/server1/`,
  `audits/servers/server1/`

## 목적과 배경

- proposal에서 온 내용: 작은 held-out diagnostic에서 same-snapshot routing
  signal이 살아남을 때만 다음 mechanism test를 연다.
- repo/protocol에서 확인한 사실: C1 pair는 technical PASS이며 model별
  analysis에서 양 model 모두 core clear input을 냈다. 실제 실행 권한은
  pair red audit의 exact `UNTOUCHED PREPARE`가 있을 때만 열린다.
- GH 추정: untouched 20은 C1보다 약 1.67배 많은 event다. C1 wall time을
  단순 환산하면 Llama 약 1시간 45분, Qwen 약 2시간 20분이며 `08:00:00`
  안이다.
- 사용자 확인 필요: 없음. 사용자는 필수 감사 후 두 model을 순차가 아니라
  함께 빠르게 제출하라고 명시했다.

## 실행 권한 envelope

- 허용 write path:
  - raw: `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_untouched_v1/`
  - log/state: `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: 위 완료 보고 경로
- Slurm 제출 허용 여부: pair post-run과 untouched preflight exact gate를
  통과한 GH `submit_session01_mv1mix_untouched_pair_server1.sh` one-shot에만
  `allowed`; 미등록 SH submit/retry/cancel은 `not allowed`
- GPU cap: server1 project 동시 최대 3; 이번 parent 총 2, child별 1.
  Overflow면 `pending_resource_cap`으로 중단
- host-memory cap / 요청: GPU당 `198117 MiB`; parent `130000M`,
  child별 `65000M`
- red-team gate:
  `audits/global/2026-07-31-mv1mix-c1-pair-v1.postrun.md`의 exact
  `UNTOUCHED PREPARE`와
  `audits/global/2026-07-31-session01-mv1mix-untouched-execution-preflight.md`의
  exact `PASS`; `block`은 제출 금지
- artifact broadcast 의무: active peer clone이 없어 즉시 broadcast
  예외를 완료 보고에 기록한다. Peer 활성화 뒤 protocol helper만 사용하며
  destructive mirror는 금지한다.
- Codex session boundary: SH가 생기면 SH 전용 ID를 이 inbox와
  `servers/active/server1.md`에 먼저 등록하고
  `scripts/check-session-boundary.sh <server1-SH-session-id>`를 통과해야
  한다. 현재는 GH one-shot helper의 GH ID exact match만 허용한다.

## 예상 산출물

- canonical untouched split exact 20 event/model
- event별 feature/action/receipt/direct-z 1, six-arm outcome 6으로
  model별 총 120 outcome
- sanitized manifest/streams/summary, model별 독립 한국어 분석,
  pair-level final MV-1 red audit
- C1과 같은 `adaptive - frozen_static` raw progress unit, replay/oracle/
  forecast summary; benchmark %, retention, ODE gain으로 환산 금지

## 중단 조건

- session/CWD/repository/job/run/model/resource envelope 불일치
- 두 child가 같은 allocation에서 동시에 시작하지 않음
- selected split이 canonical untouched exact 20이 아니거나 이미 개봉됨
- C1과 policy hash/q/six-arm/controller/threshold/seed가 다름
- fold1을 함께 실행하거나 C1 outcome으로 policy를 refit/retune함
- outcome leakage, equal-`C`, receipt ordering, rollback/hash/firewall 위반
- moments/projector 재계산·수정·download·deserialize 또는 EasyEdit write
- output/marker 중복, dirty/unpushed Git, active duplicate job, child nonzero

## 금지 사항

- 다른 repository/Codex session 조작, 미등록 SH의 제출·재제출·취소
- raw prompt/target/logit/generation, `.pt`, weights, checkpoint, dataset,
  full log 및 credential의 Git 유입
- online dependency/model download, EasyEdit/cache 수정·재계산
- untouched 결과 전 MV-2, ODE dynamics, retention 또는 paper GO 주장

---

# MV-2 refresh 추가 instruction — `session01-mv2refresh-pair-v1`

## 명령 메타데이터

- 대상 서버 / repository:
  `server1` / `/mnt/raid5/janghj/ODE-edit` / `hyunjun1127/ODE-edit`
- 대상 server-head Codex session ID:
  미지정 — 현재 SH 없음; 다른 repo/session 및 미등록 SH 실행 금지
- GH Codex session ID:
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `runs/mv2refresh_*_e0_v1/`,
  `experiment-reports/servers/server1/`,
  `audits/servers/server1/`

## 목적과 배경

- proposal에서 온 내용: fixed direct-z 아래 partial update 뒤의
  non-stationarity가 actionable한지, stale W0 direction/coefficient와
  refreshed W1 direction/coefficient를 분리해 검증한다.
- repo/protocol에서 확인한 사실: MV-1 untouched pair는 양 model 모두 locked
  clear이고 pair red audit의 exact `MV2 PREPARE`를 통과했다. MV-2는
  outcome-blind 사전등록의 exact 12 case/model, `h=1/2`, `q=1/256`,
  equal-`C` six-arm diagnostic이다.
- GH 추정: MV-1은 allocation 방향만 재현했다. ODE/relinearization의 필요성은
  아직 검증되지 않았으며 MV-2에서 direction refresh가 null이면 즉시
  coefficient/static controller로 pivot해야 한다.
- 사용자 확인 필요: 없음. 사용자는 필수 감사만 거쳐 Llama/Qwen을 순차가
  아니라 함께 빠르게 제출하라고 명시했다.

## 실행 권한 envelope

- 허용 write path:
  - raw:
    `local/results/raw/session01_motivation/mv2refresh_{llama,qwen}_e0_v1/`
  - log/state:
    `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/`
  - small report: 위 완료 보고 경로와
    `experiment-reports/global/`, `audits/global/`
- Slurm 제출 허용 여부:
  pair post-run과 MV-2 preflight exact gate를 통과한 GH
  `project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`
  one-shot에만 `allowed`; 미등록 SH의 submit/retry/cancel은 `not allowed`
- GPU cap:
  server1 project 동시 최대 3; 이번 parent 총 2, child별 1.
  parent `2 GPU / 16 CPU / 130000M / 08:00:00`, child별
  `1 GPU / 8 CPU / 65000M`; overflow면 `pending_resource_cap`
- red-team gate:
  `audits/global/2026-07-31-mv1mix-untouched-pair-v1.postrun.md`의 exact
  `MV2 PREPARE`와
  `audits/global/2026-07-31-session01-mv2-execution-preflight.md`의 exact
  `PASS`; 어느 하나라도 없거나 ambiguous하면 제출 금지
- artifact broadcast 의무:
  active peer clone이 없어 즉시 broadcast 예외를 완료 보고에 기록한다.
  Peer 활성화 뒤 protocol helper만 사용하고 destructive mirror는 금지한다.
- 완료 보고 경로:
  위 명령 metadata 경로와
  `experiment-reports/global/2026-07-31-mv2refresh-*-e0-v1-analysis.md`,
  `audits/global/2026-07-31-mv2refresh-pair-v1.postrun.md`

## 예상 산출물

- canonical salted rank `[100:112]`, 기존 first 100과 disjoint한 exact
  12 event/model
- event별 feature/action/analysis-case/receipt/direct-z 1, exact six-arm
  outcome 6으로 model별 총 72 outcome
- W0 direct-z one-compute lineage, W1 descendant receipt, matched second-`C`,
  exact rollback/RNG, replay/`h=0` sham validation
- model별 독립 한국어 분석과 compact summary, 별도 pair red audit
- 기대효과:
  `A-C` total refresh absolute rewrite-utility와
  direction/coefficient decomposition, bootstrap CI, controller incremental
  proposal/probe 비용. Downstream accuracy나 paper 성능으로 환산 금지

## 중단 조건

- session/CWD/origin/branch/HEAD/job/run/model/resource envelope 불일치
- 두 child가 같은 parent allocation에서 동시에 시작하지 않음
- salted next12가 first100과 겹치거나 selection/source hash 불일치
- direct-z recompute, target token/context/request/lineage mismatch
- covariance cache 재계산·download·write 또는 projector deserialize
- action receipt 전에 outcome 접근, unequal `C`, non-finite metric,
  rollback/RNG/firewall failure
- output/marker 중복, dirty/unpushed Git, active duplicate job, child nonzero
- 어느 child든 실패하면 fail-fast supervisor가 sibling을 종료하며 재제출은
  새 red 승인 전 금지

## 금지 사항

- 다른 repository, 다른 서버의 SH Codex session, 다른 task/job/artifact 조작
- 미등록 SH의 제출·재제출·취소
- raw prompt/target/logit/generation, `.pt`, weights, checkpoint, dataset,
  full log 및 credential의 Git 유입
- EasyEdit/model cache/dataset/stats/projector 수정, online dependency/model
  download
- MV-2 단일 model 또는 oracle을 pair success, ODE-Edit 우위, retention,
  일반화, novelty, paper GO로 과대해석

## GH 직접 제출 예외 기록

- 사유:
  현재 server1 SH가 없고 사용자가 time-critical한 Llama/Qwen 동시 제출을
  명시했다. `PROTOCOL.md:506-509`의 최소 범위 예외를 적용한다.
- exact command:
  `project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`
- 영향 범위:
  server1의 단일 one-shot parent allocation
  `2 GPU / 16 CPU / 130000M`, 두 child 동시 실행, local-only raw artifact
- 후속 보고:
  위 완료 보고 경로, model별 독립 analysis, pair red post-run audit,
  no-peer artifact-broadcast exception

---

# MV-2 technical retry 보충 instruction — `session01-mv2refresh-pair-v1-retry1`

## 목적과 배경

- proposal에서 온 내용:
  최초 lock과 동일한 refresh/stale direction·coefficient 분리 diagnostic을
  기술적으로 완주한다.
- repo/protocol에서 확인한 사실:
  job `15597`은 Llama first-case `RuntimeError`로 scientific commitment 전
  fail-closed했고 Qwen은 sibling fail-fast로 종료됐다. partial result는
  해석하지 않는다.
- GH 추정:
  synchronous proposal의 missing inference guard가 memory amplification의
  1순위 원인이다. exact stack이 없으므로 technical retry에서 safe
  stack-location 계측을 유지한다.
- 사용자 확인 필요:
  없음. 빠른 동시 pair 실행 지시 범위 안의 technical recovery다.

## 실행 권한 envelope

- 허용 write path:
  canonical raw
  `local/results/raw/session01_motivation/mv2refresh_{llama,qwen}_e0_v1/`,
  failed archive
  `local/results/raw/session01_motivation/failed/mv2refresh_pair_v1_job15597/`,
  `local/logs/slurm/session01_motivation/`,
  `local/state/slurm-submissions/session01_motivation/`,
  small report는 기존 global/server1 완료 보고 경로
- Slurm 제출 허용 여부:
  retry audit exact `PASS`, clean pushed main, failed raw archive 뒤 GH exact
  helper 1회만 `allowed`; 미등록 SH submit/retry/cancel은 `not allowed`
- GPU cap:
  server1 최대 3; parent 총 2, child별 1,
  parent `16 CPU / 130000M / 08:00:00`, child별
  `8 CPU / 65000M`
- red-team gate 통과 조건:
  기존 MV-1 `MV2 PREPARE`, 최초 MV-2 preflight `PASS`,
  `audits/global/2026-07-31-session01-mv2-technical-retry-preflight.md`의
  exact retry `PASS`, independent patch review residual P1/P2 없음
- artifact broadcast 의무:
  active peer clone이 없으므로 no-peer 예외를 완료 보고에 기록한다.
  peer 활성화 뒤 protocol helper만 사용한다.
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `experiment-reports/global/2026-07-31-mv2refresh-*-e0-v1-analysis.md`,
  `audits/global/2026-07-31-mv2refresh-pair-v1.postrun.md`
- 금지 사항:
  EasyEdit/cache/dataset/stats 수정·재계산·download, raw Git 유입,
  case/model/arm/q/h/seed/bootstrap/threshold 변경, partial rescue,
  다른 repo/session/job 조작
- 예상 산출물:
  최초 lock과 동일한 12 event/model, six-arm outcome 72/model,
  model별 독립 compact analysis와 pair red 판정; safe local stack은
  technical failure 때만 생성
- 중단 조건:
  output/retry marker 중복, dirty/unpushed main, session/cap mismatch,
  child 비동시 시작, safe trace leakage, technical gate 위반, child nonzero

## GH 직접 technical retry 예외 기록

- 사유:
  server1 SH 부재, 사용자의 time-critical 동시 실행 지시, scientific
  commitment 전 fail-closed한 job `15597` 복구
- exact archive command 범위:
  failed 두 run directory를 위 explicit ignored archive directory로
  `mv`; 삭제·덮어쓰기 금지
- exact submit command:
  `project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`
- 영향 범위:
  failed local raw의 recoverable relocation, server1 single retry parent
  `2 GPU / 16 CPU / 130000M`; scientific contract 변화 없음
- 후속 보고:
  retry job ID/state/resource, safe trace 여부, model별 독립 analysis,
  pair red post-run, no-peer broadcast exception

---

# MV-2 tensor-hash repair 보충 instruction — `session01-mv2refresh-pair-v1-repair`

## 목적과 배경

- proposal에서 온 내용:
  최초 scientific lock과 동일한 MV-2 diagnostic을 완주한다.
- repo/protocol에서 확인한 사실:
  job `15600` safe stack은 failure를
  `hooks.tensor_sha256()`의 CUDA-side byte view로 특정했다. scientific
  feature/action/outcome/receipt는 생성되지 않았다.
- GH 추정:
  없음. tensor hash plumbing line은 safe stack으로 확인됐다.
- 사용자 확인 필요:
  없음. time-critical 동시 pair technical recovery다.

## 실행 권한 envelope

- 허용 write path:
  canonical
  `local/results/raw/session01_motivation/mv2refresh_{llama,qwen}_e0_v1/`,
  failed archive
  `local/results/raw/session01_motivation/failed/mv2refresh_pair_v1_job15600/`,
  기존 local log/state 및 small completion-report path
- Slurm 제출 허용 여부:
  tensor-hash repair audit exact `PASS`, clean pushed main 뒤 GH helper 1회만
  `allowed`; 미등록 SH submit/retry/cancel은 `not allowed`
- GPU cap:
  server1 최대 3; parent 총 2, child별 1,
  `16 CPU / 130000M / 08:00:00`, child별 `8 CPU / 65000M`
- red-team gate 통과 조건:
  prior three exact gates와 새 tensor-hash repair exact `PASS`,
  full tests, actual-CUDA synthetic parity preflight, independent review
  residual P1/P2 없음
- artifact broadcast 의무:
  peer clone 부재 no-peer 예외를 완료 보고에 기록하며 다른 repo/session을
  건드리지 않는다.
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  기존 model별 global analysis path와 pair post-run audit
- 금지 사항:
  EasyEdit/cache/dataset/stats 수정·재계산, raw Git 유입, scientific lock
  변경, partial rescue, 다른 repo/session/job 조작
- 예상 산출물:
  model load 전 synthetic CUDA hash parity 통과 후 exact 12 event/model,
  72 outcome/model, 독립 model analysis 및 pair red verdict
- 중단 조건:
  CUDA parity 실패, output/repair marker 중복, dirty/unpushed main,
  cap/session mismatch, child 비동시 시작, child nonzero

## GH 직접 repair execution 예외 기록

- 사유:
  SH 부재, 사용자의 빠른 동시 실행 지시, safe stack으로 확정된 scientific
  commitment 전 hashing plumbing failure 복구
- exact archive command 범위:
  job `15600`의 두 failed run directory를 위 ignored archive로 `mv`;
  삭제·덮어쓰기 금지
- exact submit command:
  `project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`
- 영향 범위:
  failed local raw recoverable relocation, synthetic hash parity와 server1
  one parent `2 GPU / 16 CPU / 130000M`; scientific contract 변화 없음
- 후속 보고:
  repair job ID/state/resource, CUDA parity/trace 여부, model별 독립 analysis,
  pair red post-run, no-peer broadcast exception

---

# MV-2 CUDA hash RCA diagnostic instruction — `session01-mv2-hashdiag-v1`

## 목적과 배경

- 목적:
  actual proposal에서만 재발한 CUDA/hash `RuntimeError`를 과학 결과 없이
  exact originating operation 또는 fixed safe category로 좁힌다.
- repo/protocol에서 확인한 사실:
  job `15603`도 feature/action/outcome/receipt 전 fail-closed했다.
- GH 추정:
  async CUDA fault 또는 actual factor size/layout-specific transfer 문제다.
- 사용자 확인 필요:
  없음. pair 반복 제출보다 작은 time-critical technical RCA다.

## 실행 권한 envelope

- 허용 write path:
  Llama canonical raw, local Slurm log/state, 종료 후 explicit ignored failed
  archive, small technical audit/report
- Slurm 제출 허용 여부:
  exact diagnostic audit `PASS`와 clean pushed main 뒤 GH diagnostic helper
  1회만 `allowed`
- GPU cap:
  server1 최대 3 중 `1 GPU / 8 CPU / 65000M / 00:12:00`
- red-team gate:
  category non-leakage, `CUDA_LAUNCH_BLOCKING=1`, Llama-only/12분 cap,
  independent review residual P1/P2 없음
- artifact broadcast:
  peer clone 부재 no-peer 예외; 다른 repo/session 조작 금지
- 완료 보고 경로:
  `audits/global/2026-07-31-session01-mv2-cuda-hash-diagnostic-preflight.md`
  및 server1 completion message
- 금지 사항:
  Qwen 실행, scientific analysis/claim, EasyEdit/cache/data write,
  raw exception/request/target persist, direct untracked command
- 예상 산출물:
  safe stack originating line 또는 fixed tensor-hash
  phase/category/dtype/shape/device/numel, terminal/resource state
- 중단 조건:
  1 GPU/12분 cap 초과, safe field 외 leakage, dirty/unpushed Git,
  marker/output/active duplicate, session mismatch

## GH 직접 diagnostic 예외 기록

- 사유:
  SH 부재, repeated scientific-commitment-before failure, pair GPU 낭비 방지
- exact command:
  `project/run_scripts/submit_session01_mv2refresh_hashdiag_server1.sh`
- 영향 범위:
  server1 one Llama technical allocation; Qwen/scientific result 없음
- 후속 보고:
  diagnostic job ID, safe category/line, resource, failed raw archive,
  별도 root-cause fix 및 final-pair red gate

---

# MV-2 singleton-stride final repair instruction — `session01-mv2refresh-pair-v1-retry3`

## 목적과 배경

- 목적:
  exact root cause인 `(N,1)` singleton stride byte-view bug만 고치고 최초
  scientific lock의 Llama/Qwen pair를 완주한다.
- repo/protocol에서 확인한 사실:
  job `15607`은 `cpu_byte_view/tensor_layout`, float64 `(14336,1)`을
  기록해 원인을 확정했다.
- GH 추정:
  없음. pathological stride는 unit test로 재현된다.
- 사용자 확인 필요:
  없음. 빠른 동시 pair technical recovery다.

## 실행 권한 envelope

- 허용 write path:
  canonical two raw dirs, 기존 local log/state, job15607 ignored archive,
  small analysis/audit/completion paths
- Slurm 제출 허용 여부:
  singleton-stride repair exact `PASS`, clean pushed main 뒤 GH pair helper
  retry3 1회만 `allowed`
- GPU cap:
  server1 최대 3; parent `2 GPU / 16 CPU / 130000M / 08:00:00`,
  child별 `1 GPU / 8 CPU / 65000M`
- red-team gate:
  exact pathological regression, full tests, all prior marker/archive,
  independent review residual P1/P2 없음
- artifact broadcast:
  peer clone 부재 no-peer 예외; 다른 repo/session 조작 금지
- 완료 보고 경로:
  기존 model별 global analysis, pair post-run, server1 completion message
- 금지 사항:
  scientific lock 변경, EasyEdit/cache/data write, raw Git 유입,
  partial-result rescue, 다른 repo/session/job 조작
- 예상 산출물:
  exact 12 event/72 outcome per model, model별 독립 analysis/compact summary,
  pair red verdict와 refresh expected-effect/cost
- 중단 조건:
  hash/layout/category error 재발, output/retry3 marker 중복,
  dirty/unpushed main, cap/session mismatch, child 비동시 시작/nonzero

## GH 직접 final repair 예외 기록

- 사유:
  SH 부재, user time-critical simultaneous request, exact first-case plumbing
  root cause 확정과 regression closure
- exact submit command:
  `project/run_scripts/submit_session01_mv2refresh_pair_server1.sh`
- 영향 범위:
  server1 one simultaneous pair `2 GPU / 16 CPU / 130000M`;
  scientific contract 변화 없음
- 후속 보고:
  retry3 job ID/state/resource, safe trace 여부, model별 독립 analysis,
  pair red post-run, no-peer broadcast exception

---

# Motivation quarter-step pair instruction — `session01-qstep4-pair-v1`

## 목적과 배경

- proposal에서 온 내용:
  ODE-Edit의 Motivation인 경로 중 방향 재계산 이점을 작고 엄격하게
  검증한다.
- repo/protocol에서 확인한 사실:
  이전 MV-2의 첫 state treatment는 native C-distance의 약 `D/32`였고
  Llama/Qwen direction 결론이 갈렸다. 사용자가 이 under-manipulation
  가능성을 검토해 native update를 약 1/4씩 나누도록 명시했다.
- GH 추정:
  더 큰 `D/4` state 변화라면 실제 direction rotation 신호와 수치 잡음을
  구분하기 쉬워질 수 있다.
- 사용자 확인 필요:
  없음. 동시 Llama/Qwen 제출과 분석까지 명시적으로 요청됐다.

정확한 scientific contract는
`plans/global/2026-07-31-session01-quarter-step-implementation-spec.md`를
따른다. 기존 MV-2 결과를 덮어쓰지 않는 fresh rank `[112:124]` 독립
Motivation 진단이다.

## 실행 권한 envelope

- 허용 write path:
  - raw:
    `local/results/raw/session01_motivation/qstep4_llama_f0_v1/`,
    `local/results/raw/session01_motivation/qstep4_qwen_f0_v1/`
  - compact local analysis:
    `local/results/analysis/session01_motivation/qstep4_{llama,qwen}_f0_v1/`
  - logs/state:
    `local/logs/slurm/session01_motivation/`,
    `local/state/slurm-submissions/session01_motivation/qstep4_pair_v1.submitted/`
  - small completion report/audit는 아래 완료 보고 경로만 허용
- Slurm 제출 허용 여부:
  `audits/global/2026-07-31-session01-quarter-step-execution-preflight.md`의
  exact `PASS`, clean pushed `main`, session/cap check 뒤 GH one-shot helper
  `project/run_scripts/submit_session01_qstep4_pair_server1.sh` 1회만
  `allowed`; 미등록 SH의 별도 submit/retry/cancel은 `not allowed`
- GPU cap:
  server1 cap `3`; 이번 parent는 `2 GPU / 16 CPU / 130000M /
  12:00:00`, child별 `1 GPU / 8 CPU / 65000M`; active project GPU와 합이
  3을 넘거나 memory cap을 넘으면 제출하지 않고 pending
- red-team gate 통과 조건:
  fresh-case disjointness, direct-z 1회 고정, exact four-hop lineage,
  per-hop `D/4`/`E_native/16`, probe `D/64`, receipt-before-outcome,
  no best-hop selection, native one-shot/split4 control, full tests와 shell
  syntax가 모두 pass해야 한다. `warn`은 GH가 기록한 비차단 항목만 진행,
  `block` 또는 residual P1/P2면 중단
- artifact broadcast 의무:
  현재 active peer clone/SH가 없으므로 no-peer 예외를 완료 보고에
  명시한다. peer 활성화 뒤에는 protocol의
  `scripts/rsync-artifact-broadcast.sh`만 사용하며 다른 repo/session을
  건드리지 않는다.
- 완료 보고 경로:
  `messages/server-heads/server1/2026-07-31.md`,
  `experiment-reports/global/2026-07-31-qstep4-llama-f0-v1-analysis.md`,
  `experiment-reports/global/2026-07-31-qstep4-qwen-f0-v1-analysis.md`,
  `audits/global/2026-07-31-qstep4-pair-v1.postrun.md`
- Codex session boundary:
  server1 SH session ID는 아직 **미등록**이며 SH가 생기면 해당 server별
  registry에 별도 ID를 등록해야 한다. 요구 profile은 primary SH
  `Sol Ultra` (`gpt-5.6-sol`, `ultra`), CWD
  `/mnt/raid5/janghj/ODE-edit`, repository
  `hyunjun1127/ODE-edit`; 불일치하거나 다른 repo session이면 즉시 중단.
  이번 GH one-shot 실행의 session ID는
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`이다. delegated 분석 agent는
  runtime metadata로 확인된 `Terra Ultra`만 허용한다.
- 금지 사항:
  EasyEdit/cache/dataset/covariance/projector 수정·재계산·download, raw
  Git 유입, case/model/layer/K/hop/probe/seed/bootstrap/threshold 변경,
  partial-result rescue, best-hop 선택, AlphaEdit 혼합, 다른 repo/session/job
  조작, credential/private connection 정보 기록
- 예상 산출물:
  model별 exact 12 event, 12 feature/action/receipt/analysis row, 156 outcome,
  technical-valid compact analysis, step-4 A4-B4/B4-C4/A4-C4,
  A4-native proxy, native split noise control, pair post-run verdict
- 중단 조건:
  output/marker/active job 중복, dirty/unpushed main, session/cap mismatch,
  child 비동시 시작, 한 child nonzero, lineage/C budget/rollback/firewall/
  receipt/native-control 위반, unsafe raw leakage, 12시간 또는 resource cap
  초과

## GH 직접 one-shot 제출 예외 기록

- 사유:
  server1 SH가 아직 없고 사용자가 빠른 Llama/Qwen 동시 제출과 분석을
  명시했다. 이는 protocol 524--527행의 explicit user/time-critical 예외다.
- exact submit command:
  `project/run_scripts/submit_session01_qstep4_pair_server1.sh`
- 영향 범위:
  server1에서 ODE-Edit parent allocation 한 건(`2 GPU / 16 CPU /
  130000M`)과 위 explicit ignored local output/log/state path만 사용
- 후속 보고 경로:
  job ID/state/resource, child 동시 시작, raw integrity/hash, model별 compact
  analysis, pair red post-run, no-peer artifact broadcast 예외를 위 완료
  경로에 기록

## Pre-run env-parser repair retry 보충

- repo/protocol에서 확인한 사실:
  original job `15700`은 `00:00:03`, `FAILED 1:0`이며 scientific runner와
  output 생성 전에 canonical session env의 quoted `Sol Ultra` 값을 새
  wrapper가 거부했다. sibling은 pair fail-fast로 종료됐다.
- 허용 write path:
  기존 qstep raw/log/state path와 새 one-shot marker
  `local/state/slurm-submissions/session01_motivation/qstep4_pair_v1_retry1.submitted/`
- Slurm 제출 허용 여부:
  `audits/global/2026-07-31-session01-quarter-step-env-parser-repair-preflight.md`
  exact `PASS`, clean pushed main, original marker numeric ID와 exact Slurm
  `FAILED`, no canonical outputs/active qstep job, session/cap 재검증 뒤 같은
  helper retry1 1회만 `allowed`
- GPU/memory cap:
  변경 없음. server1 cap `3`, retry parent `2 GPU / 130000M`; overflow면
  pending
- red-team gate:
  exact 두 `Sol Ultra` quoted assignment만 허용하고 arbitrary quote/expansion은
  거부, 두 profile 값을 load 후 재검사, scientific code/contract diff 없음
- artifact broadcast/완료 보고:
  기존 no-peer 예외 및 동일 completion path 사용; original failure와 retry
  job ID를 함께 보고
- 금지 사항:
  env 파일 수정, generic quote parser 도입, scientific retuning, manual
  untracked sbatch, original marker/log 삭제, 다른 repo/session/job 조작
- 예상 산출물:
  retry child 둘이 parser/session/Git gate를 통과해 동시에 scientific
  runner로 진입하고 기존 exact qstep 산출물을 생성
- 중단 조건:
  original state가 exact `FAILED`가 아님, retry marker/output/active job
  중복, parser/profile/session/cap mismatch, child nonzero
- GH 직접 retry 사유/명령/영향:
  SH 부재와 사용자 time-critical 지시 아래 scientific pre-run wrapper
  defect만 고친다. exact command는
  `project/run_scripts/submit_session01_qstep4_pair_server1.sh`, 영향은 동일
  server1 pair allocation 한 건과 새 retry marker뿐이다.
